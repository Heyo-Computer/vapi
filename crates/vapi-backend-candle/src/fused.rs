//! Small fused element-wise kernels for the decode step.
//!
//! `swiglu` takes the output of a fused gate+up projection, `h = x · [W1; W3]ᵀ`
//! of shape `(rows, 2n)`, and returns `silu(h[:, :n]) * h[:, n:]` in one pass.
//! Two things motivate it. Issuing the gate and up projections as one GEMM
//! with twice the output width lets cuBLAS pick a far better kernel at
//! decode batch sizes (at M=64 the separate `[10752, 2048]` GEMMs run at
//! 217 GB/s; the fused one at 371 GB/s). And splitting the result with
//! `narrow` would hand candle strided views, whose kernels upload shape
//! metadata from a temporary host buffer at launch, which a CUDA graph
//! capture records as a dangling copy. This kernel takes every argument by
//! value, like `rows.rs`. Off CUDA, or for other dtypes, the candle ops run.

use candle_core::{Result, Tensor};

/// `silu(h[:, :n]) * h[:, n:]` for a contiguous `(rows, 2n)` tensor.
pub fn swiglu(h: &Tensor, n: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if h.device().is_cuda() && cuda::supported(h) {
        return h.apply_op1_no_bwd(&cuda::SwiGlu { n });
    }
    let gate = h.narrow(1, 0, n)?;
    let up = h.narrow(1, n, n)?;
    candle_nn::ops::silu(&gate)? * up
}

/// The short conv's cache write: `table[slots[i]] = B_i * x_i` where `B`
/// and `x` are the first and third `hidden`-wide column blocks of `h3`, the
/// `(rows, 3·hidden)` output of the fused `in_proj` GEMM. `in_bias`, when
/// present, is the `(3·hidden,)` projection bias, added here rather than
/// with a broadcast op so nothing strided is issued. CUDA only.
#[cfg(feature = "cuda")]
pub fn conv_write(
    h3: &Tensor,
    table: &Tensor,
    slots: &Tensor,
    in_bias: Option<&Tensor>,
    hidden: usize,
) -> Result<()> {
    cuda::conv_write(h3, table, slots, in_bias, hidden)
}

/// The short conv's read side: for each row, `C_i * (Σ_k w[k] *
/// table[idx[k, i]] + bias)`, with `C` the middle column block of `h3`,
/// `idx` an `(L, rows)` u32 table where [`CONV_PAD`] marks a tap before the
/// sequence start (contributes zero), and `w` the `(L, hidden)` taps. CUDA
/// only.
#[cfg(feature = "cuda")]
pub fn conv_apply(
    h3: &Tensor,
    table: &Tensor,
    idx: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    in_bias: Option<&Tensor>,
    hidden: usize,
) -> Result<Tensor> {
    cuda::conv_apply(h3, table, idx, w, bias, in_bias, hidden)
}

/// Sentinel slot index for a conv tap that precedes its sequence's start.
pub const CONV_PAD: u32 = u32::MAX;

/// Most candidates `select_candidates` returns per row.
pub const CANDIDATE_CAP: usize = 8192;

/// The sampler's candidate window in scaled nats (`vapi_engine`'s
/// `CANDIDATE_CUTOFF`): a row whose selection reaches this depth is complete.
pub const CANDIDATE_WINDOW: u32 = 30;

/// Per-row candidate selection on the device, in one launch over the
/// `(rows, vocab)` f32 logits. For each row the kernel finds the maximum,
/// bins every token by its scaled gap `(max - logit) × inv_t` into whole
/// nats, and compacts every token whose bin lies within the deepest prefix
/// of bins that fits in [`CANDIDATE_CAP`]. The result for a row is therefore
/// exactly its top `count` tokens, with `depth` the last bin included
/// (`CANDIDATE_WINDOW` means the sampler's whole window is present and the
/// selection is exact for any parameters). It also returns the row's full
/// softmax denominator, `Σ exp(-gap)`, so the host can normalise without
/// the tail. A row whose first bin alone overflows the cap gets `depth ==
/// u32::MAX` and `count` the size of that bin.
#[cfg(feature = "cuda")]
pub struct Selected {
    /// `(rows, CANDIDATE_CAP)` u32; only the first `counts[r]` of a row are set.
    pub ids: Tensor,
    /// `(rows, CANDIDATE_CAP)` f32, the raw logits of `ids`.
    pub vals: Tensor,
    /// `(rows,)` u32.
    pub counts: Tensor,
    /// `(rows,)` u32.
    pub depths: Tensor,
    /// `(rows,)` f32.
    pub maxes: Tensor,
    /// `(rows,)` f32.
    pub sums: Tensor,
}

/// See [`Selected`]. `inv_t` is `(rows,)` f32, `1 / temperature` per row.
#[cfg(feature = "cuda")]
pub fn select_candidates(logits: &Tensor, inv_t: &Tensor) -> Result<Selected> {
    cuda::select_candidates(logits, inv_t)
}

/// One row of a device draw, see [`gumbel_draw`].
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct GumbelRow {
    pub inv_temperature: f32,
    pub seed: u64,
    pub draw: u64,
    /// `(token, count)` pairs to penalise before the draw.
    pub observed: Vec<(u32, u32)>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

/// Gumbel-max draws on the device for the rows of `rows` that are `Some`.
///
/// First the penalties are applied in place to the observed tokens of
/// those rows (one thread per pair), then one block per row computes
/// `argmax_v(logit_v × inv_t + g_v)` with `g_v = -ln(-ln u_v)` and `u_v`
/// from a 64-bit hash of `(seed, draw, v)`. That is an exact sample from
/// `softmax(logit / T)`, deterministic in `(seed, draw)`. Rows given
/// `None` are untouched and their output slot is `u32::MAX`.
///
/// `logits` is modified in place for the penalised rows.
#[cfg(feature = "cuda")]
pub fn gumbel_draw(logits: &Tensor, rows: &[Option<GumbelRow>]) -> Result<Vec<u32>> {
    cuda::gumbel_draw(logits, rows)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use candle_core::cuda::{CudaStorage, CudaStorageSlice, WrapErr};
    use candle_core::{CustomOp1, DType, Layout, Result, Shape, Tensor};
    use half::bf16;

    // bf16 is handled as raw 16-bit words so the source needs no CUDA
    // headers under nvrtc: widen by a shift, narrow with round-to-nearest-even.
    const SRC: &str = r#"
__device__ __forceinline__ float bf2f(unsigned short x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (unsigned short)(u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float silu(float g) { return g / (1.0f + __expf(-g)); }

extern "C" __global__ void swiglu_bf16(
    const unsigned short* h, unsigned short* out, unsigned int n, unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / n, col = i - row * n;
    const unsigned short* r = h + (unsigned long long)row * 2u * n;
    out[i] = f2bf(silu(bf2f(r[col])) * bf2f(r[n + col]));
}

extern "C" __global__ void conv_write_bf16(
    const unsigned short* h3, const unsigned short* in_bias, unsigned short* table,
    const unsigned int* slots, unsigned int hidden, unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / hidden, col = i - row * hidden;
    const unsigned short* r = h3 + (unsigned long long)row * 3u * hidden;
    float b = bf2f(r[col]), x = bf2f(r[2u * hidden + col]);
    if (in_bias) { b += bf2f(in_bias[col]); x += bf2f(in_bias[2u * hidden + col]); }
    table[(unsigned long long)slots[row] * hidden + col] = f2bf(b * x);
}

extern "C" __global__ void conv_apply_bf16(
    const unsigned short* h3, const unsigned short* in_bias, const unsigned short* table,
    const unsigned int* idx, const unsigned short* w, const unsigned short* bias,
    unsigned short* out, unsigned int hidden, unsigned int rows, unsigned int taps,
    unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / hidden, col = i - row * hidden;
    float acc = 0.0f;
    for (unsigned int k = 0; k < taps; k++) {
        unsigned int r = idx[k * rows + row];
        if (r == 0xffffffffu) continue;
        acc += bf2f(table[(unsigned long long)r * hidden + col]) * bf2f(w[k * hidden + col]);
    }
    if (bias) acc += bf2f(bias[col]);
    float c = bf2f(h3[(unsigned long long)row * 3u * hidden + hidden + col]);
    if (in_bias) c += bf2f(in_bias[hidden + col]);
    out[i] = f2bf(c * acc);
}

extern "C" __global__ void conv_write_f32(
    const float* h3, const float* in_bias, float* table,
    const unsigned int* slots, unsigned int hidden, unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / hidden, col = i - row * hidden;
    const float* r = h3 + (unsigned long long)row * 3u * hidden;
    float b = r[col], x = r[2u * hidden + col];
    if (in_bias) { b += in_bias[col]; x += in_bias[2u * hidden + col]; }
    table[(unsigned long long)slots[row] * hidden + col] = b * x;
}

extern "C" __global__ void conv_apply_f32(
    const float* h3, const float* in_bias, const float* table,
    const unsigned int* idx, const float* w, const float* bias,
    float* out, unsigned int hidden, unsigned int rows, unsigned int taps,
    unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / hidden, col = i - row * hidden;
    float acc = 0.0f;
    for (unsigned int k = 0; k < taps; k++) {
        unsigned int r = idx[k * rows + row];
        if (r == 0xffffffffu) continue;
        acc += table[(unsigned long long)r * hidden + col] * w[k * hidden + col];
    }
    if (bias) acc += bias[col];
    float c = h3[(unsigned long long)row * 3u * hidden + hidden + col];
    if (in_bias) c += in_bias[hidden + col];
    out[i] = c * acc;
}

// One thread per (row, token) pair: the sampler's penalties, in place.
extern "C" __global__ void apply_penalties_f32(
    float* logits, unsigned int vocab, const unsigned int* rows, const unsigned int* toks,
    const float* counts, const float* rep, const float* freq, const float* pres,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int r = rows[i];
    float* l = logits + (unsigned long long)r * vocab + toks[i];
    float v = *l, p = rep[r];
    if (p != 1.0f) v = v > 0.0f ? v / p : v * p;
    v -= freq[r] * counts[i];
    if (counts[i] > 0.0f) v -= pres[r];
    *l = v;
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

// One block of 1024 threads per active row: argmax of the tempered logit
// plus Gumbel noise. Ties (practically impossible) go to the lowest id.
extern "C" __global__ void gumbel_argmax_f32(
    const float* logits, unsigned int vocab, const unsigned int* active,
    const float* inv_t, const unsigned long long* seeds, const unsigned long long* draws,
    unsigned int* out)
{
    __shared__ float best_v[1024];
    __shared__ unsigned int best_i[1024];
    const unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (!active[row]) return;
    const float* l = logits + (unsigned long long)row * vocab;
    const float it = inv_t[row];
    const unsigned long long key = mix64(seeds[row] ^ mix64(draws[row] + 0x9E3779B97F4A7C15ULL));
    float bv = -3.402823466e38f;
    unsigned int bi = 0xffffffffu;
    for (unsigned int i = tid; i < vocab; i += 1024) {
        unsigned long long h = mix64(key + (unsigned long long)i * 0xBF58476D1CE4E5B9ULL);
        float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
        float g = -__logf(-__logf(u));
        float v = l[i] * it + g;
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    best_v[tid] = bv; best_i[tid] = bi;
    __syncthreads();
    for (unsigned int s = 512; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = best_v[tid + s]; unsigned int oi = best_i[tid + s];
            if (ov > best_v[tid] || (ov == best_v[tid] && oi < best_i[tid])) {
                best_v[tid] = ov; best_i[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[row] = best_i[0];
}

// One block of 1024 threads per row. Three passes over the row: max;
// per-warp histogram of the scaled gap in whole nats plus the softmax
// sum; compaction of the bins that fit. Bins 0..=30 cover the sampler's
// 30-nat window; bin 31 is everything beyond it.
extern "C" __global__ void select_candidates_f32(
    const float* logits, const float* inv_t, unsigned int vocab, unsigned int cap,
    unsigned int* ids, float* vals, unsigned int* counts, unsigned int* depths,
    float* maxes, float* sums)
{
    __shared__ float red[1024];
    __shared__ unsigned int hist[32][33];
    __shared__ unsigned int cursor;
    __shared__ int depth;
    const unsigned int row = blockIdx.x, tid = threadIdx.x, warp = tid >> 5;
    const float* l = logits + (unsigned long long)row * vocab;
    const float it = inv_t[row];

    float m = -3.402823466e38f;
    for (unsigned int i = tid; i < vocab; i += 1024) m = fmaxf(m, l[i]);
    red[tid] = m;
    __syncthreads();
    for (unsigned int s = 512; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    if (tid < 32) for (int b = 0; b < 33; b++) hist[tid][b] = 0;
    __syncthreads();
    float sum = 0.0f;
    for (unsigned int i = tid; i < vocab; i += 1024) {
        float g = (m - l[i]) * it;
        unsigned int b = g >= 31.0f ? 31u : (unsigned int)g;
        atomicAdd(&hist[warp][b], 1u);
        sum += __expf(-g);
    }
    red[tid] = sum;
    __syncthreads();
    for (unsigned int s = 512; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int cum = 0;
        int d = -1;
        unsigned int first = 0;
        for (int b = 0; b <= 30; b++) {
            unsigned int n = 0;
            for (int w = 0; w < 32; w++) n += hist[w][b];
            if (b == 0) first = n;
            cum += n;
            if (cum <= cap) d = b; else break;
        }
        depth = d;
        cursor = 0;
        counts[row] = d < 0 ? first : 0u;
        depths[row] = d < 0 ? 0xffffffffu : (unsigned int)d;
        maxes[row] = m;
        sums[row] = red[0];
    }
    __syncthreads();
    const int d = depth;
    if (d >= 0) {
        for (unsigned int i = tid; i < vocab; i += 1024) {
            float g = (m - l[i]) * it;
            if (g < (float)(d + 1)) {
                unsigned int slot = atomicAdd(&cursor, 1u);
                if (slot < cap) {
                    ids[(unsigned long long)row * cap + slot] = i;
                    vals[(unsigned long long)row * cap + slot] = l[i];
                }
            }
        }
        __syncthreads();
        if (tid == 0) counts[row] = cursor;
    }
}

extern "C" __global__ void swiglu_f32(
    const float* h, float* out, unsigned int n, unsigned int total)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / n, col = i - row * n;
    const float* r = h + (unsigned long long)row * 2u * n;
    out[i] = silu(r[col]) * r[n + col];
}
"#;

    fn ptx() -> Result<&'static str> {
        static PTX: OnceLock<String> = OnceLock::new();
        if let Some(p) = PTX.get() {
            return Ok(p);
        }
        let compiled = candle_core::cuda::cudarc::nvrtc::compile_ptx(SRC)
            .map_err(|e| candle_core::Error::Msg(format!("nvrtc: {e}")))?;
        Ok(PTX.get_or_init(|| compiled.to_src()))
    }

    pub fn supported(h: &Tensor) -> bool {
        matches!(h.dtype(), DType::BF16 | DType::F32) && h.rank() == 2
    }

    /// Device address of a contiguous tensor's data, or 0 for `None`.
    /// The guard is dropped straight away; callers hold the tensor.
    fn addr(t: Option<&Tensor>, op: &'static str) -> Result<u64> {
        let Some(t) = t else { return Ok(0) };
        let (storage, layout) = t.storage_and_layout();
        let (o1, _) = layout
            .contiguous_offsets()
            .ok_or(candle_core::Error::RequiresContiguous { op })?;
        let candle_core::Storage::Cuda(s) = &*storage else {
            candle_core::bail!("{op}: tensor is not on CUDA");
        };
        let stream = s.device().cuda_stream();
        let elem = t.dtype().size_in_bytes() as u64;
        macro_rules! p {
            ($sl:expr) => {{
                let (ptr, _guard) = $sl.device_ptr(&stream);
                ptr + o1 as u64 * elem
            }};
        }
        Ok(match &s.slice {
            CudaStorageSlice::U32(v) => p!(v),
            CudaStorageSlice::BF16(v) => p!(v),
            CudaStorageSlice::F32(v) => p!(v),
            _ => candle_core::bail!("{op}: unsupported dtype {:?}", t.dtype()),
        })
    }

    fn kernel_name(base: &str, dtype: DType) -> Result<String> {
        Ok(match dtype {
            DType::BF16 => format!("{base}_bf16"),
            DType::F32 => format!("{base}_f32"),
            d => candle_core::bail!("{base}: unsupported dtype {d:?}"),
        })
    }

    fn cuda_device(t: &Tensor) -> Result<candle_core::CudaDevice> {
        match t.device() {
            candle_core::Device::Cuda(d) => Ok(d.clone()),
            _ => candle_core::bail!("fused conv: tensor is not on CUDA"),
        }
    }

    pub fn conv_write(
        h3: &Tensor,
        table: &Tensor,
        slots: &Tensor,
        in_bias: Option<&Tensor>,
        hidden: usize,
    ) -> Result<()> {
        let rows = slots.dim(0)?;
        if h3.dims() != [rows, 3 * hidden] {
            candle_core::bail!(
                "conv_write: h3 {:?} vs {} rows × 3·{hidden}",
                h3.dims(),
                rows
            );
        }
        let dev = cuda_device(h3)?;
        let name = kernel_name("conv_write", h3.dtype())?;
        let func = dev.get_or_load_custom_func(&name, "vapi_fused", ptx()?)?;
        let total = (rows * hidden) as u32;
        let (h3p, bp, tp, sp) = (
            addr(Some(h3), "conv_write")?,
            addr(in_bias, "conv_write")?,
            addr(Some(table), "conv_write")?,
            addr(Some(slots), "conv_write")?,
        );
        let hidden = hidden as u32;
        let mut b = func.builder();
        b.arg(&h3p);
        b.arg(&bp);
        b.arg(&tp);
        b.arg(&sp);
        b.arg(&hidden);
        b.arg(&total);
        // SAFETY: every buffer is a live tensor held by the caller; sizes
        // come from their shapes.
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }.w()?;
        Ok(())
    }

    pub fn conv_apply(
        h3: &Tensor,
        table: &Tensor,
        idx: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        in_bias: Option<&Tensor>,
        hidden: usize,
    ) -> Result<Tensor> {
        let (taps, rows) = idx.dims2()?;
        if h3.dims() != [rows, 3 * hidden] || w.dims() != [taps, hidden] {
            candle_core::bail!(
                "conv_apply: h3 {:?}, idx {:?}, w {:?} disagree for hidden {hidden}",
                h3.dims(),
                idx.dims(),
                w.dims()
            );
        }
        let dev = cuda_device(h3)?;
        let name = kernel_name("conv_apply", h3.dtype())?;
        let func = dev.get_or_load_custom_func(&name, "vapi_fused", ptx()?)?;
        let out = Tensor::zeros((rows, hidden), h3.dtype(), h3.device())?;
        let total = (rows * hidden) as u32;
        let args = [
            addr(Some(h3), "conv_apply")?,
            addr(in_bias, "conv_apply")?,
            addr(Some(table), "conv_apply")?,
            addr(Some(idx), "conv_apply")?,
            addr(Some(w), "conv_apply")?,
            addr(bias, "conv_apply")?,
            addr(Some(&out), "conv_apply")?,
        ];
        let (hidden, rows, taps) = (hidden as u32, rows as u32, taps as u32);
        let mut b = func.builder();
        for a in &args {
            b.arg(a);
        }
        b.arg(&hidden);
        b.arg(&rows);
        b.arg(&taps);
        b.arg(&total);
        // SAFETY: as in `conv_write`; `out` is fully written.
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }.w()?;
        Ok(out)
    }

    pub fn select_candidates(logits: &Tensor, inv_t: &Tensor) -> Result<super::Selected> {
        use super::CANDIDATE_CAP as CAP;
        let (rows, vocab) = logits.dims2()?;
        if logits.dtype() != DType::F32 || inv_t.dims() != [rows] {
            candle_core::bail!(
                "select_candidates: need f32 (rows, vocab) logits and (rows,) inv_t, got {:?} {:?}",
                logits.dtype(),
                inv_t.dims()
            );
        }
        let dev = cuda_device(logits)?;
        let func = dev.get_or_load_custom_func("select_candidates_f32", "vapi_fused", ptx()?)?;
        let d = logits.device();
        let ids = Tensor::zeros((rows, CAP), DType::U32, d)?;
        let vals = Tensor::zeros((rows, CAP), DType::F32, d)?;
        let counts = Tensor::zeros(rows, DType::U32, d)?;
        let depths = Tensor::zeros(rows, DType::U32, d)?;
        let maxes = Tensor::zeros(rows, DType::F32, d)?;
        let sums = Tensor::zeros(rows, DType::F32, d)?;
        let ptrs = [
            addr(Some(logits), "select_candidates")?,
            addr(Some(inv_t), "select_candidates")?,
        ];
        let outs = [
            addr(Some(&ids), "select_candidates")?,
            addr(Some(&vals), "select_candidates")?,
            addr(Some(&counts), "select_candidates")?,
            addr(Some(&depths), "select_candidates")?,
            addr(Some(&maxes), "select_candidates")?,
            addr(Some(&sums), "select_candidates")?,
        ];
        let (vocab, cap) = (vocab as u32, CAP as u32);
        let mut b = func.builder();
        b.arg(&ptrs[0]);
        b.arg(&ptrs[1]);
        b.arg(&vocab);
        b.arg(&cap);
        for o in &outs {
            b.arg(o);
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: every buffer is a live tensor sized from the shapes above;
        // the kernel bounds its writes by `cap`.
        unsafe { b.launch(cfg) }.w()?;
        Ok(super::Selected {
            ids,
            vals,
            counts,
            depths,
            maxes,
            sums,
        })
    }

    pub fn gumbel_draw(logits: &Tensor, rows: &[Option<super::GumbelRow>]) -> Result<Vec<u32>> {
        let (n_rows, vocab) = logits.dims2()?;
        if logits.dtype() != DType::F32 || rows.len() != n_rows {
            candle_core::bail!("gumbel_draw: need f32 (rows, vocab) logits and one spec per row");
        }
        let dev = cuda_device(logits)?;
        let d = logits.device();
        let mut active = vec![0u32; n_rows];
        let mut inv_t = vec![1f32; n_rows];
        let mut seeds = vec![0u64; n_rows];
        let mut draws = vec![0u64; n_rows];
        let (mut rep, mut freq, mut pres) =
            (vec![1f32; n_rows], vec![0f32; n_rows], vec![0f32; n_rows]);
        let (mut p_rows, mut p_toks, mut p_counts) = (Vec::new(), Vec::new(), Vec::new());
        for (r, spec) in rows.iter().enumerate() {
            let Some(spec) = spec else { continue };
            active[r] = 1;
            inv_t[r] = spec.inv_temperature;
            seeds[r] = spec.seed;
            draws[r] = spec.draw;
            rep[r] = spec.repetition_penalty;
            freq[r] = spec.frequency_penalty;
            pres[r] = spec.presence_penalty;
            let penalised = spec.repetition_penalty != 1.0
                || spec.frequency_penalty != 0.0
                || spec.presence_penalty != 0.0;
            if penalised {
                for &(t, c) in &spec.observed {
                    if (t as usize) < vocab {
                        p_rows.push(r as u32);
                        p_toks.push(t);
                        p_counts.push(c as f32);
                    }
                }
            }
        }
        if active.iter().all(|&a| a == 0) {
            return Ok(vec![u32::MAX; n_rows]);
        }
        let logits_p = addr(Some(logits), "gumbel_draw")?;
        let vocab_u = vocab as u32;
        let rep_t = Tensor::from_vec(rep, n_rows, d)?;
        let freq_t = Tensor::from_vec(freq, n_rows, d)?;
        let pres_t = Tensor::from_vec(pres, n_rows, d)?;
        if !p_rows.is_empty() {
            let n = p_rows.len();
            let rows_t = Tensor::from_vec(p_rows, n, d)?;
            let toks_t = Tensor::from_vec(p_toks, n, d)?;
            let counts_t = Tensor::from_vec(p_counts, n, d)?;
            let func = dev.get_or_load_custom_func("apply_penalties_f32", "vapi_fused", ptx()?)?;
            let args = [
                addr(Some(&rows_t), "gumbel_draw")?,
                addr(Some(&toks_t), "gumbel_draw")?,
                addr(Some(&counts_t), "gumbel_draw")?,
                addr(Some(&rep_t), "gumbel_draw")?,
                addr(Some(&freq_t), "gumbel_draw")?,
                addr(Some(&pres_t), "gumbel_draw")?,
            ];
            let n_u = n as u32;
            let mut b = func.builder();
            b.arg(&logits_p);
            b.arg(&vocab_u);
            for a in &args {
                b.arg(a);
            }
            b.arg(&n_u);
            // SAFETY: live tensors, sizes from their shapes; every token
            // index was bounds-checked above.
            unsafe { b.launch(LaunchConfig::for_num_elems(n_u)) }.w()?;
        }
        let active_t = Tensor::from_vec(active, n_rows, d)?;
        let inv_t_t = Tensor::from_vec(inv_t, n_rows, d)?;
        // u64 rides as two u32 words per row; the kernel reads them as one.
        let words = |v: &[u64]| -> Vec<u32> {
            v.iter()
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect()
        };
        let seeds_t = Tensor::from_vec(words(&seeds), 2 * n_rows, d)?;
        let draws_t = Tensor::from_vec(words(&draws), 2 * n_rows, d)?;
        let out = Tensor::from_vec(vec![u32::MAX; n_rows], n_rows, d)?;
        let func = dev.get_or_load_custom_func("gumbel_argmax_f32", "vapi_fused", ptx()?)?;
        let args = [
            addr(Some(&active_t), "gumbel_draw")?,
            addr(Some(&inv_t_t), "gumbel_draw")?,
            addr(Some(&seeds_t), "gumbel_draw")?,
            addr(Some(&draws_t), "gumbel_draw")?,
            addr(Some(&out), "gumbel_draw")?,
        ];
        let mut b = func.builder();
        b.arg(&logits_p);
        b.arg(&vocab_u);
        for a in &args {
            b.arg(a);
        }
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: as above; the kernel writes one u32 per row.
        unsafe { b.launch(cfg) }.w()?;
        out.to_vec1::<u32>()
    }

    pub struct SwiGlu {
        pub n: usize,
    }

    impl CustomOp1 for SwiGlu {
        fn name(&self) -> &'static str {
            "vapi-swiglu"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("swiglu is CUDA only; the CPU path uses candle ops")
        }

        fn cuda_fwd(&self, h: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let (rows, width) = l.shape().dims2()?;
            if width != 2 * self.n {
                candle_core::bail!("swiglu: width {width} is not 2 × {}", self.n);
            }
            let (o1, _) = l
                .contiguous_offsets()
                .ok_or_else(|| candle_core::Error::RequiresContiguous { op: "swiglu" })?;
            let dev = h.device().clone();
            let stream = dev.cuda_stream();
            let total = rows * self.n;
            let cfg = LaunchConfig::for_num_elems(total as u32);
            macro_rules! run {
                ($t:ty, $variant:ident, $name:literal, $src:expr) => {{
                    // SAFETY: fully written by the kernel.
                    let out = unsafe { dev.alloc::<$t>(total)? };
                    let func = dev.get_or_load_custom_func($name, "vapi_fused", ptx()?)?;
                    let (src_ptr, _g1) = $src.device_ptr(&stream);
                    let src_ptr = src_ptr + (o1 * std::mem::size_of::<$t>()) as u64;
                    let (out_ptr, _g2) = out.device_ptr(&stream);
                    let (n, total) = (self.n as u32, total as u32);
                    let mut b = func.builder();
                    b.arg(&src_ptr);
                    b.arg(&out_ptr);
                    b.arg(&n);
                    b.arg(&total);
                    // SAFETY: both buffers are live for the call and sized
                    // from the layout.
                    unsafe { b.launch(cfg) }.w()?;
                    drop((_g1, _g2));
                    CudaStorageSlice::$variant(out)
                }};
            }
            let slice = match &h.slice {
                CudaStorageSlice::BF16(v) => run!(bf16, BF16, "swiglu_bf16", v),
                CudaStorageSlice::F32(v) => run!(f32, F32, "swiglu_f32", v),
                _ => candle_core::bail!("swiglu: unsupported dtype {:?}", h.dtype()),
            };
            Ok((CudaStorage { slice, device: dev }, (rows, self.n).into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(h: &Tensor, n: usize) -> Result<Tensor> {
        candle_nn::ops::silu(&h.narrow(1, 0, n)?)? * h.narrow(1, n, n)?
    }

    #[test]
    fn cpu_matches_reference() -> Result<()> {
        let h = Tensor::randn(0f32, 2.0, (5, 16), &Device::Cpu)?;
        let got = swiglu(&h, 8)?;
        let want = reference(&h, 8)?;
        assert!((got - want)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gumbel_draws_follow_the_tempered_softmax_and_reproduce() -> Result<()> {
        use super::GumbelRow;
        let dev = Device::new_cuda(0)?;
        // A small vocabulary with a known distribution; many rows so one
        // launch gives many independent draws.
        let vocab = 512usize;
        let mut base = vec![-5.0f32; vocab];
        base[3] = 2.0;
        base[10] = 1.0;
        base[100] = 0.0;
        base[400] = 1.5;
        let temp = 0.8f32;
        let n_rows = 4096usize;
        let logits = Tensor::from_vec(base.repeat(n_rows), (n_rows, vocab), &dev)?;
        let spec = |seed: u64, draw: u64, observed: Vec<(u32, u32)>, rep: f32| GumbelRow {
            inv_temperature: 1.0 / temp,
            seed,
            draw,
            observed,
            repetition_penalty: rep,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };
        let rows: Vec<Option<GumbelRow>> = (0..n_rows)
            .map(|r| Some(spec(r as u64, 0, vec![], 1.0)))
            .collect();
        let got = gumbel_draw(&logits, &rows)?;
        let mut counts = vec![0usize; vocab];
        for &t in &got {
            counts[t as usize] += 1;
        }
        let max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = base
            .iter()
            .map(|&l| (((l - max) / temp) as f64).exp())
            .collect();
        let z: f64 = e.iter().sum();
        for &t in &[3usize, 10, 100, 400] {
            let p = e[t] / z;
            let got = counts[t] as f64 / n_rows as f64;
            // Binomial: sd = sqrt(p(1-p)/n); allow 4 sd.
            let sd = (p * (1.0 - p) / n_rows as f64).sqrt();
            assert!(
                (got - p).abs() < 4.0 * sd + 1e-3,
                "token {t}: drawn {got:.4}, expected {p:.4} (sd {sd:.4})"
            );
        }

        // Same seed and draw index: same token; a new draw index: not
        // always the same token.
        let again = gumbel_draw(&logits, &rows)?;
        assert_eq!(got, again, "draws reproduce for the same seed and index");
        let next: Vec<Option<GumbelRow>> = (0..n_rows)
            .map(|r| Some(spec(r as u64, 1, vec![], 1.0)))
            .collect();
        let got_next = gumbel_draw(&logits, &next)?;
        assert!(got_next != got, "the draw index changes the noise");

        // Inactive rows are untouched; a heavy repetition penalty on the
        // favourite tokens moves the mass away from them.
        let mut mixed: Vec<Option<GumbelRow>> = (0..n_rows)
            .map(|r| Some(spec(r as u64, 2, vec![(3, 1), (400, 1), (10, 1)], 1000.0)))
            .collect();
        mixed[0] = None;
        mixed[1] = None;
        let pen = gumbel_draw(&logits, &mixed)?;
        assert_eq!(pen[0], u32::MAX);
        assert_eq!(pen[1], u32::MAX);
        let hit = pen[2..].iter().filter(|&&t| t == 100).count() as f64 / (n_rows - 2) as f64;
        // With 3, 10 and 400 pushed to ~0, token 100 (logit 0) holds
        // 1 / (1 + 3·511·e^{-5/0.8}) of the mass... plus the near-zero ones.
        let z2: f64 = (0..vocab)
            .map(|v| {
                let l = if [3, 10, 400].contains(&v) {
                    0.0
                } else {
                    base[v]
                };
                ((l as f64) / temp as f64).exp()
            })
            .sum();
        let p100 = 1.0 / z2;
        let sd = (p100 * (1.0 - p100) / n_rows as f64).sqrt();
        assert!(
            (hit - p100).abs() < 4.0 * sd + 1e-3,
            "penalised: {hit:.4} vs {p100:.4}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn candidate_selection_matches_a_host_reference() -> Result<()> {
        use candle_core::DType;
        use rand::{RngExt, SeedableRng};
        let dev = Device::new_cuda(0)?;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let vocab = 128_000usize;
        // Row 0: peaked (a few candidates). Row 1: flat at temperature 1
        // (the first bin overflows). Row 2: moderately spread so the cap
        // cuts the window part way. Row 3: cold temperature on row 1.
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut peaked: Vec<f32> = (0..vocab)
            .map(|_| rng.random::<f32>() * 4.0 - 40.0)
            .collect();
        peaked[17] = 20.0;
        peaked[999] = 18.5;
        rows.push(peaked);
        rows.push((0..vocab).map(|_| rng.random::<f32>() * 2.0).collect());
        rows.push(
            (0..vocab)
                .map(|_| rng.random::<f32>() * 40.0 - 20.0)
                .collect(),
        );
        rows.push(rows[1].clone());
        let inv_t = vec![1.0f32, 1.0, 1.0 / 0.7, 10.0];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits = Tensor::from_vec(flat, (4, vocab), &dev)?;
        let inv = Tensor::from_vec(inv_t.clone(), 4, &dev)?;
        let sel = select_candidates(&logits, &inv)?;
        let counts = sel.counts.to_vec1::<u32>()?;
        let depths = sel.depths.to_vec1::<u32>()?;
        let maxes = sel.maxes.to_vec1::<f32>()?;
        let sums = sel.sums.to_vec1::<f32>()?;
        let ids = sel.ids.to_vec2::<u32>()?;
        let vals = sel.vals.to_vec2::<f32>()?;
        let _ = DType::F32;

        for r in 0..4 {
            let l = &rows[r];
            let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(maxes[r], max, "row {r} max");
            let sum: f64 = l
                .iter()
                .map(|&x| (((x - max) * inv_t[r]) as f64).exp())
                .sum();
            assert!(
                ((sums[r] as f64 - sum) / sum).abs() < 1e-4,
                "row {r} sum {} vs {sum}",
                sums[r]
            );
            // Bin counts as the kernel defines them.
            let bin = |x: f32| {
                let g = (max - x) * inv_t[r];
                if g >= 31.0 { 31u32 } else { g as u32 }
            };
            let mut hist = [0usize; 32];
            for &x in l {
                hist[bin(x) as usize] += 1;
            }
            let mut cum = 0;
            let mut want_depth: Option<u32> = None;
            for b in 0..=30u32 {
                cum += hist[b as usize];
                if cum <= CANDIDATE_CAP {
                    want_depth = Some(b);
                } else {
                    break;
                }
            }
            match want_depth {
                None => {
                    assert_eq!(depths[r], u32::MAX, "row {r} should overflow");
                    assert_eq!(counts[r] as usize, hist[0]);
                }
                Some(d) => {
                    assert_eq!(depths[r], d, "row {r} depth");
                    let want: std::collections::BTreeSet<u32> = l
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| bin(**x) <= d)
                        .map(|(i, _)| i as u32)
                        .collect();
                    assert_eq!(counts[r] as usize, want.len(), "row {r} count");
                    let got: std::collections::BTreeSet<u32> =
                        ids[r][..counts[r] as usize].iter().copied().collect();
                    assert_eq!(got, want, "row {r} ids");
                    for (i, &id) in ids[r][..counts[r] as usize].iter().enumerate() {
                        assert_eq!(vals[r][i], l[id as usize], "row {r} value of {id}");
                    }
                }
            }
        }
        assert_eq!(depths[0], CANDIDATE_WINDOW, "peaked row is complete");
        assert_eq!(depths[1], u32::MAX, "flat row overflows");
        assert!(depths[2] < CANDIDATE_WINDOW, "spread row is cut part way");
        assert!(depths[3] != u32::MAX && counts[3] as usize <= CANDIDATE_CAP);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn conv_kernels_match_candle_ops() -> Result<()> {
        use candle_core::DType;
        let dev = Device::new_cuda(0)?;
        let (rows, hidden, slots_n, taps) = (7usize, 32usize, 40usize, 3usize);
        let h3 = Tensor::randn(0f32, 1.0, (rows, 3 * hidden), &dev)?;
        let in_bias = Tensor::randn(0f32, 1.0, 3 * hidden, &dev)?;
        let w = Tensor::randn(0f32, 1.0, (taps, hidden), &dev)?;
        let bias = Tensor::randn(0f32, 1.0, hidden, &dev)?;
        let slots: Vec<u32> = vec![5, 6, 7, 20, 21, 33, 0];
        // Taps: two sequences [5,6,7] and [20,21], then a lone token at 33
        // and one at 0 with nothing before it.
        let idx: Vec<u32> = vec![
            u32::MAX,
            u32::MAX,
            5,
            u32::MAX,
            u32::MAX,
            31,
            u32::MAX, // two back
            u32::MAX,
            5,
            6,
            u32::MAX,
            20,
            32,
            u32::MAX, // one back
            5,
            6,
            7,
            20,
            21,
            33,
            0, // self
        ];
        let idx_t = Tensor::from_vec(idx.clone(), (taps, rows), &dev)?;
        let slots_t = Tensor::from_vec(slots.clone(), rows, &dev)?;

        for dtype in [DType::F32, DType::BF16] {
            let table = Tensor::randn(0f32, 1.0, (slots_n, hidden), &dev)?.to_dtype(dtype)?;
            let (h3, in_bias, w, bias) = (
                h3.to_dtype(dtype)?,
                in_bias.to_dtype(dtype)?,
                w.to_dtype(dtype)?,
                bias.to_dtype(dtype)?,
            );
            // Reference with candle ops, in f32.
            let h3f = h3
                .to_dtype(DType::F32)?
                .broadcast_add(&in_bias.to_dtype(DType::F32)?)?;
            let b = h3f.narrow(1, 0, hidden)?;
            let c = h3f.narrow(1, hidden, hidden)?;
            let x = h3f.narrow(1, 2 * hidden, hidden)?;
            let want_rows = (b * x)?;
            let mut ref_table = table.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let wr = want_rows.to_vec2::<f32>()?;
            for (i, s) in slots.iter().enumerate() {
                ref_table[*s as usize] = wr[i].clone();
            }
            let mut want = vec![vec![0f32; hidden]; rows];
            let wf = w.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let bf = bias.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let cf = c.to_vec2::<f32>()?;
            for i in 0..rows {
                for j in 0..hidden {
                    let mut acc = bf[j];
                    for k in 0..taps {
                        let r = idx[k * rows + i];
                        if r != u32::MAX {
                            acc += ref_table[r as usize][j] * wf[k][j];
                        }
                    }
                    want[i][j] = cf[i][j] * acc;
                }
            }

            conv_write(&h3, &table, &slots_t, Some(&in_bias), hidden)?;
            let got_table = table.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let mut d = 0f32;
            for (g, r) in got_table.iter().zip(&ref_table) {
                for (a, b) in g.iter().zip(r) {
                    d = d.max((a - b).abs() / (b.abs() + 1.0));
                }
            }
            let tol = if dtype == DType::F32 { 1e-5 } else { 0.02 };
            assert!(d < tol, "{dtype:?} conv_write: max rel diff {d}");

            let got = conv_apply(&h3, &table, &idx_t, &w, Some(&bias), Some(&in_bias), hidden)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let mut d = 0f32;
            for (g, r) in got.iter().zip(&want) {
                for (a, b) in g.iter().zip(r) {
                    d = d.max((a - b).abs() / (b.abs() + 1.0));
                }
            }
            assert!(d < tol * 2.0, "{dtype:?} conv_apply: max rel diff {d}");
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_reference() -> Result<()> {
        use candle_core::DType;
        let dev = Device::new_cuda(0)?;
        for (rows, n) in [(1usize, 8usize), (64, 10752), (7, 24)] {
            let h = Tensor::randn(0f32, 2.0, (rows, 2 * n), &dev)?;
            let got = swiglu(&h, n)?;
            let want = reference(&h, n)?;
            let d = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(d < 1e-5, "f32 rows={rows} n={n}: max diff {d}");

            // The kernel rounds once, the reference rounds after silu and
            // again after the product, so compare relative to magnitude.
            let hb = h.to_dtype(DType::BF16)?;
            let got = swiglu(&hb, n)?.to_dtype(DType::F32)?;
            let want = reference(&hb, n)?.to_dtype(DType::F32)?;
            let scale = (want.abs()? + 1.0)?;
            let d = ((got - &want)?.abs()? / scale)?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(d < 0.02, "bf16 rows={rows} n={n}: max relative diff {d}");
        }
        Ok(())
    }
}
