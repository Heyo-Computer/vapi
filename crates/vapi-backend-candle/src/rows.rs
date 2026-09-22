//! Row gather and scatter that CUDA graph capture can record.
//!
//! candle's `index_select` and `scatter_set` upload the index tensor's
//! dims and strides from a temporary host buffer at every launch. Eagerly
//! that is a small cost; under stream capture it is recorded as a copy from
//! memory that is freed the moment the call returns, and the replay reads
//! garbage. These two kernels take every argument by value, copy rows as
//! raw 16-byte chunks (so one kernel serves bf16, f16 and f32), and are
//! what the LFM2 forward uses on CUDA. On the CPU the candle ops are used.

use candle_core::{Result, Tensor};

/// `out[i, ..] = table[idx[i], ..]` for a 2-D+ `table` and `(n,)` u32 `idx`.
pub fn gather(table: &Tensor, idx: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if table.device().is_cuda() && cuda::row_bytes_ok(table)? {
        return table.apply_op2_no_bwd(idx, &cuda::GatherRows);
    }
    table.index_select(idx, 0)
}

/// `table[idx[i], ..] = src[i, ..]`, in place. `table` is 2-D+ and
/// contiguous, `src` has the same trailing shape with `n` rows.
pub fn scatter(table: &Tensor, idx: &Tensor, src: &Tensor) -> Result<()> {
    #[cfg(feature = "cuda")]
    if table.device().is_cuda() && cuda::row_bytes_ok(table)? {
        return table.inplace_op3(idx, src, &cuda::ScatterRows);
    }
    let (n, w) = (idx.dim(0)?, table.dims()[1..].iter().product::<usize>());
    let flat = table.reshape((table.dim(0)?, w))?;
    let idx = idx.reshape((n, 1))?.broadcast_as((n, w))?.contiguous()?;
    flat.scatter_set(&idx, &src.reshape((n, w))?, 0)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use candle_core::cuda::{CudaStorage, CudaStorageSlice, WrapErr};
    use candle_core::{CustomOp2, InplaceOp3, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const SRC: &str = r#"
extern "C" __global__ void gather_rows(
    const unsigned char* table, const unsigned int* idx, unsigned char* out,
    unsigned long long row_bytes, unsigned int n)
{
    unsigned long long chunks = ((unsigned long long)n * row_bytes) / 16ull;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunks) return;
    unsigned long long byte = i * 16ull;
    unsigned int row = (unsigned int)(byte / row_bytes);
    unsigned long long off = byte % row_bytes;
    const uint4* s = (const uint4*)(table + (unsigned long long)idx[row] * row_bytes + off);
    *(uint4*)(out + byte) = *s;
}

extern "C" __global__ void scatter_rows(
    unsigned char* table, const unsigned int* idx, const unsigned char* src,
    unsigned long long row_bytes, unsigned int n)
{
    unsigned long long chunks = ((unsigned long long)n * row_bytes) / 16ull;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunks) return;
    unsigned long long byte = i * 16ull;
    unsigned int row = (unsigned int)(byte / row_bytes);
    unsigned long long off = byte % row_bytes;
    const uint4* s = (const uint4*)(src + byte);
    *(uint4*)(table + (unsigned long long)idx[row] * row_bytes + off) = *s;
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

    /// Rows must be a multiple of 16 bytes for the vectorised copy.
    pub fn row_bytes_ok(table: &Tensor) -> Result<bool> {
        let w: usize = table.dims()[1..].iter().product();
        Ok((w * table.dtype().size_in_bytes()).is_multiple_of(16))
    }

    fn raw_ptr(s: &CudaStorage, l: &Layout) -> Result<u64> {
        let (o1, _) = l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: "rows" })?;
        let elem = s.dtype().size_in_bytes() as u64;
        let stream = s.device().cuda_stream();
        macro_rules! p {
            ($sl:expr) => {{
                let (ptr, _guard) = $sl.device_ptr(&stream);
                ptr + o1 as u64 * elem
            }};
        }
        Ok(match &s.slice {
            CudaStorageSlice::U8(v) => p!(v),
            CudaStorageSlice::U32(v) => p!(v),
            CudaStorageSlice::I64(v) => p!(v),
            CudaStorageSlice::BF16(v) => p!(v),
            CudaStorageSlice::F16(v) => p!(v),
            CudaStorageSlice::F32(v) => p!(v),
            CudaStorageSlice::F64(v) => p!(v),
            _ => candle_core::bail!("rows: unsupported dtype {:?}", s.dtype()),
        })
    }

    fn launch(
        dev: &candle_core::CudaDevice,
        name: &str,
        a: u64,
        idx: u64,
        b: u64,
        row_bytes: u64,
        n: u32,
    ) -> Result<()> {
        let func = dev.get_or_load_custom_func(name, "vapi_rows", ptx()?)?;
        let chunks = (n as u64 * row_bytes) / 16;
        let cfg = LaunchConfig::for_num_elems(chunks as u32);
        let mut builder = func.builder();
        builder.arg(&a);
        builder.arg(&idx);
        builder.arg(&b);
        builder.arg(&row_bytes);
        builder.arg(&n);
        // SAFETY: pointers come from live tensors held by the caller for the
        // duration of the call; sizes are computed from their layouts.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    pub struct GatherRows;

    impl CustomOp2 for GatherRows {
        fn name(&self) -> &'static str {
            "vapi-gather-rows"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("gather-rows is CUDA only; the CPU path uses index_select")
        }

        fn cuda_fwd(
            &self,
            table: &CudaStorage,
            tl: &Layout,
            idx: &CudaStorage,
            il: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let dev = table.device().clone();
            let n = il.shape().elem_count();
            let w: usize = tl.dims()[1..].iter().product();
            let row_bytes = (w * table.dtype().size_in_bytes()) as u64;
            let out_shape: Shape = [&[n][..], &tl.dims()[1..]].concat().into();
            macro_rules! run {
                ($t:ty, $variant:ident) => {{
                    // SAFETY: fully written by the kernel.
                    let out = unsafe { dev.alloc::<$t>(n * w)? };
                    let stream = dev.cuda_stream();
                    let out_ptr = {
                        let (p, _g) = out.device_ptr(&stream);
                        p
                    };
                    launch(
                        &dev,
                        "gather_rows",
                        raw_ptr(table, tl)?,
                        raw_ptr(idx, il)?,
                        out_ptr,
                        row_bytes,
                        n as u32,
                    )?;
                    CudaStorageSlice::$variant(out)
                }};
            }
            let slice = match &table.slice {
                CudaStorageSlice::BF16(_) => run!(bf16, BF16),
                CudaStorageSlice::F16(_) => run!(f16, F16),
                CudaStorageSlice::F32(_) => run!(f32, F32),
                _ => candle_core::bail!("gather-rows: unsupported dtype {:?}", table.dtype()),
            };
            Ok((CudaStorage { slice, device: dev }, out_shape))
        }
    }

    pub struct ScatterRows;

    impl InplaceOp3 for ScatterRows {
        fn name(&self) -> &'static str {
            "vapi-scatter-rows"
        }

        fn cpu_fwd(
            &self,
            _: &mut candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("scatter-rows is CUDA only; the CPU path uses scatter_set")
        }

        fn cuda_fwd(
            &self,
            table: &mut CudaStorage,
            tl: &Layout,
            idx: &CudaStorage,
            il: &Layout,
            src: &CudaStorage,
            sl: &Layout,
        ) -> Result<()> {
            if table.dtype() != src.dtype() {
                candle_core::bail!("scatter-rows: dtype mismatch");
            }
            let dev = table.device().clone();
            let n = il.shape().elem_count();
            let w: usize = tl.dims()[1..].iter().product();
            if sl.shape().elem_count() != n * w {
                candle_core::bail!("scatter-rows: {} rows of {} vs src {:?}", n, w, sl.dims());
            }
            let row_bytes = (w * table.dtype().size_in_bytes()) as u64;
            launch(
                &dev,
                "scatter_rows",
                raw_ptr(table, tl)?,
                raw_ptr(idx, il)?,
                raw_ptr(src, sl)?,
                row_bytes,
                n as u32,
            )
        }
    }
}
