//! Paged attention: the CPU reference and the CUDA fast path.
//!
//! # The shape of the problem
//!
//! A sequence's K/V is scattered across non-contiguous blocks, so attention
//! has to be told where those blocks are. Both paths below consume the same
//! [`vapi_engine::ForwardBatch`], which is laid out to match
//! `candle_flash_attn::flash_attn_varlen_paged_windowed` argument for
//! argument.
//!
//! # Why there is no custom CUDA kernel here
//!
//! `candle-flash-attn` 0.11 already ships a paged kernel. Because it is
//! *varlen* and takes a block table, prefill and decode are not separate code
//! paths: a prefill chunk is a sequence with `seqlen_q > 1`, a decode step is
//! one with `seqlen_q = 1`, and a mixed continuous batch is simply both in one
//! `cu_seqlens` array. Writing `paged_attention_v1/v2` by hand only becomes
//! worthwhile if profiling says this kernel is the bottleneck.
//!
//! Its constraints are already baked into the engine, which is why they are
//! not a surprise waiting at M6:
//!
//! - paged K/V must be `(num_blocks, page_block_size, num_kv_heads, head_dim)`
//!   — the layout [`crate::cache::PagedKvCache`] allocates;
//! - `block_table` must be a CUDA `u32`/`i32` tensor of
//!   `[batch, max_blocks_per_seq]` with a contiguous last dimension;
//! - `page_block_size % 32 == 0` — the reason
//!   [`vapi_core::config::BLOCK_SIZE`] is 32 rather than vLLM's 16.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use vapi_engine::ForwardBatch;

use crate::cache::gather_kv;

/// Pack per-sequence block tables into the `[batch, max_blocks]` tensor the
/// paged kernel wants.
///
/// Rows are padded to the widest sequence. Padding entries are never read —
/// `cu_seqlens_k` bounds how far into each row the kernel walks — but they
/// must still be in-range block ids, because an out-of-range index would fault
/// even if it is never used for output.
pub fn pack_block_tables(batch: &ForwardBatch, device: &Device) -> CandleResult<Tensor> {
    let rows = batch.block_tables.len();
    let width = batch
        .block_tables
        .iter()
        .map(|t| t.len())
        .max()
        .unwrap_or(0)
        .max(1);

    let mut flat = Vec::with_capacity(rows * width);
    for table in &batch.block_tables {
        flat.extend_from_slice(table);
        // Pad with the row's own last block rather than zero: still a valid
        // index, and it keeps the padding from aliasing another sequence's
        // block if a bug ever does read it.
        let pad = table.last().copied().unwrap_or(0);
        flat.resize(flat.len() + (width - table.len()), pad);
    }
    Tensor::from_vec(flat, (rows, width), device)
}

/// Cumulative sequence lengths as the `u32` tensor the kernel expects.
///
/// candle-flash-attn reads `seqlens_q`/`seqlens_k` with
/// `as_cuda_slice::<u32>()`, so an `i32` tensor here fails at runtime with a
/// dtype error, whatever the upstream comment next to that line says.
pub fn cu_seqlens(values: &[u32], device: &Device) -> CandleResult<Tensor> {
    Tensor::from_vec(values.to_vec(), values.len(), device)
}

/// CPU reference paged attention, one sequence at a time.
///
/// `q` is `(total_q, num_heads, head_dim)`; `k_cache` and `v_cache` are the
/// paged buffers `(num_blocks, block_size, num_kv_heads, head_dim)`. Returns
/// `(total_q, num_heads, head_dim)`, matching the CUDA path.
///
/// For each sequence its blocks are gathered in logical order and flattened,
/// K/V heads are repeated for GQA, and a query at absolute position `p` sees
/// keys `0..=p` — with `p = len_k - len_q + j` for the `j`-th query, which is
/// the bottom-right-aligned causal mask FlashAttention ≥ 2.1 uses when
/// `seqlen_q < seqlen_k`. That case is the prefix-cache-hit path. Math is done
/// in f32 whatever the cache dtype, so this is the numerical reference the
/// CUDA path is checked against.
///
/// `sliding_window = Some(w)` additionally hides keys more than `w - 1`
/// positions behind the query (HF's `q - k < w`), for architectures with
/// local-attention layers.
pub fn paged_attention_cpu(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    batch: &ForwardBatch,
    softmax_scale: f32,
    sliding_window: Option<usize>,
) -> CandleResult<Tensor> {
    let device = q.device();
    let (_, num_heads, head_dim) = q.dims3()?;
    let num_kv_heads = k_cache.dim(2)?;
    let n_rep = num_heads / num_kv_heads;
    let in_dtype = q.dtype();

    let mut outs = Vec::with_capacity(batch.batch_size());
    for i in 0..batch.batch_size() {
        let q0 = batch.cu_seqlens_q[i] as usize;
        let len_q = batch.cu_seqlens_q[i + 1] as usize - q0;
        let len_k = (batch.cu_seqlens_k[i + 1] - batch.cu_seqlens_k[i]) as usize;
        if len_q == 0 {
            continue;
        }
        let table = &batch.block_tables[i];

        // (heads, len_q, head_dim)
        let qi = q
            .narrow(0, q0, len_q)?
            .transpose(0, 1)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        // (kv_heads, len_k, head_dim) -> repeated to (heads, len_k, head_dim)
        let ki = gather_kv(k_cache, table, len_k)?
            .transpose(0, 1)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?;
        let vi = gather_kv(v_cache, table, len_k)?
            .transpose(0, 1)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?;
        let ki = candle_transformers::utils::repeat_kv(ki, n_rep)?
            .squeeze(0)?
            .contiguous()?;
        let vi = candle_transformers::utils::repeat_kv(vi, n_rep)?
            .squeeze(0)?
            .contiguous()?;

        let att = (qi.matmul(&ki.transpose(1, 2)?.contiguous()?)? * softmax_scale as f64)?;
        // Query j sits at absolute position offset + j and may not see keys
        // after it. With len_q == len_k this is the ordinary causal mask.
        let offset = len_k - len_q;
        let mask: Vec<u8> = (0..len_q)
            .flat_map(|j| {
                (0..len_k).map(move |k| {
                    let p = offset + j;
                    let too_old = sliding_window.is_some_and(|w| k < p && p - k >= w);
                    u8::from(k > p || too_old)
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (len_q, len_k), device)?
            .unsqueeze(0)?
            .broadcast_as(att.shape())?
            .contiguous()?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(att.shape())?;
        let att = mask.where_cond(&neg_inf, &att)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&vi)?; // (heads, len_q, head_dim)
        outs.push(y.transpose(0, 1)?.contiguous()?);
    }
    if outs.is_empty() {
        return Tensor::zeros((0, num_heads, head_dim), in_dtype, device);
    }
    Tensor::cat(&outs, 0)?.to_dtype(in_dtype)
}

/// CUDA paged attention: one call covers prefill and decode.
///
/// `q` is `(total_q, num_heads, head_dim)`; `k_cache` and `v_cache` are the
/// paged buffers `(num_blocks, block_size, num_kv_heads, head_dim)`.
#[cfg(feature = "cuda")]
pub fn paged_attention_cuda(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    batch: &ForwardBatch,
    softmax_scale: f32,
    block_size: usize,
    sliding_window: Option<usize>,
) -> CandleResult<Tensor> {
    // Causal attention is `window_size_left < 0 && window_size_right == 0`
    // in candle-flash-attn, and `None` maps to -1 on both sides. So
    // `(None, None)` is *bidirectional*: prefill tokens would attend to
    // their own future. `Some(0)` on the right is what upstream's
    // `flash_attn_varlen(.., causal = true)` passes. A sliding window of
    // `w` (HF: `q - k < w`) is `window_size_left = w - 1`.
    paged_attention_cuda_windowed(
        q,
        k_cache,
        v_cache,
        batch,
        softmax_scale,
        block_size,
        sliding_window.map(|w| w.saturating_sub(1)),
        Some(0),
    )
}

/// The kernel call with the right-hand window exposed, so a test can show
/// that `None` (bidirectional) really does produce different output.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)] // mirrors the kernel's own argument list
fn paged_attention_cuda_windowed(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    batch: &ForwardBatch,
    softmax_scale: f32,
    block_size: usize,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> CandleResult<Tensor> {
    let device = q.device();
    let block_table = pack_block_tables(batch, device)?;
    let seqlens_q = cu_seqlens(&batch.cu_seqlens_q, device)?;
    let seqlens_k = cu_seqlens(&batch.cu_seqlens_k, device)?;
    let prepared = PreparedAttention {
        block_table,
        seqlens_q,
        seqlens_k,
        max_seqlen_q: batch.max_seqlen_q,
        max_seqlen_k: batch.max_seqlen_k,
    };
    paged_attention_cuda_prepared(
        q,
        k_cache,
        v_cache,
        &prepared,
        softmax_scale,
        block_size,
        window_size_left,
        window_size_right,
    )
}

/// The kernel's per-step device inputs, built once per step (or once per
/// captured CUDA graph) rather than once per layer.
#[cfg(feature = "cuda")]
pub struct PreparedAttention {
    pub block_table: Tensor,
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}

#[cfg(feature = "cuda")]
impl PreparedAttention {
    pub fn from_batch(batch: &ForwardBatch, device: &Device) -> CandleResult<Self> {
        Ok(Self {
            block_table: pack_block_tables(batch, device)?,
            seqlens_q: cu_seqlens(&batch.cu_seqlens_q, device)?,
            seqlens_k: cu_seqlens(&batch.cu_seqlens_k, device)?,
            max_seqlen_q: batch.max_seqlen_q,
            max_seqlen_k: batch.max_seqlen_k,
        })
    }
}

/// `paged_attention_cuda` over prepared device inputs. No host-to-device
/// copies happen here, which is what lets it be captured into a graph.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda_prepared(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    prepared: &PreparedAttention,
    softmax_scale: f32,
    block_size: usize,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> CandleResult<Tensor> {
    candle_flash_attn::flash_attn_varlen_paged_windowed(
        q,
        k_cache,
        v_cache,
        &prepared.seqlens_q,
        &prepared.seqlens_k,
        &prepared.block_table,
        None, // mm_prefix_ranges: multimodal only
        prepared.max_seqlen_q,
        prepared.max_seqlen_k,
        softmax_scale,
        window_size_left,
        window_size_right,
        block_size,
        None, // softcap
    )
}

#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;
    use crate::cache::{PagedKvCache, write_kv_to_cache};

    /// Plain dense causal attention over one sequence, written the long way,
    /// as an oracle for the paged path.
    fn dense_causal(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
        let len_k = k.len();
        let len_q = q.len();
        let offset = len_k - len_q;
        q.iter()
            .enumerate()
            .map(|(j, qj)| {
                let visible = offset + j + 1;
                let scores: Vec<f32> = k[..visible]
                    .iter()
                    .map(|kk| qj.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale)
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = w.iter().sum();
                let mut out = vec![0.0; v[0].len()];
                for (wi, vi) in w.iter().zip(&v[..visible]) {
                    for (o, x) in out.iter_mut().zip(vi) {
                        *o += wi / z * x;
                    }
                }
                out
            })
            .collect()
    }

    #[test]
    fn a_sliding_window_hides_keys_older_than_the_window() {
        // One head, head_dim 2, six keys, six queries. With window 3 the
        // query at position p sees keys p-2..=p. Zero queries make the
        // attention uniform over visible keys, so the output's second
        // component is the mean visible key index.
        let dev = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 1, 2, DType::F32, &dev).unwrap();
        let k: Vec<f32> = (0..6).flat_map(|i| [i as f32, 0.0]).collect();
        let v: Vec<f32> = (0..6).flat_map(|i| [1.0, i as f32]).collect();
        let kt = Tensor::from_vec(k, (6, 1, 2), &dev).unwrap();
        let vt = Tensor::from_vec(v, (6, 1, 2), &dev).unwrap();
        write_kv_to_cache(&mut cache, 0, &kt, &vt, &[0, 1, 2, 3, 4, 5]).unwrap();
        let q = Tensor::zeros((6, 1, 2), DType::F32, &dev).unwrap();
        let batch = ForwardBatch {
            cu_seqlens_q: vec![0, 6],
            cu_seqlens_k: vec![0, 6],
            block_tables: vec![vec![0, 1]],
            ..Default::default()
        };
        let out = paged_attention_cpu(&q, &cache.k[0], &cache.v[0], &batch, 1.0, Some(3)).unwrap();
        let means: Vec<f32> = out
            .reshape((6, 2))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| r[1])
            .collect();
        assert_eq!(means, vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn the_cpu_reference_matches_dense_attention_for_a_mixed_batch() {
        // One kv head, one q head, head_dim 4, block size 4: small enough to
        // check by hand. Sequence A is a prefill chunk of 3 on top of 5 cached
        // tokens (seqlen_q < seqlen_k); sequence B is a decode over 6 keys.
        // Blocks are deliberately not contiguous.
        let dev = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 8, 4, 1, 4, DType::F32, &dev).unwrap();
        let rnd = |n: usize, seed: f32| -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    (0..4)
                        .map(|d| ((i * 7 + d * 3) as f32 * 0.37 + seed).sin())
                        .collect()
                })
                .collect()
        };
        let (ka, va) = (rnd(8, 0.1), rnd(8, 0.2));
        let (kb, vb) = (rnd(6, 0.3), rnd(6, 0.4));
        let table_a = [5u32, 2];
        let table_b = [7u32, 0];
        let to_t =
            |rows: &[Vec<f32>]| Tensor::from_vec(rows.concat(), (rows.len(), 1, 4), &dev).unwrap();
        let slots = |table: &[u32], n: usize| -> Vec<u32> {
            (0..n).map(|p| table[p / 4] * 4 + (p % 4) as u32).collect()
        };
        write_kv_to_cache(&mut cache, 0, &to_t(&ka), &to_t(&va), &slots(&table_a, 8)).unwrap();
        write_kv_to_cache(&mut cache, 0, &to_t(&kb), &to_t(&vb), &slots(&table_b, 6)).unwrap();

        let qa = rnd(3, 0.5);
        let qb = rnd(1, 0.6);
        let q = to_t(&[qa.clone(), qb.clone()].concat());
        let batch = ForwardBatch {
            cu_seqlens_q: vec![0, 3, 4],
            cu_seqlens_k: vec![0, 8, 14],
            block_tables: vec![table_a.to_vec(), table_b.to_vec()],
            ..Default::default()
        };
        let scale = 0.5;
        let out = paged_attention_cpu(&q, &cache.k[0], &cache.v[0], &batch, scale, None).unwrap();
        assert_eq!(out.dims(), &[4, 1, 4]);
        let out = out.reshape((4, 4)).unwrap().to_vec2::<f32>().unwrap();

        let want_a = dense_causal(&qa, &ka, &va, scale);
        let want_b = dense_causal(&qb, &kb, &vb, scale);
        for (got, want) in out.iter().zip(want_a.iter().chain(&want_b)) {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
            }
        }
    }

    fn batch_with_tables(tables: Vec<Vec<u32>>) -> ForwardBatch {
        ForwardBatch {
            block_tables: tables,
            ..Default::default()
        }
    }

    #[test]
    fn ragged_block_tables_are_padded_to_a_rectangle() {
        let b = batch_with_tables(vec![vec![0, 1, 2], vec![7]]);
        let t = pack_block_tables(&b, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        let rows: Vec<Vec<u32>> = t.to_vec2().unwrap();
        assert_eq!(rows[0], vec![0, 1, 2]);
        // Padded with the row's own last block: a valid index either way.
        assert_eq!(rows[1], vec![7, 7, 7]);
    }

    #[test]
    fn cumulative_lengths_are_u32_as_the_kernel_reads_them() {
        let t = cu_seqlens(&[0, 4, 9], &Device::Cpu).unwrap();
        assert_eq!(t.dtype(), candle_core::DType::U32);
        assert_eq!(t.to_vec1::<u32>().unwrap(), vec![0, 4, 9]);
    }
}

/// Step 1 of the GPU handoff: prove the paged FlashAttention kernel against
/// the CPU reference before building anything on it.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cache::{PagedKvCache, write_kv_to_cache};
    use vapi_core::config::BLOCK_SIZE;

    const HEADS: usize = 4;
    const KV_HEADS: usize = 2; // GQA
    const HEAD_DIM: usize = 64;

    fn rows(n: usize, seed: f32, dev: &Device, dtype: DType) -> Tensor {
        // Deterministic, smooth values with no huge magnitudes, so bf16
        // rounding stays within the tolerance we assert.
        let data: Vec<f32> = (0..n * KV_HEADS * HEAD_DIM)
            .map(|i| ((i as f32) * 0.173 + seed).sin() * 0.5)
            .collect();
        Tensor::from_vec(data, (n, KV_HEADS, HEAD_DIM), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn q_rows(n: usize, seed: f32, dev: &Device, dtype: DType) -> Tensor {
        let data: Vec<f32> = (0..n * HEADS * HEAD_DIM)
            .map(|i| ((i as f32) * 0.131 + seed).cos() * 0.5)
            .collect();
        Tensor::from_vec(data, (n, HEADS, HEAD_DIM), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn slots(table: &[u32], positions: std::ops::Range<usize>) -> Vec<u32> {
        positions
            .map(|p| table[p / BLOCK_SIZE] * BLOCK_SIZE as u32 + (p % BLOCK_SIZE) as u32)
            .collect()
    }

    /// A: 70 cached tokens plus a 20-token prefill chunk (seqlen_q < seqlen_k,
    /// the prefix-cache-hit path). B: a decode over 40 keys. Blocks are
    /// deliberately out of order.
    struct Mixed {
        cache: PagedKvCache,
        q: Tensor,
        batch: ForwardBatch,
    }

    fn mixed(dev: &Device, dtype: DType) -> Mixed {
        let mut cache =
            PagedKvCache::new(1, 8, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype, dev).unwrap();
        let table_a = [5u32, 1, 6];
        let table_b = [7u32, 2];
        let (len_a, chunk_a, len_b) = (90usize, 20usize, 40usize);
        write_kv_to_cache(
            &mut cache,
            0,
            &rows(len_a, 0.1, dev, dtype),
            &rows(len_a, 0.2, dev, dtype),
            &slots(&table_a, 0..len_a),
        )
        .unwrap();
        write_kv_to_cache(
            &mut cache,
            0,
            &rows(len_b, 0.3, dev, dtype),
            &rows(len_b, 0.4, dev, dtype),
            &slots(&table_b, 0..len_b),
        )
        .unwrap();
        let q = q_rows(chunk_a + 1, 0.5, dev, dtype);
        let batch = ForwardBatch {
            cu_seqlens_q: vec![0, chunk_a as u32, chunk_a as u32 + 1],
            cu_seqlens_k: vec![0, len_a as u32, (len_a + len_b) as u32],
            block_tables: vec![table_a.to_vec(), table_b.to_vec()],
            max_seqlen_q: chunk_a,
            max_seqlen_k: len_a,
            ..Default::default()
        };
        Mixed { cache, q, batch }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn check_dtype(dtype: DType, tol: f32) {
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let m = mixed(&dev, dtype);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let want =
            paged_attention_cpu(&m.q, &m.cache.k[0], &m.cache.v[0], &m.batch, scale, None).unwrap();
        let got = paged_attention_cuda(
            &m.q,
            &m.cache.k[0],
            &m.cache.v[0],
            &m.batch,
            scale,
            BLOCK_SIZE,
            None,
        )
        .unwrap();
        assert_eq!(got.dims(), want.dims());
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < tol,
            "{dtype:?}: max abs diff {diff} vs CPU reference"
        );
    }

    /// Laguna-XS-2.1's attention shapes at toy size: head_dim 16, 6 or 8
    /// query heads over 2 KV heads (GQA 3:1 and 4:1), a 6-token window, a
    /// pure prefill (seqlen_q == seqlen_k) plus a decode. The generic mixed
    /// test above uses head_dim 64 and GQA 2:1, which is not the same kernel
    /// template.
    #[test]
    fn laguna_shapes_match_the_cpu_reference() {
        let dev = Device::new_cuda(0).expect("a CUDA device");
        for (heads, window) in [(6usize, None), (8, Some(6)), (6, Some(6))] {
            let (kvh, hd) = (2usize, 16usize);
            let mut cache =
                PagedKvCache::new(1, 8, BLOCK_SIZE, kvh, hd, DType::BF16, &dev).unwrap();
            let mk = |n: usize, h: usize, seed: f32| {
                let data: Vec<f32> = (0..n * h * hd)
                    .map(|i| ((i as f32) * 0.173 + seed).sin() * 0.5)
                    .collect();
                Tensor::from_vec(data, (n, h, hd), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
            };
            let table_a = [4u32];
            let table_b = [6u32, 1];
            let (len_a, len_b) = (18usize, 40usize);
            let sl = |t: &[u32], r: std::ops::Range<usize>| -> Vec<u32> {
                r.map(|p| t[p / BLOCK_SIZE] * BLOCK_SIZE as u32 + (p % BLOCK_SIZE) as u32)
                    .collect()
            };
            write_kv_to_cache(
                &mut cache,
                0,
                &mk(len_a, kvh, 0.1),
                &mk(len_a, kvh, 0.2),
                &sl(&table_a, 0..len_a),
            )
            .unwrap();
            write_kv_to_cache(
                &mut cache,
                0,
                &mk(len_b, kvh, 0.3),
                &mk(len_b, kvh, 0.4),
                &sl(&table_b, 0..len_b),
            )
            .unwrap();
            let q = mk(len_a + 1, heads, 0.5);
            let batch = ForwardBatch {
                cu_seqlens_q: vec![0, len_a as u32, len_a as u32 + 1],
                cu_seqlens_k: vec![0, len_a as u32, (len_a + len_b) as u32],
                block_tables: vec![table_a.to_vec(), table_b.to_vec()],
                max_seqlen_q: len_a,
                max_seqlen_k: len_b,
                ..Default::default()
            };
            let scale = 1.0 / (hd as f32).sqrt();
            let want =
                paged_attention_cpu(&q, &cache.k[0], &cache.v[0], &batch, scale, window).unwrap();
            let got = paged_attention_cuda(
                &q,
                &cache.k[0],
                &cache.v[0],
                &batch,
                scale,
                BLOCK_SIZE,
                window,
            )
            .unwrap();
            let d = max_abs_diff(&got, &want);
            assert!(
                d < 1e-2,
                "heads {heads} window {window:?}: max abs diff {d}"
            );
        }
    }

    #[test]
    fn a_sliding_window_matches_the_cpu_reference_and_differs_from_full() {
        // Laguna's local layers: a 16-token window over a 90-key sequence.
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let m = mixed(&dev, DType::BF16);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let want = paged_attention_cpu(
            &m.q,
            &m.cache.k[0],
            &m.cache.v[0],
            &m.batch,
            scale,
            Some(16),
        )
        .unwrap();
        let got = paged_attention_cuda(
            &m.q,
            &m.cache.k[0],
            &m.cache.v[0],
            &m.batch,
            scale,
            BLOCK_SIZE,
            Some(16),
        )
        .unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-2);
        let full =
            paged_attention_cpu(&m.q, &m.cache.k[0], &m.cache.v[0], &m.batch, scale, None).unwrap();
        assert!(
            max_abs_diff(&full, &want) > 1e-2,
            "the window must hide something"
        );
    }

    #[test]
    fn the_flash_kernel_matches_the_cpu_reference_in_bf16() {
        check_dtype(DType::BF16, 1e-2);
    }

    #[test]
    fn the_flash_kernel_matches_the_cpu_reference_in_f16() {
        check_dtype(DType::F16, 1e-3);
    }

    #[test]
    fn a_bidirectional_window_is_not_causal() {
        // F1: with window_size_right = None the kernel lets prefill tokens see
        // their own future. This pins the fix: reverting it changes the output
        // of the seqlen_q > 1 rows and this test fails.
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let m = mixed(&dev, DType::BF16);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let causal = paged_attention_cuda_windowed(
            &m.q,
            &m.cache.k[0],
            &m.cache.v[0],
            &m.batch,
            scale,
            BLOCK_SIZE,
            None,
            Some(0),
        )
        .unwrap();
        let bidirectional = paged_attention_cuda_windowed(
            &m.q,
            &m.cache.k[0],
            &m.cache.v[0],
            &m.batch,
            scale,
            BLOCK_SIZE,
            None,
            None,
        )
        .unwrap();
        let want =
            paged_attention_cpu(&m.q, &m.cache.k[0], &m.cache.v[0], &m.batch, scale, None).unwrap();
        assert!(max_abs_diff(&causal, &want) < 1e-2);
        // The decode row (last one) sees every key either way; the prefill
        // rows must differ.
        let prefill_diff = max_abs_diff(
            &bidirectional.narrow(0, 0, 20).unwrap(),
            &want.narrow(0, 0, 20).unwrap(),
        );
        assert!(
            prefill_diff > 1e-2,
            "window_size_right = None must not look causal (diff {prefill_diff})"
        );
    }
}
