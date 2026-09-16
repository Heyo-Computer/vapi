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

use candle_core::{Device, Result as CandleResult, Tensor};
use vapi_engine::ForwardBatch;

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

/// Cumulative sequence lengths as the `i32` tensor the kernel expects.
pub fn cu_seqlens(values: &[u32], device: &Device) -> CandleResult<Tensor> {
    let v: Vec<i32> = values.iter().map(|&x| x as i32).collect();
    Tensor::from_vec(v, values.len(), device)
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
) -> CandleResult<Tensor> {
    let device = q.device();
    let block_table = pack_block_tables(batch, device)?;
    let seqlens_q = cu_seqlens(&batch.cu_seqlens_q, device)?;
    let seqlens_k = cu_seqlens(&batch.cu_seqlens_k, device)?;

    candle_flash_attn::flash_attn_varlen_paged_windowed(
        q,
        k_cache,
        v_cache,
        &seqlens_q,
        &seqlens_k,
        &block_table,
        None, // mm_prefix_ranges: multimodal only
        batch.max_seqlen_q,
        batch.max_seqlen_k,
        softmax_scale,
        None, // window_size_left: None = full causal attention
        None, // window_size_right
        block_size,
        None, // softcap
    )
}

#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;

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
    fn cumulative_lengths_convert_to_i32() {
        let t = cu_seqlens(&[0, 4, 9], &Device::Cpu).unwrap();
        assert_eq!(t.to_vec1::<i32>().unwrap(), vec![0, 4, 9]);
    }
}
