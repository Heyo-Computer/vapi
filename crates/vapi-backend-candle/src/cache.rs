//! Device-side paged KV buffers.
//!
//! The engine deals in block *ids*; this is the memory those ids index into.

use candle_core::{DType as CandleDType, Device, Result as CandleResult, Tensor};

/// Per-layer K and V buffers, each `(num_blocks, block_size, num_kv_heads, head_dim)`.
///
/// That layout is not a free choice — it is exactly what
/// `flash_attn_varlen_paged_windowed` requires, so the CUDA path needs no
/// reshaping or staging copy on the hot path.
pub struct PagedKvCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub num_blocks: usize,
    pub block_size: usize,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: CandleDType,
        device: &Device,
    ) -> CandleResult<Self> {
        let shape = (num_blocks, block_size, num_kv_heads, head_dim);
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(Tensor::zeros(shape, dtype, device)?);
            v.push(Tensor::zeros(shape, dtype, device)?);
        }
        Ok(Self {
            k,
            v,
            num_blocks,
            block_size,
        })
    }

    /// Flat slot count: the cache viewed as `(num_blocks * block_size, ...)`,
    /// which is the addressing `ForwardBatch::slot_mapping` uses.
    pub fn num_slots(&self) -> usize {
        self.num_blocks * self.block_size
    }
}

/// Write a step's K/V into the paged cache.
///
/// This is `reshape_and_cache` in vLLM. View the layer's cache as
/// `(num_blocks * block_size, num_kv_heads, head_dim)` and scatter each
/// token's K/V to `slot_mapping[i]`; the engine has already computed those
/// slots as `block_table[pos / block_size] * block_size + pos % block_size`.
///
/// Worth prototyping standalone before building on it: candle's exact
/// index-rank and broadcast semantics for `scatter`/`index_add` decide whether
/// this is a one-liner or needs an `InplaceOp2` with a hand-written kernel.
/// It sits in the innermost loop, so it is the one op whose cost is worth
/// measuring early.
pub fn write_kv_to_cache(
    _cache: &mut PagedKvCache,
    _layer: usize,
    _k: &Tensor,
    _v: &Tensor,
    _slot_mapping: &[u32],
) -> CandleResult<()> {
    todo!("scatter K/V into the flattened cache view at slot_mapping")
}
