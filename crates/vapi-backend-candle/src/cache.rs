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
    /// One per layer. Layers that keep only a single per-token buffer (a
    /// short-convolution layer's input history, say) hold an empty tensor
    /// here; nothing reads it.
    pub v: Vec<Tensor>,
    pub num_blocks: usize,
    pub block_size: usize,
}

/// What one layer stores per token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerLayout {
    /// K and V, each `(num_kv_heads, head_dim)` per token.
    Kv {
        num_kv_heads: usize,
        head_dim: usize,
    },
    /// A single `(1, width)` row per token, K only.
    Rows { width: usize },
}

impl LayerLayout {
    /// Elements stored per token for this layer.
    pub fn elems_per_token(&self) -> usize {
        match *self {
            LayerLayout::Kv {
                num_kv_heads,
                head_dim,
            } => 2 * num_kv_heads * head_dim,
            LayerLayout::Rows { width } => width,
        }
    }
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
        let layout = LayerLayout::Kv {
            num_kv_heads,
            head_dim,
        };
        Self::with_layouts(
            &vec![layout; num_layers],
            num_blocks,
            block_size,
            dtype,
            device,
        )
    }

    /// Allocate with a possibly different layout per layer.
    pub fn with_layouts(
        layouts: &[LayerLayout],
        num_blocks: usize,
        block_size: usize,
        dtype: CandleDType,
        device: &Device,
    ) -> CandleResult<Self> {
        let mut k = Vec::with_capacity(layouts.len());
        let mut v = Vec::with_capacity(layouts.len());
        for layout in layouts {
            match *layout {
                LayerLayout::Kv {
                    num_kv_heads,
                    head_dim,
                } => {
                    let shape = (num_blocks, block_size, num_kv_heads, head_dim);
                    k.push(Tensor::zeros(shape, dtype, device)?);
                    v.push(Tensor::zeros(shape, dtype, device)?);
                }
                LayerLayout::Rows { width } => {
                    k.push(Tensor::zeros(
                        (num_blocks, block_size, 1, width),
                        dtype,
                        device,
                    )?);
                    v.push(Tensor::zeros((0,), dtype, device)?);
                }
            }
        }
        Ok(Self {
            k,
            v,
            num_blocks,
            block_size,
        })
    }

    /// Elements per token across all layers, for sizing.
    pub fn elems_per_token(layouts: &[LayerLayout]) -> usize {
        layouts.iter().map(|l| l.elems_per_token()).sum()
    }

    /// Flat slot count: the cache viewed as `(num_blocks * block_size, ...)`,
    /// which is the addressing `ForwardBatch::slot_mapping` uses.
    pub fn num_slots(&self) -> usize {
        self.num_blocks * self.block_size
    }
}

/// Write a step's K/V into the paged cache, in place.
///
/// This is `reshape_and_cache` in vLLM. `k` and `v` are
/// `(num_tokens, num_kv_heads, head_dim)`; token `i` lands at flat slot
/// `slot_mapping[i]`, which the engine computed as
/// `block_table[pos / block_size] * block_size + pos % block_size`.
///
/// The layer's cache is viewed as `(num_blocks * block_size, kv_heads,
/// head_dim)` — a reshape of a contiguous tensor shares storage in candle, so
/// writing through the view mutates the paged buffer the attention kernel
/// reads — and each maximal run of consecutive slots is written with one
/// `Tensor::slice_set`. `slice_set` and `scatter_set` are the two in-place
/// options in candle 0.11; `scatter` and `slice_assign` return a fresh tensor,
/// which would copy the whole layer cache every step. A prefill chunk is
/// mostly long runs inside blocks, and a decode batch is one run per
/// sequence, so the number of copies stays small either way.
pub fn write_kv_to_cache(
    cache: &mut PagedKvCache,
    layer: usize,
    k: &Tensor,
    v: &Tensor,
    slot_mapping: &[u32],
) -> CandleResult<()> {
    if v.dims() != k.dims() {
        candle_core::bail!("k {:?} and v {:?} differ in shape", k.dims(), v.dims());
    }
    write_rows(&cache.k[layer], cache.num_slots(), k, slot_mapping)?;
    write_rows(&cache.v[layer], cache.num_slots(), v, slot_mapping)
}

/// Write a step's rows into a K-only layer (`LayerLayout::Rows`): `rows` is
/// `(num_tokens, 1, width)`.
pub fn write_k_to_cache(
    cache: &mut PagedKvCache,
    layer: usize,
    rows: &Tensor,
    slot_mapping: &[u32],
) -> CandleResult<()> {
    write_rows(&cache.k[layer], cache.num_slots(), rows, slot_mapping)
}

/// The scatter index for a step, built once and reused by every layer that
/// shares a row shape: `slot_mapping` broadcast to `(n, heads, width)`, the
/// shape candle's `scatter_set` wants. One host-to-device copy per shape per
/// step instead of one per layer.
pub fn scatter_index(
    slot_mapping: &[u32],
    heads: usize,
    width: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    Tensor::from_vec(slot_mapping.to_vec(), (slot_mapping.len(), 1, 1), device)?
        .broadcast_as((slot_mapping.len(), heads, width))?
        .contiguous()
}

/// Write K rows `(n, heads, width)` at the slots a prebuilt
/// [`scatter_index`] names, as one scatter.
pub fn write_rows_indexed(
    cache: &mut PagedKvCache,
    layer: usize,
    src: &Tensor,
    idx: &Tensor,
) -> CandleResult<()> {
    let num_slots = cache.num_slots();
    scatter_rows(&cache.k[layer], num_slots, src, idx)
}

/// [`write_rows_indexed`] for V.
pub fn write_v_rows_indexed(
    cache: &mut PagedKvCache,
    layer: usize,
    src: &Tensor,
    idx: &Tensor,
) -> CandleResult<()> {
    let num_slots = cache.num_slots();
    scatter_rows(&cache.v[layer], num_slots, src, idx)
}

fn scatter_rows(
    cache_layer: &Tensor,
    num_slots: usize,
    src: &Tensor,
    idx: &Tensor,
) -> CandleResult<()> {
    let (n, heads, width) = src.dims3()?;
    if idx.dims() != [n, heads, width] {
        candle_core::bail!(
            "scatter index {:?} does not match rows {:?}",
            idx.dims(),
            src.dims()
        );
    }
    let flat = cache_layer.reshape((num_slots, heads, width))?;
    flat.scatter_set(idx, &src.contiguous()?, 0)
}

fn write_rows(
    cache_layer: &Tensor,
    num_slots: usize,
    src: &Tensor,
    slot_mapping: &[u32],
) -> CandleResult<()> {
    let (n, heads, width) = src.dims3()?;
    if n != slot_mapping.len() {
        candle_core::bail!("{n} tokens but {} cache slots", slot_mapping.len());
    }
    if cache_layer.dtype() != src.dtype() {
        candle_core::bail!(
            "cache is {:?} but the rows are {:?}",
            cache_layer.dtype(),
            src.dtype()
        );
    }
    let flat = cache_layer.reshape((num_slots, heads, width))?;
    let src = src.contiguous()?;
    if let Some(&s) = slot_mapping.iter().max()
        && s as usize >= num_slots
    {
        candle_core::bail!("slot {s} is past the end of the cache");
    }

    // Runs of consecutive slots: a prefill chunk is a few long runs, a
    // decode batch is one run per sequence.
    let mut runs = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && slot_mapping[end] == slot_mapping[end - 1] + 1 {
            end += 1;
        }
        runs.push((start, end));
        start = end;
    }

    if runs.len() <= SCATTER_ABOVE_RUNS {
        for (start, end) in runs {
            flat.slice_set(
                &src.narrow(0, start, end - start)?,
                0,
                slot_mapping[start] as usize,
            )?;
        }
        return Ok(());
    }
    // Many small runs: one scatter for the whole step instead of one copy
    // per run. candle's scatter wants an index shaped like the source, so
    // the slot ids are broadcast across the row (n × heads × width u32,
    // 128 KB for a 64-sequence decode over 8 × 64).
    let idx = Tensor::from_vec(slot_mapping.to_vec(), (n, 1, 1), src.device())?
        .broadcast_as((n, heads, width))?
        .contiguous()?;
    flat.scatter_set(&idx, &src, 0)
}

/// Above this many runs a step is written with one `scatter_set` rather
/// than one `slice_set` per run. A decode batch of this size or more is
/// where the per-launch cost dominates.
const SCATTER_ABOVE_RUNS: usize = 4;

/// Gather one sequence's K (or V) rows from the paged cache, in logical order:
/// `(len_k, kv_heads, head_dim)`. The CPU reference attention and the tests
/// both read the cache through this.
pub fn gather_kv(cache_layer: &Tensor, block_table: &[u32], len_k: usize) -> CandleResult<Tensor> {
    let (_, block_size, kv_heads, head_dim) = cache_layer.dims4()?;
    let table = Tensor::from_slice(block_table, block_table.len(), cache_layer.device())?;
    cache_layer
        .index_select(&table, 0)?
        .reshape((block_table.len() * block_size, kv_heads, head_dim))?
        .narrow(0, 0, len_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 4;

    fn cache(num_blocks: usize) -> PagedKvCache {
        PagedKvCache::new(2, num_blocks, BS, 2, 3, CandleDType::F32, &Device::Cpu).unwrap()
    }

    fn rows(n: usize, base: f32) -> Tensor {
        // Every element distinct, so a misplaced row is detectable.
        let data: Vec<f32> = (0..n * 2 * 3).map(|i| base + i as f32).collect();
        Tensor::from_vec(data, (n, 2, 3), &Device::Cpu).unwrap()
    }

    #[test]
    fn a_write_spanning_non_adjacent_blocks_lands_in_exactly_those_slots() {
        let mut c = cache(8);
        // A prefill chunk of 9 tokens over block table [6, 1, 3]: slots
        // 24..28, 4..8, 12 — three separate runs.
        let slots: Vec<u32> = vec![24, 25, 26, 27, 4, 5, 6, 7, 12];
        let k = rows(9, 100.0);
        let v = rows(9, 1000.0);
        write_kv_to_cache(&mut c, 1, &k, &v, &slots).unwrap();

        let got_k = gather_kv(&c.k[1], &[6, 1, 3], 9).unwrap();
        let got_v = gather_kv(&c.v[1], &[6, 1, 3], 9).unwrap();
        assert_eq!(got_k.to_vec3::<f32>().unwrap(), k.to_vec3::<f32>().unwrap());
        assert_eq!(got_v.to_vec3::<f32>().unwrap(), v.to_vec3::<f32>().unwrap());

        // Nothing else moved: the other layer is untouched, and every slot
        // not in the mapping is still zero.
        let other: f32 = c.k[0].sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(other, 0.0);
        let flat = c.k[1]
            .reshape((32, 2, 3))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for (slot, row) in flat.iter().enumerate() {
            let written = slots.contains(&(slot as u32));
            let nonzero = row.iter().flatten().any(|&x| x != 0.0);
            assert_eq!(nonzero, written, "slot {slot}");
        }
    }

    #[test]
    fn the_flat_view_really_shares_storage_with_the_paged_tensor() {
        // The whole scheme rests on this: if reshape copied, the attention
        // kernel would read a cache nothing was ever written to.
        let mut c = cache(2);
        write_kv_to_cache(&mut c, 0, &rows(1, 7.0), &rows(1, 9.0), &[5]).unwrap();
        let paged = c.k[0].to_vec3::<f32>();
        assert!(paged.is_err(), "4-D tensor; sanity check on the layout");
        let block1 = c.k[0].get(1).unwrap(); // (BS, 2, 3)
        let row = block1.get(1).unwrap().to_vec2::<f32>().unwrap(); // slot 5 = block 1, offset 1
        assert_eq!(row, vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]]);
    }

    #[test]
    fn a_decode_batch_is_many_single_slot_runs() {
        // 6 runs: above the scatter threshold, so this is the scatter path.
        let mut c = cache(4);
        let slots = [3u32, 9, 0, 14, 6, 11];
        let k = rows(6, 1.0);
        write_kv_to_cache(&mut c, 0, &k, &k, &slots).unwrap();
        let flat = c.k[0].reshape((16, 2, 3)).unwrap();
        for (i, &s) in slots.iter().enumerate() {
            let want = k.get(i).unwrap().to_vec2::<f32>().unwrap();
            let got = flat.get(s as usize).unwrap().to_vec2::<f32>().unwrap();
            assert_eq!(got, want, "token {i} at slot {s}");
        }
    }

    #[test]
    fn the_scatter_path_and_the_run_path_agree() {
        // Same rows, same slots, written both ways into two caches.
        let slots: Vec<u32> = vec![30, 31, 5, 6, 7, 12, 20, 2]; // 5 runs
        let k = rows(8, 3.0);
        let mut a = cache(8);
        let mut b = cache(8);
        write_kv_to_cache(&mut a, 0, &k, &k, &slots).unwrap(); // scatter (5 > 4)
        for (i, &s) in slots.iter().enumerate() {
            let row = k.narrow(0, i, 1).unwrap();
            write_kv_to_cache(&mut b, 0, &row, &row, &[s]).unwrap(); // 1 run each
        }
        let fa = a.k[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let fb = b.k[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(fa, fb);
        assert!(fa.iter().filter(|&&x| x != 0.0).count() > 0);
    }

    #[test]
    fn a_prebuilt_scatter_index_writes_the_same_as_the_slot_mapping() {
        let slots: Vec<u32> = vec![30, 31, 5, 6, 7, 12, 20, 2];
        let k = rows(8, 3.0);
        let mut a = cache(8);
        let mut b = cache(8);
        write_kv_to_cache(&mut a, 0, &k, &k, &slots).unwrap();
        let idx = scatter_index(&slots, 2, 3, &Device::Cpu).unwrap();
        write_rows_indexed(&mut b, 0, &k, &idx).unwrap();
        write_v_rows_indexed(&mut b, 0, &k, &idx).unwrap();
        for t in [(&a.k[0], &b.k[0]), (&a.v[0], &b.v[0])] {
            let fa = t.0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let fb = t.1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(fa, fb);
        }
        // A mismatched index shape is an error, not a silent misplacement.
        let wrong = scatter_index(&slots, 1, 3, &Device::Cpu).unwrap();
        assert!(write_rows_indexed(&mut b, 0, &k, &wrong).is_err());
    }

    #[test]
    fn a_second_write_overwrites_rather_than_accumulates() {
        let mut c = cache(2);
        write_kv_to_cache(&mut c, 0, &rows(1, 1.0), &rows(1, 1.0), &[2]).unwrap();
        write_kv_to_cache(&mut c, 0, &rows(1, 50.0), &rows(1, 50.0), &[2]).unwrap();
        let flat = c.k[0].reshape((8, 2, 3)).unwrap();
        assert_eq!(flat.get(2).unwrap().to_vec2::<f32>().unwrap()[0][0], 50.0);
    }

    #[test]
    fn a_rows_layer_stores_one_row_per_token_and_no_v() {
        let layouts = [
            LayerLayout::Kv {
                num_kv_heads: 2,
                head_dim: 3,
            },
            LayerLayout::Rows { width: 5 },
        ];
        let mut c =
            PagedKvCache::with_layouts(&layouts, 4, BS, CandleDType::F32, &Device::Cpu).unwrap();
        assert_eq!(c.k[1].dims(), &[4, BS, 1, 5]);
        assert_eq!(c.v[1].elem_count(), 0);
        assert_eq!(PagedKvCache::elems_per_token(&layouts), 2 * 2 * 3 + 5);
        let rows =
            Tensor::from_vec((0..10).map(|i| i as f32).collect(), (2, 1, 5), &Device::Cpu).unwrap();
        write_k_to_cache(&mut c, 1, &rows, &[9, 10]).unwrap();
        let got = gather_kv(&c.k[1], &[2], 3).unwrap(); // block 2 = slots 8..12
        let got = got.reshape((3, 5)).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(got[0], vec![0.0; 5]);
        assert_eq!(got[1], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(got[2], vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn mismatched_shapes_and_out_of_range_slots_are_errors() {
        let mut c = cache(2);
        assert!(write_kv_to_cache(&mut c, 0, &rows(2, 0.0), &rows(2, 0.0), &[0]).is_err());
        assert!(write_kv_to_cache(&mut c, 0, &rows(1, 0.0), &rows(1, 0.0), &[8]).is_err());
        let wrong_dtype = rows(1, 0.0).to_dtype(CandleDType::F16).unwrap();
        assert!(write_kv_to_cache(&mut c, 0, &wrong_dtype, &wrong_dtype, &[0]).is_err());
    }
}
