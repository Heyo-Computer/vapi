//! [`ExecutionBackend`] over real weights.

use std::path::{Path, PathBuf};

use candle_core::{DType as CandleDType, Device, Tensor};
use candle_nn::VarBuilder;
use vapi_cache::BlockId;
use vapi_core::config::{BLOCK_SIZE, DType, DeviceKind};
use vapi_core::{Error, Result};
use vapi_engine::backend::{
    CandidateNeed, ExecutionBackend, ForwardBatch, Logits, ModelSpec, StepLogits,
};
#[cfg(feature = "cuda")]
use vapi_engine::backend::{RowCandidates, RowLogits};

use crate::cache::{LayerLayout, PagedKvCache};
use crate::models::laguna::{LagunaConfig, PagedLaguna};
use crate::models::lfm2::{Lfm2Config, PagedLfm2};
use crate::models::llama::{AttentionImpl, LlamaConfig, PagedLlama};

/// The architectures the backend can run.
enum Model {
    Llama(PagedLlama),
    Laguna(PagedLaguna),
    Lfm2(PagedLfm2),
}

impl Model {
    fn forward(
        &self,
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> candle_core::Result<Option<Tensor>> {
        match self {
            Model::Llama(m) => m.forward(batch, cache),
            Model::Laguna(m) => m.forward(batch, cache),
            Model::Lfm2(m) => m.forward(batch, cache),
        }
    }
}

/// How to load a model.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub dtype: DType,
    pub device: DeviceKind,
    /// Fixed block count. Required on CPU; on CUDA it overrides sizing from
    /// free memory.
    pub num_blocks: Option<usize>,
    /// Fraction of *free* device memory the KV cache may claim, on CUDA.
    pub kv_cache_fraction: f32,
    /// Capture pure-decode steps into CUDA graphs (LFM2 on CUDA only).
    pub cuda_graphs: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Auto,
            device: DeviceKind::Auto,
            num_blocks: None,
            kv_cache_fraction: 0.85,
            cuda_graphs: false,
        }
    }
}

/// candle-backed execution: a paged model plus its KV buffers.
pub struct CandleBackend {
    model: Model,
    cache: PagedKvCache,
    spec: ModelSpec,
    device: Device,
    /// Blocks at the top of the cache that the pool never hands out: the
    /// warm-up target and, with CUDA graphs, the padding rows' home.
    reserved_blocks: usize,
    #[cfg(feature = "cuda")]
    graphs: Option<graphs::GraphRunner>,
}

fn engine_err(e: impl std::fmt::Display) -> Error {
    Error::Engine(e.to_string())
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))
}

/// `eos_token_id` is a scalar or a list; instruct models usually stop on
/// several tokens.
fn eos_ids(v: &serde_json::Value) -> Vec<u32> {
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as u32)
            .collect(),
        _ => Vec::new(),
    }
}

/// Safetensors shards in a model directory: the index's `weight_map` when
/// sharded, else the single `model.safetensors`.
fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if let Some(index) = read_json(&dir.join("model.safetensors.index.json"))? {
        let mut names: Vec<String> = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Err(Error::Config(
                "model.safetensors.index.json has an empty weight_map".into(),
            ));
        }
        return Ok(names.into_iter().map(|n| dir.join(n)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(Error::Config(format!(
        "{}: no model.safetensors or model.safetensors.index.json",
        dir.display()
    )))
}

fn pick_device(kind: DeviceKind) -> Result<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Device::new_cuda(0).map_err(engine_err),
        #[cfg(feature = "cuda")]
        DeviceKind::Auto => Ok(Device::new_cuda(0).unwrap_or(Device::Cpu)),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => Err(Error::Config(
            "model.device = cuda but this binary was built without the `cuda` feature".into(),
        )),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Auto => Ok(Device::Cpu),
    }
}

fn pick_dtype(dtype: DType, device: &Device) -> CandleDType {
    match dtype {
        DType::F32 => CandleDType::F32,
        DType::F16 => CandleDType::F16,
        DType::Bf16 => CandleDType::BF16,
        DType::Auto if device.is_cuda() => CandleDType::BF16,
        DType::Auto => CandleDType::F32,
    }
}

/// Free device memory in bytes, when the device can report it.
#[cfg(feature = "cuda")]
fn free_memory(device: &Device) -> Option<u64> {
    if !device.is_cuda() {
        return None;
    }
    candle_core::cuda::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

#[cfg(not(feature = "cuda"))]
fn free_memory(_device: &Device) -> Option<u64> {
    None
}

impl CandleBackend {
    /// Load a Llama-architecture model from a local directory holding
    /// `config.json`, the safetensors weights and optionally
    /// `generation_config.json`.
    pub fn load(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let config_json = read_json(&dir.join("config.json"))?
            .ok_or_else(|| Error::Config(format!("{}: config.json not found", dir.display())))?;
        let arch = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();
        if !["llama", "laguna", "lfm2"].contains(&arch.as_str()) {
            return Err(Error::Config(format!(
                "model_type {arch:?} is not supported; llama, laguna and lfm2 are"
            )));
        }

        let device = pick_device(opts.device)?;
        let dtype = pick_dtype(opts.dtype, &device);
        let attention = match () {
            #[cfg(feature = "cuda")]
            () if device.is_cuda() => AttentionImpl::FlashPaged,
            () => AttentionImpl::Reference,
        };

        let files = weight_files(dir)?;
        // Safety: the mmap is read-only and the files are not modified while
        // the process runs; this is candle's standard loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(engine_err)?;
        // (model, num_layers, num_kv_heads, head_dim, vocab, max_context).
        // `layouts` is set by architectures whose layers do not all store
        // a K/V pair per token.
        let mut layouts: Option<Vec<LayerLayout>> = None;
        let (model, num_layers, num_kv_heads, head_dim, vocab_size, max_context) = if arch == "lfm2"
        {
            let cfg = Lfm2Config::from_json(&config_json).map_err(engine_err)?;
            let model = PagedLfm2::load(vb, &cfg, dtype, &device, attention).map_err(engine_err)?;
            layouts = Some(cfg.cache_layouts());
            (
                Model::Lfm2(model),
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.vocab_size,
                cfg.max_position_embeddings,
            )
        } else if arch == "laguna" {
            let cfg = LagunaConfig::from_json(&config_json).map_err(engine_err)?;
            let model =
                PagedLaguna::load(vb, &cfg, dtype, &device, attention).map_err(engine_err)?;
            (
                Model::Laguna(model),
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.vocab_size,
                cfg.max_position_embeddings,
            )
        } else {
            let llama_cfg: LlamaConfig = serde_json::from_value(config_json.clone())
                .map_err(|e| Error::Config(format!("config.json: {e}")))?;
            let cfg = llama_cfg.into_config(false);
            let model =
                PagedLlama::load(vb, &cfg, dtype, &device, attention).map_err(engine_err)?;
            let head_dim = config_json
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
            (
                Model::Llama(model),
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                head_dim,
                cfg.vocab_size,
                cfg.max_position_embeddings,
            )
        };

        let gen_config = read_json(&dir.join("generation_config.json"))?.unwrap_or_default();
        let mut eos = eos_ids(&gen_config);
        if eos.is_empty() {
            eos = eos_ids(&config_json);
        }
        eos.sort_unstable();
        eos.dedup();

        let spec = ModelSpec {
            num_layers,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_context,
            eos_token_ids: eos,
        };

        let layouts = layouts.unwrap_or_else(|| {
            vec![
                LayerLayout::Kv {
                    num_kv_heads: spec.num_kv_heads,
                    head_dim: spec.head_dim,
                };
                spec.num_layers
            ]
        });
        let bytes_per_token = PagedKvCache::elems_per_token(&layouts) * dtype.size_in_bytes();
        let num_blocks = match (opts.num_blocks, free_memory(&device)) {
            (Some(n), _) => n,
            (None, Some(free)) => {
                let budget = (free as f64 * opts.kv_cache_fraction as f64) as u64;
                (budget / (bytes_per_token * BLOCK_SIZE) as u64) as usize
            }
            (None, None) => {
                return Err(Error::Config(
                    "model.num_blocks is required when the device cannot report free memory".into(),
                ));
            }
        };
        if num_blocks == 0 {
            return Err(Error::Config("KV cache budget fits zero blocks".into()));
        }
        tracing::info!(
            num_blocks,
            ?dtype,
            device = if device.is_cuda() { "cuda" } else { "cpu" },
            kv_mib = num_blocks * BLOCK_SIZE * bytes_per_token / (1 << 20),
            "kv cache allocated"
        );
        let cache = PagedKvCache::with_layouts(&layouts, num_blocks, BLOCK_SIZE, dtype, &device)
            .map_err(engine_err)?;

        #[cfg(feature = "cuda")]
        let graphs = if opts.cuda_graphs && device.is_cuda() && matches!(model, Model::Lfm2(_)) {
            Some(graphs::GraphRunner::new(&device))
        } else {
            if opts.cuda_graphs {
                tracing::warn!("cuda_graphs is set but only applies to LFM2 on a CUDA device");
            }
            None
        };
        #[cfg(feature = "cuda")]
        let reserved_blocks = if graphs.is_some() {
            graphs::RESERVED_BLOCKS
        } else {
            1
        };
        #[cfg(not(feature = "cuda"))]
        let reserved_blocks = 1;
        if num_blocks <= reserved_blocks {
            return Err(Error::Config(format!(
                "num_blocks must exceed the {reserved_blocks} reserved block(s)"
            )));
        }
        let mut backend = Self {
            model,
            cache,
            spec,
            device,
            reserved_blocks,
            #[cfg(feature = "cuda")]
            graphs,
        };
        backend.warm_up()?;
        Ok(backend)
    }

    /// One tiny forward at load, so the first real request does not pay for
    /// CUDA context and kernel initialisation (~180 ms measured). Uses the
    /// last block, which a sequence always writes before it reads.
    fn warm_up(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let block = (self.cache.num_blocks - 1) as u32;
        let batch = ForwardBatch {
            tokens: vec![0],
            positions: vec![0],
            cu_seqlens_q: vec![0, 1],
            cu_seqlens_k: vec![0, 1],
            slot_mapping: vec![block * BLOCK_SIZE as u32],
            block_tables: vec![vec![block]],
            logits_indices: vec![0],
            max_seqlen_q: 1,
            max_seqlen_k: 1,
        };
        self.forward_greedy(&batch)?;
        tracing::info!(ms = t0.elapsed().as_millis(), "warm-up forward");
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl CandleBackend {
    /// The forward pass proper: f32 logits still on the device (or `None`
    /// when the batch samples nothing) and the instant the launches
    /// finished, so callers can attribute the wait that follows.
    fn run(&mut self, batch: &ForwardBatch) -> Result<(Option<Tensor>, std::time::Instant)> {
        // Same discipline as MockBackend: a malformed batch here means wrong
        // text, not a crash, so check it while the cost is affordable.
        #[cfg(debug_assertions)]
        batch.validate(self.cache.num_blocks, self.cache.block_size)?;

        let t0 = std::time::Instant::now();
        let logits = self
            .model
            .forward(batch, &mut self.cache)
            .map_err(engine_err)?;
        let t1 = std::time::Instant::now();
        // The device work is asynchronous until a host copy forces it, so
        // this is launch time; the copy histogram is mostly the wait.
        metrics::histogram!("vapi_forward_launch_ms").record((t1 - t0).as_secs_f64() * 1e3);
        Ok((logits, t1))
    }
}

impl ExecutionBackend for CandleBackend {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn num_blocks(&self) -> usize {
        self.cache.num_blocks - self.reserved_blocks
    }

    fn forward(&mut self, batch: &ForwardBatch) -> Result<Logits> {
        let (logits, t1) = self.run(batch)?;
        let data = match logits {
            None => Vec::new(),
            Some(t) => t
                .flatten_all()
                .map_err(engine_err)?
                .to_vec1::<f32>()
                .map_err(engine_err)?,
        };
        metrics::histogram!("vapi_forward_logits_copy_ms").record(t1.elapsed().as_secs_f64() * 1e3);
        Ok(Logits {
            data,
            vocab_size: self.spec.vocab_size,
        })
    }

    fn forward_candidates(
        &mut self,
        batch: &ForwardBatch,
        needs: &[CandidateNeed],
    ) -> Result<StepLogits> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            use crate::fused::{CANDIDATE_CAP, CANDIDATE_WINDOW, select_candidates};
            let (logits, t1) = self.run(batch)?;
            let Some(t) = logits else {
                return Ok(StepLogits::Full(Logits {
                    data: Vec::new(),
                    vocab_size: self.spec.vocab_size,
                }));
            };
            let rows = t.dim(0).map_err(engine_err)?;
            if needs.len() != rows {
                return Err(vapi_core::Error::Engine(format!(
                    "{} candidate needs for {rows} logits rows",
                    needs.len()
                )));
            }
            // Pure-temperature rows are drawn on the device outright; the
            // rest go through candidate selection.
            let gumbel: Vec<Option<crate::fused::GumbelRow>> = needs
                .iter()
                .map(|n| {
                    n.gumbel
                        .as_ref()
                        .filter(|_| !n.full)
                        .map(|g| crate::fused::GumbelRow {
                            inv_temperature: n.inv_temperature,
                            seed: g.seed,
                            draw: g.draw,
                            observed: g.observed.clone(),
                            repetition_penalty: g.repetition_penalty,
                            frequency_penalty: g.frequency_penalty,
                            presence_penalty: g.presence_penalty,
                        })
                })
                .collect();
            let drawn = if gumbel.iter().any(|g| g.is_some()) {
                crate::fused::gumbel_draw(&t, &gumbel).map_err(engine_err)?
            } else {
                vec![u32::MAX; rows]
            };
            let inv_t: Vec<f32> = needs.iter().map(|n| n.inv_temperature).collect();
            let inv_t = Tensor::from_vec(inv_t, rows, &self.device).map_err(engine_err)?;
            let sel = select_candidates(&t, &inv_t).map_err(engine_err)?;
            let counts = sel.counts.to_vec1::<u32>().map_err(engine_err)?;
            let depths = sel.depths.to_vec1::<u32>().map_err(engine_err)?;
            let covered: Vec<bool> = needs
                .iter()
                .zip(&counts)
                .zip(&depths)
                .zip(&gumbel)
                .map(|(((need, &c), &d), g)| {
                    let c = c as usize;
                    !need.full
                        && (g.is_some()
                            || (c <= CANDIDATE_CAP
                                && (d == CANDIDATE_WINDOW
                                    || (need.needed > 0 && c >= need.needed))))
                })
                .collect();
            // Rows the selection cannot serve exactly get their full logits;
            // only those rows are copied.
            let uncovered: Vec<u32> = (0..rows as u32).filter(|&r| !covered[r as usize]).collect();
            let mut full_rows: Vec<Vec<f32>> = Vec::new();
            if !uncovered.is_empty() {
                metrics::counter!("vapi_candidates_fallback_rows_total")
                    .increment(uncovered.len() as u64);
                let idx = Tensor::from_vec(uncovered.clone(), uncovered.len(), &self.device)
                    .map_err(engine_err)?;
                full_rows = t
                    .index_select(&idx, 0)
                    .map_err(engine_err)?
                    .to_vec2::<f32>()
                    .map_err(engine_err)?;
            }
            let widest = counts
                .iter()
                .zip(&covered)
                .zip(&gumbel)
                .filter(|((_, c), g)| **c && g.is_none())
                .map(|((&n, _), _)| n as usize)
                .max()
                .unwrap_or(0)
                .max(1);
            let ids = sel
                .ids
                .narrow(1, 0, widest)
                .map_err(engine_err)?
                .contiguous()
                .map_err(engine_err)?
                .to_vec2::<u32>()
                .map_err(engine_err)?;
            let vals = sel
                .vals
                .narrow(1, 0, widest)
                .map_err(engine_err)?
                .contiguous()
                .map_err(engine_err)?
                .to_vec2::<f32>()
                .map_err(engine_err)?;
            let maxes = sel.maxes.to_vec1::<f32>().map_err(engine_err)?;
            let sums = sel.sums.to_vec1::<f32>().map_err(engine_err)?;
            metrics::histogram!("vapi_forward_logits_copy_ms")
                .record(t1.elapsed().as_secs_f64() * 1e3);
            let mut full_rows = full_rows.into_iter();
            let out = (0..rows)
                .map(|r| {
                    if gumbel[r].is_some() {
                        return RowLogits::Token(drawn[r]);
                    }
                    if !covered[r] {
                        return RowLogits::Full(
                            full_rows.next().expect("one copy per uncovered row"),
                        );
                    }
                    let n = counts[r] as usize;
                    RowLogits::Candidates(RowCandidates {
                        entries: ids[r][..n]
                            .iter()
                            .copied()
                            .zip(vals[r][..n].iter().copied())
                            .collect(),
                        max: maxes[r],
                        sum: sums[r],
                    })
                })
                .collect();
            return Ok(StepLogits::Rows(out));
        }
        let _ = needs;
        Ok(StepLogits::Full(self.forward(batch)?))
    }

    fn forward_greedy(&mut self, batch: &ForwardBatch) -> Result<Option<Vec<u32>>> {
        #[cfg(feature = "cuda")]
        if let (Some(runner), Model::Lfm2(model)) = (&mut self.graphs, &self.model)
            && let Some(tokens) = runner.try_decode(model, &mut self.cache, batch)?
        {
            return Ok(Some(tokens));
        }
        let (logits, t1) = self.run(batch)?;
        let tokens = match logits {
            None => Vec::new(),
            // argmax on the device; only `rows × 4` bytes come back.
            Some(t) => t
                .argmax(1)
                .map_err(engine_err)?
                .to_vec1::<u32>()
                .map_err(engine_err)?,
        };
        metrics::histogram!("vapi_forward_logits_copy_ms").record(t1.elapsed().as_secs_f64() * 1e3);
        Ok(Some(tokens))
    }

    fn block_bytes(&self) -> usize {
        self.cache
            .k
            .iter()
            .chain(self.cache.v.iter())
            .map(|t| {
                if t.dims().len() < 2 {
                    return 0;
                }
                let per_block: usize = t.dims()[1..].iter().product();
                per_block * t.dtype().size_in_bytes()
            })
            .sum()
    }

    /// One block's K and V from every layer, concatenated in layer order and
    /// kept in the cache's own dtype. Read back only by the same build
    /// against the same model, which the content hash already guarantees.
    ///
    /// The per-layer slices are joined on the device and copied to the host
    /// once. Sixty small copies instead cost 52 ms a block here, twenty
    /// times what recomputing the block costs, which would make the tier
    /// worse than useless.
    fn export_block(&self, id: BlockId) -> Result<Vec<u8>> {
        let mut parts = Vec::with_capacity(self.cache.k.len() * 2);
        for t in self.cache.k.iter().chain(self.cache.v.iter()) {
            if t.dims().len() < 2 {
                continue;
            }
            parts.push(
                t.narrow(0, id.index(), 1)
                    .and_then(|b| b.flatten_all())
                    .map_err(engine_err)?,
            );
        }
        let flat = Tensor::cat(&parts, 0).map_err(engine_err)?;
        let mut out = Vec::with_capacity(self.block_bytes());
        append_bytes(&flat, &mut out)?;
        Ok(out)
    }

    /// The inverse: one host-to-device copy, then device-side scatters into
    /// each layer's buffer.
    fn import_block(&mut self, id: BlockId, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.block_bytes() {
            return Err(vapi_core::Error::Engine(format!(
                "spilled block is {} bytes, this cache wants {}",
                bytes.len(),
                self.block_bytes()
            )));
        }
        let Some(dtype) = self.cache.k.first().map(|t| t.dtype()) else {
            return Ok(());
        };
        let flat = tensor_from_bytes(bytes, dtype, &self.device).map_err(engine_err)?;
        let buffers = self.cache.k.len();
        let mut at = 0usize;
        for i in 0..buffers * 2 {
            let target = if i < buffers {
                &self.cache.k[i]
            } else {
                &self.cache.v[i - buffers]
            };
            if target.dims().len() < 2 {
                continue;
            }
            let dims: Vec<usize> = std::iter::once(1usize)
                .chain(target.dims()[1..].iter().copied())
                .collect();
            let elems: usize = dims.iter().product();
            let part = flat
                .narrow(0, at, elems)
                .and_then(|p| p.reshape(dims))
                .map_err(engine_err)?;
            target.slice_set(&part, 0, id.index()).map_err(engine_err)?;
            at += elems;
        }
        Ok(())
    }

    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]) -> Result<()> {
        // Copy-on-write for n > 1 forking. Not on any hot path yet.
        for &(src, dst) in pairs {
            for layer in 0..self.spec.num_layers {
                for buf in [&self.cache.k[layer], &self.cache.v[layer]] {
                    let block: Tensor = buf.get(src.0 as usize).map_err(engine_err)?;
                    buf.slice_set(&block.unsqueeze(0).map_err(engine_err)?, 0, dst.0 as usize)
                        .map_err(engine_err)?;
                }
            }
        }
        Ok(())
    }
}

/// A contiguous tensor's raw bytes, appended to `out`.
///
/// The bytes are the element type's native little-endian representation,
/// which is what the spill tier stores; nothing else reads them.
fn append_bytes(t: &Tensor, out: &mut Vec<u8>) -> Result<()> {
    fn extend<T: Copy>(v: &[T], out: &mut Vec<u8>) {
        // SAFETY: `T` here is only ever f32/f16/bf16/u8/u32, all plain data
        // with no padding, and the slice is read as bytes and never written.
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        out.extend_from_slice(bytes);
    }
    match t.dtype() {
        candle_core::DType::F32 => extend(&t.to_vec1::<f32>().map_err(engine_err)?, out),
        candle_core::DType::F16 => extend(&t.to_vec1::<half::f16>().map_err(engine_err)?, out),
        candle_core::DType::BF16 => extend(&t.to_vec1::<half::bf16>().map_err(engine_err)?, out),
        candle_core::DType::U8 => extend(&t.to_vec1::<u8>().map_err(engine_err)?, out),
        candle_core::DType::U32 => extend(&t.to_vec1::<u32>().map_err(engine_err)?, out),
        other => {
            return Err(vapi_core::Error::Engine(format!(
                "cannot export a {other:?} cache"
            )));
        }
    }
    Ok(())
}

/// Inverse of [`append_bytes`] for one buffer.
fn tensor_from_bytes(
    bytes: &[u8],
    dtype: candle_core::DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    fn read<T: Copy>(bytes: &[u8]) -> Vec<T> {
        let n = bytes.len() / std::mem::size_of::<T>();
        let mut out = Vec::with_capacity(n);
        // SAFETY: as in `append_bytes`; the source may be unaligned, so the
        // reads go through `read_unaligned`.
        unsafe {
            let p = bytes.as_ptr() as *const T;
            for i in 0..n {
                out.push(p.add(i).read_unaligned());
            }
        }
        out
    }
    match dtype {
        candle_core::DType::F32 => Tensor::from_vec(read::<f32>(bytes), bytes.len() / 4, device),
        candle_core::DType::F16 => {
            Tensor::from_vec(read::<half::f16>(bytes), bytes.len() / 2, device)
        }
        candle_core::DType::BF16 => {
            Tensor::from_vec(read::<half::bf16>(bytes), bytes.len() / 2, device)
        }
        candle_core::DType::U8 => Tensor::from_vec(read::<u8>(bytes), bytes.len(), device),
        candle_core::DType::U32 => Tensor::from_vec(read::<u32>(bytes), bytes.len() / 4, device),
        other => candle_core::bail!("cannot import a {other:?} cache"),
    }
}

/// CUDA graphs for pure-decode steps of LFM2.
///
/// A decode step's kernels are the same for every step of the same shape;
/// only the input buffers change. So the step is captured once per
/// (batch-size bucket, block-table width bucket) with inputs living in
/// persistent device buffers, and later steps overwrite those buffers and
/// replay. Batches are padded up to the bucket with rows that write to
/// reserved cache blocks and attend over one key, so padding never touches
/// a real sequence.
#[cfg(feature = "cuda")]
mod graphs {
    use std::collections::HashMap;

    use candle_core::cuda::cudarc::driver::{CudaGraph, sys};
    use candle_core::{DType as CandleDType, Device, Tensor};
    use vapi_core::config::BLOCK_SIZE;
    use vapi_core::{Error, Result};
    use vapi_engine::ForwardBatch;

    use crate::cache::PagedKvCache;
    use crate::models::lfm2::{Lfm2Inputs, PagedLfm2};

    /// Two blocks hold up to 64 padding rows (one slot each).
    pub const RESERVED_BLOCKS: usize = 2;
    const MAX_BUCKET: usize = 64;
    /// Block-table widths are rounded up to this many blocks.
    const TABLE_BUCKET: usize = 8;

    /// `CudaGraph` holds raw driver handles and is not `Send`, but the
    /// backend is owned by exactly one engine thread and cudarc binds the
    /// context to the calling thread on every launch, so moving the whole
    /// backend between threads (which `ExecutionBackend: Send` allows) is
    /// sound as long as it is never *shared*, which it is not.
    struct SendGraph(CudaGraph);
    // SAFETY: see above; the graph is only ever launched from the thread
    // that currently owns the backend, one launch at a time.
    unsafe impl Send for SendGraph {}

    struct DecodeGraph {
        inputs: Lfm2Inputs,
        /// `(bucket, vocab)` f32, filled by the graph; argmax runs on it
        /// afterwards, since candle's reduce uploads metadata at launch.
        logits: Tensor,
        graph: SendGraph,
    }

    pub struct GraphRunner {
        graphs: HashMap<(usize, usize), DecodeGraph>,
        /// Set after a capture fails; every later step runs eagerly.
        disabled: bool,
        device: Device,
    }

    fn engine_err(e: impl std::fmt::Display) -> Error {
        Error::Engine(e.to_string())
    }

    impl GraphRunner {
        pub fn new(device: &Device) -> Self {
            if let Device::Cuda(dev) = device {
                // Per-allocation event records would land inside the
                // capture; everything runs on the one stream anyway.
                // SAFETY: cudarc's tracking exists to order work across
                // streams; this backend issues all device work on this one
                // stream, from one thread, so there is nothing to order.
                unsafe { dev.cuda_stream().context().disable_event_tracking() };
            }
            Self {
                graphs: HashMap::new(),
                disabled: false,
                device: device.clone(),
            }
        }

        /// Whether this batch is a pure decode step every sequence samples.
        fn is_pure_decode(batch: &ForwardBatch) -> bool {
            let b = batch.batch_size();
            b > 0
                && b <= MAX_BUCKET
                && batch.tokens.len() == b
                && batch.logits_indices.len() == b
                && batch
                    .logits_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &x)| x as usize == i)
        }

        /// Pad a pure-decode batch to its buckets. Padding rows use token 0
        /// at position 0 in the reserved blocks, each with a one-key table.
        fn padded(batch: &ForwardBatch, num_blocks: usize) -> (ForwardBatch, usize, usize) {
            let b = batch.batch_size();
            let bucket = b.next_power_of_two().max(1);
            let widest = batch
                .block_tables
                .iter()
                .map(|t| t.len())
                .max()
                .unwrap_or(1);
            let blocks = widest.div_ceil(TABLE_BUCKET) * TABLE_BUCKET;
            let reserved = (num_blocks - RESERVED_BLOCKS) as u32;
            let mut p = batch.clone();
            for table in &mut p.block_tables {
                let last = *table.last().expect("non-empty table");
                table.resize(blocks, last);
            }
            for j in 0..bucket - b {
                let block = reserved + (j / BLOCK_SIZE) as u32;
                let slot = block * BLOCK_SIZE as u32 + (j % BLOCK_SIZE) as u32;
                p.tokens.push(0);
                p.positions.push(0);
                p.slot_mapping.push(slot);
                p.cu_seqlens_q.push(*p.cu_seqlens_q.last().unwrap() + 1);
                p.cu_seqlens_k.push(*p.cu_seqlens_k.last().unwrap() + 1);
                p.block_tables.push(vec![block; blocks]);
                p.logits_indices.push((b + j) as u32);
            }
            p.max_seqlen_q = 1;
            p.max_seqlen_k = blocks * BLOCK_SIZE;
            (p, bucket, blocks)
        }

        /// Run a pure-decode step through a captured graph, capturing it
        /// first if this shape has not been seen. `Ok(None)` means the batch
        /// is not a pure decode (or graphs are disabled): run it eagerly.
        pub fn try_decode(
            &mut self,
            model: &PagedLfm2,
            cache: &mut PagedKvCache,
            batch: &ForwardBatch,
        ) -> Result<Option<Vec<u32>>> {
            if self.disabled || !Self::is_pure_decode(batch) {
                return Ok(None);
            }
            if !model.graph_safe() {
                tracing::warn!("this checkpoint's conv bias is not graph-safe; running eagerly");
                self.disabled = true;
                return Ok(None);
            }
            let (padded, bucket, blocks) = Self::padded(batch, cache.num_blocks);
            let key = (bucket, blocks);
            let fresh = model.prepare(&padded, cache).map_err(engine_err)?;
            let t0 = std::time::Instant::now();
            if !self.graphs.contains_key(&key) {
                match self.capture(model, cache, fresh) {
                    Ok(g) => {
                        tracing::info!(
                            bucket,
                            blocks,
                            ms = t0.elapsed().as_millis(),
                            "captured a decode graph"
                        );
                        self.graphs.insert(key, g);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA graph capture failed; running eagerly from now on");
                        self.disabled = true;
                        return Ok(None);
                    }
                }
            } else {
                let g = self.graphs.get_mut(&key).expect("present");
                g.inputs.copy_from(&fresh).map_err(engine_err)?;
            }
            let g = &self.graphs[&key];
            g.graph.0.launch().map_err(engine_err)?;
            let out = g
                .logits
                .argmax(1)
                .map_err(engine_err)?
                .to_vec1::<u32>()
                .map_err(engine_err)?;
            metrics::histogram!("vapi_forward_graph_ms").record(t0.elapsed().as_secs_f64() * 1e3);
            Ok(Some(out[..batch.batch_size()].to_vec()))
        }

        fn capture(
            &self,
            model: &PagedLfm2,
            cache: &mut PagedKvCache,
            inputs: Lfm2Inputs,
        ) -> Result<DecodeGraph> {
            let Device::Cuda(dev) = &self.device else {
                return Err(Error::Engine("not a CUDA device".into()));
            };
            let stream = dev.cuda_stream();
            let rows = inputs.logits_rows();
            // Allocated before capture so it is not graph-owned memory and
            // can be read after every replay.
            let logits_out =
                Tensor::zeros((rows, model.vocab_size()), CandleDType::F32, &self.device)
                    .map_err(engine_err)?;
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(engine_err)?;
            // Every temporary must be dropped inside the capture so its free
            // becomes a graph node; the closure scope does that.
            let body = (|| -> candle_core::Result<()> {
                let logits = model.forward_prepared(&inputs, cache)?.ok_or_else(|| {
                    candle_core::Error::Msg("decode step produced no logits".into())
                })?;
                logits_out.slice_set(&logits, 0, 0)
            })();
            let ended = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            body.map_err(engine_err)?;
            let graph = ended
                .map_err(engine_err)?
                .ok_or_else(|| Error::Engine("stream capture produced no graph".into()))?;
            Ok(DecodeGraph {
                inputs,
                logits: logits_out,
                graph: SendGraph(graph),
            })
        }
    }
}
