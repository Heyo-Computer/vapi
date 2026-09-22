//! Llama with a paged KV cache and a continuous-batch forward pass.
//!
//! Vendored from `candle-transformers` 0.11 `models/llama.rs`. Kept as-is:
//! the config types (reused from upstream so `config.json` parses the same
//! way), RMSNorm, the MLP, embeddings, `lm_head` and `tie_word_embeddings`,
//! weight loading through `VarBuilder`, the precomputed RoPE tables including
//! Llama 3 frequency scaling, and `repeat_kv` for GQA.
//!
//! Replaced, exactly two things:
//!
//! 1. Upstream's `Cache` (`Vec<Option<(Tensor, Tensor)>>`, concatenated on
//!    every step) is gone. K/V go into a [`PagedKvCache`] through
//!    [`write_kv_to_cache`] and attention reads them back by block table.
//! 2. `forward(x, index_pos, cache)` took one position for the whole input,
//!    so it could not batch sequences sitting at different offsets. It is now
//!    `forward(&ForwardBatch, &mut PagedKvCache)`: the input is the flat token
//!    list, RoPE is gathered per token from `positions`, and hidden states are
//!    picked at `logits_indices` before `lm_head` so a 2048-token prefill
//!    chunk does not produce 2048 × vocab logits.

use std::f32::consts::PI;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_no_bias as linear};
use vapi_engine::ForwardBatch;

pub use candle_transformers::models::llama::{
    Config, Llama3RopeConfig, Llama3RopeType, LlamaConfig, LlamaEosToks,
};

use crate::attention::paged_attention_cpu;
use crate::cache::{PagedKvCache, write_kv_to_cache};

/// Which attention implementation runs over the paged cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionImpl {
    /// Gather-and-matmul reference, any device. The numerical baseline.
    Reference,
    /// `candle-flash-attn`'s paged varlen kernel. CUDA only.
    #[cfg(feature = "cuda")]
    FlashPaged,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

/// Precomputed `cos`/`sin` for every position, `(max_position_embeddings, head_dim / 2)`.
fn rope_tables(cfg: &Config, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta = match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => calculate_default_inv_freq(cfg),
        Some(rope_scaling) => {
            let low_freq_wavelen =
                rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
            let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.high_freq_factor;

            calculate_default_inv_freq(cfg)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rope_scaling.factor
                    } else {
                        let smooth = (rope_scaling.original_max_position_embeddings as f32
                            / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    };
    let theta = Tensor::new(theta, device)?;
    let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_position_embeddings, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    // Not the paper's interleaved layout; HF's rotate-half, see
    // modeling_llama.py.
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Everything a layer needs from the batch besides the hidden states.
struct StepContext<'a> {
    batch: &'a ForwardBatch,
    /// Per-token RoPE rows, `(num_tokens, head_dim / 2)`.
    cos: Tensor,
    sin: Tensor,
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attention: AttentionImpl,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl CausalSelfAttention {
    /// `x` is `(num_tokens, heads, head_dim)`: `rope_thd`'s `(b, t, h, d)`
    /// layout with `b = 1`, so no transposes or copies are needed.
    fn apply_rotary_emb(&self, x: &Tensor, ctx: &StepContext<'_>) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        candle_nn::rotary_emb::rope_thd(&x.contiguous()?.unsqueeze(0)?, &ctx.cos, &ctx.sin)?
            .squeeze(0)
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext<'_>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (num_tokens, hidden_size) = x.dims2()?;
        let q = self.q_proj.forward(x)?.reshape((
            num_tokens,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let k = self.k_proj.forward(x)?.reshape((
            num_tokens,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((num_tokens, self.num_key_value_heads, self.head_dim))?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, ctx)?;
        let k = self.apply_rotary_emb(&k, ctx)?;

        // The cache write is the side effect the engine relies on: after this
        // step every token in the batch has K/V at its slot.
        write_kv_to_cache(cache, layer, &k, &v, &ctx.batch.slot_mapping)?;

        let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
        let y = match self.attention {
            AttentionImpl::Reference => paged_attention_cpu(
                &q,
                &cache.k[layer],
                &cache.v[layer],
                ctx.batch,
                softmax_scale,
                None,
            )?,
            #[cfg(feature = "cuda")]
            AttentionImpl::FlashPaged => crate::attention::paged_attention_cuda(
                &q,
                &cache.k[layer],
                &cache.v[layer],
                ctx.batch,
                softmax_scale,
                cache.block_size,
                None,
            )?,
        };
        let y = y.reshape((num_tokens, hidden_size))?;
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &Config, attention: AttentionImpl) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            attention,
            span,
            span_rot,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext<'_>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, layer, ctx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config, attention: AttentionImpl) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, attention)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

/// Llama over a paged KV cache.
#[derive(Debug, Clone)]
pub struct PagedLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl PagedLlama {
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        // Llama 3.2 1B/3B and SmolLM tie the output projection to the
        // embedding; there is no `lm_head.weight` in those checkpoints.
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg, attention))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = rope_tables(cfg, dtype, device)?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cos,
            sin,
            device: device.clone(),
        })
    }

    /// Run one continuous batch. Returns f32 logits
    /// `(batch.logits_indices.len(), vocab_size)`, or `None` when the batch
    /// samples nothing (a non-final prefill chunk), and leaves every token's
    /// K/V written at its `slot_mapping` entry either way.
    ///
    /// The `None` case is not just an optimisation: a zero-row
    /// `index_select` or matmul is a zero-sized kernel launch, which CUDA
    /// rejects with `CUDA_ERROR_INVALID_VALUE` even though the CPU backend
    /// shrugs at it.
    pub fn forward(
        &self,
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> Result<Option<Tensor>> {
        let n = batch.tokens.len();
        let tokens = Tensor::from_slice(&batch.tokens, n, &self.device)?;
        let positions = Tensor::from_slice(&batch.positions, n, &self.device)?;
        let ctx = StepContext {
            batch,
            cos: self.cos.index_select(&positions, 0)?,
            sin: self.sin.index_select(&positions, 0)?,
        };

        let mut x = self.wte.forward(&tokens)?; // (n, hidden)
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, layer, &ctx, cache)?;
        }
        if batch.logits_indices.is_empty() {
            return Ok(None);
        }
        let x = self.ln_f.forward(&x)?;
        // Only the rows the sampler needs go through lm_head.
        let idx = Tensor::from_slice(
            &batch.logits_indices,
            batch.logits_indices.len(),
            &self.device,
        )?;
        let x = x.index_select(&idx, 0)?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        Ok(Some(logits.to_dtype(DType::F32)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::llama::{Cache, Llama};
    use std::collections::HashMap;

    const BS: usize = 4;

    fn tiny_config() -> Config {
        Config {
            hidden_size: 32,
            intermediate_size: 48,
            vocab_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2, // GQA, so repeat_kv is exercised
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 64,
            tie_word_embeddings: false,
        }
    }

    /// Random weights under the exact names the loader expects, shared by the
    /// stock and paged models so any difference is in the forward pass.
    fn random_weights(cfg: &Config, dev: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let mut add = |name: &str, shape: (usize, usize)| {
            m.insert(
                name.to_string(),
                Tensor::randn(0f32, 0.2, shape, dev).unwrap(),
            );
        };
        let h = cfg.hidden_size;
        let hd = h / cfg.num_attention_heads;
        add("model.embed_tokens.weight", (cfg.vocab_size, h));
        add("lm_head.weight", (cfg.vocab_size, h));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            add(&format!("{p}.self_attn.q_proj.weight"), (h, h));
            add(
                &format!("{p}.self_attn.k_proj.weight"),
                (cfg.num_key_value_heads * hd, h),
            );
            add(
                &format!("{p}.self_attn.v_proj.weight"),
                (cfg.num_key_value_heads * hd, h),
            );
            add(&format!("{p}.self_attn.o_proj.weight"), (h, h));
            add(
                &format!("{p}.mlp.gate_proj.weight"),
                (cfg.intermediate_size, h),
            );
            add(
                &format!("{p}.mlp.up_proj.weight"),
                (cfg.intermediate_size, h),
            );
            add(
                &format!("{p}.mlp.down_proj.weight"),
                (h, cfg.intermediate_size),
            );
        }
        // Norm weights around 1, as trained models have them.
        for name in (0..cfg.num_hidden_layers)
            .flat_map(|i| {
                [
                    format!("model.layers.{i}.input_layernorm.weight"),
                    format!("model.layers.{i}.post_attention_layernorm.weight"),
                ]
            })
            .chain(["model.norm.weight".to_string()])
        {
            let w = (Tensor::randn(0f32, 0.1, h, dev).unwrap() + 1.0).unwrap();
            m.insert(name, w);
        }
        m
    }

    struct Pair {
        stock: Llama,
        stock_cache: Cache,
        paged: PagedLlama,
        cache: PagedKvCache,
        cfg: Config,
    }

    fn pair(num_blocks: usize) -> Pair {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let weights = random_weights(&cfg, &dev);
        let vb = VarBuilder::from_tensors(weights, DType::F32, &dev);
        let stock = Llama::load(vb.clone(), &cfg).unwrap();
        let stock_cache = Cache::new(true, DType::F32, &cfg, &dev).unwrap();
        let paged = PagedLlama::load(vb, &cfg, DType::F32, &dev, AttentionImpl::Reference).unwrap();
        let cache = PagedKvCache::new(
            cfg.num_hidden_layers,
            num_blocks,
            BS,
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            DType::F32,
            &dev,
        )
        .unwrap();
        Pair {
            stock,
            stock_cache,
            paged,
            cache,
            cfg,
        }
    }

    fn slots(table: &[u32], positions: std::ops::Range<usize>) -> Vec<u32> {
        positions
            .map(|p| table[p / BS] * BS as u32 + (p % BS) as u32)
            .collect()
    }

    /// A single-sequence batch forwarding `tokens[start..]` over `table`,
    /// requesting logits for the last token only.
    fn chunk(tokens: &[u32], start: usize, table: &[u32]) -> ForwardBatch {
        let q = tokens.len() - start;
        ForwardBatch {
            tokens: tokens[start..].to_vec(),
            positions: (start as u32..tokens.len() as u32).collect(),
            cu_seqlens_q: vec![0, q as u32],
            cu_seqlens_k: vec![0, tokens.len() as u32],
            slot_mapping: slots(table, start..tokens.len()),
            block_tables: vec![table.to_vec()],
            logits_indices: vec![q as u32 - 1],
            max_seqlen_q: q,
            max_seqlen_k: tokens.len(),
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
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

    #[test]
    fn a_full_prompt_matches_the_stock_model() {
        let mut p = pair(8);
        let prompt: Vec<u32> = vec![5, 17, 3, 42, 8, 9, 60];
        let input = Tensor::new(&prompt[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want = p.stock.forward(&input, 0, &mut p.stock_cache).unwrap(); // (1, vocab)

        let table = [6u32, 1]; // non-contiguous on purpose
        let got = p
            .paged
            .forward(&chunk(&prompt, 0, &table), &mut p.cache)
            .unwrap()
            .unwrap();
        assert_eq!(got.dims(), &[1, p.cfg.vocab_size]);
        let diff = max_abs_diff(&got, &want);
        assert!(diff < 1e-4, "max abs diff {diff}");
    }

    #[test]
    fn chunked_prefill_then_decode_matches_the_stock_model_step_for_step() {
        let mut p = pair(8);
        let prompt: Vec<u32> = vec![5, 17, 3, 42, 8, 9, 60];
        let dev = Device::Cpu;
        let table = [3u32, 7, 0];

        // Stock: whole prompt, then two decode steps against its own cache.
        let input = Tensor::new(&prompt[..], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want0 = p.stock.forward(&input, 0, &mut p.stock_cache).unwrap();
        let next1 = Tensor::new(&[11u32][..], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want1 = p
            .stock
            .forward(&next1, prompt.len(), &mut p.stock_cache)
            .unwrap();
        let next2 = Tensor::new(&[23u32][..], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want2 = p
            .stock
            .forward(&next2, prompt.len() + 1, &mut p.stock_cache)
            .unwrap();

        // Paged: the prompt in two chunks (the first requests no logits, as
        // the scheduler would), then decodes at seqlen_q = 1 over the cache.
        let mut first = chunk(&prompt[..3], 0, &table);
        first.logits_indices.clear();
        let none = p.paged.forward(&first, &mut p.cache).unwrap();
        assert!(none.is_none(), "no logits requested, none produced");
        let got0 = p
            .paged
            .forward(&chunk(&prompt, 3, &table), &mut p.cache)
            .unwrap()
            .unwrap();
        assert!(max_abs_diff(&got0, &want0) < 1e-4);

        let mut seq = prompt.clone();
        seq.push(11);
        let got1 = p
            .paged
            .forward(&chunk(&seq, seq.len() - 1, &table), &mut p.cache)
            .unwrap()
            .unwrap();
        assert!(max_abs_diff(&got1, &want1) < 1e-4);
        seq.push(23);
        let got2 = p
            .paged
            .forward(&chunk(&seq, seq.len() - 1, &table), &mut p.cache)
            .unwrap()
            .unwrap();
        assert!(max_abs_diff(&got2, &want2) < 1e-4);
    }

    #[test]
    fn a_mixed_batch_gives_each_sequence_what_it_would_get_alone() {
        // Sequence A: decode at position 6 over its own cached prompt.
        // Sequence B: a fresh 5-token prefill. One batch, two logits rows.
        let mut p = pair(8);
        let dev = Device::Cpu;
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let b: Vec<u32> = vec![40, 41, 42, 43, 44];
        let table_a = [2u32, 5];
        let table_b = [7u32, 1];

        // References from the stock model, one sequence at a time.
        let input = Tensor::new(&a[..], &dev).unwrap().unsqueeze(0).unwrap();
        p.stock.forward(&input, 0, &mut p.stock_cache).unwrap();
        let next = Tensor::new(&[30u32][..], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want_a = p.stock.forward(&next, a.len(), &mut p.stock_cache).unwrap();
        let mut fresh = Cache::new(true, DType::F32, &p.cfg, &dev).unwrap();
        let input = Tensor::new(&b[..], &dev).unwrap().unsqueeze(0).unwrap();
        let want_b = p.stock.forward(&input, 0, &mut fresh).unwrap();

        // Paged: A's prompt first, then the mixed batch.
        p.paged
            .forward(&chunk(&a, 0, &table_a), &mut p.cache)
            .unwrap()
            .unwrap();
        let mut tokens = vec![30u32];
        tokens.extend(&b);
        let batch = ForwardBatch {
            tokens,
            positions: [vec![a.len() as u32], (0..b.len() as u32).collect()].concat(),
            cu_seqlens_q: vec![0, 1, 1 + b.len() as u32],
            cu_seqlens_k: vec![0, a.len() as u32 + 1, a.len() as u32 + 1 + b.len() as u32],
            slot_mapping: [
                slots(&table_a, a.len()..a.len() + 1),
                slots(&table_b, 0..b.len()),
            ]
            .concat(),
            block_tables: vec![table_a.to_vec(), table_b.to_vec()],
            logits_indices: vec![0, b.len() as u32],
            max_seqlen_q: b.len(),
            max_seqlen_k: a.len() + 1,
        };
        batch.validate(8, BS).unwrap();
        let got = p.paged.forward(&batch, &mut p.cache).unwrap().unwrap();
        assert_eq!(got.dims(), &[2, p.cfg.vocab_size]);
        assert!(max_abs_diff(&got.get(0).unwrap().unsqueeze(0).unwrap(), &want_a) < 1e-4);
        assert!(max_abs_diff(&got.get(1).unwrap().unsqueeze(0).unwrap(), &want_b) < 1e-4);
    }
}
