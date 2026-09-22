//! Qwen2 and Qwen3 over the paged KV cache.
//!
//! Both are Llama-shaped: RMSNorm, SwiGLU, GQA, rotary embeddings. Three
//! things differ, and between them they are why this is a separate file
//! rather than a flag on [`super::llama`]:
//!
//! - **`head_dim` is its own number.** Qwen3-0.6B has 16 heads of 128 over
//!   a hidden size of 1024, so the projections are wider than the residual
//!   stream. Assuming `hidden_size / num_attention_heads` silently builds
//!   the wrong shapes.
//! - **Qwen2 biases Q, K and V**; Qwen3 does not.
//! - **Qwen3 normalises Q and K per head** before the rotary embedding,
//!   the same trick Laguna uses.
//!
//! Everything else follows `llama.rs`, including the `(tokens, heads,
//! head_dim)` layout that lets `rope_thd` run without a transpose.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};
use candle_transformers::models::with_tracing::{
    Linear, RmsNorm, linear as linear_bias, linear_no_bias,
};
use candle_transformers::quantized_nn as qnn;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

/// A projection whose weights are either dense or GGUF-quantised.
///
/// Quantised weights are what let a model larger than the card run at all:
/// the arithmetic is the same, the weights are read at 4 or 8 bits and
/// dequantised into the multiply by candle's kernels.
enum Proj {
    Dense(Linear),
    Quant(qnn::Linear),
}

impl Module for Proj {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            // candle's quantised kernels take bf16 directly, so the
            // residual stream stays in one dtype; the bias is cast at load
            // rather than converting the activations twice a projection.
            Self::Quant(l) => l.forward(x),
        }
    }
}

/// The same for a norm. GGUF stores norm weights quantised too, but they
/// are one vector per layer, so they are dequantised once at load into the
/// model's dtype rather than converted on every step.
enum Norm {
    Dense(RmsNorm),
    Plain(candle_nn::RmsNorm),
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(n) => n.forward(x),
            Self::Plain(n) => n.forward(x),
        }
    }
}
use vapi_engine::ForwardBatch;

use crate::attention::paged_attention_cpu;
use crate::cache::{PagedKvCache, write_kv_to_cache};
use crate::models::llama::AttentionImpl;

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    /// Qwen2 carries a bias on Q, K and V; Qwen3 does not.
    pub attention_bias: bool,
    /// Qwen3 normalises Q and K per head.
    pub qk_norm: bool,
    pub max_position_embeddings: usize,
}

fn get_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| candle_core::Error::Msg(format!("config.json: {key} is missing")))
}

impl QwenConfig {
    /// Read the config out of GGUF metadata, where llama.cpp keeps it
    /// under an architecture prefix rather than in a `config.json`.
    pub fn from_gguf(
        md: &std::collections::HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Result<Self> {
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_else(|| "qwen3".to_string());
        let get_u = |k: &str| -> Result<usize> {
            md.get(&format!("{arch}.{k}"))
                .ok_or_else(|| candle_core::Error::Msg(format!("gguf: {arch}.{k} is missing")))
                .and_then(|v| v.to_u32())
                .map(|v| v as usize)
        };
        let hidden_size = get_u("embedding_length")?;
        let num_attention_heads = get_u("attention.head_count")?;
        Ok(Self {
            hidden_size,
            intermediate_size: get_u("feed_forward_length")?,
            vocab_size: md
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.to_vec().ok().map(|t| t.len()))
                .ok_or_else(|| candle_core::Error::Msg("gguf: no tokenizer tokens".into()))?,
            num_hidden_layers: get_u("block_count")?,
            num_attention_heads,
            num_key_value_heads: get_u("attention.head_count_kv")?,
            head_dim: get_u("attention.key_length").unwrap_or(hidden_size / num_attention_heads),
            rms_norm_eps: md
                .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1e-6) as f64,
            rope_theta: md
                .get(&format!("{arch}.rope.freq_base"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1_000_000.0),
            // GGUF drops the output weight when it is tied.
            tie_word_embeddings: false,
            attention_bias: arch == "qwen2",
            qk_norm: arch != "qwen2",
            max_position_embeddings: get_u("context_length").unwrap_or(32768),
        })
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let hidden_size = get_usize(v, "hidden_size")?;
        let num_attention_heads = get_usize(v, "num_attention_heads")?;
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("qwen3");
        Ok(Self {
            hidden_size,
            intermediate_size: get_usize(v, "intermediate_size")?,
            vocab_size: get_usize(v, "vocab_size")?,
            num_hidden_layers: get_usize(v, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: get_usize(v, "num_key_value_heads")?,
            head_dim: v
                .get("head_dim")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(hidden_size / num_attention_heads),
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or(1_000_000.0) as f32,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            // Qwen2 defaults the bias on, Qwen3 off, and either can say so
            // explicitly.
            attention_bias: v
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(model_type == "qwen2"),
            qk_norm: model_type != "qwen2",
            max_position_embeddings: v
                .get("max_position_embeddings")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(32768),
        })
    }
}

/// GGUF metadata, read without loading any tensor data.
fn gguf_metadata(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<String, candle_core::quantized::gguf_file::Value>> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut f)?;
    Ok(content.metadata)
}

/// Rotary tables in the `(positions, head_dim / 2)` layout `rope_thd` wants.
fn rope_tables(cfg: &QwenConfig, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = cfg.head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_position_embeddings, 1))?;
    let freqs = positions.matmul(&inv_freq)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

/// Everything one step needs, already on the device.
///
/// Built once per step by [`PagedQwen::prepare`], which is where all the
/// host-to-device traffic happens. [`PagedQwen::forward_prepared`] then
/// touches nothing on the host, which is what makes a step capturable
/// into a CUDA graph.
pub struct QwenInputs {
    /// The host batch, for the CPU reference attention only.
    batch: ForwardBatch,
    tokens: Tensor,
    cos: Tensor,
    sin: Tensor,
    /// Flat cache slot per token, `(n,)` u32.
    slots: Tensor,
    #[cfg(feature = "cuda")]
    attention: Option<crate::attention::PreparedAttention>,
    logits_indices: Option<Tensor>,
}

impl QwenInputs {
    /// Row count of the logits this step produces.
    pub fn logits_rows(&self) -> usize {
        self.batch.logits_indices.len()
    }

    /// Overwrite every device tensor in place from a freshly prepared step
    /// of the same shape, so a captured graph reads the new step's inputs.
    pub fn copy_from(&mut self, other: &QwenInputs) -> Result<()> {
        fn set(dst: &Tensor, src: &Tensor) -> Result<()> {
            if dst.dims() != src.dims() {
                candle_core::bail!("input shape changed: {:?} vs {:?}", dst.dims(), src.dims());
            }
            dst.slice_set(src, 0, 0)
        }
        set(&self.tokens, &other.tokens)?;
        set(&self.cos, &other.cos)?;
        set(&self.sin, &other.sin)?;
        set(&self.slots, &other.slots)?;
        #[cfg(feature = "cuda")]
        if let (Some(a), Some(b)) = (&self.attention, &other.attention) {
            set(&a.block_table, &b.block_table)?;
            set(&a.seqlens_q, &b.seqlens_q)?;
            set(&a.seqlens_k, &b.seqlens_k)?;
            if a.max_seqlen_q != b.max_seqlen_q || a.max_seqlen_k != b.max_seqlen_k {
                candle_core::bail!("max_seqlen changed between prepared steps");
            }
        }
        if let (Some(a), Some(b)) = (&self.logits_indices, &other.logits_indices) {
            set(a, b)?;
        }
        self.batch = other.batch.clone();
        Ok(())
    }
}

type StepContext = QwenInputs;

struct Attention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    /// Qwen3 only.
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention: AttentionImpl,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &QwenConfig, attention: AttentionImpl) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let q_out = cfg.num_attention_heads * hd;
        let kv_out = cfg.num_key_value_heads * hd;
        let proj = |i, o, name: &str| -> Result<Proj> {
            Ok(Proj::Dense(if cfg.attention_bias {
                linear_bias(i, o, vb.pp(name))?
            } else {
                linear_no_bias(i, o, vb.pp(name))?
            }))
        };
        let (q_norm, k_norm) = if cfg.qk_norm {
            (
                Some(Norm::Dense(RmsNorm::new(
                    hd,
                    cfg.rms_norm_eps,
                    vb.pp("q_norm"),
                )?)),
                Some(Norm::Dense(RmsNorm::new(
                    hd,
                    cfg.rms_norm_eps,
                    vb.pp("k_norm"),
                )?)),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj: proj(h, q_out, "q_proj")?,
            k_proj: proj(h, kv_out, "k_proj")?,
            v_proj: proj(h, kv_out, "v_proj")?,
            // The output projection never carries a bias in either version.
            o_proj: Proj::Dense(linear_no_bias(q_out, h, vb.pp("o_proj"))?),
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            attention,
        })
    }

    /// RMSNorm over the last dimension of `(tokens, heads, head_dim)`.
    fn head_norm(norm: &Option<Norm>, x: &Tensor) -> Result<Tensor> {
        let Some(norm) = norm else {
            return Ok(x.clone());
        };
        let (n, h, d) = x.dims3()?;
        norm.forward(&x.reshape((n * h, d))?)?.reshape((n, h, d))
    }

    fn rope(x: &Tensor, ctx: &StepContext) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_thd(&x.contiguous()?.unsqueeze(0)?, &ctx.cos, &ctx.sin)?
            .squeeze(0)
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let n = x.dim(0)?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((n, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((n, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((n, self.num_kv_heads, self.head_dim))?
            .contiguous()?;

        // Normalise per head first, then rotate: the order matters, and it
        // is the one transformers uses.
        let q = Self::rope(&Self::head_norm(&self.q_norm, &q)?, ctx)?;
        let k = Self::rope(&Self::head_norm(&self.k_norm, &k)?, ctx)?;

        // The cache write and the attention both have to be free of host
        // traffic for a capture to replay correctly, so on CUDA they go
        // through the row kernels and the prepared descriptors.
        #[cfg(feature = "cuda")]
        let wrote = if k.device().is_cuda() {
            let (nb, bs, kvh, hd) = cache.k[layer].dims4()?;
            let flat_k = cache.k[layer].reshape((nb * bs, kvh * hd))?;
            let flat_v = cache.v[layer].reshape((nb * bs, kvh * hd))?;
            crate::rows::scatter(&flat_k, &ctx.slots, &k.reshape((n, kvh * hd))?)?;
            crate::rows::scatter(&flat_v, &ctx.slots, &v.reshape((n, kvh * hd))?)?;
            true
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let wrote = false;
        if !wrote {
            write_kv_to_cache(cache, layer, &k, &v, &ctx.batch.slot_mapping)?;
        }

        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let y = match self.attention {
            AttentionImpl::Reference => paged_attention_cpu(
                &q,
                &cache.k[layer],
                &cache.v[layer],
                &ctx.batch,
                scale,
                None,
            )?,
            #[cfg(feature = "cuda")]
            AttentionImpl::FlashPaged => match &ctx.attention {
                Some(prepared) => crate::attention::paged_attention_cuda_prepared(
                    &q,
                    &cache.k[layer],
                    &cache.v[layer],
                    prepared,
                    scale,
                    cache.block_size,
                    None,
                    Some(0),
                )?,
                None => crate::attention::paged_attention_cuda(
                    &q,
                    &cache.k[layer],
                    &cache.v[layer],
                    &ctx.batch,
                    scale,
                    cache.block_size,
                    None,
                )?,
            },
        };
        self.o_proj
            .forward(&y.reshape((n, self.num_heads * self.head_dim))?)
    }
}

struct Mlp {
    /// Dense weights: gate and up stacked along the output dimension, so
    /// the two projections are one GEMM. At decode batch sizes cuBLAS
    /// picks a much better kernel for the doubled width than for each half
    /// (measured on LFM2.5: 217 GB/s against 371), and the fused SwiGLU
    /// splits the result without materialising strided views.
    w13: Option<Tensor>,
    /// Used when the weights are quantised, where stacking is not possible.
    gate_proj: Option<Proj>,
    up_proj: Option<Proj>,
    down_proj: Proj,
    intermediate: usize,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &QwenConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate = vb.get((i, h), "gate_proj.weight")?;
        let up = vb.get((i, h), "up_proj.weight")?;
        Ok(Self {
            w13: Some(Tensor::cat(&[&gate, &up], 0)?),
            gate_proj: None,
            up_proj: None,
            down_proj: Proj::Dense(linear_no_bias(i, h, vb.pp("down_proj"))?),
            intermediate: i,
        })
    }

    /// The quantised variant keeps its projections apart: a `QMatMul`'s
    /// weights are blocks of packed integers, not a tensor to concatenate.
    fn quantised(gate: Proj, up: Proj, down: Proj, intermediate: usize) -> Self {
        Self {
            w13: None,
            gate_proj: Some(gate),
            up_proj: Some(up),
            down_proj: down,
            intermediate,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = match &self.w13 {
            Some(w13) => {
                let h = x.matmul(&w13.t()?)?;
                crate::fused::swiglu(&h, self.intermediate)?
            }
            None => {
                let gate = self.gate_proj.as_ref().expect("quantised mlp");
                let up = self.up_proj.as_ref().expect("quantised mlp");
                (candle_nn::ops::silu(&gate.forward(x)?)? * up.forward(x)?)?
            }
        };
        self.down_proj.forward(&hidden)
    }
}

struct Block {
    input_layernorm: Norm,
    attn: Attention,
    post_attention_layernorm: Norm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &QwenConfig, attention: AttentionImpl) -> Result<Self> {
        Ok(Self {
            input_layernorm: Norm::Dense(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?),
            attn: Attention::load(vb.pp("self_attn"), cfg, attention)?,
            post_attention_layernorm: Norm::Dense(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?),
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self
            .attn
            .forward(&self.input_layernorm.forward(x)?, layer, ctx, cache)?;
        let x = (h + residual)?;
        let residual = &x;
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        h + residual
    }
}

/// Qwen2 or Qwen3 over a paged KV cache.
pub struct PagedQwen {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: Norm,
    lm_head: Proj,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl PagedQwen {
    pub fn load(
        vb: VarBuilder,
        cfg: &QwenConfig,
        dtype: DType,
        device: &Device,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        // Qwen3-0.6B ties the output projection to the embedding; the
        // larger ones do not.
        let lm_head = Proj::Dense(if cfg.tie_word_embeddings {
            Linear::from_weights(embed.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        });
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg, attention))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = rope_tables(cfg, dtype, device)?;
        Ok(Self {
            embed,
            blocks,
            norm: Norm::Dense(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("model.norm"),
            )?),
            lm_head,
            cos,
            sin,
            device: device.clone(),
        })
    }

    /// Load from a GGUF file: the same architecture with quantised weights.
    ///
    /// GGUF carries its own metadata and llama.cpp's tensor names, so the
    /// config comes from the file rather than a `config.json` and the
    /// prefixes differ throughout (`blk.0.attn_q` for
    /// `model.layers.0.self_attn.q_proj`). Everything after loading is the
    /// shared forward pass.
    pub fn load_gguf(
        path: &std::path::Path,
        dtype: DType,
        device: &Device,
        attention: AttentionImpl,
    ) -> Result<(Self, QwenConfig)> {
        let vb = QVarBuilder::from_gguf(path, device)?;
        // A norm is one vector; dequantise it once into the model's dtype.
        let norm = |name: &str, size: usize, eps: f64| -> Result<Norm> {
            let w = vb
                .get_no_shape(&format!("{name}.weight"))?
                .dequantize(device)?
                .to_dtype(dtype)?;
            let _ = size;
            Ok(Norm::Plain(candle_nn::RmsNorm::new(w, eps)))
        };
        // A projection keeps its quantised weights; only the bias, when
        // there is one, is cast so the add matches the activations.
        let proj = |name: &str, bias: bool| -> Result<Proj> {
            let w = vb.get_no_shape(&format!("{name}.weight"))?;
            let b = if bias {
                Some(
                    vb.get_no_shape(&format!("{name}.bias"))?
                        .dequantize(device)?
                        .to_dtype(dtype)?,
                )
            } else {
                None
            };
            Ok(Proj::Quant(qnn::Linear::from_arc(w, b)?))
        };
        let md = gguf_metadata(path)?;
        let cfg = QwenConfig::from_gguf(&md)?;

        let embed_q = vb.get_no_shape("token_embd.weight")?;
        let embed_w = embed_q.dequantize(device)?.to_dtype(dtype)?;
        let embed = Embedding::new(embed_w.clone(), cfg.hidden_size);
        // `output.weight` is absent when the model ties its embedding.
        let lm_head = if vb.contains_key("output.weight") {
            proj("output", false)?
        } else {
            // Tied embedding: GGUF simply omits the output weight.
            Proj::Quant(qnn::Linear::from_arc(embed_q, None)?)
        };

        let bias = cfg.attention_bias;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let hd = cfg.head_dim;
            let (q_norm, k_norm) = if cfg.qk_norm {
                (
                    Some(norm(&format!("blk.{i}.attn_q_norm"), hd, cfg.rms_norm_eps)?),
                    Some(norm(&format!("blk.{i}.attn_k_norm"), hd, cfg.rms_norm_eps)?),
                )
            } else {
                (None, None)
            };
            blocks.push(Block {
                input_layernorm: norm(
                    &format!("blk.{i}.attn_norm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )?,
                attn: Attention {
                    q_proj: proj(&format!("blk.{i}.attn_q"), bias)?,
                    k_proj: proj(&format!("blk.{i}.attn_k"), bias)?,
                    v_proj: proj(&format!("blk.{i}.attn_v"), bias)?,
                    o_proj: proj(&format!("blk.{i}.attn_output"), false)?,
                    q_norm,
                    k_norm,
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    head_dim: hd,
                    attention,
                },
                post_attention_layernorm: norm(
                    &format!("blk.{i}.ffn_norm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )?,
                mlp: Mlp::quantised(
                    proj(&format!("blk.{i}.ffn_gate"), false)?,
                    proj(&format!("blk.{i}.ffn_up"), false)?,
                    proj(&format!("blk.{i}.ffn_down"), false)?,
                    cfg.intermediate_size,
                ),
            });
        }
        let (cos, sin) = rope_tables(&cfg, dtype, device)?;
        let model = Self {
            embed,
            blocks,
            norm: norm("output_norm", cfg.hidden_size, cfg.rms_norm_eps)?,
            lm_head,
            cos,
            sin,
            device: device.clone(),
        };
        Ok((model, cfg))
    }

    /// One continuous batch. `None` when the batch samples nothing, which
    /// is a non-final prefill chunk; the K/V writes happen either way.
    pub fn forward(
        &self,
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> Result<Option<Tensor>> {
        let inputs = self.prepare(batch, cache)?;
        self.forward_prepared(&inputs, cache)
    }

    /// Build every device input a step needs. All host-to-device traffic
    /// of a step happens here.
    pub fn prepare(&self, batch: &ForwardBatch, cache: &PagedKvCache) -> Result<QwenInputs> {
        let _ = cache;
        let n = batch.tokens.len();
        let dev = &self.device;
        let positions = Tensor::from_slice(&batch.positions, n, dev)?;
        Ok(QwenInputs {
            batch: batch.clone(),
            tokens: Tensor::from_slice(&batch.tokens, n, dev)?,
            cos: self.cos.index_select(&positions, 0)?,
            sin: self.sin.index_select(&positions, 0)?,
            slots: Tensor::from_slice(&batch.slot_mapping, n, dev)?,
            #[cfg(feature = "cuda")]
            attention: if dev.is_cuda() {
                Some(crate::attention::PreparedAttention::from_batch(batch, dev)?)
            } else {
                None
            },
            logits_indices: if batch.logits_indices.is_empty() {
                None
            } else {
                Some(Tensor::from_slice(
                    &batch.logits_indices,
                    batch.logits_indices.len(),
                    dev,
                )?)
            },
        })
    }

    /// The forward pass over prepared inputs: device work only, no
    /// host-to-device copies and no host synchronisation, so on CUDA it can
    /// be captured into a graph and replayed.
    pub fn forward_prepared(
        &self,
        inputs: &QwenInputs,
        cache: &mut PagedKvCache,
    ) -> Result<Option<Tensor>> {
        // `index_select` uploads its shape metadata from a temporary host
        // buffer, which a capture records as a copy from freed memory, so
        // the row kernels stand in for it.
        let mut x = crate::rows::gather(self.embed.embeddings(), &inputs.tokens)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, layer, inputs, cache)?;
        }
        let Some(idx) = &inputs.logits_indices else {
            return Ok(None);
        };
        let x = self.norm.forward(&x)?;
        let x = crate::rows::gather(&x, idx)?.contiguous()?;
        Ok(Some(self.lm_head.forward(&x)?.to_dtype(DType::F32)?))
    }

    pub fn vocab_size(&self) -> usize {
        self.embed.embeddings().dim(0).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LayerLayout;

    /// The fixture `tools/gen_qwen_goldens.py` writes.
    #[derive(serde::Deserialize)]
    struct Golden {
        config: serde_json::Value,
        tokens: Vec<u32>,
        full_logits: Vec<Vec<f32>>,
        greedy_ids: Vec<u32>,
    }

    fn fixture() -> Option<(std::path::PathBuf, Golden)> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("models/qwen-tiny");
        let json = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/goldens/qwen-tiny.json");
        if !dir.join("config.json").exists() || !json.exists() {
            eprintln!("SKIP: no qwen-tiny fixture; run tools/gen_qwen_goldens.py");
            return None;
        }
        let g: Golden = serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        Some((dir, g))
    }

    fn load(
        dir: &std::path::Path,
        g: &Golden,
        dtype: DType,
        dev: &Device,
        attention: AttentionImpl,
    ) -> (PagedQwen, PagedKvCache) {
        let cfg = QwenConfig::from_json(&g.config).unwrap();
        let files = vec![dir.join("model.safetensors")];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, dev).unwrap() };
        let model = PagedQwen::load(vb, &cfg, dtype, dev, attention).unwrap();
        let layouts = vec![
            LayerLayout::Kv {
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            };
            cfg.num_hidden_layers
        ];
        // The flash kernel wants a page size that is a multiple of 32, which
        // is what `BLOCK_SIZE` is everywhere else.
        let cache = PagedKvCache::with_layouts(&layouts, 8, 32, dtype, dev).unwrap();
        (model, cache)
    }

    /// One batch holding the whole prompt, asking for every position's
    /// logits so the comparison covers the sequence rather than its end.
    fn batch(tokens: &[u32]) -> ForwardBatch {
        let n = tokens.len();
        ForwardBatch {
            tokens: tokens.to_vec(),
            positions: (0..n as u32).collect(),
            cu_seqlens_q: vec![0, n as u32],
            cu_seqlens_k: vec![0, n as u32],
            slot_mapping: (0..n as u32).collect(),
            block_tables: vec![vec![0, 1, 2, 3]],
            logits_indices: (0..n as u32).collect(),
            max_seqlen_q: n,
            max_seqlen_k: n,
        }
    }

    fn max_abs_diff(got: &Tensor, want: &[Vec<f32>]) -> f32 {
        let got = got.to_vec2::<f32>().unwrap();
        let mut d = 0f32;
        for (g, w) in got.iter().zip(want) {
            for (a, b) in g.iter().zip(w) {
                d = d.max((a - b).abs());
            }
        }
        d
    }

    #[test]
    fn the_tiny_fixture_matches_transformers_in_f32() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::Cpu;
        let (model, mut cache) = load(&dir, &g, DType::F32, &dev, AttentionImpl::Reference);
        let got = model
            .forward(&batch(&g.tokens), &mut cache)
            .unwrap()
            .unwrap();
        let d = max_abs_diff(&got, &g.full_logits);
        assert!(d < 1e-4, "max abs diff {d}");
    }

    #[test]
    fn greedy_decoding_matches_transformers() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::Cpu;
        let (model, mut cache) = load(&dir, &g, DType::F32, &dev, AttentionImpl::Reference);
        // Prefill, then one token at a time out of the paged cache.
        let mut tokens = g.tokens.clone();
        let mut got = Vec::new();
        for step in 0..g.greedy_ids.len() {
            let n = tokens.len();
            let b = if step == 0 {
                batch(&tokens)
            } else {
                ForwardBatch {
                    tokens: vec![tokens[n - 1]],
                    positions: vec![n as u32 - 1],
                    cu_seqlens_q: vec![0, 1],
                    cu_seqlens_k: vec![0, n as u32],
                    slot_mapping: vec![n as u32 - 1],
                    block_tables: vec![vec![0, 1, 2, 3]],
                    logits_indices: vec![0],
                    max_seqlen_q: 1,
                    max_seqlen_k: n,
                }
            };
            let logits = model.forward(&b, &mut cache).unwrap().unwrap();
            let row: Vec<f32> = logits
                .to_vec2::<f32>()
                .unwrap()
                .pop()
                .expect("a logits row");
            let next = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                    if v > b.1 { (i, v) } else { b }
                })
                .0 as u32;
            got.push(next);
            tokens.push(next);
        }
        assert_eq!(got, g.greedy_ids);
    }

    /// The quantisation itself, with no model in the way: how far a weight
    /// moves when it is stored at four bits and read back.
    ///
    /// The standing rule is to prove a numerical claim before trusting it.
    /// Q4_K keeps a scale per 32-value block, so the error is bounded
    /// relative to the block's own range rather than to the tensor's.
    #[test]
    fn four_bit_weights_come_back_close_enough() {
        use candle_core::quantized::{GgmlDType, QTensor};
        let dev = Device::Cpu;
        // A weight-like tensor: mostly small values with a few large ones.
        let w = Tensor::randn(0f32, 0.5, (256, 512), &dev).unwrap();
        for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
            let q = QTensor::quantize(&w, dtype).unwrap();
            let back = q.dequantize(&dev).unwrap();
            let err = (&back - &w)
                .unwrap()
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let scale = w
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let relative = err / scale;
            eprintln!("{dtype:?}: mean error {err:.5} on a mean magnitude of {scale:.5}");
            let bound = if dtype == GgmlDType::Q8_0 { 0.02 } else { 0.10 };
            assert!(relative < bound, "{dtype:?}: {relative} of the magnitude");
            // And the size is what was paid for: four bits a weight, plus
            // the block scales.
            let bytes = q.storage_size_in_bytes();
            let dense = w.elem_count() * 4;
            // Four bits a weight plus block scales is about a seventh of
            // f32; eight bits is about a quarter.
            let shrink = if dtype == GgmlDType::Q8_0 { 3 } else { 6 };
            assert!(
                bytes * shrink < dense,
                "{dtype:?}: {bytes} bytes against {dense} dense"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_tiny_fixture_matches_on_cuda_in_bf16() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let (model, mut cache) = load(&dir, &g, DType::BF16, &dev, AttentionImpl::FlashPaged);
        let got = model
            .forward(&batch(&g.tokens), &mut cache)
            .unwrap()
            .unwrap();
        let range = g
            .full_logits
            .iter()
            .flatten()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        let d = max_abs_diff(&got, &g.full_logits);
        eprintln!("qwen-tiny on CUDA bf16: max abs diff {d:.4} over a range of {range:.2}");
        assert!(d < 0.05 * range);
    }
}
