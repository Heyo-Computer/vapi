//! LFM2 (Liquid AI) with a paged cache and a continuous-batch forward.
//!
//! Written against transformers' `modeling_lfm2.py` and checked against it
//! on a random tiny fixture (`tools/gen_lfm2_goldens.py`) and on
//! `LiquidAI/LFM2.5-2.6B`.
//!
//! LFM2 is a hybrid. Most layers are a **short gated convolution**:
//! `in_proj` maps hidden → 3×hidden, split into `B`, `C`, `x`; `h = B * x`;
//! a causal depthwise conv over time with kernel `conv_L_cache` (3);
//! `y = C * conv(h)`; `out_proj`. The rest are grouped-query attention with
//! per-head RMSNorm on q and k. Each layer is `operator_norm` → operator →
//! residual → `ffn_norm` → SwiGLU (`w1`, `w3`, `w2`) → residual; the model
//! ends with `embedding_norm` and a tied `lm_head`.
//!
//! The conv's state is the last `L - 1` rows of `h` per sequence. Rather
//! than a second kind of cache, each conv layer stores `h` per token at the
//! token's paged slot (`LayerLayout::Rows`), and a step gathers the `L - 1`
//! rows before its chunk through the block table. That makes chunked
//! prefill, prefix-cache hits and preemption exact by construction; the
//! cost is `hidden` elements per token per conv layer.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_no_bias as linear};
use vapi_engine::ForwardBatch;

use crate::attention::paged_attention_cpu;
use crate::cache::{
    LayerLayout, PagedKvCache, scatter_index, write_rows_indexed, write_v_rows_indexed,
};
use crate::models::llama::AttentionImpl;

#[derive(Clone, Debug)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub conv_kernel: usize,
    pub conv_bias: bool,
    pub tie_word_embeddings: bool,
    /// `"conv"` or `"full_attention"` per layer.
    pub layer_types: Vec<String>,
    pub eos_token_ids: Vec<u32>,
}

fn get_usize(v: &serde_json::Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| candle_core::Error::Msg(format!("config.json: missing integer `{k}`")))
}

impl Lfm2Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let hidden_size = get_usize(v, "hidden_size")?;
        let num_attention_heads = get_usize(v, "num_attention_heads")?;
        let num_hidden_layers = get_usize(v, "num_hidden_layers")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        // `block_auto_adjust_ff_dim` rescales the FFN width; the released
        // LFM2.5 checkpoints ship it off with the final width in
        // `intermediate_size`.
        let mut intermediate_size = get_usize(v, "intermediate_size")?;
        if v.get("block_auto_adjust_ff_dim")
            .and_then(|x| x.as_bool())
            .unwrap_or(false)
        {
            let mut i = (2 * intermediate_size) / 3;
            if let Some(m) = v.get("block_ffn_dim_multiplier").and_then(|x| x.as_f64()) {
                i = (m * i as f64) as usize;
            }
            let mult = v
                .get("block_multiple_of")
                .and_then(|x| x.as_u64())
                .unwrap_or(256) as usize;
            intermediate_size = mult * i.div_ceil(mult);
        }
        let layer_types: Vec<String> = match v.get("layer_types").and_then(|x| x.as_array()) {
            Some(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            None => vec!["conv".to_string(); num_hidden_layers],
        };
        if layer_types.len() != num_hidden_layers {
            candle_core::bail!("layer_types does not cover every layer");
        }
        let rope_theta = v
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| v.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .unwrap_or(1_000_000.0) as f32;
        let eos_token_ids = match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|x| x as u32).into_iter().collect()
            }
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: get_usize(v, "num_key_value_heads")?,
            head_dim,
            norm_eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5),
            max_position_embeddings: get_usize(v, "max_position_embeddings")?,
            rope_theta,
            conv_kernel: v.get("conv_L_cache").and_then(|x| x.as_u64()).unwrap_or(3) as usize,
            conv_bias: v
                .get("conv_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            layer_types,
            eos_token_ids,
        })
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.layer_types[layer] == "full_attention"
    }

    /// What each layer stores per token in the paged cache.
    pub fn cache_layouts(&self) -> Vec<LayerLayout> {
        (0..self.num_hidden_layers)
            .map(|l| {
                if self.is_attention(l) {
                    LayerLayout::Kv {
                        num_kv_heads: self.num_key_value_heads,
                        head_dim: self.head_dim,
                    }
                } else {
                    LayerLayout::Rows {
                        width: self.hidden_size,
                    }
                }
            })
            .collect()
    }
}

fn rope_tables(cfg: &Lfm2Config, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let d = cfg.head_dim;
    let inv: Vec<f32> = (0..d)
        .step_by(2)
        .map(|i| 1.0 / cfg.rope_theta.powf(i as f32 / d as f32))
        .collect();
    let max_pos = cfg.max_position_embeddings;
    let mut cos = Vec::with_capacity(max_pos * inv.len());
    let mut sin = Vec::with_capacity(max_pos * inv.len());
    for p in 0..max_pos {
        for f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_pos, inv.len()), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (max_pos, inv.len()), device)?.to_dtype(dtype)?,
    ))
}

/// Everything a layer needs from the batch besides the hidden states, built
/// once per step: RoPE rows, the scatter indices for both row shapes, and
/// the conv tap indices. Each is a host-to-device copy, and building them
/// per layer was ~200 synchronous copies per step.
/// Everything a step needs on the device, built once per step by
/// [`PagedLfm2::prepare`]: token ids, RoPE rows, the scatter indices for
/// both row shapes, the conv tap indices and masks, and the attention
/// kernel's block table and cumulative lengths. Building these per layer
/// was ~200 synchronous host-to-device copies per step; keeping them out
/// of [`PagedLfm2::forward_prepared`] is also what makes that function
/// capturable into a CUDA graph.
pub struct Lfm2Inputs {
    /// The host batch, used only by the CPU reference attention.
    batch: ForwardBatch,
    tokens: Tensor,
    cos: Tensor,
    sin: Tensor,
    /// Flat cache slot per token, `(n,)` u32, for the CUDA row kernels.
    slots: Tensor,
    /// Scatter index for attention K/V rows `(n, kv_heads, head_dim)`.
    idx_kv: Tensor,
    /// Scatter index for conv rows `(n, 1, hidden)`.
    idx_conv: Tensor,
    /// Per tap `k`: the flat slot of each token's row `L - 1 - k` positions
    /// back, and a `(n, hidden)` 0/1 mask zeroing rows before the sequence
    /// start (full shape, so the in-graph multiply is contiguous).
    conv_taps: Vec<(Tensor, Tensor)>,
    /// CUDA: the same taps as one `(L, n)` u32 table with
    /// [`crate::fused::CONV_PAD`] for rows before the sequence start, read
    /// by the fused conv kernel. `conv_taps` and `taps` stay empty there.
    conv_idx: Option<Tensor>,
    /// Per conv layer, its taps expanded to `(n, hidden)`; constants shared
    /// with the layer's memo.
    taps: std::collections::HashMap<usize, Vec<Tensor>>,
    #[cfg(feature = "cuda")]
    attention: Option<crate::attention::PreparedAttention>,
    logits_indices: Option<Tensor>,
}

impl Lfm2Inputs {
    /// For every token in the batch, the flat cache slot of the row `back`
    /// positions before it in its own sequence, and a 0/1 mask zeroing
    /// tokens that sit closer than `back` to their sequence start.
    fn conv_taps(batch: &ForwardBatch, back: usize, block_size: usize) -> (Vec<u32>, Vec<f32>) {
        let b = batch;
        let mut idx = Vec::with_capacity(b.tokens.len());
        let mut mask = Vec::with_capacity(b.tokens.len());
        for i in 0..b.batch_size() {
            let table = &b.block_tables[i];
            for t in b.cu_seqlens_q[i] as usize..b.cu_seqlens_q[i + 1] as usize {
                let p = b.positions[t] as usize;
                if p >= back {
                    let q = p - back;
                    idx.push(table[q / block_size] * block_size as u32 + (q % block_size) as u32);
                    mask.push(1.0);
                } else {
                    idx.push(b.slot_mapping[t]);
                    mask.push(0.0);
                }
            }
        }
        (idx, mask)
    }

    /// Number of token rows this step carries.
    pub fn rows(&self) -> usize {
        self.batch.tokens.len()
    }

    /// Row count of the logits this step produces.
    pub fn logits_rows(&self) -> usize {
        self.batch.logits_indices.len()
    }

    /// Overwrite every device tensor in place from a freshly prepared step
    /// of the same shape. The CUDA graph path does this outside the
    /// captured region so the replay reads the new step's inputs.
    pub fn copy_from(&mut self, other: &Lfm2Inputs) -> Result<()> {
        fn set(dst: &Tensor, src: &Tensor) -> Result<()> {
            if dst.dims() != src.dims() {
                candle_core::bail!("input shape changed: {:?} vs {:?}", dst.dims(), src.dims());
            }
            dst.slice_set(src, 0, 0)
        }
        set(&self.tokens, &other.tokens)?;
        set(&self.slots, &other.slots)?;
        set(&self.cos, &other.cos)?;
        set(&self.sin, &other.sin)?;
        set(&self.idx_kv, &other.idx_kv)?;
        set(&self.idx_conv, &other.idx_conv)?;
        for ((i, m), (oi, om)) in self.conv_taps.iter().zip(&other.conv_taps) {
            set(i, oi)?;
            set(m, om)?;
        }
        if let (Some(a), Some(b)) = (&self.conv_idx, &other.conv_idx) {
            set(a, b)?;
        }
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

type StepContext = Lfm2Inputs;

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention: AttentionImpl,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Lfm2Config, attention: AttentionImpl) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        Ok(Self {
            q_proj: linear(h, cfg.num_attention_heads * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            out_proj: linear(cfg.num_attention_heads * hd, h, vb.pp("out_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: RmsNorm::new(hd, cfg.norm_eps, vb.pp("k_layernorm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            attention,
        })
    }

    fn head_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
        let (n, h, d) = x.dims3()?;
        norm.forward(&x.reshape((n * h, d))?)?.reshape((n, h, d))
    }

    /// `x` is `(n, heads, head_dim)`, which is exactly `rope_thd`'s
    /// `(b, t, h, d)` layout with `b = 1`, so RoPE runs on the projection
    /// output as it is: no transposes and no copies on either side.
    fn rope(x: &Tensor, ctx: &StepContext) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_thd(&x.unsqueeze(0)?, &ctx.cos, &ctx.sin)?.squeeze(0)
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let (n, _) = x.dims2()?;
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
        let q = Self::rope(&Self::head_norm(&self.q_norm, &q)?, ctx)?;
        let k = Self::rope(&Self::head_norm(&self.k_norm, &k)?, ctx)?;
        if k.device().is_cuda() {
            let (nb, bs, kvh, hd) = cache.k[layer].dims4()?;
            let flat_k = cache.k[layer].reshape((nb * bs, kvh * hd))?;
            let flat_v = cache.v[layer].reshape((nb * bs, kvh * hd))?;
            crate::rows::scatter(&flat_k, &ctx.slots, &k.reshape((n, kvh * hd))?)?;
            crate::rows::scatter(&flat_v, &ctx.slots, &v.reshape((n, kvh * hd))?)?;
        } else {
            write_rows_indexed(cache, layer, &k, &ctx.idx_kv)?;
            write_v_rows_indexed(cache, layer, &v, &ctx.idx_kv)?;
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
            AttentionImpl::FlashPaged => {
                let prepared = ctx.attention.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("inputs were prepared without attention".into())
                })?;
                crate::attention::paged_attention_cuda_prepared(
                    &q,
                    &cache.k[layer],
                    &cache.v[layer],
                    prepared,
                    scale,
                    cache.block_size,
                    None,
                    Some(0),
                )?
            }
        };
        self.out_proj
            .forward(&y.reshape((n, self.num_heads * self.head_dim))?)
    }
}

/// The gated short convolution.
struct ShortConv {
    /// CPU: `in_proj` split into its three row blocks (B, C, x in
    /// checkpoint order), so each output is a contiguous matmul result
    /// rather than a strided view of one. Strided views make candle upload
    /// shape metadata from a temporary host buffer at launch, which a CUDA
    /// graph capture would record as a dangling copy.
    parts: Option<[Linear; 3]>,
    /// CUDA: the whole `(3·hidden, hidden)` `in_proj` as one GEMM, whose
    /// output the fused kernels in `fused.rs` read B, C and x out of
    /// directly. One launch instead of three, and cuBLAS runs the wider
    /// GEMM closer to memory bandwidth. The bias, if any, is applied inside
    /// those kernels.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    in_proj: Option<(Tensor, Option<Tensor>)>,
    out_proj: Linear,
    /// Per tap `k`, the `(1, hidden)` channel weights, oldest first.
    taps: Vec<Tensor>,
    /// The same taps as one `(L, hidden)` tensor, for the fused kernel.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    taps_w: Tensor,
    /// Taps expanded to `(n, hidden)` per batch size `n`, built on first
    /// use outside any capture, so the in-graph multiply is contiguous.
    taps_expanded: std::sync::Mutex<std::collections::HashMap<usize, Vec<Tensor>>>,
    bias: Option<Tensor>,
    hidden: usize,
}

impl ShortConv {
    fn load(vb: VarBuilder, cfg: &Lfm2Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let weight = vb.get((h, 1, cfg.conv_kernel), "conv.weight")?;
        let bias = if cfg.conv_bias {
            Some(vb.get(h, "conv.bias")?)
        } else {
            None
        };
        let w = vb.get((3 * h, h), "in_proj.weight")?;
        let in_bias = if cfg.conv_bias {
            Some(vb.get(3 * h, "in_proj.bias")?)
        } else {
            None
        };
        let part = |i: usize| -> Result<Linear> {
            let wi = w.narrow(0, i * h, h)?.contiguous()?;
            let bi = match &in_bias {
                Some(b) => Some(b.narrow(0, i * h, h)?.contiguous()?),
                None => None,
            };
            Ok(Linear::from_weights(wi, bi))
        };
        let out_proj = if cfg.conv_bias {
            candle_transformers::models::with_tracing::linear(h, h, vb.pp("out_proj"))?
        } else {
            linear(h, h, vb.pp("out_proj"))?
        };
        let taps2d = weight.squeeze(1)?; // (hidden, kernel)
        let taps = (0..cfg.conv_kernel)
            .map(|k| taps2d.narrow(1, k, 1)?.t()?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        let taps_w = taps2d.t()?.contiguous()?;
        let (parts, in_proj) = if vb.device().is_cuda() {
            (None, Some((w, in_bias)))
        } else {
            (Some([part(0)?, part(1)?, part(2)?]), None)
        };
        Ok(Self {
            parts,
            in_proj,
            out_proj,
            taps,
            taps_w,
            taps_expanded: std::sync::Mutex::new(std::collections::HashMap::new()),
            bias,
            hidden: h,
        })
    }

    /// The taps broadcast to `(n, hidden)`, memoised per `n`.
    fn taps_for(&self, n: usize) -> Result<Vec<Tensor>> {
        let mut memo = self.taps_expanded.lock().expect("taps memo");
        if let Some(t) = memo.get(&n) {
            return Ok(t.clone());
        }
        let expanded = self
            .taps
            .iter()
            .map(|t| t.broadcast_as((n, self.hidden))?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        memo.insert(n, expanded.clone());
        Ok(expanded)
    }

    /// `x` is `(n, hidden)` for the whole batch. Writes each token's `h`
    /// row to its slot first, then reads, for every token, its own row and
    /// the `L - 1` rows before it straight from the cache by slot: one
    /// `index_select` per tap over the flat `(num_slots, hidden)` view, for
    /// the whole batch at once. Rows before a sequence's start are masked
    /// to zero (the conv's causal padding). No per-sequence work, and no
    /// gathering a whole history to use two rows of it.
    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let flat = cache.k[layer].reshape((cache.num_slots(), self.hidden))?;
        #[cfg(feature = "cuda")]
        if let Some((w, in_bias)) = &self.in_proj {
            let h3 = x.matmul(&w.t()?)?; // (n, 3·hidden): B | C | x
            let idx = ctx
                .conv_idx
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("no conv_idx for CUDA".into()))?;
            crate::fused::conv_write(&h3, &flat, &ctx.slots, in_bias.as_ref(), self.hidden)?;
            let y = crate::fused::conv_apply(
                &h3,
                &flat,
                idx,
                &self.taps_w,
                self.bias.as_ref(),
                in_bias.as_ref(),
                self.hidden,
            )?;
            return self.out_proj.forward(&y);
        }
        let [b_proj, c_proj, x_proj] = self
            .parts
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("conv parts missing off CUDA".into()))?;
        let b = b_proj.forward(x)?;
        let c = c_proj.forward(x)?;
        let xx = x_proj.forward(x)?;
        let h = (b * xx)?; // (n, hidden), contiguous
        write_rows_indexed(cache, layer, &h.unsqueeze(1)?, &ctx.idx_conv)?;

        let taps = ctx.taps.get(&layer).ok_or_else(|| {
            candle_core::Error::Msg(format!("no expanded taps for layer {layer}"))
        })?;
        let mut acc: Option<Tensor> = None;
        for (k, (idx, mask)) in ctx.conv_taps.iter().enumerate() {
            let rows = crate::rows::gather(&flat, idx)?; // (n, hidden)
            // Same-shape, contiguous multiplies only (see ShortConv).
            let term = (rows * &taps[k])?.mul(mask)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let mut conv = acc.expect("kernel >= 1");
        if let Some(bias) = &self.bias {
            conv = conv.broadcast_add(bias)?;
        }
        let y = (c * conv)?;
        self.out_proj.forward(&y)
    }
}

struct Mlp {
    /// `w1` and `w3` stacked along the output dim, `(2 × intermediate,
    /// hidden)`, so gate and up are one GEMM. At decode batch sizes cuBLAS
    /// picks a kernel for the doubled width that is nearly twice as fast per
    /// byte as the one it picks for each half; see `fused::swiglu`.
    w13: Tensor,
    w2: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Lfm2Config) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let w1 = vb.get((i, h), "w1.weight")?;
        let w3 = vb.get((i, h), "w3.weight")?;
        Ok(Self {
            w13: Tensor::cat(&[&w1, &w3], 0)?,
            w2: linear(i, h, vb.pp("w2"))?,
            intermediate: i,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.w13.t()?)?;
        self.w2
            .forward(&crate::fused::swiglu(&h, self.intermediate)?)
    }
}

enum Operator {
    Attention(Box<Attention>),
    Conv(Box<ShortConv>),
}

struct Block {
    operator_norm: RmsNorm,
    operator: Operator,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(
        vb: VarBuilder,
        cfg: &Lfm2Config,
        layer: usize,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let operator = if cfg.is_attention(layer) {
            Operator::Attention(Box::new(Attention::load(
                vb.pp("self_attn"),
                cfg,
                attention,
            )?))
        } else {
            Operator::Conv(Box::new(ShortConv::load(vb.pp("conv"), cfg)?))
        };
        Ok(Self {
            operator_norm: RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?,
            operator,
            ffn_norm: RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?,
            mlp: Mlp::load(vb.pp("feed_forward"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let n = self.operator_norm.forward(x)?;
        let h = match &self.operator {
            Operator::Attention(a) => a.forward(&n, layer, ctx, cache)?,
            Operator::Conv(c) => c.forward(&n, layer, ctx, cache)?,
        };
        let x = (x + h)?;
        let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

/// LFM2 over a paged cache.
pub struct PagedLfm2 {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    conv_kernel: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
}

impl PagedLfm2 {
    pub fn load(
        vb: VarBuilder,
        cfg: &Lfm2Config,
        dtype: DType,
        device: &Device,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings || !vb.contains_tensor("lm_head.weight") {
            Linear::from_weights(embed.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("model.embedding_norm"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|l| Block::load(vb.pp(format!("model.layers.{l}")), cfg, l, attention))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = rope_tables(cfg, dtype, device)?;
        Ok(Self {
            embed,
            blocks,
            norm,
            lm_head,
            cos,
            sin,
            device: device.clone(),
            conv_kernel: cfg.conv_kernel,
            hidden: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
        })
    }

    /// Whether `forward_prepared` stays on contiguous ops end to end, which
    /// is what CUDA graph capture needs. A conv bias would add a broadcast.
    pub fn graph_safe(&self) -> bool {
        self.blocks.iter().all(|b| match &b.operator {
            Operator::Conv(c) => c.bias.is_none(),
            Operator::Attention(_) => true,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Same contract as `PagedLlama::forward`: f32 logits for
    /// `batch.logits_indices`, or `None` when nothing is sampled, with every
    /// token's K/V (and conv input row) written to the cache either way.
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
    pub fn prepare(&self, batch: &ForwardBatch, cache: &PagedKvCache) -> Result<Lfm2Inputs> {
        let n = batch.tokens.len();
        let dev = &self.device;
        let tokens = Tensor::from_slice(&batch.tokens, n, dev)?;
        let positions = Tensor::from_slice(&batch.positions, n, dev)?;
        let dt = self.cos.dtype();
        let mut conv_taps = Vec::with_capacity(self.conv_kernel);
        let mut conv_idx = None;
        let mut taps = std::collections::HashMap::new();
        if dev.is_cuda() {
            let mut flat = Vec::with_capacity(self.conv_kernel * n);
            for k in 0..self.conv_kernel {
                let back = self.conv_kernel - 1 - k;
                let (idx, mask) = Lfm2Inputs::conv_taps(batch, back, cache.block_size);
                flat.extend(
                    idx.iter().zip(&mask).map(
                        |(&i, &m)| {
                            if m > 0.0 { i } else { crate::fused::CONV_PAD }
                        },
                    ),
                );
            }
            conv_idx = Some(Tensor::from_vec(flat, (self.conv_kernel, n), dev)?);
        } else {
            for k in 0..self.conv_kernel {
                let back = self.conv_kernel - 1 - k;
                let (idx, mask) = Lfm2Inputs::conv_taps(batch, back, cache.block_size);
                conv_taps.push((
                    Tensor::from_vec(idx, n, dev)?,
                    Tensor::from_vec(mask, (n, 1), dev)?
                        .to_dtype(dt)?
                        .broadcast_as((n, self.hidden))?
                        .contiguous()?,
                ));
            }
            for (layer, block) in self.blocks.iter().enumerate() {
                if let Operator::Conv(conv) = &block.operator {
                    taps.insert(layer, conv.taps_for(n)?);
                }
            }
        }
        let logits_indices = if batch.logits_indices.is_empty() {
            None
        } else {
            Some(Tensor::from_slice(
                &batch.logits_indices,
                batch.logits_indices.len(),
                dev,
            )?)
        };
        Ok(Lfm2Inputs {
            batch: batch.clone(),
            tokens,
            slots: Tensor::from_slice(&batch.slot_mapping, n, dev)?,
            cos: self.cos.index_select(&positions, 0)?,
            sin: self.sin.index_select(&positions, 0)?,
            idx_kv: scatter_index(&batch.slot_mapping, self.num_kv_heads, self.head_dim, dev)?,
            idx_conv: scatter_index(&batch.slot_mapping, 1, self.hidden, dev)?,
            conv_taps,
            conv_idx,
            taps,
            #[cfg(feature = "cuda")]
            attention: if dev.is_cuda() {
                Some(crate::attention::PreparedAttention::from_batch(batch, dev)?)
            } else {
                None
            },
            logits_indices,
        })
    }

    /// The forward pass over prepared inputs: device work only, no
    /// host-to-device copies and no host synchronisation, so on CUDA it can
    /// be captured into a graph and replayed.
    pub fn forward_prepared(
        &self,
        inputs: &Lfm2Inputs,
        cache: &mut PagedKvCache,
    ) -> Result<Option<Tensor>> {
        let mut x = crate::rows::gather(self.embed.embeddings(), &inputs.tokens)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, layer, inputs, cache)?;
        }
        let Some(idx) = &inputs.logits_indices else {
            return Ok(None);
        };
        let x = self.norm.forward(&x)?;
        let x = crate::rows::gather(&x, idx)?;
        Ok(Some(self.lm_head.forward(&x)?.to_dtype(DType::F32)?))
    }
}

#[cfg(test)]
mod tests {
    //! Parity with transformers on the fixture `tools/gen_lfm2_goldens.py`
    //! writes. Skips without it.

    use super::*;
    use std::path::{Path, PathBuf};
    use vapi_core::config::BLOCK_SIZE;

    #[derive(serde::Deserialize)]
    struct Golden {
        prompt_ids: Vec<u32>,
        full_logits: Vec<Vec<f32>>,
        decode_token: u32,
        decode_logits: Vec<f32>,
    }

    fn fixture() -> Option<(PathBuf, Golden)> {
        let home = std::env::var("HOME").unwrap_or_default();
        let dir = Path::new(&home).join("models/lfm2-tiny");
        if !dir.join("config.json").exists() {
            eprintln!(
                "SKIP: no fixture at {}; run tools/gen_lfm2_goldens.py",
                dir.display()
            );
            return None;
        }
        let golden =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/goldens/lfm2-tiny.json");
        Some((
            dir,
            serde_json::from_str(&std::fs::read_to_string(golden).ok()?).ok()?,
        ))
    }

    fn load(
        dir: &Path,
        dtype: DType,
        dev: &Device,
        attention: AttentionImpl,
    ) -> (PagedLfm2, PagedKvCache) {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let cfg = Lfm2Config::from_json(&v).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, dev)
                .unwrap()
        };
        let model = PagedLfm2::load(vb, &cfg, dtype, dev, attention).unwrap();
        let cache =
            PagedKvCache::with_layouts(&cfg.cache_layouts(), 8, BLOCK_SIZE, dtype, dev).unwrap();
        (model, cache)
    }

    fn slots(table: &[u32], positions: std::ops::Range<usize>) -> Vec<u32> {
        positions
            .map(|p| table[p / BLOCK_SIZE] * BLOCK_SIZE as u32 + (p % BLOCK_SIZE) as u32)
            .collect()
    }

    fn chunk(tokens: &[u32], start: usize, table: &[u32], want: Vec<u32>) -> ForwardBatch {
        let q = tokens.len() - start;
        ForwardBatch {
            tokens: tokens[start..].to_vec(),
            positions: (start as u32..tokens.len() as u32).collect(),
            cu_seqlens_q: vec![0, q as u32],
            cu_seqlens_k: vec![0, tokens.len() as u32],
            slot_mapping: slots(table, start..tokens.len()),
            block_tables: vec![table.to_vec()],
            logits_indices: want,
            max_seqlen_q: q,
            max_seqlen_k: tokens.len(),
        }
    }

    fn max_abs_diff(got: &Tensor, want: &[Vec<f32>]) -> f32 {
        let got = got
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len(), "row count");
        got.iter()
            .zip(want)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_fixture_matches_transformers_for_prompt_decode_and_chunks() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::Cpu;
        let (model, mut cache) = load(&dir, DType::F32, &dev, AttentionImpl::Reference);
        let table = [3u32, 6];
        let n = g.prompt_ids.len();

        let all = (0..n as u32).collect();
        let got = model
            .forward(&chunk(&g.prompt_ids, 0, &table, all), &mut cache)
            .unwrap()
            .unwrap();
        let d = max_abs_diff(&got, &g.full_logits);
        assert!(d < 1e-4, "full prompt max abs diff {d}");

        let mut seq = g.prompt_ids.clone();
        seq.push(g.decode_token);
        let got = model
            .forward(&chunk(&seq, n, &table, vec![0]), &mut cache)
            .unwrap()
            .unwrap();
        let d = max_abs_diff(&got, std::slice::from_ref(&g.decode_logits));
        assert!(d < 1e-4, "decode max abs diff {d}");

        // Chunk boundaries at every position: each one puts a different
        // number of the conv window's rows in the cache rather than the
        // chunk, so the gather path is exercised for take = 0, 1 and 2.
        for split in 1..n {
            let (m2, mut c2) = load(&dir, DType::F32, &dev, AttentionImpl::Reference);
            assert!(
                m2.forward(&chunk(&g.prompt_ids[..split], 0, &table, vec![]), &mut c2)
                    .unwrap()
                    .is_none()
            );
            let rest = chunk(
                &g.prompt_ids,
                split,
                &table,
                (0..(n - split) as u32).collect(),
            );
            let got = m2.forward(&rest, &mut c2).unwrap().unwrap();
            let d = max_abs_diff(&got, &g.full_logits[split..]);
            assert!(d < 1e-4, "prefill split at {split}: max abs diff {d}");
        }
        eprintln!("lfm2-tiny: full, decode and every chunk split within 1e-4");
    }

    #[test]
    fn a_mixed_batch_gives_each_sequence_what_it_would_get_alone() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::Cpu;
        let n = g.prompt_ids.len();
        let a = &g.prompt_ids;
        let b: Vec<u32> = g.prompt_ids.iter().rev().cloned().collect();
        let (table_a, table_b) = ([1u32, 5], [7u32, 2]);
        let (m1, mut c1) = load(&dir, DType::F32, &dev, AttentionImpl::Reference);
        let want_b = m1
            .forward(&chunk(&b, 0, &table_b, vec![(n - 1) as u32]), &mut c1)
            .unwrap()
            .unwrap();

        let (model, mut cache) = load(&dir, DType::F32, &dev, AttentionImpl::Reference);
        model
            .forward(&chunk(a, 0, &table_a, vec![]), &mut cache)
            .unwrap();
        let mut tokens = vec![g.decode_token];
        tokens.extend(&b);
        let batch = ForwardBatch {
            tokens,
            positions: [vec![n as u32], (0..n as u32).collect()].concat(),
            cu_seqlens_q: vec![0, 1, 1 + n as u32],
            cu_seqlens_k: vec![0, n as u32 + 1, 2 * n as u32 + 1],
            slot_mapping: [slots(&table_a, n..n + 1), slots(&table_b, 0..n)].concat(),
            block_tables: vec![table_a.to_vec(), table_b.to_vec()],
            logits_indices: vec![0, n as u32],
            max_seqlen_q: n,
            max_seqlen_k: n + 1,
        };
        batch.validate(8, BLOCK_SIZE).unwrap();
        let got = model.forward(&batch, &mut cache).unwrap().unwrap();
        let d = max_abs_diff(
            &got.narrow(0, 0, 1).unwrap(),
            std::slice::from_ref(&g.decode_logits),
        );
        assert!(d < 1e-4, "A's decode in the mixed batch: {d}");
        let d = max_abs_diff(
            &got.narrow(0, 1, 1).unwrap(),
            &want_b.to_vec2::<f32>().unwrap(),
        );
        assert!(d < 1e-4, "B's prefill in the mixed batch: {d}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_fixture_matches_on_cuda_in_bf16() {
        let Some((dir, g)) = fixture() else { return };
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let (model, mut cache) = load(&dir, DType::BF16, &dev, AttentionImpl::FlashPaged);
        let n = g.prompt_ids.len();
        let got = model
            .forward(
                &chunk(&g.prompt_ids, 0, &[3, 6], (0..n as u32).collect()),
                &mut cache,
            )
            .unwrap()
            .unwrap();
        let range = g
            .full_logits
            .iter()
            .flatten()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        let d = max_abs_diff(&got, &g.full_logits);
        eprintln!("lfm2-tiny on CUDA bf16: max abs diff {d:.4} over a logit range of {range:.2}");
        assert!(d < 0.05 * range);
    }
}
