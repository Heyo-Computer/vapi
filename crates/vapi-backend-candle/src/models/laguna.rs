//! Laguna (poolside) with a paged KV cache and a continuous-batch forward.
//!
//! Written against poolside's `modeling_laguna.py` (shipped with every
//! Laguna checkpoint) and checked against it on `Laguna-tiny-per-element`
//! and a random-initialised config carrying every Laguna-XS-2.1 feature.
//! What differs from the vendored Llama:
//!
//! - **Attention** has per-head RMSNorm on q and k before RoPE, *partial*
//!   rotary (only the first `head_dim × partial_rotary_factor` dims rotate),
//!   an output gate `softplus(g_proj(x))` applied per head or per element
//!   before `o_proj`, and a per-layer head count. Layers are either
//!   full-attention (YaRN RoPE) or sliding-window (plain RoPE, window
//!   `sliding_window`), each with its own cos/sin tables.
//! - **MLP** is dense on `mlp_only_layers` and a sparse MoE elsewhere:
//!   sigmoid router, a per-expert selection bias that does not affect the
//!   weights, top-k renormalisation, a routed scaling factor, plus a shared
//!   expert added to the routed output.
//!
//! The router's top-k runs on the host over an `(n, num_experts)` matrix
//! and the experts are applied one at a time with `index_select` /
//! `index_add`, exactly as the reference does. That is a correctness-first
//! implementation; a fused MoE kernel is a later optimisation behind the
//! same interface.

use std::collections::HashMap;
use std::f32::consts::PI;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_no_bias as linear};
use vapi_engine::ForwardBatch;

use crate::attention::paged_attention_cpu;
use crate::cache::{PagedKvCache, write_kv_to_cache};
use crate::models::llama::AttentionImpl;

/// `gating` in config.json: `true`/`"per-element"` (one gate per channel),
/// `"per-head"`, or `false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    None,
    PerHead,
    PerElement,
}

/// RoPE parameters for one layer type.
#[derive(Clone, Debug)]
pub struct RopeSpec {
    pub theta: f32,
    pub partial_rotary_factor: f32,
    /// `None` is plain RoPE; `Some` is YaRN.
    pub yarn: Option<Yarn>,
}

#[derive(Clone, Debug)]
pub struct Yarn {
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// Explicit `attention_factor`, else derived from `factor`.
    pub attention_factor: Option<f32>,
    pub truncate: bool,
}

#[derive(Clone, Debug)]
pub struct LagunaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub gating: Gating,
    pub sliding_window: Option<usize>,
    /// `"full_attention"` or `"sliding_attention"` per layer.
    pub layer_types: Vec<String>,
    pub heads_per_layer: Vec<usize>,
    pub rope_full: RopeSpec,
    pub rope_sliding: RopeSpec,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    pub routed_scaling_factor: f32,
    pub router_logit_softcapping: f32,
    pub eos_token_ids: Vec<u32>,
}

fn get_usize(v: &serde_json::Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| candle_core::Error::Msg(format!("config.json: missing integer `{k}`")))
}

fn get_or<T>(
    v: &serde_json::Value,
    k: &str,
    default: T,
    f: impl Fn(&serde_json::Value) -> Option<T>,
) -> T {
    v.get(k).and_then(f).unwrap_or(default)
}

fn rope_spec(v: &serde_json::Value, default_theta: f32, default_partial: f32) -> Result<RopeSpec> {
    let theta = get_or(v, "rope_theta", default_theta, |x| {
        x.as_f64().map(|f| f as f32)
    });
    let partial = get_or(v, "partial_rotary_factor", default_partial, |x| {
        x.as_f64().map(|f| f as f32)
    });
    let rope_type = get_or(v, "rope_type", "default".to_string(), |x| {
        x.as_str().map(str::to_string)
    });
    let yarn = match rope_type.as_str() {
        "default" => None,
        "yarn" => Some(Yarn {
            factor: get_or(v, "factor", 1.0, |x| x.as_f64().map(|f| f as f32)),
            original_max_position_embeddings: get_usize(v, "original_max_position_embeddings")?,
            beta_fast: get_or(v, "beta_fast", 32.0, |x| x.as_f64().map(|f| f as f32)),
            beta_slow: get_or(v, "beta_slow", 1.0, |x| x.as_f64().map(|f| f as f32)),
            attention_factor: v
                .get("attention_factor")
                .and_then(|x| x.as_f64())
                .map(|f| f as f32),
            truncate: get_or(v, "truncate", true, |x| x.as_bool()),
        }),
        other => candle_core::bail!("unsupported rope_type {other:?} for Laguna"),
    };
    Ok(RopeSpec {
        theta,
        partial_rotary_factor: partial,
        yarn,
    })
}

impl LagunaConfig {
    /// Parse a Laguna `config.json`. Defaults follow `configuration_laguna.py`.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let num_hidden_layers = get_usize(v, "num_hidden_layers")?;
        let num_attention_heads = get_usize(v, "num_attention_heads")?;
        let hidden_size = get_usize(v, "hidden_size")?;
        let head_dim = get_or(v, "head_dim", hidden_size / num_attention_heads, |x| {
            x.as_u64().map(|x| x as usize)
        });
        let gating = match v.get("gating") {
            None | Some(serde_json::Value::Bool(true)) => Gating::PerElement,
            Some(serde_json::Value::Bool(false)) => Gating::None,
            Some(serde_json::Value::String(s)) if s == "per-head" => Gating::PerHead,
            Some(serde_json::Value::String(s)) if s == "per-element" => Gating::PerElement,
            Some(other) => candle_core::bail!("unsupported gating {other}"),
        };
        if get_or(v, "swa_attention_sink_enabled", false, |x| x.as_bool()) {
            candle_core::bail!("swa_attention_sink_enabled is not supported");
        }
        let layer_types: Vec<String> = match v.get("layer_types").and_then(|x| x.as_array()) {
            Some(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            None => vec!["full_attention".to_string(); num_hidden_layers],
        };
        let heads_per_layer = match v
            .get("num_attention_heads_per_layer")
            .and_then(|x| x.as_array())
        {
            Some(a) => a
                .iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect(),
            None => vec![num_attention_heads; num_hidden_layers],
        };
        if layer_types.len() != num_hidden_layers || heads_per_layer.len() != num_hidden_layers {
            candle_core::bail!(
                "layer_types / num_attention_heads_per_layer do not cover every layer"
            );
        }

        // rope_parameters is either flat or nested by layer type; the
        // top-level partial_rotary_factor, if any, fills in for dicts that
        // lack one (configuration_laguna.py does the same).
        let top_partial = v
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);
        let rp = v
            .get("rope_parameters")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let (full_v, sliding_v) = if rp.get("full_attention").is_some() {
            (
                rp["full_attention"].clone(),
                rp.get("sliding_attention")
                    .cloned()
                    .unwrap_or(rp["full_attention"].clone()),
            )
        } else {
            let swa = v.get("swa_rope_parameters").cloned().unwrap_or(rp.clone());
            (rp.clone(), swa)
        };
        let rope_full = rope_spec(&full_v, 500_000.0, top_partial)?;
        let rope_sliding = rope_spec(&sliding_v, 500_000.0, top_partial)?;

        let num_experts = get_or(v, "num_experts", 0, |x| x.as_u64().map(|x| x as usize));
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
            intermediate_size: get_usize(v, "intermediate_size")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: get_usize(v, "num_key_value_heads")?,
            head_dim,
            rms_norm_eps: get_or(v, "rms_norm_eps", 1e-6, |x| x.as_f64()),
            max_position_embeddings: get_usize(v, "max_position_embeddings")?,
            tie_word_embeddings: get_or(v, "tie_word_embeddings", false, |x| x.as_bool()),
            gating,
            sliding_window: v
                .get("sliding_window")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize),
            layer_types,
            heads_per_layer,
            rope_full,
            rope_sliding,
            num_experts,
            num_experts_per_tok: get_or(v, "num_experts_per_tok", 0, |x| {
                x.as_u64().map(|x| x as usize)
            }),
            moe_intermediate_size: get_or(v, "moe_intermediate_size", 0, |x| {
                x.as_u64().map(|x| x as usize)
            }),
            shared_expert_intermediate_size: get_or(v, "shared_expert_intermediate_size", 0, |x| {
                x.as_u64().map(|x| x as usize)
            }),
            norm_topk_prob: get_or(v, "norm_topk_prob", true, |x| x.as_bool()),
            decoder_sparse_step: get_or(v, "decoder_sparse_step", 1, |x| {
                x.as_u64().map(|x| x as usize)
            }),
            mlp_only_layers: match v.get("mlp_only_layers").and_then(|x| x.as_array()) {
                Some(a) => a
                    .iter()
                    .filter_map(|x| x.as_u64().map(|x| x as usize))
                    .collect(),
                None => vec![0],
            },
            routed_scaling_factor: get_or(v, "moe_routed_scaling_factor", 1.0, |x| {
                x.as_f64().map(|f| f as f32)
            }),
            router_logit_softcapping: get_or(v, "moe_router_logit_softcapping", 0.0, |x| {
                x.as_f64().map(|f| f as f32)
            }),
            eos_token_ids,
        })
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types[layer] == "sliding_attention"
    }

    pub fn is_sparse(&self, layer: usize) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && self.num_experts > 0
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

/// Inverse frequencies as `_compute_yarn_parameters` /
/// `compute_default_rope_parameters` in transformers produce them, plus the
/// cos/sin attention scaling.
fn inv_freq(spec: &RopeSpec, head_dim: usize) -> (Vec<f32>, f32) {
    let dim = (head_dim as f32 * spec.partial_rotary_factor) as usize;
    let pos_freqs: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| spec.theta.powf(i as f32 / dim as f32))
        .collect();
    let Some(y) = &spec.yarn else {
        return (pos_freqs.iter().map(|f| 1.0 / f).collect(), 1.0);
    };
    let attention_factor = y.attention_factor.unwrap_or_else(|| {
        if y.factor <= 1.0 {
            1.0
        } else {
            0.1 * y.factor.ln() + 1.0
        }
    });
    let correction_dim = |num_rotations: f32| -> f32 {
        (dim as f32 * (y.original_max_position_embeddings as f32 / (num_rotations * 2.0 * PI)).ln())
            / (2.0 * spec.theta.ln())
    };
    let (mut low, mut high) = (correction_dim(y.beta_fast), correction_dim(y.beta_slow));
    if y.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let high = high.min(dim as f32 - 1.0);
    let half = dim / 2;
    let (lo, hi) = if low == high {
        (low, high + 0.001)
    } else {
        (low, high)
    };
    let inv: Vec<f32> = pos_freqs
        .iter()
        .enumerate()
        .map(|(i, pf)| {
            let ramp = ((i as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            let interpolation = 1.0 / (y.factor * pf);
            let extrapolation = 1.0 / pf;
            interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
        })
        .collect();
    debug_assert_eq!(inv.len(), half);
    (inv, attention_factor)
}

/// `(cos, sin)` tables `(max_pos, rot_dim / 2)`, scaled by the attention
/// factor, in the model dtype.
fn rope_tables(
    spec: &RopeSpec,
    head_dim: usize,
    max_pos: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (inv, scaling) = inv_freq(spec, head_dim);
    let half = inv.len();
    let mut cos = Vec::with_capacity(max_pos * half);
    let mut sin = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos() * scaling);
            sin.push(a.sin() * scaling);
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_pos, half), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (max_pos, half), device)?.to_dtype(dtype)?,
    ))
}

/// `ln(1 + e^x)`, computed as `max(x, 0) + ln(1 + e^-|x|)` so large inputs
/// neither overflow nor lose precision.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let tail = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    pos + tail
}

struct RopePair {
    cos: Tensor,
    sin: Tensor,
}

/// Per-token RoPE rows for both layer types plus the batch.
struct StepContext<'a> {
    batch: &'a ForwardBatch,
    full: RopePair,
    sliding: RopePair,
}

/// Apply partial rotary to `x: (n, heads, head_dim)` using per-token
/// `(n, rot/2)` tables.
fn apply_partial_rope(x: &Tensor, pair: &RopePair) -> Result<Tensor> {
    let (_n, _h, head_dim) = x.dims3()?;
    let rot = pair.cos.dim(1)? * 2;
    let x_t = x.transpose(0, 1)?.unsqueeze(0)?.contiguous()?; // (1, heads, n, d)
    let rotated = if rot == head_dim {
        candle_nn::rotary_emb::rope(&x_t, &pair.cos, &pair.sin)?
    } else {
        let head = x_t.narrow(3, 0, rot)?.contiguous()?;
        let pass = x_t.narrow(3, rot, head_dim - rot)?;
        let head = candle_nn::rotary_emb::rope(&head, &pair.cos, &pair.sin)?;
        Tensor::cat(&[&head, &pass], 3)?
    };
    rotated.squeeze(0)?.transpose(0, 1)?.contiguous()
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    g_proj: Option<Linear>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gating: Gating,
    sliding_window: Option<usize>,
    attention: AttentionImpl,
}

impl Attention {
    fn load(
        vb: VarBuilder,
        cfg: &LagunaConfig,
        layer: usize,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.heads_per_layer[layer];
        let hd = cfg.head_dim;
        let g_proj = match cfg.gating {
            Gating::None => None,
            Gating::PerHead => Some(linear(h, num_heads, vb.pp("g_proj"))?),
            Gating::PerElement => Some(linear(h, num_heads * hd, vb.pp("g_proj"))?),
        };
        Ok(Self {
            q_proj: linear(h, num_heads * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o_proj: linear(num_heads * hd, h, vb.pp("o_proj"))?,
            g_proj,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            gating: cfg.gating,
            sliding_window: if cfg.is_sliding(layer) {
                cfg.sliding_window
            } else {
                None
            },
            attention,
        })
    }

    /// RMSNorm over `head_dim` for every (token, head).
    fn head_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
        let (n, h, d) = x.dims3()?;
        norm.forward(&x.reshape((n * h, d))?)?.reshape((n, h, d))
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext<'_>,
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
        let q = Self::head_norm(&self.q_norm, &q)?;
        let k = Self::head_norm(&self.k_norm, &k)?;
        let pair = if self.sliding_window.is_some() {
            &ctx.sliding
        } else {
            &ctx.full
        };
        let q = apply_partial_rope(&q, pair)?;
        let k = apply_partial_rope(&k, pair)?;

        write_kv_to_cache(cache, layer, &k, &v, &ctx.batch.slot_mapping)?;

        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let y = match self.attention {
            AttentionImpl::Reference => paged_attention_cpu(
                &q,
                &cache.k[layer],
                &cache.v[layer],
                ctx.batch,
                scale,
                self.sliding_window,
            )?,
            #[cfg(feature = "cuda")]
            AttentionImpl::FlashPaged => crate::attention::paged_attention_cuda(
                &q,
                &cache.k[layer],
                &cache.v[layer],
                ctx.batch,
                scale,
                cache.block_size,
                self.sliding_window,
            )?,
        };
        // (n, heads, head_dim)
        let y = match (&self.g_proj, self.gating) {
            (Some(g), Gating::PerHead) => {
                let gate = softplus(&g.forward(x)?.to_dtype(DType::F32)?)?
                    .to_dtype(y.dtype())?
                    .unsqueeze(2)?; // (n, heads, 1)
                y.broadcast_mul(&gate)?
            }
            (Some(g), Gating::PerElement) => {
                let gate = softplus(&g.forward(x)?.to_dtype(DType::F32)?)?
                    .to_dtype(y.dtype())?
                    .reshape((n, self.num_heads, self.head_dim))?;
                (y * gate)?
            }
            _ => y,
        };
        self.o_proj
            .forward(&y.reshape((n, self.num_heads * self.head_dim))?)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, hidden: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(hidden, inter, vb.pp("gate_proj"))?,
            up_proj: linear(hidden, inter, vb.pp("up_proj"))?,
            down_proj: linear(inter, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }
}

/// Sigmoid-routed MoE with a shared expert.
struct MoeBlock {
    /// `(num_experts, hidden)`
    router: Tensor,
    /// `(num_experts)`, added to the scores for *selection* only.
    selection_bias: Tensor,
    /// Per expert `(2 × inter, hidden)`: gate rows then up rows.
    gate_up: Vec<Tensor>,
    /// Per expert `(hidden, inter)`.
    down: Vec<Tensor>,
    shared: Mlp,
    top_k: usize,
    norm_topk: bool,
    scaling: f32,
    softcap: f32,
    inter: usize,
}

impl MoeBlock {
    fn load(vb: VarBuilder, cfg: &LagunaConfig) -> Result<Self> {
        let (e, h, i) = (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size);
        let router = vb.get((e, h), "gate.weight")?;
        let selection_bias = vb
            .get((e,), "gate.e_score_correction_bias")
            .or_else(|_| vb.get((e,), "experts.e_score_correction_bias"))
            .or_else(|_| Tensor::zeros((e,), vb.dtype(), vb.device()))?;
        // Two checkpoint layouts: fused 3-D tensors (as `state_dict()` saves
        // them) or one `experts.N.{gate,up,down}_proj.weight` per expert (as
        // poolside ships them).
        let ex = vb.pp("experts");
        let (gate_up, down) = if ex.contains_tensor("gate_up_proj") {
            let gu = ex.get((e, 2 * i, h), "gate_up_proj")?;
            let d = ex.get((e, h, i), "down_proj")?;
            (
                (0..e).map(|k| gu.get(k)).collect::<Result<Vec<_>>>()?,
                (0..e).map(|k| d.get(k)).collect::<Result<Vec<_>>>()?,
            )
        } else {
            let mut gate_up = Vec::with_capacity(e);
            let mut down = Vec::with_capacity(e);
            for k in 0..e {
                let p = ex.pp(k.to_string());
                let g = p.get((i, h), "gate_proj.weight")?;
                let u = p.get((i, h), "up_proj.weight")?;
                gate_up.push(Tensor::cat(&[&g, &u], 0)?);
                down.push(p.get((h, i), "down_proj.weight")?);
            }
            (gate_up, down)
        };
        Ok(Self {
            router,
            selection_bias,
            gate_up,
            down,
            shared: Mlp::load(
                vb.pp("shared_expert"),
                h,
                cfg.shared_expert_intermediate_size,
            )?,
            top_k: cfg.num_experts_per_tok,
            norm_topk: cfg.norm_topk_prob,
            scaling: cfg.routed_scaling_factor,
            softcap: cfg.router_logit_softcapping,
            inter: i,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, _h) = x.dims2()?;
        let shared = self.shared.forward(x)?;

        // Router in f32, as the reference does.
        let logits = x
            .to_dtype(DType::F32)?
            .matmul(&self.router.to_dtype(DType::F32)?.t()?)?; // (n, E)
        let logits = if self.softcap > 0.0 {
            ((logits / self.softcap as f64)?.tanh()? * self.softcap as f64)?
        } else {
            logits
        };
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let selection = scores.broadcast_add(&self.selection_bias.to_dtype(DType::F32)?)?;
        let scores = scores.to_vec2::<f32>()?;
        let selection = selection.to_vec2::<f32>()?;

        // Host-side top-k on the biased scores; weights are the unbiased ones.
        let mut per_expert: HashMap<usize, (Vec<u32>, Vec<f32>)> = HashMap::new();
        for (t, sel) in selection.iter().enumerate() {
            let mut idx: Vec<usize> = (0..sel.len()).collect();
            idx.sort_by(|&a, &b| {
                sel[b]
                    .partial_cmp(&sel[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let chosen = &idx[..self.top_k];
            let mut w: Vec<f32> = chosen.iter().map(|&e| scores[t][e]).collect();
            if self.norm_topk {
                let z: f32 = w.iter().sum();
                for v in &mut w {
                    *v /= z;
                }
            }
            for (&e, &wt) in chosen.iter().zip(&w) {
                let slot = per_expert.entry(e).or_default();
                slot.0.push(t as u32);
                slot.1.push(wt * self.scaling);
            }
        }

        let mut out = Tensor::zeros_like(&shared)?;
        let mut experts: Vec<_> = per_expert.into_iter().collect();
        experts.sort_by_key(|(e, _)| *e);
        for (e, (tokens, weights)) in experts {
            let idx = Tensor::from_vec(tokens, (weights.len(),), x.device())?;
            let xs = x.index_select(&idx, 0)?;
            let gu = xs.matmul(&self.gate_up[e].t()?)?;
            let gate = gu.narrow(1, 0, self.inter)?;
            let up = gu.narrow(1, self.inter, self.inter)?;
            let hdn = (candle_nn::ops::silu(&gate)? * up)?;
            let y = hdn.matmul(&self.down[e].t()?)?;
            let w = Tensor::from_vec(weights, (idx.dim(0)?, 1), x.device())?.to_dtype(y.dtype())?;
            let y = y.broadcast_mul(&w)?;
            out = out.index_add(&idx, &y, 0)?;
        }
        let _ = n;
        out + shared
    }
}

enum FeedForward {
    Dense(Mlp),
    Sparse(MoeBlock),
}

struct Block {
    input_norm: RmsNorm,
    attn: Attention,
    post_norm: RmsNorm,
    ff: FeedForward,
}

impl Block {
    fn load(
        vb: VarBuilder,
        cfg: &LagunaConfig,
        layer: usize,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let ff = if cfg.is_sparse(layer) {
            FeedForward::Sparse(MoeBlock::load(vb.pp("mlp"), cfg)?)
        } else {
            FeedForward::Dense(Mlp::load(
                vb.pp("mlp"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?)
        };
        Ok(Self {
            input_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::load(vb.pp("self_attn"), cfg, layer, attention)?,
            post_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            ff,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        ctx: &StepContext<'_>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let h = self
            .attn
            .forward(&self.input_norm.forward(x)?, layer, ctx, cache)?;
        let x = (x + h)?;
        let n = self.post_norm.forward(&x)?;
        let h = match &self.ff {
            FeedForward::Dense(m) => m.forward(&n)?,
            FeedForward::Sparse(m) => m.forward(&n)?,
        };
        x + h
    }
}

/// Laguna over a paged KV cache.
pub struct PagedLaguna {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    full: (Tensor, Tensor),
    sliding: (Tensor, Tensor),
    device: Device,
}

impl PagedLaguna {
    pub fn load(
        vb: VarBuilder,
        cfg: &LagunaConfig,
        dtype: DType,
        device: &Device,
        attention: AttentionImpl,
    ) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|l| Block::load(vb.pp(format!("model.layers.{l}")), cfg, l, attention))
            .collect::<Result<Vec<_>>>()?;
        let max_pos = cfg.max_position_embeddings;
        Ok(Self {
            embed,
            blocks,
            norm,
            lm_head,
            full: rope_tables(&cfg.rope_full, cfg.head_dim, max_pos, dtype, device)?,
            sliding: rope_tables(&cfg.rope_sliding, cfg.head_dim, max_pos, dtype, device)?,
            device: device.clone(),
        })
    }

    /// Same contract as `PagedLlama::forward`: f32 logits for
    /// `batch.logits_indices`, or `None` when nothing is sampled, with every
    /// token's K/V written to the cache either way.
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
            full: RopePair {
                cos: self.full.0.index_select(&positions, 0)?,
                sin: self.full.1.index_select(&positions, 0)?,
            },
            sliding: RopePair {
                cos: self.sliding.0.index_select(&positions, 0)?,
                sin: self.sliding.1.index_select(&positions, 0)?,
            },
        };
        let mut x = self.embed.forward(&tokens)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, layer, &ctx, cache)?;
        }
        if batch.logits_indices.is_empty() {
            return Ok(None);
        }
        let x = self.norm.forward(&x)?;
        let idx = Tensor::from_slice(
            &batch.logits_indices,
            batch.logits_indices.len(),
            &self.device,
        )?;
        let x = x.index_select(&idx, 0)?.contiguous()?;
        Ok(Some(self.lm_head.forward(&x)?.to_dtype(DType::F32)?))
    }
}

#[cfg(test)]
mod tests {
    //! Parity with poolside's reference implementation on the fixtures
    //! `tools/gen_laguna_goldens.py` produces. Skips when the fixture model
    //! directories are absent.

    use super::*;
    use std::path::{Path, PathBuf};
    use vapi_core::config::BLOCK_SIZE;

    #[derive(serde::Deserialize)]
    struct Golden {
        name: String,
        prompt_ids: Vec<u32>,
        full_logits: Vec<Vec<f32>>,
        decode_token: u32,
        decode_logits: Vec<f32>,
    }

    fn fixture(name: &str) -> Option<(PathBuf, Golden)> {
        let home = std::env::var("HOME").unwrap_or_default();
        let dir = Path::new(&home).join("models").join(name);
        if !dir.join("config.json").exists() {
            eprintln!(
                "SKIP: no fixture at {}; run tools/gen_laguna_goldens.py",
                dir.display()
            );
            return None;
        }
        let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/goldens")
            .join(format!("{name}.json"));
        let g: Golden = serde_json::from_str(&std::fs::read_to_string(golden).ok()?).ok()?;
        Some((dir, g))
    }

    fn load(dir: &Path) -> (PagedLaguna, LagunaConfig, PagedKvCache) {
        let dev = Device::Cpu;
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let cfg = LagunaConfig::from_json(&v).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)
                .unwrap()
        };
        let model =
            PagedLaguna::load(vb, &cfg, DType::F32, &dev, AttentionImpl::Reference).unwrap();
        let cache = PagedKvCache::new(
            cfg.num_hidden_layers,
            8,
            BLOCK_SIZE,
            cfg.num_key_value_heads,
            cfg.head_dim,
            DType::F32,
            &dev,
        )
        .unwrap();
        (model, cfg, cache)
    }

    fn slots(table: &[u32], positions: std::ops::Range<usize>) -> Vec<u32> {
        positions
            .map(|p| table[p / BLOCK_SIZE] * BLOCK_SIZE as u32 + (p % BLOCK_SIZE) as u32)
            .collect()
    }

    /// One sequence forwarding `tokens[start..]`, logits for `want` indices
    /// (relative to the chunk).
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
        let got = got.to_vec2::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "row count");
        got.iter()
            .zip(want)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f32::max)
    }

    fn check(name: &str) {
        let Some((dir, g)) = fixture(name) else {
            return;
        };
        let (model, _cfg, mut cache) = load(&dir);
        let table = [3u32, 6];
        let n = g.prompt_ids.len();

        // Whole prompt, logits at every position.
        let all = (0..n as u32).collect();
        let got = model
            .forward(&chunk(&g.prompt_ids, 0, &table, all), &mut cache)
            .unwrap()
            .unwrap();
        let d = max_abs_diff(&got, &g.full_logits);
        assert!(d < 1e-4, "{}: full prompt max abs diff {d}", g.name);

        // One decode step against the cache.
        let mut seq = g.prompt_ids.clone();
        seq.push(g.decode_token);
        let got = model
            .forward(&chunk(&seq, n, &table, vec![0]), &mut cache)
            .unwrap()
            .unwrap();
        let d = max_abs_diff(&got, std::slice::from_ref(&g.decode_logits));
        assert!(d < 1e-4, "{}: decode max abs diff {d}", g.name);

        // Chunked prefill on a fresh cache lands on the same last-row logits.
        let (model2, _, mut cache2) = load(&dir);
        let split = n / 2;
        assert!(
            model2
                .forward(
                    &chunk(&g.prompt_ids[..split], 0, &table, vec![]),
                    &mut cache2
                )
                .unwrap()
                .is_none()
        );
        let rest = chunk(&g.prompt_ids, split, &table, vec![(n - split - 1) as u32]);
        let got = model2.forward(&rest, &mut cache2).unwrap().unwrap();
        let d = max_abs_diff(&got, std::slice::from_ref(&g.full_logits[n - 1]));
        assert!(d < 1e-4, "{}: chunked prefill max abs diff {d}", g.name);
        eprintln!(
            "{}: full, decode and chunked prefill all within 1e-4",
            g.name
        );
    }

    #[test]
    fn the_tiny_checkpoint_matches_the_reference() {
        check("laguna-tiny");
    }

    #[test]
    fn the_xs_feature_set_matches_the_reference() {
        // Sliding layers with their own RoPE, YaRN with an explicit
        // attention factor, per-head gating, per-layer head counts, routed
        // scaling and a non-zero selection bias.
        check("laguna-tiny-xs");
    }

    /// The CUDA path (flash kernel, sliding window through
    /// `window_size_left`, 64-way GQA in the XS clone) against the f32
    /// reference logits, in bf16. Tolerance is relative to the logit range,
    /// since bf16 keeps about two decimal digits.
    #[cfg(feature = "cuda")]
    fn check_cuda(name: &str) {
        let Some((dir, g)) = fixture(name) else {
            return;
        };
        let dev = Device::new_cuda(0).expect("a CUDA device");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let cfg = LagunaConfig::from_json(&v).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::BF16, &dev)
                .unwrap()
        };
        let n = g.prompt_ids.len();
        let range = g
            .full_logits
            .iter()
            .flatten()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        // Same weights, same device, same dtype: the flash kernel path and
        // the gather-and-matmul reference path. Their difference isolates
        // the kernel; the reference path's distance from the f32 goldens is
        // bf16 itself (a random-weight MoE flips near-tied routing easily).
        let run = |attention: AttentionImpl| {
            let model = PagedLaguna::load(vb.clone(), &cfg, DType::BF16, &dev, attention).unwrap();
            let mut cache = PagedKvCache::new(
                cfg.num_hidden_layers,
                8,
                BLOCK_SIZE,
                cfg.num_key_value_heads,
                cfg.head_dim,
                DType::BF16,
                &dev,
            )
            .unwrap();
            let all = (0..n as u32).collect();
            model
                .forward(&chunk(&g.prompt_ids, 0, &[3, 6], all), &mut cache)
                .unwrap()
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
        };
        let got = run(AttentionImpl::FlashPaged);
        let via_reference = run(AttentionImpl::Reference);
        let d_kernel = max_abs_diff(&got, &via_reference.to_vec2::<f32>().unwrap());
        let d_flash = max_abs_diff(&got, &g.full_logits);
        let d_ref = max_abs_diff(&via_reference, &g.full_logits);
        eprintln!(
            "{name} on CUDA bf16: flash vs f32 goldens {d_flash:.4}, reference attention vs goldens {d_ref:.4}, flash vs reference {d_kernel:.4}, logit range {range:.2}"
        );
        // On a trained-shaped checkpoint both paths sit within bf16 noise of
        // the f32 goldens. On the random XS clone, bf16 alone flips near-tied
        // expert routing and moves logits by O(1); the kernel may not add
        // error beyond what bf16 already does. The kernel itself is proven
        // at Laguna's shapes in `attention::cuda_tests::laguna_shapes_match_the_cpu_reference`.
        assert!(
            d_kernel <= d_ref.max(0.05 * range),
            "{name}: the flash kernel path adds error beyond bf16's own"
        );
        // The greedy choice at the last position agrees with the f32
        // reference unless bf16 alone already moved it.
        let last = got.get(n - 1).unwrap().to_vec1::<f32>().unwrap();
        let argmax = (0..last.len())
            .max_by(|&a, &b| last[a].total_cmp(&last[b]))
            .unwrap();
        let want = &g.full_logits[n - 1];
        let want_argmax = (0..want.len())
            .max_by(|&a, &b| want[a].total_cmp(&want[b]))
            .unwrap();
        if d_ref < 0.05 * range {
            assert_eq!(argmax, want_argmax, "{name}: greedy next token on CUDA");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_tiny_checkpoint_matches_the_reference_on_cuda() {
        check_cuda("laguna-tiny");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_xs_feature_set_matches_the_reference_on_cuda() {
        // Exercises the sliding-window kernel path and per-layer head counts.
        check_cuda("laguna-tiny-xs");
    }

    #[test]
    fn a_mixed_batch_gives_each_sequence_what_it_would_get_alone() {
        let Some((dir, g)) = fixture("laguna-tiny-xs") else {
            return;
        };
        let (model, _, mut cache) = load(&dir);
        let n = g.prompt_ids.len();
        let a = &g.prompt_ids;
        let b: Vec<u32> = g.prompt_ids.iter().rev().cloned().collect();
        let table_a = [1u32, 5];
        let table_b = [7u32, 2];
        // References: each alone.
        let (m1, _, mut c1) = load(&dir);
        let want_b = m1
            .forward(&chunk(&b, 0, &table_b, vec![(n - 1) as u32]), &mut c1)
            .unwrap()
            .unwrap();
        // A's prompt into the shared cache, then a mixed batch: A decodes
        // one token while B prefills.
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
}
