//! ModernBERT: the bidirectional encoder the decision head sits on.
//!
//! Unlike everything else in this directory there is no KV cache here. What
//! it shares with them is the **flat, unpadded layout**: every row's tokens
//! end to end in one `[total, hidden]` tensor, with cumulative lengths saying
//! where each begins. A padded `[rows, longest, hidden]` batch would be the
//! obvious shape and is the wrong one — a batch of 64 questions costs
//! `rows × longest` positions of work whatever the rows actually are, and the
//! attention it implies materialises `[rows, heads, longest, longest]`
//! scores. At 64 rows of 512 tokens that is 537 MB per layer of pure padding
//! arithmetic, and it measured 2.5 seconds a pass.
//!
//! `candle-transformers` ships a ModernBERT. This is a separate one because
//! that one expects a `model.` tensor prefix where a decision checkpoint has
//! `encoder.`, reads the pre-5.0 flat `global_rope_theta` keys rather than the
//! nested `rope_parameters` that current exports write, and is padded-dense.
//!
//! Two details are easy to get wrong and expensive to notice:
//!
//! - **Layer 0 has no attention norm.** It is `nn.Identity` in the reference,
//!   so the checkpoint has no tensor for it, and normalising anyway shifts
//!   every activation after it.
//! - **`local_attention: 128` means ±64, inclusive.** It is the full width of
//!   the window, not the reach on each side.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder, embedding, ops::softmax_last_dim};
use candle_transformers::models::with_tracing::{Linear, linear_no_bias};

/// How much a masked-out position is penalised before the softmax on the CPU
/// path. Finite on purpose: `-inf` in a fully masked row gives `NaN`.
const MASK_PENALTY: f32 = -1e4;

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    /// Every nth layer attends globally; the rest use the sliding window.
    pub global_attn_every_n_layers: usize,
    /// Total width of the sliding window, so each side sees half of it.
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub pad_token_id: u32,
}

impl Config {
    /// Read a `config.json`, accepting both the flat rope keys and the nested
    /// `rope_parameters` block that transformers 5 writes.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let num = |key: &str| v.get(key).and_then(serde_json::Value::as_u64);
        let need = |key: &str| {
            num(key)
                .map(|n| n as usize)
                .ok_or_else(|| candle_core::Error::Msg(format!("modernbert config: missing {key}")))
        };
        let theta = |kind: &str, flat: &str, default: f64| -> f64 {
            v.get("rope_parameters")
                .and_then(|r| r.get(kind))
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .or_else(|| v.get(flat).and_then(serde_json::Value::as_f64))
                .unwrap_or(default)
        };
        Ok(Self {
            vocab_size: need("vocab_size")?,
            hidden_size: need("hidden_size")?,
            num_hidden_layers: need("num_hidden_layers")?,
            num_attention_heads: need("num_attention_heads")?,
            intermediate_size: need("intermediate_size")?,
            max_position_embeddings: num("max_position_embeddings").unwrap_or(8192) as usize,
            norm_eps: v
                .get("norm_eps")
                .or_else(|| v.get("layer_norm_eps"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5),
            global_attn_every_n_layers: num("global_attn_every_n_layers").unwrap_or(3) as usize,
            local_attention: num("local_attention").unwrap_or(128) as usize,
            global_rope_theta: theta("full_attention", "global_rope_theta", 160_000.0),
            local_rope_theta: theta("sliding_attention", "local_rope_theta", 10_000.0),
            pad_token_id: num("pad_token_id").unwrap_or(0) as u32,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether layer `i` attends over the whole sequence.
    pub fn is_global(&self, layer: usize) -> bool {
        self.global_attn_every_n_layers == 0
            || layer.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// How far a windowed layer reaches on each side.
    pub fn window(&self) -> usize {
        self.local_attention / 2
    }
}

/// Where each row sits in the flat batch.
///
/// Built once per forward and shared by every layer: the rotary gathers, the
/// attention kernel and the head all index by it.
pub struct Layout {
    /// Cumulative lengths, `rows + 1` entries starting at 0.
    pub cu_seqlens: Vec<u32>,
    pub max_len: usize,
    /// The same, on the device, for the kernel. The CPU reference walks the
    /// host copy instead.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    cu_device: Tensor,
}

impl Layout {
    pub fn new(lengths: &[usize], device: &Device) -> Result<Self> {
        let mut cu = Vec::with_capacity(lengths.len() + 1);
        cu.push(0u32);
        let mut at = 0u32;
        for &len in lengths {
            at += len as u32;
            cu.push(at);
        }
        let cu_device = Tensor::from_slice(&cu, (cu.len(),), device)?;
        Ok(Self {
            max_len: lengths.iter().copied().max().unwrap_or(0),
            cu_seqlens: cu,
            cu_device,
        })
    }

    pub fn rows(&self) -> usize {
        self.cu_seqlens.len().saturating_sub(1)
    }

    pub fn total(&self) -> usize {
        self.cu_seqlens.last().copied().unwrap_or(0) as usize
    }

    /// `(start, len)` of each row.
    pub fn spans(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.cu_seqlens
            .windows(2)
            .map(|w| (w[0] as usize, (w[1] - w[0]) as usize))
    }

    /// Absolute position of every token within its own row.
    pub fn positions(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total());
        for (_, len) in self.spans() {
            out.extend(0..len as u32);
        }
        out
    }
}

/// Rotary tables for one theta, gathered per forward to the batch's positions.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(cfg: &Config, theta: f64, dtype: DType, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let half = dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
            .collect();
        // f32 angles: at theta 160k over 8192 positions, computing these in
        // bf16 moves a token by a fraction of a position.
        let inv = Tensor::from_vec(inv, (1, half), device)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// `[total, head_dim / 2]` for this batch's positions.
    fn gather(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            crate::rows::gather(&self.cos, positions)?,
            crate::rows::gather(&self.sin, positions)?,
        ))
    }
}

/// Everything a layer needs that does not vary by layer.
pub struct Prepared<'a> {
    layout: &'a Layout,
    /// Gathered `(cos, sin)` for the global and the windowed thetas.
    global: (Tensor, Tensor),
    local: (Tensor, Tensor),
    window: usize,
    /// Whether the FlashAttention kernel will run. Decided once, here,
    /// because the reference path needs masks built and the kernel does not:
    /// deriving it twice is how a batch ends up on the reference path with no
    /// window mask, attending over everything.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    flash: bool,
    /// Per-row additive window masks, for the reference path only.
    masks: Vec<Option<Tensor>>,
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            wqkv: linear_no_bias(d, 3 * d, vb.pp("Wqkv"))?,
            wo: linear_no_bias(d, d, vb.pp("Wo"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `x` is `[total, hidden]`.
    fn forward(&self, x: &Tensor, ctx: &Prepared<'_>, global: bool) -> Result<Tensor> {
        let (total, d) = x.dims2()?;
        let qkv = self.wqkv.forward(x)?;
        let part = |i: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, i * d, d)?
                .reshape((total, self.heads, self.head_dim))?
                .contiguous()
        };
        let (cos, sin) = if global { &ctx.global } else { &ctx.local };
        // `rope_thd` takes `(batch, tokens, heads, dim)`; with the tables
        // already gathered to this batch's positions, one row of "batch" is
        // the whole flat sequence.
        let rope = |t: Tensor| -> Result<Tensor> {
            candle_nn::rotary_emb::rope_thd(&t.unsqueeze(0)?, cos, sin)?.squeeze(0)
        };
        let q = rope(part(0)?)?;
        let k = rope(part(1)?)?;
        let v = part(2)?;

        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let window = (!global).then_some(ctx.window);
        let out = attend(&q, &k, &v, ctx, scale, window)?;
        self.wo.forward(&out.reshape((total, d))?)
    }
}

/// Bidirectional varlen attention: FlashAttention on CUDA, a per-row reference
/// on the CPU.
///
/// `window` is the reach on each side, or `None` for a layer that attends over
/// the whole row. Nothing is causal here — an encoder reads both ways, which
/// is `causal = false` and a window open on the right.
#[cfg(feature = "cuda")]
fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ctx: &Prepared<'_>,
    scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    if ctx.flash {
        return candle_flash_attn::flash_attn_varlen_windowed(
            q,
            k,
            v,
            &ctx.layout.cu_device,
            &ctx.layout.cu_device,
            ctx.layout.max_len,
            ctx.layout.max_len,
            scale,
            window,
            window,
        );
    }
    attend_reference(q, k, v, ctx, scale, window)
}

#[cfg(not(feature = "cuda"))]
fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ctx: &Prepared<'_>,
    scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    attend_reference(q, k, v, ctx, scale, window)
}

/// One row at a time, the long way. The numerical reference, and the only
/// path on a machine with no GPU.
fn attend_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ctx: &Prepared<'_>,
    scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(ctx.layout.rows());
    for (row, (start, len)) in ctx.layout.spans().enumerate() {
        // [len, heads, dim] -> [heads, len, dim]
        let take = |t: &Tensor| t.narrow(0, start, len)?.transpose(0, 1)?.contiguous();
        let (qr, kr, vr) = (take(q)?, take(k)?, take(v)?);
        let att = (qr.matmul(&kr.transpose(D::Minus2, D::Minus1)?)? * scale as f64)?;
        let att = match (window, &ctx.masks[row]) {
            (Some(_), Some(mask)) => att.broadcast_add(mask)?,
            _ => att,
        };
        let out = softmax_last_dim(&att)?.matmul(&vr)?;
        rows.push(out.transpose(0, 1)?.contiguous()?);
    }
    Tensor::cat(&rows, 0)
}

/// Full bidirectional attention within each row, for the decision head's own
/// layers, which have no window and no rotary.
pub(crate) fn attend_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ctx: &Prepared<'_>,
    scale: f32,
) -> Result<Tensor> {
    attend(q, k, v, ctx, scale, None)
}

struct Mlp {
    wi: Linear,
    wo: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            // One projection producing the value and its gate, split below.
            wi: linear_no_bias(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("Wi"))?,
            wo: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?,
            intermediate: cfg.intermediate_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wide = self.wi.forward(x)?;
        let value = wide.narrow(D::Minus1, 0, self.intermediate)?;
        let gate = wide.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        // The exact erf gelu, which is what `hidden_activation: "gelu"` means
        // in this config — not the tanh approximation.
        self.wo.forward(&(value.gelu_erf()? * gate)?)
    }
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
    global: bool,
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &Config, index: usize) -> Result<Self> {
        let d = cfg.hidden_size;
        // Layer 0's attn_norm is Identity in the reference, and the
        // checkpoint has no tensor for it.
        let attn_norm = if index == 0 {
            None
        } else {
            Some(norm_no_bias(d, cfg.norm_eps, vb.pp("attn_norm"))?)
        };
        Ok(Self {
            attn_norm,
            attn: Attention::load(vb.pp("attn"), cfg)?,
            mlp_norm: norm_no_bias(d, cfg.norm_eps, vb.pp("mlp_norm"))?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            global: cfg.is_global(index),
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Prepared<'_>) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(n) => n.forward(x)?,
            None => x.clone(),
        };
        let x = (x + self.attn.forward(&normed, ctx, self.global)?)?;
        let normed = self.mlp_norm.forward(&x)?;
        &x + self.mlp.forward(&normed)?
    }
}

fn norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(vb.get(size, "weight")?, eps))
}

/// The encoder. `vb` must already point at the prefix the layers live under.
pub struct ModernBert {
    embeddings: Embedding,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    global_rotary: Rotary,
    local_rotary: Rotary,
    window: usize,
    pub config: Config,
    dtype: DType,
    device: Device,
    force_reference: bool,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let embed_norm = norm_no_bias(cfg.hidden_size, cfg.norm_eps, vb.pp("embeddings.norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::load(vb.pp(format!("layers.{i}")), &cfg, i)?);
        }
        let final_norm = norm_no_bias(cfg.hidden_size, cfg.norm_eps, vb.pp("final_norm"))?;
        Ok(Self {
            global_rotary: Rotary::new(&cfg, cfg.global_rope_theta, dtype, &device)?,
            local_rotary: Rotary::new(&cfg, cfg.local_rope_theta, dtype, &device)?,
            window: cfg.window(),
            embeddings,
            embed_norm,
            layers,
            final_norm,
            config: cfg,
            dtype,
            device,
            force_reference: false,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// FlashAttention is a half-precision CUDA kernel. Anything else — the
    /// CPU, or f32 on the device for debugging — takes the reference path.
    fn uses_flash(&self) -> bool {
        cfg!(feature = "cuda")
            && self.device.is_cuda()
            && matches!(self.dtype, DType::BF16 | DType::F16)
            && !self.force_reference
    }

    /// Run the reference attention even where the kernel would serve.
    ///
    /// An escape hatch for exactly the bug this file has already had once:
    /// when the kernel and the reference disagree, running the same weights
    /// at the same dtype both ways is what separates a wrong window from
    /// ordinary half-precision drift.
    pub fn use_reference_attention(&mut self, yes: bool) {
        self.force_reference = yes;
    }

    /// Build the per-forward context for a batch shaped by `layout`.
    pub fn prepare<'a>(&self, layout: &'a Layout) -> Result<Prepared<'a>> {
        let positions = Tensor::from_vec(layout.positions(), (layout.total(),), &self.device)?;
        let flash = self.uses_flash();
        // The reference path needs one additive window mask per row; the
        // kernel takes the window as two integers and needs none.
        let mut masks = Vec::with_capacity(layout.rows());
        if flash {
            masks.resize_with(layout.rows(), || None);
        } else {
            for (_, len) in layout.spans() {
                masks.push(Some(self.window_mask(len)?));
            }
        }
        Ok(Prepared {
            layout,
            global: self.global_rotary.gather(&positions)?,
            local: self.local_rotary.gather(&positions)?,
            window: self.window,
            flash,
            masks,
        })
    }

    /// `ids` is the flat `[total]` token sequence described by `ctx`.
    pub fn forward(&self, ids: &Tensor, ctx: &Prepared<'_>) -> Result<Tensor> {
        let mut x = self.embed_norm.forward(&self.embeddings.forward(ids)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, ctx)?;
        }
        self.final_norm.forward(&x)
    }

    fn window_mask(&self, len: usize) -> Result<Tensor> {
        let w = self.window;
        let mut data = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..len {
                if i.abs_diff(j) > w {
                    data[i * len + j] = MASK_PENALTY;
                }
            }
        }
        Tensor::from_vec(data, (1, len, len), &self.device)?.to_dtype(self.dtype)
    }
}
