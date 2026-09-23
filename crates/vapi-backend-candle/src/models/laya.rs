//! The decision head: markers in, calibrated-ready logits out.
//!
//! On top of a [`ModernBert`](super::modernbert::ModernBert) sit four small
//! pieces, all trained from scratch:
//!
//! 1. a **type embedding** added at every position, so the same weights can
//!    answer a choice, a score and a yes/no differently;
//! 2. two **post-encoder transformer layers**, pre-norm with a ReLU feedforward
//!    — PyTorch's `TransformerEncoderLayer` defaults, which are not the
//!    encoder's own conventions and are easy to assume away;
//! 3. a **scorer** read at each option's `[MASK]` marker, giving one logit per
//!    option, softmaxed over that question's options alone;
//! 4. an **act head** reading the pooled `[CLS]` position together with four
//!    summary features of the answer distribution, which is how the model says
//!    it would rather escalate than answer.
//!
//! The logits leave here **uncalibrated**. The temperature that turns them
//! into honest probabilities is fitted per deployment, so baking one in would
//! quietly make it unfittable.

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder, embedding, layer_norm, ops::softmax_last_dim};
use candle_transformers::models::with_tracing::{Linear, linear};

use super::modernbert::{Config as EncoderConfig, Layout, ModernBert, Prepared};

/// What a marker that has no option is scored, before the softmax over a
/// row's options. Matches the reference; large enough to vanish, small enough
/// not to overflow a half-precision exponential.
const ABSENT_MARKER: f64 = -1e4;

/// Features the act head sees alongside the pooled state.
const ACT_FEATURES: usize = 4;

/// One question's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct RowOutput {
    pub logits: Vec<f32>,
    pub act: f32,
}

/// PyTorch's `nn.TransformerEncoderLayer`, pre-norm.
///
/// Worth spelling out rather than reusing an encoder block: the feedforward
/// is ReLU and not gated, the norms carry biases, and the attention keeps its
/// three projections fused in one `in_proj_weight`. None of that matches the
/// ModernBERT layers underneath.
struct HeadLayer {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize, heads: usize, eps: f64) -> Result<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            norm1: layer_norm(hidden, eps, vb.pp("norm1"))?,
            in_proj: fused_in_proj(hidden, attn.clone())?,
            out_proj: linear(hidden, hidden, attn.pp("out_proj"))?,
            norm2: layer_norm(hidden, eps, vb.pp("norm2"))?,
            linear1: linear(hidden, 4 * hidden, vb.pp("linear1"))?,
            linear2: linear(4 * hidden, hidden, vb.pp("linear2"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Prepared<'_>) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let x = (x + self.attention(&normed, ctx)?)?;
        let normed = self.norm2.forward(&x)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&normed)?.relu()?)?;
        &x + ff
    }

    /// Full attention within each row, no rotary — the head layers see
    /// positions only through what the encoder already put there.
    fn attention(&self, x: &Tensor, ctx: &Prepared<'_>) -> Result<Tensor> {
        let (total, d) = x.dims2()?;
        let qkv = self.in_proj.forward(x)?;
        let part = |i: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, i * d, d)?
                .reshape((total, self.heads, self.head_dim))?
                .contiguous()
        };
        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let out = super::modernbert::attend_full(&part(0)?, &part(1)?, &part(2)?, ctx, scale)?;
        self.out_proj.forward(&out.reshape((total, d))?)
    }
}

/// `in_proj_weight` / `in_proj_bias` hold Q, K and V stacked, under names
/// `candle_nn::linear` does not look for.
fn fused_in_proj(hidden: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((3 * hidden, hidden), "in_proj_weight")?;
    let bias = vb.get(3 * hidden, "in_proj_bias")?;
    Ok(Linear::from_weights(weight, Some(bias)))
}

pub struct Laya {
    encoder: ModernBert,
    type_emb: Embedding,
    head: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer_in: Linear,
    scorer_out: Linear,
    act_in: Linear,
    act_out: Linear,
    hidden: usize,
}

impl Laya {
    /// `vb` points at the root of the checkpoint: the encoder lives under
    /// `encoder.`, the head at the top level.
    pub fn load(vb: VarBuilder, encoder_config: EncoderConfig, head_layers: usize) -> Result<Self> {
        let hidden = encoder_config.hidden_size;
        let eps = encoder_config.norm_eps;
        // The reference derives the head's head count from the width, rather
        // than reusing the encoder's.
        let heads = (hidden / 64).max(1);
        let encoder = ModernBert::load(vb.pp("encoder"), encoder_config)?;

        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            head.push(HeadLayer::load(
                vb.pp(format!("head.layers.{i}")),
                hidden,
                heads,
                eps,
            )?);
        }
        Ok(Self {
            type_emb: embedding(3, hidden, vb.pp("type_emb"))?,
            // `nn.Sequential` names its children by position: 0 is the norm,
            // 1 and 3 the projections, 2 the activation.
            scorer_norm: layer_norm(hidden, eps, vb.pp("scorer.0"))?,
            scorer_in: linear(hidden, hidden, vb.pp("scorer.1"))?,
            scorer_out: linear(hidden, 1, vb.pp("scorer.3"))?,
            act_in: linear(hidden + ACT_FEATURES, 256, vb.pp("act_head.0"))?,
            act_out: linear(256, 2, vb.pp("act_head.2"))?,
            encoder,
            head,
            hidden,
        })
    }

    pub fn device(&self) -> &candle_core::Device {
        self.encoder.device()
    }

    pub fn dtype(&self) -> DType {
        self.encoder.dtype()
    }

    /// See [`ModernBert::use_reference_attention`].
    pub fn use_reference_attention(&mut self, yes: bool) {
        self.encoder.use_reference_attention(yes);
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.encoder.config
    }

    /// The encoder's own output for one sequence, with no head on top.
    ///
    /// Only used to localise a parity failure: when the answers are wrong,
    /// this says whether the encoder or the head is the one that drifted.
    pub fn encoder_hidden(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let device = self.encoder.device().clone();
        let layout = Layout::new(&[tokens.len()], &device)?;
        let ctx = self.encoder.prepare(&layout)?;
        let ids = Tensor::from_slice(tokens, (tokens.len(),), &device)?;
        self.encoder
            .forward(&ids, &ctx)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()
    }

    /// Answer a batch.
    ///
    /// `tokens` is every row's sequence end to end, `lengths` says where they
    /// divide, and `markers` gives each row's option positions relative to
    /// its own start.
    pub fn forward(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        markers: &[Vec<u32>],
        qtypes: &[u32],
    ) -> Result<Vec<RowOutput>> {
        let rows = lengths.len();
        if markers.len() != rows || qtypes.len() != rows {
            candle_core::bail!(
                "{rows} rows, {} markers, {} types",
                markers.len(),
                qtypes.len()
            );
        }
        let device = self.encoder.device().clone();
        let layout = Layout::new(lengths, &device)?;
        if layout.total() != tokens.len() {
            candle_core::bail!(
                "{} tokens for rows totalling {}",
                tokens.len(),
                layout.total()
            );
        }
        let ctx = self.encoder.prepare(&layout)?;
        let ids = Tensor::from_slice(tokens, (tokens.len(),), &device)?;
        let mut h = self.encoder.forward(&ids, &ctx)?;

        // The type embedding is added at every position of its own row,
        // before the head. One gather rather than a broadcast, because rows
        // of different types share the flat sequence.
        let mut per_token = Vec::with_capacity(tokens.len());
        for (row, (_, len)) in layout.spans().enumerate() {
            per_token.extend(std::iter::repeat_n(qtypes[row], len));
        }
        let per_token = Tensor::from_vec(per_token, (tokens.len(),), &device)?;
        h = (h + self.type_emb.forward(&per_token)?)?;

        for layer in &self.head {
            h = layer.forward(&h, &ctx)?;
        }

        // Markers are row-relative; the flat tensor wants them absolute.
        let width = markers.iter().map(Vec::len).max().unwrap_or(0);
        if width == 0 {
            candle_core::bail!("no row has an option marker");
        }
        let mut flat = Vec::with_capacity(rows * width);
        for (row, (start, len)) in layout.spans().enumerate() {
            for slot in 0..width {
                // Absent markers point at the row's own first token and are
                // masked out of the softmax below.
                let at = markers[row].get(slot).copied().unwrap_or(0) as usize;
                if at >= len {
                    candle_core::bail!("row {row} marker at {at} is past its {len} tokens");
                }
                flat.push((start + at) as u32);
            }
        }
        let index = Tensor::from_vec(flat, (rows * width,), &device)?;
        let picked = crate::rows::gather(&h, &index)?.reshape((rows, width, self.hidden))?;

        let scored = self.scorer_norm.forward(&picked)?;
        let scored = self.scorer_in.forward(&scored)?.gelu_erf()?;
        let logits = self
            .scorer_out
            .forward(&scored)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        // The act head reads each row's `[CLS]`, which is its first token.
        let cls: Vec<u32> = layout.spans().map(|(start, _)| start as u32).collect();
        let cls = Tensor::from_vec(cls, (rows,), &device)?;
        let pooled = crate::rows::gather(&h, &cls)?;
        let acts = self.act(&pooled, &logits, markers, width)?;

        Ok(logits
            .into_iter()
            .zip(acts)
            .zip(markers)
            .map(|((row, act), positions)| RowOutput {
                logits: row[..positions.len()].to_vec(),
                act,
            })
            .collect())
    }

    /// The probability of acting rather than escalating, per row.
    ///
    /// Its features are computed from the answer distribution the model just
    /// produced — deliberately, so the head can be unsure *because* the
    /// answer is unsure — and in f32, as the reference does.
    fn act(
        &self,
        pooled: &Tensor,
        logits: &[Vec<f32>],
        markers: &[Vec<u32>],
        width: usize,
    ) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(logits.len() * ACT_FEATURES);
        for (row, positions) in logits.iter().zip(markers) {
            let k = positions.len().max(2) as f64;
            let masked: Vec<f32> = (0..width)
                .map(|i| {
                    if i < positions.len() {
                        row[i]
                    } else {
                        ABSENT_MARKER as f32
                    }
                })
                .collect();
            let p = vapi_core::calibrated_probabilities(&masked, 1.0);
            let entropy: f64 = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f64>() / k.ln();
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top = sorted.first().copied().unwrap_or(0.0);
            let second = sorted.get(1).copied().unwrap_or(0.0);
            features.extend([
                top as f32,
                (top - second) as f32,
                entropy as f32,
                (k / 255.0) as f32,
            ]);
        }
        let b = logits.len();
        let features = Tensor::from_vec(features, (b, ACT_FEATURES), pooled.device())?
            .to_dtype(pooled.dtype())?;
        let x = Tensor::cat(&[pooled, &features], D::Minus1)?;
        let act = self
            .act_out
            .forward(&self.act_in.forward(&x)?.gelu_erf()?)?;
        let act = softmax_last_dim(&act.to_dtype(DType::F32)?)?.to_vec2::<f32>()?;
        Ok(act.into_iter().map(|r| r[0]).collect())
    }
}
