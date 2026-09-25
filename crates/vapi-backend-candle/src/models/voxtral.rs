//! Voxtral Realtime: speech in, text out.
//!
//! Two stacks and a projector, and the surprise on reading the checkpoint is
//! how little of it is new. Both stacks are Llama-shaped — RMSNorm, gated
//! SiLU feedforward, rotary embeddings, causal attention with a sliding
//! window — so the only genuinely unfamiliar pieces are the convolutional
//! stem over log-mel bins, the projector that groups four encoder frames into
//! one decoder position, and a per-layer adaptive norm.
//!
//! # The mechanism that decides everything else
//!
//! One decoder position sits on **every 80 ms of audio**: eight 10 ms mel
//! frames, halved by the stem's strided convolution and quartered again by
//! the projector's grouping. The merge is addition —
//! `inputs_embeds = embed(previous token) + audio_embed(this frame)` — not
//! interleaving, so the audio does not occupy positions of its own. The model
//! emits exactly one text token per position, `[STREAMING_PAD]` when it has
//! nothing to say yet, which is what "a single text-token is worth 80 ms"
//! means and why the decode is lock-step with wall-clock audio.
//!
//! # The adaptive norm
//!
//! Every decoder layer scales its post-attention activations by
//! `1 + linear2(gelu(linear1(t_cond)))`, where `t_cond` is a **parameter-free**
//! sinusoidal embedding of the delay in tokens. That is how one set of weights
//! serves every delay from 80 ms to 2.4 s. It varies with neither position nor
//! step, so the per-layer scales are computed once per session and then cost
//! nothing — see [`Conditioning`].

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Conv1d, Conv1dConfig, Embedding, RmsNorm, VarBuilder, embedding, rms_norm};
use candle_transformers::models::with_tracing::{Linear, linear as linear_bias, linear_no_bias};

/// Penalty applied to a masked position before the softmax on the reference
/// path. Finite, so a fully masked row cannot produce `NaN`.
const MASK_PENALTY: f32 = -1e4;

// ----------------------------------------------------------------- config

#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_mel_bins: usize,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
}

#[derive(Clone, Debug)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
}

#[derive(Clone, Debug)]
pub struct VoxtralConfig {
    pub audio: AudioConfig,
    pub text: TextConfig,
    /// Mel frames per decoder position: eight, or 80 ms.
    pub audio_length_per_tok: usize,
    /// Encoder frames the projector groups into one position.
    pub downsample_factor: usize,
    pub default_num_delay_tokens: usize,
    /// How the audio is padded before it reaches the model. Not in
    /// `config.json` — it comes from the tokenizer's `audio` block, and it is
    /// part of the input contract rather than the architecture.
    pub padding: PaddingConfig,
}

/// Silence the model expects around the audio.
///
/// Both pads are zeros, and both matter: the left pad is documented as giving
/// the model more compute before the speech starts, and without the right pad
/// the last words are never transcribed, because the decoder needs positions
/// after the audio ends to emit them into.
#[derive(Clone, Copy, Debug)]
pub struct PaddingConfig {
    /// Positions of silence before the audio.
    pub left_tokens: usize,
    /// Slack for a long final word, past the delay and the opening token.
    pub buffer_tokens: usize,
}

impl Default for PaddingConfig {
    fn default() -> Self {
        // The published checkpoint's numbers.
        Self {
            left_tokens: 32,
            buffer_tokens: 10,
        }
    }
}

impl PaddingConfig {
    /// Read the `audio` block of a `tekken.json`.
    pub fn from_tekken(v: &serde_json::Value) -> Self {
        let d = Self::default();
        let audio = v.get("audio").unwrap_or(v);
        Self {
            left_tokens: audio
                .get("streaming_n_left_pad_tokens")
                .and_then(serde_json::Value::as_u64)
                .map_or(d.left_tokens, |n| n as usize),
            buffer_tokens: d.buffer_tokens,
        }
    }

    /// `(delay + 1) + buffer`: the induced delay, the opening token, and the
    /// long-word allowance.
    pub fn right_tokens(&self, num_delay_tokens: usize) -> usize {
        num_delay_tokens + 1 + self.buffer_tokens
    }
}

fn rope_theta_of(v: &serde_json::Value, default: f64) -> f64 {
    v.get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| v.get("rope_theta").and_then(serde_json::Value::as_f64))
        .unwrap_or(default)
}

impl VoxtralConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let need = |o: &serde_json::Value, key: &str| -> Result<usize> {
            o.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| candle_core::Error::Msg(format!("voxtral config: missing {key}")))
        };
        let a = v
            .get("audio_config")
            .ok_or_else(|| candle_core::Error::Msg("voxtral config: no audio_config".into()))?;
        let t = v
            .get("text_config")
            .ok_or_else(|| candle_core::Error::Msg("voxtral config: no text_config".into()))?;
        let eps = |o: &serde_json::Value| {
            o.get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5)
        };
        Ok(Self {
            audio: AudioConfig {
                hidden_size: need(a, "hidden_size")?,
                num_layers: need(a, "num_hidden_layers")?,
                num_heads: need(a, "num_attention_heads")?,
                head_dim: need(a, "head_dim")?,
                intermediate_size: need(a, "intermediate_size")?,
                num_mel_bins: need(a, "num_mel_bins")?,
                sliding_window: need(a, "sliding_window")?,
                rope_theta: rope_theta_of(a, 1e6),
                rms_norm_eps: eps(a),
                max_position_embeddings: need(a, "max_position_embeddings")?,
            },
            text: TextConfig {
                vocab_size: need(t, "vocab_size")?,
                hidden_size: need(t, "hidden_size")?,
                num_layers: need(t, "num_hidden_layers")?,
                num_heads: need(t, "num_attention_heads")?,
                num_kv_heads: need(t, "num_key_value_heads")?,
                head_dim: need(t, "head_dim")?,
                intermediate_size: need(t, "intermediate_size")?,
                sliding_window: need(t, "sliding_window")?,
                rope_theta: rope_theta_of(t, 1e6),
                rms_norm_eps: eps(t),
                max_position_embeddings: need(t, "max_position_embeddings")?,
            },
            audio_length_per_tok: need(v, "audio_length_per_tok")?,
            downsample_factor: need(v, "downsample_factor")?,
            default_num_delay_tokens: need(v, "default_num_delay_tokens")?,
            padding: PaddingConfig::default(),
        })
    }

    /// Mel frames consumed by one decoder position.
    pub fn mel_per_position(&self) -> usize {
        self.audio_length_per_tok
    }

    /// Audio samples one decoder position covers, at the frontend's hop.
    pub fn samples_per_position(&self, hop_length: usize) -> usize {
        self.audio_length_per_tok * hop_length
    }

    /// The prompt for a transcription: an opening token, then one pad per
    /// left-pad and delay position.
    pub fn prompt(&self, bos: u32, streaming_pad: u32, num_delay_tokens: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(1 + self.padding.left_tokens + num_delay_tokens);
        out.push(bos);
        out.resize(
            1 + self.padding.left_tokens + num_delay_tokens,
            streaming_pad,
        );
        out
    }

    /// Surround the audio with the silence the model expects, and round it up
    /// to a whole number of 80 ms positions — a partial position has no
    /// decoder step to land on.
    pub fn pad_audio(
        &self,
        samples: &[f32],
        hop_length: usize,
        num_delay_tokens: usize,
    ) -> Vec<f32> {
        let per = self.samples_per_position(hop_length);
        let whole = samples.len().div_ceil(per) * per;
        let left = self.padding.left_tokens * per;
        let right = self.padding.right_tokens(num_delay_tokens) * per;
        let mut out = vec![0f32; left + whole + right];
        out[left..left + samples.len()].copy_from_slice(samples);
        out
    }
}

// ----------------------------------------------------------------- rotary

/// Rotary frequencies, with the angles computed per forward rather than
/// tabulated.
///
/// A precomputed table has to be sized to something, and sizing it to
/// `max_position_embeddings` caps the encoder at 1500 frames — thirty
/// seconds of audio, which a meeting exceeds in its first minute. The
/// reference computes from `position_ids` for the same reason; the cost is
/// one small matmul per forward.
struct Rotary {
    inv_freq: Tensor,
    dtype: DType,
}

/// `cos` and `sin` for one contiguous span of positions.
struct RopeSlice {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f64, dtype: DType, dev: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        Ok(Self {
            // f32 angles: at theta 1e6 the low bits matter, and computing
            // them in bf16 moves a token by a fraction of a position.
            inv_freq: Tensor::from_vec(inv, (1, half), dev)?,
            dtype,
        })
    }

    fn slice(&self, offset: usize, len: usize) -> Result<RopeSlice> {
        let pos = Tensor::arange(offset as u32, (offset + len) as u32, self.inv_freq.device())?
            .to_dtype(DType::F32)?
            .reshape((len, 1))?;
        let freqs = pos.matmul(&self.inv_freq)?;
        Ok(RopeSlice {
            cos: freqs.cos()?.to_dtype(self.dtype)?,
            sin: freqs.sin()?.to_dtype(self.dtype)?,
        })
    }
}

impl RopeSlice {
    /// `x` is `(tokens, heads, head_dim)`.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_thd(&x.unsqueeze(0)?.contiguous()?, &self.cos, &self.sin)?
            .squeeze(0)
    }
}

// -------------------------------------------------------------- attention

/// Additive mask for `(queries, keys)`, causal and optionally windowed.
///
/// `offset` is where the queries start, so a single decoding step attends to
/// everything already cached.
fn causal_mask(
    queries: usize,
    keys: usize,
    offset: usize,
    window: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; queries * keys];
    for q in 0..queries {
        let pos = offset + q;
        for k in 0..keys {
            let too_late = k > pos;
            let too_old = window.is_some_and(|w| pos.saturating_sub(k) >= w);
            if too_late || too_old {
                data[q * keys + k] = MASK_PENALTY;
            }
        }
    }
    Tensor::from_vec(data, (1, queries, keys), dev)?.to_dtype(dtype)
}

/// Grouped-query attention needs each KV head repeated across its group.
fn repeat_kv(x: &Tensor, times: usize) -> Result<Tensor> {
    if times == 1 {
        return Ok(x.clone());
    }
    let (heads, tokens, dim) = x.dims3()?;
    x.unsqueeze(1)?
        .expand((heads, times, tokens, dim))?
        .reshape((heads * times, tokens, dim))
}

/// `q`, `k`, `v` are `(tokens, heads, head_dim)`. Returns `(tokens, heads *
/// head_dim)`.
fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    let (tq, hq, d) = q.dims3()?;
    let (tk, hk, _) = k.dims3()?;
    let scale = 1f32 / (d as f32).sqrt();

    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && matches!(q.dtype(), DType::BF16 | DType::F16) {
        // FlashAttention's causal mask is bottom-right aligned, which is what
        // a single decoding step against a full cache needs: the one query is
        // the newest position, not the oldest.
        let out = candle_flash_attn::flash_attn_windowed(
            &q.unsqueeze(0)?,
            &k.unsqueeze(0)?,
            &v.unsqueeze(0)?,
            scale,
            window.map(|w| w - 1),
            Some(0),
        )?;
        return out.squeeze(0)?.reshape((tq, hq * d));
    }

    // Reference: (heads, tokens, dim) with an explicit mask.
    let qh = q.transpose(0, 1)?.contiguous()?;
    let kh = repeat_kv(&k.transpose(0, 1)?.contiguous()?, hq / hk)?;
    let vh = repeat_kv(&v.transpose(0, 1)?.contiguous()?, hq / hk)?;
    let att = (qh.matmul(&kh.transpose(D::Minus2, D::Minus1)?)? * scale as f64)?;
    let mask = causal_mask(tq, tk, offset, window, q.dtype(), q.device())?;
    let att = softmax_last_dim(&att.broadcast_add(&mask)?)?;
    att.matmul(&vh)?.transpose(0, 1)?.reshape((tq, hq * d))
}

// ------------------------------------------------------------------ audio

struct AudioAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let inner = cfg.num_heads * cfg.head_dim;
        Ok(Self {
            // Whisper's convention, and easy to get wrong: q, v and o carry a
            // bias, k does not.
            q: linear_bias(d, inner, vb.pp("q_proj"))?,
            k: linear_no_bias(d, inner, vb.pp("k_proj"))?,
            v: linear_bias(d, inner, vb.pp("v_proj"))?,
            o: linear_bias(inner, d, vb.pp("o_proj"))?,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeSlice, window: usize) -> Result<Tensor> {
        let t = x.dim(0)?;
        let shape = |p: Tensor| p.reshape((t, self.heads, self.head_dim));
        let q = rope.apply(&shape(self.q.forward(x)?)?)?;
        let k = rope.apply(&shape(self.k.forward(x)?)?)?;
        let v = shape(self.v.forward(x)?)?.contiguous()?;
        self.o.forward(&attend(&q, &k, &v, 0, Some(window))?)
    }
}

/// Gated SiLU feedforward. The audio encoder biases `down_proj` and nothing
/// else; the text decoder biases nothing.
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize, down_bias: bool) -> Result<Self> {
        let down = if down_bias {
            linear_bias(intermediate, hidden, vb.pp("down_proj"))?
        } else {
            linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?
        };
        Ok(Self {
            gate: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(self.gate.forward(x)?.silu()? * self.up.forward(x)?)?)
    }
}

struct AudioLayer {
    attn_norm: RmsNorm,
    attn: AudioAttention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl AudioLayer {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            attn: AudioAttention::load(vb.pp("self_attn"), cfg)?,
            mlp_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("final_layer_norm"))?,
            mlp: Mlp::load(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size, true)?,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeSlice, window: usize) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&self.attn_norm.forward(x)?, rope, window)?)?;
        &h + self.mlp.forward(&self.mlp_norm.forward(&h)?)?
    }
}

/// The convolutional stem: 128 mel bins to the encoder width, halving time.
///
/// Both convolutions are **causal** — padded on the left only — which is what
/// lets the encoder run on a stream without seeing the future.
struct Embedder {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl Embedder {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        Ok(Self {
            conv1: candle_nn::conv1d(
                cfg.num_mel_bins,
                cfg.hidden_size,
                3,
                Conv1dConfig::default(),
                vb.pp("conv1"),
            )?,
            conv2: candle_nn::conv1d(
                cfg.hidden_size,
                cfg.hidden_size,
                3,
                Conv1dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv2"),
            )?,
        })
    }

    /// `mel` is `(1, mel_bins, frames)`; returns `(frames / 2, hidden)`.
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // `left_pad = (kernel - 1) * dilation + 1 - stride`: 2 for conv1,
        // 1 for conv2.
        let x = left_pad(mel, 2)?;
        let x = self.conv1.forward(&x)?.gelu_erf()?;
        let x = left_pad(&x, 1)?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        // (1, hidden, frames) -> (frames, hidden)
        x.squeeze(0)?.transpose(0, 1)?.contiguous()
    }
}

fn left_pad(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let (b, c, _) = x.dims3()?;
    let zeros = Tensor::zeros((b, c, pad), x.dtype(), x.device())?;
    Tensor::cat(&[&zeros, x], 2)
}

pub struct AudioEncoder {
    embedder: Embedder,
    layers: Vec<AudioLayer>,
    norm: RmsNorm,
    rotary: Rotary,
    window: usize,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &AudioConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(AudioLayer::load(vb.pp(format!("layers.{i}")), cfg)?);
        }
        Ok(Self {
            embedder: Embedder::load(vb.pp("embedder"), cfg)?,
            layers,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, dtype, dev)?,
            window: cfg.sliding_window,
        })
    }

    /// `mel` is `(1, mel_bins, frames)`; returns `(frames / 2, hidden)`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.embedder.forward(mel)?;
        // Once per forward, not once per layer.
        let rope = self.rotary.slice(0, x.dim(0)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, &rope, self.window)?;
        }
        self.norm.forward(&x)
    }
}

/// Four encoder frames grouped into one decoder position.
struct Projector {
    linear_1: Linear,
    linear_2: Linear,
    group: usize,
    audio_hidden: usize,
}

impl Projector {
    fn load(vb: VarBuilder, cfg: &VoxtralConfig) -> Result<Self> {
        Ok(Self {
            linear_1: linear_no_bias(
                cfg.audio.hidden_size * cfg.downsample_factor,
                cfg.text.hidden_size,
                vb.pp("linear_1"),
            )?,
            linear_2: linear_no_bias(
                cfg.text.hidden_size,
                cfg.text.hidden_size,
                vb.pp("linear_2"),
            )?,
            group: cfg.downsample_factor,
            audio_hidden: cfg.audio.hidden_size,
        })
    }

    /// `(frames, audio_hidden)` to `(frames / group, text_hidden)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let frames = x.dim(0)?;
        if !frames.is_multiple_of(self.group) {
            candle_core::bail!(
                "{frames} encoder frames do not group into {}s; the audio must be a whole \
                 number of positions",
                self.group
            );
        }
        let grouped = x.reshape((frames / self.group, self.audio_hidden * self.group))?;
        self.linear_2
            .forward(&self.linear_1.forward(&grouped)?.gelu_erf()?)
    }
}

// ------------------------------------------------------------------- text

/// The per-layer scales the delay conditioning produces.
///
/// Computed once per session: `t_cond` depends only on the delay, so nothing
/// here varies by position or step.
pub struct Conditioning {
    scales: Vec<Tensor>,
}

struct AdaRmsNorm {
    linear1: Linear,
    linear2: Linear,
}

impl AdaRmsNorm {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear_no_bias(hidden, 32, vb.pp("linear1"))?,
            linear2: linear_no_bias(32, hidden, vb.pp("linear2"))?,
        })
    }

    fn scale(&self, t_cond: &Tensor) -> Result<Tensor> {
        self.linear2
            .forward(&self.linear1.forward(t_cond)?.gelu_erf()?)
    }
}

struct TextAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl TextAttention {
    fn load(vb: VarBuilder, cfg: &TextConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            q: linear_no_bias(d, cfg.num_heads * cfg.head_dim, vb.pp("q_proj"))?,
            k: linear_no_bias(d, cfg.num_kv_heads * cfg.head_dim, vb.pp("k_proj"))?,
            v: linear_no_bias(d, cfg.num_kv_heads * cfg.head_dim, vb.pp("v_proj"))?,
            o: linear_no_bias(cfg.num_heads * cfg.head_dim, d, vb.pp("o_proj"))?,
            heads: cfg.num_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeSlice,
        cache: &mut LayerCache,
        window: usize,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let offset = cache.len();
        let q = rope.apply(&self.q.forward(x)?.reshape((t, self.heads, self.head_dim))?)?;
        let k = rope.apply(
            &self
                .k
                .forward(x)?
                .reshape((t, self.kv_heads, self.head_dim))?,
        )?;
        let v = self
            .v
            .forward(x)?
            .reshape((t, self.kv_heads, self.head_dim))?
            .contiguous()?;

        let (k, v) = cache.push(&k, &v, window)?;
        self.o
            .forward(&attend(&q, &k, &v, offset - cache.dropped(), Some(window))?)
    }
}

struct TextLayer {
    input_norm: RmsNorm,
    attn: TextAttention,
    post_norm: RmsNorm,
    ada: AdaRmsNorm,
    mlp: Mlp,
}

impl TextLayer {
    fn load(vb: VarBuilder, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            input_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: TextAttention::load(vb.pp("self_attn"), cfg)?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            ada: AdaRmsNorm::load(vb.pp("ada_rms_norm"), cfg.hidden_size)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size, false)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeSlice,
        cache: &mut LayerCache,
        window: usize,
        scale: &Tensor,
    ) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&self.input_norm.forward(x)?, rope, cache, window)?)?;
        // The residual is taken before the norm, and the conditioning is
        // applied between the norm and the feedforward — not around it.
        let normed = self.post_norm.forward(&h)?;
        let modulated = normed.broadcast_mul(&(scale + 1.0)?)?;
        &h + self.mlp.forward(&modulated)?
    }
}

/// One layer's K/V, kept contiguous and trimmed to the sliding window.
#[derive(Default)]
struct LayerCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// Positions evicted from the front, so absolute positions stay right.
    dropped: usize,
}

impl LayerCache {
    fn len(&self) -> usize {
        match &self.k {
            Some(k) => k.dim(0).unwrap_or(0) + self.dropped,
            None => 0,
        }
    }

    fn dropped(&self) -> usize {
        self.dropped
    }

    fn push(&mut self, k: &Tensor, v: &Tensor, window: usize) -> Result<(Tensor, Tensor)> {
        let (mut nk, mut nv) = match (&self.k, &self.v) {
            (Some(pk), Some(pv)) => (Tensor::cat(&[pk, k], 0)?, Tensor::cat(&[pv, v], 0)?),
            _ => (k.clone(), v.clone()),
        };
        // Past the window the oldest positions can never be attended to
        // again, so holding them would grow without bound over a long
        // recording.
        let len = nk.dim(0)?;
        if len > window {
            let cut = len - window;
            nk = nk.narrow(0, cut, window)?.contiguous()?;
            nv = nv.narrow(0, cut, window)?.contiguous()?;
            self.dropped += cut;
        }
        self.k = Some(nk.clone());
        self.v = Some(nv.clone());
        Ok((nk, nv))
    }
}

pub struct TextDecoder {
    embed: Embedding,
    layers: Vec<TextLayer>,
    norm: RmsNorm,
    rotary: Rotary,
    window: usize,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &TextConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(TextLayer::load(vb.pp(format!("layers.{i}")), cfg)?);
        }
        Ok(Self {
            embed: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?,
            layers,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, dtype, dev)?,
            window: cfg.sliding_window,
        })
    }
}

/// Everything a running transcription carries between steps.
#[derive(Default)]
pub struct Session {
    caches: Vec<LayerCache>,
}

impl Session {
    fn new(layers: usize) -> Self {
        Self {
            caches: (0..layers).map(|_| LayerCache::default()).collect(),
        }
    }

    pub fn positions(&self) -> usize {
        self.caches.first().map_or(0, LayerCache::len)
    }
}

// ------------------------------------------------------------------ model

pub struct Voxtral {
    audio: AudioEncoder,
    projector: Projector,
    decoder: TextDecoder,
    pub config: VoxtralConfig,
    device: Device,
    dtype: DType,
}

impl Voxtral {
    pub fn load(vb: VarBuilder, config: VoxtralConfig) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        Ok(Self {
            audio: AudioEncoder::load(vb.pp("audio_tower"), &config.audio, dtype, &device)?,
            projector: Projector::load(vb.pp("multi_modal_projector"), &config)?,
            decoder: TextDecoder::load(
                vb.pp("language_model.model"),
                &config.text,
                dtype,
                &device,
            )?,
            config,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Sinusoidal embedding of the delay, and the per-layer scales it implies.
    ///
    /// Parameter-free by design — there is no `time_embedding` tensor in the
    /// checkpoint.
    pub fn conditioning(&self, num_delay_tokens: usize) -> Result<Conditioning> {
        let dim = self.config.text.hidden_size;
        let half = dim / 2;
        let theta = 10_000f64;
        let t = num_delay_tokens as f64;
        let mut values = Vec::with_capacity(dim);
        let freqs: Vec<f64> = (0..half)
            .map(|i| (-theta.ln() * i as f64 / half as f64).exp() * t)
            .collect();
        values.extend(freqs.iter().map(|f| f.cos() as f32));
        values.extend(freqs.iter().map(|f| f.sin() as f32));
        let t_cond = Tensor::from_vec(values, (1, dim), &self.device)?.to_dtype(self.dtype)?;

        let mut scales = Vec::with_capacity(self.decoder.layers.len());
        for layer in &self.decoder.layers {
            scales.push(layer.ada.scale(&t_cond)?.reshape((dim,))?);
        }
        Ok(Conditioning { scales })
    }

    /// The sinusoidal conditioning vector itself, for parity checks.
    pub fn time_embedding(&self, num_delay_tokens: usize) -> Result<Vec<f32>> {
        let dim = self.config.text.hidden_size;
        let half = dim / 2;
        let t = num_delay_tokens as f64;
        let freqs: Vec<f64> = (0..half)
            .map(|i| (-10_000f64.ln() * i as f64 / half as f64).exp() * t)
            .collect();
        Ok(freqs
            .iter()
            .map(|f| f.cos() as f32)
            .chain(freqs.iter().map(|f| f.sin() as f32))
            .collect())
    }

    /// Log-mel frames, `[bins][frames]`, to one embedding per 80 ms.
    pub fn audio_embeds(&self, mel: &[Vec<f32>]) -> Result<Tensor> {
        let bins = mel.len();
        if bins != self.config.audio.num_mel_bins {
            candle_core::bail!(
                "{bins} mel bins, expected {}",
                self.config.audio.num_mel_bins
            );
        }
        let frames = mel.first().map_or(0, Vec::len);
        let per = self.config.mel_per_position();
        if !frames.is_multiple_of(per) {
            candle_core::bail!(
                "{frames} mel frames is not a whole number of {per}-frame positions"
            );
        }
        let flat: Vec<f32> = mel.iter().flatten().copied().collect();
        let mel = Tensor::from_vec(flat, (1, bins, frames), &self.device)?.to_dtype(self.dtype)?;
        let hidden = self.audio.forward(&mel)?;
        self.projector.forward(&hidden)
    }

    /// The encoder's output before the projector, for parity checks.
    pub fn encoder_hidden(&self, mel: &[Vec<f32>]) -> Result<Tensor> {
        let bins = mel.len();
        let frames = mel.first().map_or(0, Vec::len);
        let flat: Vec<f32> = mel.iter().flatten().copied().collect();
        let mel = Tensor::from_vec(flat, (1, bins, frames), &self.device)?.to_dtype(self.dtype)?;
        self.audio.forward(&mel)
    }

    pub fn session(&self) -> Session {
        Session::new(self.decoder.layers.len())
    }

    /// One forward over `tokens`, each position's audio embedding added to
    /// its token embedding. Returns logits for every position.
    pub fn step(
        &self,
        session: &mut Session,
        tokens: &[u32],
        audio: &Tensor,
        cond: &Conditioning,
    ) -> Result<Tensor> {
        let n = tokens.len();
        if audio.dim(0)? != n {
            candle_core::bail!("{n} tokens against {} audio positions", audio.dim(0)?);
        }
        let ids = Tensor::from_vec(tokens.to_vec(), (n,), &self.device)?;
        // The merge: addition at the same position, not interleaving.
        let mut x = (self.decoder.embed.forward(&ids)? + audio.to_dtype(self.dtype)?)?;
        let rope = self.decoder.rotary.slice(session.positions(), n)?;
        for (i, layer) in self.decoder.layers.iter().enumerate() {
            x = layer.forward(
                &x,
                &rope,
                &mut session.caches[i],
                self.decoder.window,
                &cond.scales[i],
            )?;
        }
        let x = self.decoder.norm.forward(&x)?;
        // Tied embeddings: the vocabulary projection is the embedding matrix.
        x.matmul(&self.decoder.embed.embeddings().t()?)?
            .to_dtype(DType::F32)
    }

    /// Transcribe a whole clip, greedily.
    ///
    /// `prompt` is the token sequence the processor builds — a `<s>` and then
    /// one `[STREAMING_PAD]` per left-pad and delay token — and it already
    /// covers the first `prompt.len()` audio positions. From there the loop is
    /// lock-step: one token in, one token out, one 80 ms frame consumed, until
    /// the audio runs out.
    pub fn transcribe(
        &self,
        mel: &[Vec<f32>],
        prompt: &[u32],
        num_delay_tokens: usize,
    ) -> Result<Vec<u32>> {
        let audio = self.audio_embeds(mel)?;
        let positions = audio.dim(0)?;
        if positions < prompt.len() {
            candle_core::bail!(
                "{positions} audio positions is shorter than the {}-token prompt",
                prompt.len()
            );
        }
        let cond = self.conditioning(num_delay_tokens)?;
        let mut session = self.session();
        let mut tokens = prompt.to_vec();

        let logits = self.step(
            &mut session,
            prompt,
            &audio.narrow(0, 0, prompt.len())?,
            &cond,
        )?;
        tokens.push(argmax(&logits.i(prompt.len() - 1)?)?);

        for p in prompt.len()..positions.saturating_sub(1) {
            let logits = self.step(&mut session, &tokens[p..=p], &audio.narrow(0, p, 1)?, &cond)?;
            tokens.push(argmax(&logits.i(0)?)?);
        }
        Ok(tokens)
    }
}

fn argmax(row: &Tensor) -> Result<u32> {
    row.argmax(D::Minus1)?.to_scalar::<u32>()
}
