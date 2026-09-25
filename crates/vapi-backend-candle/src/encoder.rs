//! The single-pass backend: a decision checkpoint behind
//! [`EncoderBackend`](vapi_engine::EncoderBackend).
//!
//! Short, because there is no cache to manage and no state between calls. The
//! only real decision here is padding: rows in a batch are padded to the
//! longest one, and the mask keeps the padding out of every attention. That
//! makes a batch cost `rows × longest` positions of work, which is why the
//! engine that feeds this sorts by length before it fills a batch.

use std::path::Path;

use candle_core::Device;
use vapi_core::config::{DType, DeviceKind};
use vapi_core::{Error, Result};
use vapi_engine::{EncoderBackend, EncoderBatch, EncoderSpec, MarkerLogits, RowScores};

use crate::backend::{engine_err, pick_device, pick_dtype, read_json, weight_files};
use crate::models::laya::Laya;
use crate::models::modernbert::Config as EncoderConfig;

/// How to load a decision model.
#[derive(Clone, Debug)]
pub struct EncoderLoadOptions {
    pub dtype: DType,
    pub device: DeviceKind,
}

impl Default for EncoderLoadOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Auto,
            device: DeviceKind::Auto,
        }
    }
}

pub struct CandleEncoder {
    model: Laya,
    spec: EncoderSpec,
    device: Device,
}

impl CandleEncoder {
    /// Whether this directory holds a decision checkpoint rather than a
    /// decoder.
    ///
    /// Checked by layout rather than by name: the encoder's config in its own
    /// subdirectory beside a head config is what distinguishes one, and the
    /// three published checkpoints all differ in model id.
    pub fn looks_like_decision_model(dir: &Path) -> bool {
        dir.join("rl_agent_config.json").exists() && dir.join("encoder/config.json").exists()
    }

    pub fn load(dir: impl AsRef<Path>, opts: &EncoderLoadOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let encoder_json = read_json(&dir.join("encoder/config.json"))?.ok_or_else(|| {
            Error::Config(format!(
                "{}/encoder/config.json not found; this is not a decision checkpoint",
                dir.display()
            ))
        })?;
        let arch = encoder_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if arch != "modernbert" {
            return Err(Error::Config(format!(
                "encoder model_type {arch:?} is not supported; modernbert is"
            )));
        }
        let cfg = EncoderConfig::from_json(&encoder_json).map_err(engine_err)?;

        let head_json = read_json(&dir.join("rl_agent_config.json"))?.unwrap_or_default();
        let head_layers = head_json
            .get("head_layers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2) as usize;
        let decision = vapi_core::DecisionConfig::from_json(&head_json.to_string())?;

        let device = pick_device(opts.device)?;
        let dtype = pick_dtype(opts.dtype, &device);
        let files = weight_files(dir)?;
        // Safety: as elsewhere, the mmap is read-only and nothing rewrites
        // the checkpoint while the process runs.
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(engine_err)?;

        let spec = EncoderSpec {
            hidden_size: cfg.hidden_size,
            // The checkpoint's own budget, not the encoder's architectural
            // limit: the scorer was fitted against sequences of this length,
            // and a longer one is outside what was calibrated.
            max_context: decision.max_len.min(cfg.max_position_embeddings),
            head_context: decision.head_max_len,
            pad_token_id: cfg.pad_token_id,
        };
        let model = Laya::load(vb, cfg, head_layers).map_err(engine_err)?;

        tracing::info!(
            dir = %dir.display(),
            ?dtype,
            hidden = spec.hidden_size,
            max_context = spec.max_context,
            head_layers,
            "decision model loaded"
        );
        Ok(Self {
            model,
            spec,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run the reference attention instead of the kernel. See
    /// [`crate::models::modernbert::ModernBert::use_reference_attention`].
    pub fn use_reference_attention(&mut self, yes: bool) {
        self.model.use_reference_attention(yes);
    }

    /// The encoder's hidden states for one sequence, for parity debugging.
    pub fn encoder_hidden(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.model.encoder_hidden(tokens).map_err(engine_err)
    }
}

impl EncoderBackend for CandleEncoder {
    fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    fn forward(&mut self, batch: &EncoderBatch) -> Result<MarkerLogits> {
        batch.validate(self.spec.max_context)?;
        if batch.is_empty() {
            return Ok(MarkerLogits::default());
        }
        // Flat and unpadded: rows end to end, with their lengths alongside.
        // Nothing here pads to the longest row, so a batch costs what its
        // rows actually are.
        let mut tokens = Vec::with_capacity(batch.num_tokens());
        let mut lengths = Vec::with_capacity(batch.num_rows());
        let mut markers = Vec::with_capacity(batch.num_rows());
        let mut qtypes = Vec::with_capacity(batch.num_rows());
        for row in &batch.rows {
            tokens.extend_from_slice(&row.tokens);
            lengths.push(row.tokens.len());
            markers.push(row.markers.clone());
            qtypes.push(row.qtype.index());
        }

        let out = self
            .model
            .forward(&tokens, &lengths, &markers, &qtypes)
            .map_err(engine_err)?;

        Ok(MarkerLogits {
            rows: out
                .into_iter()
                .map(|r| RowScores {
                    logits: r.logits,
                    act: r.act,
                })
                .collect(),
        })
    }
}

// ------------------------------------------------------------------ speech

/// A loaded Voxtral, with its frontend settings.
///
/// Not an [`EncoderBackend`] and not an [`ExecutionBackend`]: it is a decoder
/// with a paged-cache-shaped future, but the engine integration — audio
/// arriving over time, embeddings as a batch input — is phase 9b. This is the
/// model, loadable and runnable on its own, which is what the parity tests
/// and the benchmark need.
pub struct CandleVoxtral {
    pub model: crate::models::voxtral::Voxtral,
    pub mel: vapi_audio::MelSettings,
    tokenizer: vapi_tokenize::Tekken,
}

impl CandleVoxtral {
    /// Whether this directory holds a Voxtral checkpoint.
    pub fn looks_like_speech_model(dir: &Path) -> bool {
        read_json(&dir.join("config.json"))
            .ok()
            .flatten()
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|m| m.as_str())
                    .map(|m| m == "voxtral_realtime")
            })
            .unwrap_or(false)
    }

    pub fn load(dir: impl AsRef<Path>, opts: &EncoderLoadOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let config_json = read_json(&dir.join("config.json"))?
            .ok_or_else(|| Error::Config(format!("{}/config.json not found", dir.display())))?;
        let arch = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if arch != "voxtral_realtime" {
            return Err(Error::Config(format!(
                "model_type {arch:?} is not a speech model; voxtral_realtime is"
            )));
        }
        let mut config =
            crate::models::voxtral::VoxtralConfig::from_json(&config_json).map_err(engine_err)?;
        // The padding is part of the input contract and lives with the
        // tokenizer, not the architecture.
        if let Some(tekken) = read_json(&dir.join("tekken.json"))? {
            config.padding = crate::models::voxtral::PaddingConfig::from_tekken(&tekken);
        }

        let mel = match std::fs::read_to_string(dir.join("processor_config.json")) {
            Ok(text) => vapi_audio::MelSettings::from_json(&text)?,
            // The defaults are this checkpoint's own numbers; say so rather
            // than failing, since a hand-assembled directory is a fair thing
            // to point at.
            Err(_) => {
                tracing::warn!("no processor_config.json; using the default mel settings");
                vapi_audio::MelSettings::default()
            }
        };

        let device = pick_device(opts.device)?;
        let dtype = pick_dtype(opts.dtype, &device);
        let files = weight_files(dir)?;
        // Safety: read-only mmap, as elsewhere in this crate.
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(engine_err)?;
        let model = crate::models::voxtral::Voxtral::load(vb, config).map_err(engine_err)?;
        // The tokenizer comes along because the prompt is built out of
        // control tokens and the answer has to be turned back into text;
        // splitting those across crates would mean passing four ids around.
        let tokenizer = vapi_tokenize::Tekken::from_file(dir.join("tekken.json"))?;

        tracing::info!(
            dir = %dir.display(),
            ?dtype,
            audio_layers = model.config.audio.num_layers,
            text_layers = model.config.text.num_layers,
            "speech model loaded"
        );
        Ok(Self {
            model,
            mel,
            tokenizer,
        })
    }

    /// `(<s>, [STREAMING_PAD])`, the two the prompt is built from.
    pub fn control_tokens(&self) -> Result<(u32, u32)> {
        let need = |name: &str| {
            self.tokenizer.special_id(name).ok_or_else(|| {
                Error::Tokenizer(format!("this tokenizer has no {name}; it is not Voxtral's"))
            })
        };
        Ok((need("<s>")?, need("[STREAMING_PAD]")?))
    }

    /// Token ids to text, dropping the control tokens — the padding the model
    /// emits while it has nothing to say is not part of the transcript.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true)
    }

    /// Transcribe 16 kHz mono samples, returning the generated token ids
    /// including the prompt.
    ///
    /// `bos` and `streaming_pad` are the tokenizer's `<s>` and
    /// `[STREAMING_PAD]`; the prompt and the surrounding silence are built
    /// from them here, because getting either wrong changes what the model
    /// reads without changing anything that errors.
    pub fn transcribe(&self, samples: &[f32], bos: u32, streaming_pad: u32) -> Result<Vec<u32>> {
        let cfg = &self.model.config;
        let delay = cfg.default_num_delay_tokens;
        let padded = cfg.pad_audio(samples, self.mel.hop_length, delay);
        let mel = vapi_audio::MelSpectrogram::new(self.mel).compute(&padded);
        let prompt = cfg.prompt(bos, streaming_pad, delay);
        self.model
            .transcribe(&mel, &prompt, delay)
            .map_err(engine_err)
    }

    /// How many tokens of `transcribe`'s output are prompt rather than
    /// transcript.
    pub fn prompt_len(&self) -> usize {
        1 + self.model.config.padding.left_tokens + self.model.config.default_num_delay_tokens
    }
}
