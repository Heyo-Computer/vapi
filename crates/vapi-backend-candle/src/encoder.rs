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
