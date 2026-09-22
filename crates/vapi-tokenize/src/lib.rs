//! Chat templates and tokenization.
//!
//! Lives on the gateway rather than the worker: rendering and tokenizing there
//! keeps the tokenizer off the engine's hot path, lets an over-length prompt
//! be rejected with a 400 before any queue work happens, and makes a queued
//! job self-describing — the worker never has to reconstruct what the client
//! asked for.

pub mod bytes;
pub mod template;

pub use bytes::{BYTE_EOS, ByteTokenizer};
pub use template::{ChatTemplate, TemplateSource};

use std::path::Path;

use tokenizers::Tokenizer;
use vapi_core::{Error, Result};

/// A tokenizer plus the model's chat template and special tokens.
pub struct TokenizerBundle {
    tokenizer: Tokenizer,
    template: Option<ChatTemplate>,
    pub eos_token_ids: Vec<u32>,
    /// The BOS *string* (e.g. `<|begin_of_text|>`), handed to templates that
    /// emit it themselves.
    pub bos_token: Option<String>,
    /// The EOS string, likewise; some templates close assistant turns with it.
    pub eos_token: Option<String>,
}

impl TokenizerBundle {
    /// Load from a directory holding `tokenizer.json`, and optionally
    /// `tokenizer_config.json` and `generation_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            // A repo shipping only a sentencepiece `tokenizer.model` cannot be
            // loaded by the `tokenizers` crate. Say so plainly rather than
            // failing somewhere deeper with a confusing error.
            return Err(Error::Tokenizer(format!(
                "{} not found. If this repo ships only tokenizer.model (sentencepiece), \
                 convert it to tokenizer.json first",
                tok_path.display()
            )));
        }
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(e.to_string()))?;

        let config: serde_json::Value =
            read_json(dir.join("tokenizer_config.json"))?.unwrap_or_default();
        let gen_config: serde_json::Value =
            read_json(dir.join("generation_config.json"))?.unwrap_or_default();

        // Newer repos ship the template as `chat_template.jinja` next to the
        // tokenizer instead of inside tokenizer_config.json; HF prefers the
        // file when both exist.
        let template = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(text) => Some(ChatTemplate::new(text, TemplateSource::TemplateFile)?),
            Err(_) => ChatTemplate::from_tokenizer_config(&config)?,
        };
        let bos_token = config.get("bos_token").and_then(token_str);
        let eos_token = config.get("eos_token").and_then(token_str);

        let mut eos_token_ids = collect_eos(&gen_config, &tokenizer);
        if eos_token_ids.is_empty() {
            eos_token_ids = collect_eos(&config, &tokenizer);
        }

        Ok(Self {
            tokenizer,
            template,
            eos_token_ids,
            bos_token,
            eos_token,
        })
    }

    /// Build from an already-loaded tokenizer, for tests.
    pub fn from_parts(
        tokenizer: Tokenizer,
        template: Option<ChatTemplate>,
        eos_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            tokenizer,
            template,
            eos_token_ids,
            bos_token: None,
            eos_token: None,
        }
    }

    /// Set the special-token strings a template may interpolate.
    pub fn with_special_tokens(mut self, bos: Option<String>, eos: Option<String>) -> Self {
        self.bos_token = bos;
        self.eos_token = eos;
        self
    }

    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn template(&self) -> Option<&ChatTemplate> {
        self.template.as_ref()
    }

    /// Tokenize raw text, as `/v1/completions` does.
    ///
    /// `add_special_tokens` is the caller's decision because it differs by
    /// path: a chat-templated prompt already contains BOS, so adding another
    /// is a classic silent quality bug.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode_fast(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decoder closure for [`vapi_engine::IncrementalDetokenizer`].
    pub fn decoder(&self) -> impl Fn(&[u32]) -> String + '_ {
        move |ids: &[u32]| self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    /// Render a conversation through the chat template and tokenize it.
    pub fn encode_chat(
        &self,
        messages: &[vapi_openai::ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| Error::ChatTemplate("model has no chat template".into()))?;
        // The template decides where BOS goes (Llama 3 opens with
        // `{{- bos_token }}`; ChatML never mentions it), so it gets the real
        // string and the tokenizer must not add its own — that would double it.
        let rendered = template.render_with(
            messages,
            true,
            self.bos_token.as_deref().unwrap_or(""),
            self.eos_token.as_deref().unwrap_or(""),
            tools,
        )?;
        self.encode(&rendered, false)
    }

    /// The id of a single token by its text, if the vocabulary has it.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}

fn read_json(path: std::path::PathBuf) -> Result<Option<serde_json::Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))
}

/// `eos_token` may be a string, and `eos_token_id` may be a scalar *or a
/// list* — newer instruct models commonly stop on several tokens, and reading
/// only the first is why a model appears to ramble past its turn.
fn collect_eos(config: &serde_json::Value, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut out = Vec::new();
    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            if let Some(v) = n.as_u64() {
                out.push(v as u32);
            }
        }
        Some(serde_json::Value::Array(a)) => {
            out.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32));
        }
        _ => {}
    }
    if let Some(tok) = config.get("eos_token").and_then(token_str)
        && let Some(id) = tokenizer.token_to_id(&tok)
    {
        out.push(id);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A special token is either a bare string or `{"content": "..."}`.
fn token_str(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// The tokenizer a gateway or worker is running with.
///
/// `Bytes` exists so the full request path works with no model present; `Hf`
/// is the real thing.
pub enum Tokenization {
    Hf(Box<TokenizerBundle>),
    Bytes(Box<ByteTokenizer>),
}

impl Tokenization {
    pub fn bytes() -> Result<Self> {
        Ok(Self::Bytes(Box::new(ByteTokenizer::new()?)))
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::Hf(Box::new(TokenizerBundle::from_dir(dir)?)))
    }

    pub fn encode_chat(
        &self,
        messages: &[vapi_openai::ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Hf(b) => b.encode_chat(messages, tools),
            Self::Bytes(b) => b.encode_chat(messages),
        }
    }

    /// Whether the vocabulary has this exact token; how the gateway learns
    /// which output markers a model uses.
    pub fn has_token(&self, token: &str) -> bool {
        self.token_id(token).is_some()
    }

    /// The id of a single token by its text, when the vocabulary has one.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Hf(b) => b.token_id(token),
            Self::Bytes(_) => None,
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            // Raw completions get the model's special tokens, matching what
            // `/v1/completions` does elsewhere.
            Self::Hf(b) => b.encode(text, true),
            Self::Bytes(b) => Ok(b.encode(text)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::Hf(b) => b.decode(ids, true).unwrap_or_default(),
            Self::Bytes(b) => b.decode(ids),
        }
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Hf(b) => b.eos_token_ids.clone(),
            Self::Bytes(b) => b.eos_token_ids(),
        }
    }

    pub fn vocab_hint(&self) -> usize {
        match self {
            Self::Hf(_) => 0,
            Self::Bytes(_) => 257,
        }
    }
}
