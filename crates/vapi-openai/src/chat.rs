use serde::{Deserialize, Serialize};
use vapi_core::{FinishReason, SamplingParams, StopCondition};

use crate::{StringOrVec, Usage, unix_now};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    /// Deprecated by OpenAI but still emitted by older clients; chat templates
    /// generally treat it as `tool`.
    Function,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    /// Optional because tool-call messages carry no text. Multimodal content
    /// arrays are not accepted yet; a client sending one gets a 400 rather
    /// than having its images silently dropped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: Some(content.into()),
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: Some(content.into()),
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: Some(content.into()),
            name: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's replacement for `max_tokens`; takes precedence when both are
    /// present, which is what the official clients now send.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    /// When set, a final chunk with an empty `choices` array carries `usage`.
    #[serde(default)]
    pub include_usage: bool,
}

impl ChatCompletionRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .map(|o| o.include_usage)
            .unwrap_or(false)
    }

    /// Translate into engine sampling parameters, applying OpenAI's defaults
    /// for anything the client omitted.
    ///
    /// `default_max_tokens` is supplied by the caller because the sensible
    /// ceiling depends on the model's context window, which this crate does
    /// not know about.
    pub fn to_sampling_params(&self, default_max_tokens: usize) -> Result<SamplingParams, String> {
        if let Some(n) = self.n
            && n != 1
        {
            return Err("n > 1 is not supported yet".into());
        }
        // `logprobs` here is a bool; the count lives in `top_logprobs`.
        let logprobs = match (self.logprobs.unwrap_or(false), self.top_logprobs) {
            (true, Some(n)) => Some(n),
            (true, None) => Some(1),
            (false, _) => None,
        };

        let params = SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k,
            min_p: self.min_p.unwrap_or(0.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(default_max_tokens),
            seed: self.seed,
            logprobs,
            stop: StopCondition {
                stop_token_ids: Vec::new(),
                stop_strings: self
                    .stop
                    .clone()
                    .map(StringOrVec::into_vec)
                    .unwrap_or_default(),
                include_stop_str: false,
            },
        };
        params.validate()?;
        Ok(params)
    }
}

// ---------------------------------------------------------------- responses

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(
        id: String,
        model: String,
        content: String,
        finish: FinishReason,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object: "chat.completion",
            created: unix_now(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::assistant(content),
                finish_reason: Some(finish.as_openai().to_string()),
            }],
            usage,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// One `data:` frame of an SSE stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Only present on the final chunk, and only when the client asked for it
    /// via `stream_options.include_usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl StreamChunk {
    fn base(id: &str, model: &str, created: u64, choices: Vec<ChunkChoice>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices,
            usage: None,
        }
    }

    /// The first chunk announces the assistant role and carries no content.
    pub fn role(id: &str, model: &str, created: u64) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some(ChatRole::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        )
    }

    pub fn content(id: &str, model: &str, created: u64, text: impl Into<String>) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(text.into()),
                },
                finish_reason: None,
            }],
        )
    }

    pub fn finish(id: &str, model: &str, created: u64, reason: FinishReason) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(reason.as_openai().to_string()),
            }],
        )
    }

    /// Trailing usage-only chunk. `choices` must be empty here — clients that
    /// validate the shape will reject a usage chunk that also carries a choice.
    pub fn usage_only(id: &str, model: &str, created: u64, usage: Usage) -> Self {
        let mut c = Self::base(id, model, created, Vec::new());
        c.usage = Some(usage);
        c
    }

    /// Render as a complete SSE frame, trailing blank line included.
    pub fn to_sse_frame(&self) -> String {
        match serde_json::to_string(self) {
            Ok(json) => format!("data: {json}\n\n"),
            // Serializing our own owned types cannot realistically fail; if it
            // somehow does, end the stream cleanly rather than hanging it.
            Err(_) => "data: [DONE]\n\n".to_string(),
        }
    }

    pub const DONE_FRAME: &'static str = "data: [DONE]\n\n";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("parse")
    }

    #[test]
    fn parses_a_minimal_official_client_payload() {
        let r = req(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(r.messages.len(), 1);
        assert!(!r.is_streaming());
        let p = r.to_sampling_params(256).unwrap();
        assert_eq!(p.max_tokens, 256);
        assert_eq!(p.temperature, 1.0);
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        // The official SDK migrated to max_completion_tokens; some clients
        // send both during the transition.
        let r = req(r#"{"model":"m","messages":[],"max_tokens":10,"max_completion_tokens":99}"#);
        assert_eq!(r.to_sampling_params(256).unwrap().max_tokens, 99);
    }

    #[test]
    fn stop_accepts_string_or_array() {
        let one = req(r#"{"model":"m","messages":[],"stop":"END"}"#);
        assert_eq!(
            one.to_sampling_params(16).unwrap().stop.stop_strings,
            vec!["END"]
        );
        let many = req(r#"{"model":"m","messages":[],"stop":["A","B"]}"#);
        assert_eq!(
            many.to_sampling_params(16).unwrap().stop.stop_strings,
            vec!["A", "B"]
        );
    }

    #[test]
    fn top_logprobs_drives_the_count() {
        let r = req(r#"{"model":"m","messages":[],"logprobs":true,"top_logprobs":5}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().logprobs, Some(5));
        let r = req(r#"{"model":"m","messages":[],"logprobs":true}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().logprobs, Some(1));
        let r = req(r#"{"model":"m","messages":[]}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().logprobs, None);
    }

    #[test]
    fn n_greater_than_one_is_rejected_not_ignored() {
        let r = req(r#"{"model":"m","messages":[],"n":3}"#);
        assert!(r.to_sampling_params(16).is_err());
    }

    #[test]
    fn unsupported_fields_are_rejected_loudly() {
        // deny_unknown_fields: better a 400 than silently ignoring `tools`
        // and returning a plain completion the caller will misinterpret.
        let r: Result<ChatCompletionRequest, _> =
            serde_json::from_str(r#"{"model":"m","messages":[],"tools":[]}"#);
        assert!(r.is_err());
    }

    #[test]
    fn sse_frame_shape_matches_the_spec() {
        let c = StreamChunk::content("chatcmpl-1", "m", 1700000000, "He");
        let frame = c.to_sse_frame();
        assert!(frame.starts_with("data: {"));
        assert!(frame.ends_with("\n\n"));
        let v: serde_json::Value = serde_json::from_str(frame[6..frame.len() - 2].trim()).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["content"], "He");
        assert!(v["choices"][0]["finish_reason"].is_null());
        // role must be absent, not null, on a content chunk
        assert!(v["choices"][0]["delta"].get("role").is_none());
    }

    #[test]
    fn usage_chunk_has_no_choices() {
        let c = StreamChunk::usage_only("id", "m", 0, Usage::new(3, 4));
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["total_tokens"], 7);
    }

    #[test]
    fn non_usage_chunks_omit_usage_entirely() {
        let v = serde_json::to_value(StreamChunk::role("id", "m", 0)).unwrap();
        assert!(v.get("usage").is_none());
    }
}
