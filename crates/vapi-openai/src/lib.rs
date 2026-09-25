//! OpenAI-compatible wire types.
//!
//! Serde definitions only, with the conversion into vapi's own
//! [`SamplingParams`] as the single piece of logic. Keeping this crate free of
//! engine and transport dependencies means the shapes can be unit-tested
//! against captured payloads from real client SDKs.

pub mod chat;
pub mod completion;
pub mod decision;
pub mod models;
pub mod transcription;

pub use chat::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatLogprobs, ChatMessage, ChatRole,
    ChunkChoice, Delta, FunctionCall, LogprobEntry, StreamChunk, StreamOptions, ToolCall,
    ToolCallDelta, TopLogprob,
};
pub use completion::{CompletionChoice, CompletionRequest, CompletionResponse};
pub use decision::{
    Answer, Criteria, DecisionRequest, DecisionResponse, Ordered, Question, python_json, round4,
};
pub use models::{Model, ModelList};
pub use transcription::{TranscriptionFormat, TranscriptionResponse, VerboseTranscriptionResponse};

use serde::{Deserialize, Serialize};

/// Token accounting returned to the client.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// `stop` accepts either a single string or an array of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
