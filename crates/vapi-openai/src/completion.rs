//! Legacy `/v1/completions`. Same engine path as chat, minus the chat
//! template: the prompt is tokenized verbatim.

use serde::{Deserialize, Serialize};
use vapi_core::{FinishReason, SamplingParams, StopCondition};

use crate::{StringOrVec, Usage, unix_now};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    /// Only a single string prompt is supported. Batched array prompts would
    /// need `n`-way fan-out, which the engine does not do yet.
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
}

impl CompletionRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub fn to_sampling_params(&self, default_max_tokens: usize) -> Result<SamplingParams, String> {
        let params = SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k,
            max_tokens: self.max_tokens.unwrap_or(default_max_tokens),
            seed: self.seed,
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            stop: StopCondition {
                stop_strings: self
                    .stop
                    .clone()
                    .map(StringOrVec::into_vec)
                    .unwrap_or_default(),
                ..Default::default()
            },
            ..Default::default()
        };
        params.validate()?;
        Ok(params)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(
        id: String,
        model: String,
        text: String,
        finish: FinishReason,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object: "text_completion",
            created: unix_now(),
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: Some(finish.as_openai().to_string()),
                logprobs: None,
            }],
            usage,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_completion_payload() {
        let r: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"once upon","max_tokens":5}"#).unwrap();
        assert_eq!(r.prompt, "once upon");
        assert_eq!(r.to_sampling_params(16).unwrap().max_tokens, 5);
    }

    #[test]
    fn response_object_tag_is_text_completion() {
        let r = CompletionResponse::new(
            "cmpl-1".into(),
            "m".into(),
            "hi".into(),
            FinishReason::Stop,
            Usage::new(1, 1),
        );
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }
}
