use serde::{Deserialize, Serialize};
use vapi_core::{FinishReason, ResponseFormat, SamplingParams, StopCondition};

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
    /// On an assistant message: calls the model made earlier in the
    /// conversation. On a response: calls it is making now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// On a `tool` message: which call this is the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// The model's thinking, when its output format separates it from the
    /// answer (the convention DeepSeek and vLLM use). Accepted on input and
    /// ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    fn with(role: ChatRole, content: Option<String>) -> Self {
        Self {
            role,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::with(ChatRole::System, Some(content.into()))
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::with(ChatRole::User, Some(content.into()))
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::with(ChatRole::Assistant, Some(content.into()))
    }
}

/// A function call the model makes, in OpenAI's shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, as OpenAI sends them.
    pub arguments: String,
}

impl ToolCall {
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: String) -> Self {
        Self {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        }
    }
}

/// One tool call in a streaming delta: the whole call arrives in one chunk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
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
    /// OpenAI tool definitions, passed to the chat template as given.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// `"auto"` (the default) or `"none"`. Forcing a particular function is
    /// not supported and gets a 400.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Accepted for compatibility; the model decides.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// `{"type": "text"}`, `{"type": "json_object"}`, or
    /// `{"type": "json_schema", "json_schema": {"schema": {...}}}`.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
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

    /// Translate `response_format` into a decoding constraint.
    fn constraint(&self) -> Result<Option<ResponseFormat>, String> {
        let Some(rf) = &self.response_format else {
            return Ok(None);
        };
        let kind = rf
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| "response_format needs a type".to_string())?;
        match kind {
            "text" => Ok(None),
            "json_object" => Ok(Some(ResponseFormat::JsonObject)),
            "json_schema" => {
                // OpenAI nests the schema under `json_schema.schema`; some
                // clients pass it bare, so accept both.
                let js = rf
                    .get("json_schema")
                    .ok_or_else(|| "json_schema is missing".to_string())?;
                let schema = js.get("schema").unwrap_or(js).clone();
                Ok(Some(ResponseFormat::JsonSchema { schema }))
            }
            other => Err(format!(
                "response_format {other:?} is not supported; use text, json_object or json_schema"
            )),
        }
    }

    /// The tools to render into the prompt, honouring `tool_choice`.
    pub fn active_tools(&self) -> Result<Option<&[serde_json::Value]>, String> {
        match &self.tool_choice {
            None => {}
            Some(serde_json::Value::String(s)) if s == "auto" => {}
            Some(serde_json::Value::String(s)) if s == "none" => return Ok(None),
            Some(other) => {
                return Err(format!(
                    "tool_choice {other} is not supported; use \"auto\" or \"none\""
                ));
            }
        }
        Ok(self.tools.as_deref().filter(|t| !t.is_empty()))
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
            n: self.n.unwrap_or(1),
            seed: self.seed,
            logprobs,
            response_format: self.constraint()?,
            // The gateway fills this in: it knows the model's markers.
            constraint_starts_after: None,
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
                logprobs: None,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

impl ChatCompletionResponse {
    /// A response carrying several completions, in choice order.
    pub fn with_choices(
        id: String,
        model: String,
        choices: Vec<(String, FinishReason)>,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object: "chat.completion",
            created: unix_now(),
            model,
            choices: choices
                .into_iter()
                .enumerate()
                .map(|(index, (content, finish))| ChatChoice {
                    index,
                    message: ChatMessage::assistant(content),
                    finish_reason: Some(finish.as_openai().to_string()),
                    logprobs: None,
                })
                .collect(),
            usage,
        }
    }

    /// Attach parsed tool calls and/or reasoning to the message.
    pub fn with_parsed(mut self, reasoning: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        if let Some(c) = self.choices.first_mut() {
            c.message.reasoning_content = reasoning.filter(|r| !r.is_empty());
            if !tool_calls.is_empty() {
                if c.message.content.as_deref() == Some("") {
                    c.message.content = None;
                }
                c.message.tool_calls = Some(tool_calls);
                c.finish_reason = Some(FinishReason::ToolCalls.as_openai().to_string());
            }
        }
        self
    }

    pub fn with_logprobs(mut self, content: Vec<LogprobEntry>) -> Self {
        if let Some(c) = self.choices.first_mut() {
            c.logprobs = Some(ChatLogprobs { content });
        }
        self
    }
}

/// `choices[].logprobs` in OpenAI's chat format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatLogprobs {
    pub content: Vec<LogprobEntry>,
}

/// One generated token's log-probability and the alternatives at that
/// position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    /// UTF-8 bytes of `token`, as OpenAI reports them (a token can be a
    /// partial character).
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

impl LogprobEntry {
    pub fn new(token: String, logprob: f32, top: Vec<(String, f32)>) -> Self {
        Self {
            bytes: Some(token.as_bytes().to_vec()),
            top_logprobs: top
                .into_iter()
                .map(|(token, logprob)| TopLogprob {
                    bytes: Some(token.as_bytes().to_vec()),
                    token,
                    logprob,
                })
                .collect(),
            token,
            logprob,
        }
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
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
                    ..Delta::default()
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// A content chunk for choice `index`, which OpenAI clients use to
    /// keep several completions apart in one stream.
    pub fn content_for(
        id: &str,
        model: &str,
        created: u64,
        index: usize,
        text: impl Into<String>,
    ) -> Self {
        let mut c = Self::content(id, model, created, text);
        if let Some(choice) = c.choices.first_mut() {
            choice.index = index;
        }
        c
    }

    /// The finish chunk for choice `index`.
    pub fn finish_for(
        id: &str,
        model: &str,
        created: u64,
        index: usize,
        reason: FinishReason,
    ) -> Self {
        let mut c = Self::finish(id, model, created, reason);
        if let Some(choice) = c.choices.first_mut() {
            choice.index = index;
        }
        c
    }

    pub fn content(id: &str, model: &str, created: u64, text: impl Into<String>) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some(text.into()),
                    ..Delta::default()
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// A chunk of the model's thinking.
    pub fn reasoning(id: &str, model: &str, created: u64, text: impl Into<String>) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    reasoning_content: Some(text.into()),
                    ..Delta::default()
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// A chunk carrying whole tool calls.
    pub fn tool_calls(id: &str, model: &str, created: u64, calls: &[ToolCall]) -> Self {
        Self::base(
            id,
            model,
            created,
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    tool_calls: Some(
                        calls
                            .iter()
                            .enumerate()
                            .map(|(index, c)| ToolCallDelta {
                                index,
                                id: c.id.clone(),
                                kind: c.kind.clone(),
                                function: c.function.clone(),
                            })
                            .collect(),
                    ),
                    ..Delta::default()
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// A content chunk carrying this token's logprobs.
    pub fn with_logprobs(mut self, entry: LogprobEntry) -> Self {
        if let Some(c) = self.choices.first_mut() {
            c.logprobs = Some(ChatLogprobs {
                content: vec![entry],
            });
        }
        self
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
                logprobs: None,
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
    fn n_is_carried_through_and_bounded() {
        let r = req(r#"{"model":"m","messages":[],"n":3}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().n, 3);
        let r = req(r#"{"model":"m","messages":[]}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().n, 1);
        // Every choice is a sequence in the same cache, so there is a cap.
        let r = req(r#"{"model":"m","messages":[],"n":99}"#);
        assert!(r.to_sampling_params(16).is_err());
        let r = req(r#"{"model":"m","messages":[],"n":0}"#);
        assert!(r.to_sampling_params(16).is_err());
    }

    #[test]
    fn unsupported_fields_are_rejected_loudly() {
        // deny_unknown_fields: better a 400 than silently ignoring a field
        // and returning a completion the caller will misinterpret.
        let r: Result<ChatCompletionRequest, _> =
            serde_json::from_str(r#"{"model":"m","messages":[],"frequency_bias":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn response_format_becomes_a_decoding_constraint() {
        let r = req(r#"{"model":"m","messages":[],"response_format":{"type":"json_object"}}"#);
        assert_eq!(
            r.to_sampling_params(16).unwrap().response_format,
            Some(ResponseFormat::JsonObject)
        );
        // OpenAI nests the schema; a bare one is accepted too.
        let nested = r#"{"model":"m","messages":[],"response_format":{"type":"json_schema",
            "json_schema":{"name":"x","schema":{"type":"object"}}}}"#;
        let bare = r#"{"model":"m","messages":[],"response_format":{"type":"json_schema",
            "json_schema":{"type":"object"}}}"#;
        for body in [nested, bare] {
            let got = req(body).to_sampling_params(16).unwrap().response_format;
            assert_eq!(
                got,
                Some(ResponseFormat::JsonSchema {
                    schema: serde_json::json!({"type": "object"})
                }),
                "{body}"
            );
        }
        // `text` is the default and constrains nothing.
        let r = req(r#"{"model":"m","messages":[],"response_format":{"type":"text"}}"#);
        assert_eq!(r.to_sampling_params(16).unwrap().response_format, None);
        let r = req(r#"{"model":"m","messages":[],"response_format":{"type":"yaml"}}"#);
        assert!(r.to_sampling_params(16).is_err());
    }

    #[test]
    fn tool_choice_decides_whether_tools_reach_the_prompt() {
        let with_tools = r#""tools":[{"type":"function","function":{"name":"f"}}]"#;
        let r = req(&format!(r#"{{"model":"m","messages":[],{with_tools}}}"#));
        assert_eq!(r.active_tools().unwrap().map(<[_]>::len), Some(1));
        let r = req(&format!(
            r#"{{"model":"m","messages":[],{with_tools},"tool_choice":"none"}}"#
        ));
        assert_eq!(r.active_tools().unwrap(), None);
        let r = req(&format!(
            r#"{{"model":"m","messages":[],{with_tools},"tool_choice":"auto"}}"#
        ));
        assert!(r.active_tools().unwrap().is_some());
        // A named function is not supported, and says so rather than
        // quietly letting the model choose.
        let r = req(&format!(
            r#"{{"model":"m","messages":[],{with_tools},"tool_choice":{{"type":"function"}}}}"#
        ));
        assert!(r.active_tools().is_err());
        // No tools at all, or an empty list: nothing to render.
        let r = req(r#"{"model":"m","messages":[],"tools":[]}"#);
        assert_eq!(r.active_tools().unwrap(), None);
    }

    #[test]
    fn a_tool_result_conversation_round_trips() {
        let r = req(r#"{"model":"m","messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"18C"}
            ]}"#);
        assert_eq!(r.messages.len(), 3);
        let calls = r.messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(r.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn tool_calls_shape_the_response_and_its_finish_reason() {
        let call = ToolCall::function("call_1", "f", r#"{"a":1}"#.into());
        let resp = ChatCompletionResponse::new(
            "chatcmpl-1".into(),
            "m".into(),
            String::new(),
            FinishReason::Stop,
            Usage::new(1, 1),
        )
        .with_parsed(Some("thinking".into()), vec![call.clone()]);
        let c = &resp.choices[0];
        assert_eq!(c.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(c.message.tool_calls.as_deref(), Some(&[call][..]));
        assert_eq!(c.message.reasoning_content.as_deref(), Some("thinking"));
        // Empty content is dropped rather than sent as "".
        assert_eq!(c.message.content, None);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            r#"{"a":1}"#
        );
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
