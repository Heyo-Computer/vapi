use serde::{Deserialize, Serialize};

/// Everything that controls token selection for one sequence.
///
/// This type is hashed into the tier-2 response cache key, so its
/// serialization must be stable and it must capture *every* input that can
/// change the output. Adding a field here without adding it to
/// [`SamplingParams::is_deterministic`] is how a response cache starts
/// returning wrong answers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub max_tokens: usize,
    /// How many completions to generate. Above 1 the engine prefills once
    /// and forks, so the choices share the prompt's KV.
    pub n: usize,
    pub seed: Option<u64>,
    /// Number of top logprobs to report per position, if any.
    pub logprobs: Option<usize>,
    /// Constrain decoding so the answer parses. The schema travels as JSON
    /// because it is the client's, and the engine compiles it once per
    /// request.
    pub response_format: Option<ResponseFormat>,
    /// Hold the constraint back until this marker appears in the output.
    ///
    /// A model whose prompt opens a reasoning block has to be allowed to
    /// finish thinking before its answer is forced into a shape; the
    /// gateway sets this to the model's closing marker.
    pub constraint_starts_after: Option<String>,
    pub stop: StopCondition,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
            n: 1,
            seed: None,
            logprobs: None,
            response_format: None,
            constraint_starts_after: None,
            stop: StopCondition::default(),
        }
    }
}

impl SamplingParams {
    /// True when re-running this request is guaranteed to produce the same
    /// tokens, which is the precondition for storing it in the response cache.
    ///
    /// Greedy decoding qualifies. A fixed seed also qualifies, but only
    /// because the engine seeds a per-sequence `ChaCha8Rng` rather than drawing
    /// from a shared generator — with a shared RNG the result would depend on
    /// what else happened to be in the batch.
    pub fn is_deterministic(&self) -> bool {
        self.temperature <= f32::EPSILON || self.seed.is_some()
    }

    /// The seed for choice `i`. Choices must differ, so a seeded request
    /// offsets rather than reusing one seed; greedy choices are identical
    /// by definition and the engine does not fork them.
    pub fn seed_for(&self, choice: usize) -> Option<u64> {
        self.seed.map(|s| s.wrapping_add(choice as u64))
    }

    /// Greedy decoding: take the argmax and skip the sampling pipeline.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= f32::EPSILON
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err("temperature must be in [0, 2]".into());
        }
        if !(0.0..=1.0).contains(&self.top_p) || self.top_p == 0.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err("min_p must be in [0, 1]".into());
        }
        if let Some(k) = self.top_k
            && k == 0
        {
            return Err("top_k must be >= 1".into());
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be >= 1".into());
        }
        // A cap, because every choice is a sequence competing for the same
        // KV cache and the caller pays for all of them.
        if self.n == 0 || self.n > MAX_CHOICES {
            return Err(format!("n must be in [1, {MAX_CHOICES}]"));
        }
        if self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be > 0".into());
        }
        if let Some(n) = self.logprobs
            && n > 20
        {
            return Err("logprobs must be <= 20".into());
        }
        Ok(())
    }
}

/// Most completions one request may ask for.
pub const MAX_CHOICES: usize = 8;

/// What the output must conform to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Any JSON document.
    JsonObject,
    /// A document matching this JSON Schema.
    JsonSchema { schema: serde_json::Value },
}

/// Why generation stopped for a sequence. Mirrors OpenAI's `finish_reason`
/// plus the internal-only variants the API never surfaces directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Hit an EOS token or a configured stop condition.
    Stop,
    /// Reached `max_tokens`, or the model's context limit.
    Length,
    /// The gateway parsed tool calls out of the answer. Never produced by
    /// the engine.
    ToolCalls,
    /// The client went away and we published a cancellation.
    Cancelled,
    /// The engine failed; the gateway turns this into a 500.
    Error,
}

impl FinishReason {
    /// The string OpenAI clients expect. `Cancelled` and `Error` never reach a
    /// well-formed response body, but map them to something sane rather than
    /// inventing a value a client SDK will reject.
    pub fn as_openai(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::Cancelled | Self::Error => "stop",
        }
    }
}

/// Stop tokens and stop strings.
///
/// Stop *strings* are the awkward half: a stop string can straddle a token
/// boundary and can even be produced halfway through a token, so detecting one
/// requires the incrementally detokenized text, not the token ids.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopCondition {
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    /// When false (the OpenAI default) the stop string is trimmed from the
    /// returned text.
    pub include_stop_str: bool,
}

impl StopCondition {
    /// Returns the byte index at which `text` should be truncated if any stop
    /// string is present.
    pub fn find_stop_string(&self, text: &str) -> Option<usize> {
        self.stop_strings
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| {
                text.find(s.as_str()).map(|i| {
                    if self.include_stop_str {
                        i + s.len()
                    } else {
                        i
                    }
                })
            })
            .min()
    }

    /// The longest stop string, which is how many bytes of already-emitted
    /// text must be held back before streaming it to the client: a stop string
    /// could still be completed by the next token.
    pub fn max_stop_len(&self) -> usize {
        self.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic() {
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(p.is_deterministic());
        assert!(p.is_greedy());
    }

    #[test]
    fn sampled_without_seed_is_not_cacheable() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: None,
            ..Default::default()
        };
        assert!(!p.is_deterministic());
    }

    #[test]
    fn sampled_with_seed_is_cacheable() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: Some(42),
            ..Default::default()
        };
        assert!(p.is_deterministic());
    }

    #[test]
    fn stop_string_trimmed_by_default() {
        let s = StopCondition {
            stop_strings: vec!["\nUser:".into()],
            ..Default::default()
        };
        assert_eq!(s.find_stop_string("hello\nUser: hi"), Some(5));
    }

    #[test]
    fn stop_string_included_when_requested() {
        let s = StopCondition {
            stop_strings: vec!["END".into()],
            include_stop_str: true,
            ..Default::default()
        };
        assert_eq!(s.find_stop_string("abcENDxyz"), Some(6));
    }

    #[test]
    fn earliest_stop_string_wins() {
        let s = StopCondition {
            stop_strings: vec!["zzz".into(), "b".into()],
            ..Default::default()
        };
        assert_eq!(s.find_stop_string("abzzz"), Some(1));
    }

    #[test]
    fn validation_rejects_nonsense() {
        assert!(
            SamplingParams {
                temperature: -1.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingParams {
                top_p: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingParams {
                max_tokens: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(SamplingParams::default().validate().is_ok());
    }
}
