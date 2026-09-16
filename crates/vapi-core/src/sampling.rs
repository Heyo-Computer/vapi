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
    pub seed: Option<u64>,
    /// Number of top logprobs to report per position, if any.
    pub logprobs: Option<usize>,
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
            seed: None,
            logprobs: None,
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

/// Why generation stopped for a sequence. Mirrors OpenAI's `finish_reason`
/// plus the internal-only variants the API never surfaces directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Hit an EOS token or a configured stop condition.
    Stop,
    /// Reached `max_tokens`, or the model's context limit.
    Length,
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
