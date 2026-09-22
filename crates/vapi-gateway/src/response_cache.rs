//! Tier 2: an exact-match cache of finished completions in NATS KV.
//!
//! A request is cacheable only when its output is a pure function of its
//! input: greedy, or sampled with a `seed`. The key hashes everything that
//! decides the output — the model's weight fingerprint, the prompt token
//! ids, every sampling parameter, `max_tokens`, the stop strings — so a
//! response is never served to a request that could have produced a
//! different one. A shorter `max_tokens` is a different key on purpose:
//! truncating a longer cached answer would report the wrong finish reason
//! and could cut inside a stop string.
//!
//! Hits skip the queue entirely: no job is published, no worker is
//! touched, and a streaming client is replayed the same token texts the
//! original client saw.

use std::time::Duration;

use async_nats::jetstream::{self, kv};
use serde::{Deserialize, Serialize};
use vapi_core::{FinishReason, SamplingParams};
use vapi_proto::JobKind;

/// A finished completion, as stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedResponse {
    /// The text deltas in order, so a streaming replay has the same shape.
    pub tokens: Vec<String>,
    pub finish: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl CachedResponse {
    pub fn text(&self) -> String {
        self.tokens.concat()
    }
}

/// Whether this request's answer is a function of its input alone.
pub fn cacheable(params: &SamplingParams) -> bool {
    params.is_deterministic() && params.logprobs.is_none()
}

/// The cache key. Hex, so it is a valid KV key.
pub fn key(fingerprint: &str, kind: JobKind, prompt: &[u32], params: &SamplingParams) -> String {
    let mut h = blake3::Hasher::new();
    h.update(fingerprint.as_bytes());
    h.update(&[0]);
    h.update(format!("{kind:?}").as_bytes());
    h.update(&[0]);
    for &t in prompt {
        h.update(&t.to_le_bytes());
    }
    h.update(&[0]);
    // Every field that shapes the output, in a fixed order. Floats are
    // hashed by their bits so 0.7 and 0.7000001 are different keys, as they
    // are different requests.
    for f in [
        params.temperature,
        params.top_p,
        params.min_p,
        params.repetition_penalty,
        params.frequency_penalty,
        params.presence_penalty,
    ] {
        h.update(&f.to_bits().to_le_bytes());
    }
    h.update(&(params.top_k.map_or(u64::MAX, |k| k as u64)).to_le_bytes());
    h.update(&(params.max_tokens as u64).to_le_bytes());
    h.update(&params.seed.map_or(u64::MAX, |s| s ^ 1).to_le_bytes());
    h.update(&[params.seed.is_some() as u8]);
    for s in &params.stop.stop_strings {
        h.update(s.as_bytes());
        h.update(&[0]);
    }
    h.update(&(params.stop.stop_token_ids.len() as u64).to_le_bytes());
    for &t in &params.stop.stop_token_ids {
        h.update(&t.to_le_bytes());
    }
    h.finalize().to_hex().to_string()
}

#[derive(Clone)]
pub struct ResponseCache {
    store: kv::Store,
}

impl ResponseCache {
    /// Open (or create) the bucket. Entries expire after `ttl`.
    pub async fn open(
        js: &jetstream::Context,
        bucket: &str,
        ttl: Duration,
    ) -> Result<Self, String> {
        let store = js
            .create_key_value(kv::Config {
                bucket: bucket.to_string(),
                max_age: ttl,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("response cache bucket {bucket}: {e}"))?;
        Ok(Self { store })
    }

    pub async fn get(&self, key: &str) -> Option<CachedResponse> {
        match self.store.get(key).await {
            Ok(Some(bytes)) => match serde_json::from_slice(&bytes) {
                Ok(v) => {
                    metrics::counter!("vapi_response_cache_hits_total").increment(1);
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "undecodable response cache entry; ignoring");
                    None
                }
            },
            Ok(None) => {
                metrics::counter!("vapi_response_cache_misses_total").increment(1);
                None
            }
            Err(e) => {
                // The cache is an optimisation; a NATS hiccup must not fail
                // the request.
                tracing::warn!(error = %e, "response cache read failed");
                None
            }
        }
    }

    /// Store a finished completion. Best effort, off the request path.
    pub fn put(&self, key: String, entry: CachedResponse) {
        let store = self.store.clone();
        tokio::spawn(async move {
            match serde_json::to_vec(&entry) {
                Ok(bytes) => {
                    if let Err(e) = store.put(key, bytes.into()).await {
                        tracing::warn!(error = %e, "response cache write failed");
                    } else {
                        metrics::counter!("vapi_response_cache_writes_total").increment(1);
                    }
                }
                Err(e) => tracing::warn!(error = %e, "response cache encode failed"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            max_tokens: 16,
            ..Default::default()
        }
    }

    #[test]
    fn only_deterministic_requests_without_logprobs_are_cacheable() {
        assert!(cacheable(&greedy()));
        assert!(cacheable(&SamplingParams {
            temperature: 0.8,
            seed: Some(1),
            ..Default::default()
        }));
        assert!(!cacheable(&SamplingParams {
            temperature: 0.8,
            ..Default::default()
        }));
        assert!(!cacheable(&SamplingParams {
            logprobs: Some(1),
            ..greedy()
        }));
    }

    #[test]
    fn the_key_covers_everything_that_shapes_the_answer() {
        let base = key("fp", JobKind::Chat, &[1, 2, 3], &greedy());
        assert_eq!(base, key("fp", JobKind::Chat, &[1, 2, 3], &greedy()));
        assert_ne!(base, key("fp2", JobKind::Chat, &[1, 2, 3], &greedy()));
        assert_ne!(base, key("fp", JobKind::Completion, &[1, 2, 3], &greedy()));
        assert_ne!(base, key("fp", JobKind::Chat, &[1, 2], &greedy()));
        let shorter = SamplingParams {
            max_tokens: 8,
            ..greedy()
        };
        assert_ne!(base, key("fp", JobKind::Chat, &[1, 2, 3], &shorter));
        let stop = SamplingParams {
            stop: vapi_core::StopCondition {
                stop_strings: vec!["\n".into()],
                ..Default::default()
            },
            ..greedy()
        };
        assert_ne!(base, key("fp", JobKind::Chat, &[1, 2, 3], &stop));
        let seeded = |s: u64| SamplingParams {
            temperature: 0.5,
            seed: Some(s),
            ..greedy()
        };
        assert_ne!(
            key("fp", JobKind::Chat, &[1], &seeded(1)),
            key("fp", JobKind::Chat, &[1], &seeded(2))
        );
        assert!(base.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
