use serde::{Deserialize, Serialize};
use vapi_core::{FinishReason, ModelId, RequestId, SamplingParams};

/// A unit of work on the durable queue.
///
/// The prompt is carried as **token ids, already templated and tokenized by
/// the gateway**. That placement is deliberate: it keeps the tokenizer and the
/// chat template off the worker's hot path, lets the gateway reject an
/// over-length prompt with a 400 before any queue work happens, and makes the
/// job self-describing — a worker never has to reconstruct what the client
/// actually asked for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub request_id: RequestId,
    pub model: ModelId,
    pub kind: JobKind,
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    /// Folded into every KV block hash. Requests sharing a namespace share
    /// cached prefixes; a tenant needing isolation gets its own.
    pub namespace: String,
    /// Subject the worker publishes deltas to. Carried explicitly rather than
    /// derived, because a JetStream message's own `reply` subject is the ack
    /// subject and cannot be reused for this.
    pub reply_to: String,
    /// Gateway clock at enqueue, milliseconds since epoch. Used for queue-wait
    /// metrics only — never for ordering, since gateway clocks are not synced.
    pub enqueued_at_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// `/v1/chat/completions` — prompt was rendered through the chat template.
    Chat,
    /// `/v1/completions` — prompt tokenized verbatim.
    Completion,
}

/// One message on a request's core-NATS token stream.
///
/// `Token` carries the incrementally detokenized text rather than the raw id,
/// because turning ids into text needs tokenizer state (a multi-byte character
/// can straddle two tokens) that only the generating worker has. The id rides
/// along for logprobs and for tests that assert on exact token sequences.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Delta {
    /// Worker has admitted the sequence. Lets the gateway distinguish "still
    /// queued" from "running but slow" when a timeout fires.
    Started {
        worker_id: String,
        /// Prompt tokens served from the shared prefix cache; 0 on a miss.
        cached_prefix_tokens: usize,
    },
    Token {
        /// Which completion this belongs to, for `n > 1`.
        #[serde(default)]
        choice: u32,
        text: String,
        token_id: u32,
        /// Log-probability of this token under the model's own distribution
        /// (before temperature and penalties), when the request asked.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
        /// The most likely tokens at this position, most likely first, when
        /// the request asked for them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_logprobs: Option<Vec<TokenLogprob>>,
    },
    Done {
        #[serde(default)]
        choice: u32,
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Generation failed. The gateway turns this into an error frame, or a
    /// non-2xx body if nothing has been sent yet.
    Failed { message: String },
}

/// One alternative token and its log-probability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// A delta plus its position in the request's stream.
///
/// Core NATS does not retry or replay, so a dropped message is a token that
/// silently never arrives — the client sees a coherent-looking answer with a
/// word missing. The sequence number makes that detectable: the gateway tracks
/// the expected next value and reports a gap instead of quietly serving
/// corrupted output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaMsg {
    pub seq: u64,
    #[serde(flatten)]
    pub delta: Delta,
}

impl DeltaMsg {
    pub fn new(seq: u64, delta: Delta) -> Self {
        Self { seq, delta }
    }
}

/// Assigns consecutive sequence numbers to one request's deltas.
#[derive(Debug, Default)]
pub struct DeltaSeq(u64);

impl DeltaSeq {
    pub fn next(&mut self, delta: Delta) -> DeltaMsg {
        let seq = self.0;
        self.0 += 1;
        DeltaMsg::new(seq, delta)
    }
}

/// Gateway-side check that no delta went missing.
#[derive(Debug, Default)]
pub struct DeltaSeqCheck {
    expected: u64,
    gaps: u64,
}

impl DeltaSeqCheck {
    /// Returns how many messages were skipped before this one, if any.
    pub fn observe(&mut self, seq: u64) -> Option<u64> {
        let missing = seq.saturating_sub(self.expected);
        self.expected = seq + 1;
        if missing > 0 {
            self.gaps += missing;
            Some(missing)
        } else {
            None
        }
    }

    pub fn total_gaps(&self) -> u64 {
        self.gaps
    }
}

impl Delta {
    /// Whether this delta ends the stream. The gateway unsubscribes on it.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Failed { .. })
    }
}

/// Periodic worker liveness and load, published to `vapi.ctl.worker.<id>` and
/// mirrored into the `VAPI_WORKERS` KV bucket with a TTL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerStats {
    pub worker_id: String,
    pub model: ModelId,
    pub running: usize,
    pub waiting: usize,
    pub max_concurrent: usize,
    /// Fraction of KV blocks currently allocated, 0.0..=1.0.
    pub block_utilization: f32,
    /// Cumulative prefix-cache hit rate since start, 0.0..=1.0.
    pub prefix_hit_rate: f32,
    pub uptime_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_tagging_is_stable_on_the_wire() {
        // Workers and gateways are deployed independently, so these tags are a
        // compatibility surface; renaming a variant breaks a rolling upgrade.
        let json = serde_json::to_value(Delta::Token {
            choice: 0,
            text: "hi".into(),
            token_id: 7,
            logprob: None,
            top_logprobs: None,
        })
        .unwrap();
        assert_eq!(json["t"], "token");
        assert_eq!(json["text"], "hi");
        assert!(json.get("logprob").is_none(), "absent, not null");

        let done = serde_json::to_value(Delta::Done {
            choice: 0,
            reason: FinishReason::Length,
            prompt_tokens: 3,
            completion_tokens: 4,
        })
        .unwrap();
        assert_eq!(done["t"], "done");
        assert_eq!(done["reason"], "length");
    }

    #[test]
    fn only_done_and_failed_terminate_a_stream() {
        assert!(
            !Delta::Started {
                worker_id: "w".into(),
                cached_prefix_tokens: 0
            }
            .is_terminal()
        );
        assert!(
            !Delta::Token {
                choice: 0,
                text: "x".into(),
                token_id: 1,
                logprob: None,
                top_logprobs: None,
            }
            .is_terminal()
        );
        assert!(
            Delta::Done {
                choice: 0,
                reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1
            }
            .is_terminal()
        );
        assert!(
            Delta::Failed {
                message: "boom".into()
            }
            .is_terminal()
        );
    }

    #[test]
    fn delta_msg_flattens_seq_alongside_the_tag() {
        let m = DeltaMsg::new(
            4,
            Delta::Token {
                choice: 0,
                text: "x".into(),
                token_id: 1,
                logprob: None,
                top_logprobs: None,
            },
        );
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["seq"], 4);
        assert_eq!(v["t"], "token");
        let back: DeltaMsg = serde_json::from_value(v).unwrap();
        assert_eq!(back.seq, 4);
    }

    #[test]
    fn a_dropped_delta_is_detected_not_silently_accepted() {
        let mut c = DeltaSeqCheck::default();
        assert_eq!(c.observe(0), None);
        assert_eq!(c.observe(1), None);
        // Message 2 was lost in transit.
        assert_eq!(c.observe(3), Some(1));
        assert_eq!(c.observe(4), None);
        assert_eq!(c.total_gaps(), 1);
    }

    #[test]
    fn sequencer_and_checker_agree_on_a_clean_stream() {
        let mut seq = DeltaSeq::default();
        let mut check = DeltaSeqCheck::default();
        for i in 0..10u32 {
            let m = seq.next(Delta::Token {
                choice: 0,
                text: i.to_string(),
                token_id: i,
                logprob: None,
                top_logprobs: None,
            });
            assert_eq!(check.observe(m.seq), None);
        }
        assert_eq!(check.total_gaps(), 0);
    }

    #[test]
    fn job_round_trips() {
        let job = Job {
            request_id: RequestId::parse("r1").unwrap(),
            model: ModelId("m".into()),
            kind: JobKind::Chat,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            namespace: "global".into(),
            reply_to: "vapi.stream.r1".into(),
            enqueued_at_ms: 1700000000000,
        };
        let bytes = crate::encode(&job).unwrap();
        let back: Job = crate::decode(&bytes).unwrap();
        assert_eq!(back.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(back.request_id.as_str(), "r1");
        assert_eq!(back.kind, JobKind::Chat);
    }
}
