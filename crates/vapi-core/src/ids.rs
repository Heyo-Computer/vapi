use std::fmt;

use serde::{Deserialize, Serialize};

/// Identifies one client request end to end: it is the JetStream `Nats-Msg-Id`
/// used for publish deduplication, and it names the core-NATS subject that
/// carries the token deltas back to the gateway holding the SSE connection.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().simple().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// NATS subject tokens may not contain `.`, ` `, `*` or `>`. Request ids
    /// reach us from our own generator today, but a future resumable-request
    /// feature would let a client supply one, so validate rather than trust.
    pub fn parse(s: impl Into<String>) -> Option<Self> {
        let s = s.into();
        let ok = !s.is_empty()
            && s.len() <= 128
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        ok.then_some(Self(s))
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RequestId({})", self.0)
    }
}

/// A model as named on the wire (`"meta-llama/Llama-3.2-1B-Instruct"`).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    /// The subject token for this model's work queue. `/` is not legal in a
    /// NATS subject token, so it is mapped to `_`.
    pub fn subject_token(&self) -> String {
        self.0.replace(['/', '.', ' ', '*', '>'], "_")
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Engine-local sequence handle. Unlike [`RequestId`] this is never seen
/// outside a single worker process and is cheap to copy.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct SeqId(pub u64);

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq{}", self.0)
    }
}
