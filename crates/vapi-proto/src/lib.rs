//! The wire contract between gateway and worker.
//!
//! The governing decision here is the split between durable and ephemeral
//! traffic. Prompts go through JetStream because losing one loses a user's
//! request. Token deltas go over **core NATS** because a JetStream publish is
//! an acknowledged, persisted write, and doing that per token at a
//! few-milliseconds cadence would fsync thousands of times a second for data
//! whose entire lifetime is one HTTP connection. If the SSE connection dies,
//! the deltas are worthless anyway.

pub mod codec;
pub mod envelope;
pub mod subjects;

pub use codec::{decode, encode};
pub use envelope::{
    AudioClip, DecisionRow, Delta, DeltaMsg, DeltaSeq, DeltaSeqCheck, Job, JobKind, RowScores,
    TokenLogprob, WorkerEntry, WorkerStats,
};
pub use subjects::Subjects;
