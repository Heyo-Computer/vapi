//! Scheduling, batching and sampling.
//!
//! This crate owns the engine loop but not the tensors: model execution sits
//! behind [`ExecutionBackend`], so the scheduler and cache logic are testable
//! with no model, no device and no runtime.

pub mod backend;
pub mod detokenize;
pub mod encoder;
pub mod sampler;
pub mod scheduler;
pub mod sequence;
pub mod structured;

pub use backend::{
    CandidateNeed, ExecutionBackend, ForwardBatch, GumbelDraw, MockBackend, ModelSpec,
    RowCandidates, RowLogits, StepLogits,
};
pub use detokenize::IncrementalDetokenizer;
pub use encoder::{
    EncoderBackend, EncoderBatch, EncoderRow, EncoderSpec, MarkerLogits, MockEncoder, RowScores,
};
pub use sampler::{RowLogprobs, Sampler, SamplerState};
pub use scheduler::{Scheduler, SchedulerConfig, StepPlan};
pub use sequence::{SeqStatus, Sequence};
pub use structured::{Machine, Schema, TokenMasker};
