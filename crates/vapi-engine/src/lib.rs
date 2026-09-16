//! Scheduling, batching and sampling.
//!
//! This crate owns the engine loop but not the tensors: model execution sits
//! behind [`ExecutionBackend`], so the scheduler and cache logic are testable
//! with no model, no device and no runtime.

pub mod backend;
pub mod detokenize;
pub mod sampler;
pub mod scheduler;
pub mod sequence;

pub use backend::{ExecutionBackend, ForwardBatch, MockBackend, ModelSpec};
pub use detokenize::IncrementalDetokenizer;
pub use sampler::{Sampler, SamplerState};
pub use scheduler::{Scheduler, SchedulerConfig, StepPlan};
pub use sequence::{SeqStatus, Sequence};
