//! Shared vocabulary for every vapi crate: identifiers, sampling parameters,
//! finish reasons, configuration, and the error type.
//!
//! This crate deliberately has no async runtime, no NATS, and no tensor
//! dependency so that it compiles in a fraction of a second and can be used
//! from tests everywhere.

pub mod config;
pub mod error;
pub mod ids;
pub mod sampling;

pub use config::Config;
pub use error::{Error, Result};
pub use ids::{ModelId, RequestId, SeqId};
pub use sampling::{FinishReason, SamplingParams, StopCondition};
