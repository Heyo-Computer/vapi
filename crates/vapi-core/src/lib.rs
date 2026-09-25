//! Shared vocabulary for every vapi crate: identifiers, sampling parameters,
//! finish reasons, configuration, and the error type.
//!
//! This crate deliberately has no async runtime, no NATS, and no tensor
//! dependency so that it compiles in a fraction of a second and can be used
//! from tests everywhere.

pub mod auth;
pub mod cli;
pub mod config;
pub mod decision;
pub mod error;
pub mod ids;
pub mod sampling;
pub mod script;

pub use auth::{ApiKey, AuthConfig, Principal};
pub use cli::Args;
pub use config::Config;
pub use decision::{
    Calibration, DecisionConfig, QuestionType, calibrated_probabilities, confidence, expected_level,
};
pub use error::{Error, Result};
pub use ids::{ModelId, RequestId, SeqId};
pub use sampling::{FinishReason, MAX_CHOICES, ResponseFormat, SamplingParams, StopCondition};
pub use script::{Reading, Script};
