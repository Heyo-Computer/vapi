//! The audio frontend: PCM in, log-mel frames out.
//!
//! Deliberately dependency-free and free of tensors. It is the one part of
//! the speech path that has to agree exactly with a Python reference, and the
//! easiest place to hide a disagreement is inside a library that resamples or
//! windows slightly differently.
//!
//! The pipeline is Whisper's, with one change that matters for streaming: the
//! log-mel floor is a **global** constant rather than the maximum of the
//! current utterance. A per-utterance maximum cannot be computed from a
//! stream that has not finished, and using one would make the first second of
//! a live transcription depend on the last.

pub mod mel;
pub mod resample;

pub use mel::{MelSettings, MelSpectrogram};
pub use resample::{resample_linear, to_mono};
