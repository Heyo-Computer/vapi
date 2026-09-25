//! Model architectures adapted to the paged, continuous-batched forward pass.
//!
//! Each file here starts life as a copy of the stock `candle-transformers`
//! implementation and changes exactly two things: the per-layer KV cache
//! becomes [`crate::cache::PagedKvCache`], and `forward` takes a
//! [`vapi_engine::ForwardBatch`] instead of `(x, index_pos)` so sequences at
//! different positions can share one batch.

pub mod laguna;
pub mod laya;
pub mod lfm2;
pub mod llama;
pub mod modernbert;
pub mod qwen;
pub mod voxtral;
