//! Tier 1 of vapi's caching: the paged KV block pool and the prefix cache
//! shared across every user.
//!
//! This crate is pure data structures — no tensors, no device, no runtime — so
//! the allocator and the sharing logic can be property-tested in milliseconds
//! on a machine with no GPU. The tensor-side counterpart (the actual K/V
//! buffers the block ids index into) lives in the backend crate.

pub mod hash;
pub mod pool;
pub mod spill;

pub use hash::{BlockHash, CacheNamespace, hash_block_chain};
pub use pool::{AllocError, BlockId, BlockPool, PoolStats, PrefixMatch};
pub use spill::{SpillStats, SpillStore};

/// Tokens per block.
///
/// Re-exported from `vapi-core` so callers of this crate get the constant and
/// the reasoning together: it is 32 because `candle-flash-attn`'s paged kernel
/// rejects a `page_block_size` that is not a multiple of 32.
pub use vapi_core::config::BLOCK_SIZE;
