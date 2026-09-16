//! The candle-backed execution backend: where vapi meets real tensors.
//!
//! Nothing here is on by default. Build with `--features candle` for the CPU
//! path, or `--features cuda` on a GPU box.
//!
//! # Where this picks up
//!
//! The engine is already complete and tested against [`vapi_engine::MockBackend`].
//! What remains is implementing [`vapi_engine::ExecutionBackend`] for real
//! weights — see [`attention`] for the part that actually matters.

#[cfg(feature = "candle")]
pub mod attention;
#[cfg(feature = "candle")]
pub mod cache;

/// Bytes per element for a dtype, used to turn a VRAM budget into a block
/// count. Available without the `candle` feature so sizing can be reasoned
/// about (and tested) on any machine.
pub fn dtype_bytes(dtype: vapi_core::config::DType) -> usize {
    use vapi_core::config::DType;
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        // `Auto` resolves to bf16 on CUDA and f32 on CPU; assume the GPU case,
        // since that is where the sizing arithmetic matters.
        DType::Auto => 2,
    }
}

/// How many KV blocks fit in a memory budget.
///
/// Worth keeping separate from the device code: getting this wrong means
/// either wasting VRAM or OOMing partway through the first large batch, and it
/// is pure arithmetic that deserves a test.
pub fn blocks_for_budget(
    budget_bytes: u64,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: vapi_core::config::DType,
    block_size: usize,
) -> usize {
    // K and V, per layer, per token.
    let per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes(dtype);
    let per_block = per_token * block_size;
    if per_block == 0 {
        return 0;
    }
    (budget_bytes / per_block as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use vapi_core::config::{BLOCK_SIZE, DType};

    #[test]
    fn block_sizing_matches_hand_arithmetic() {
        // Llama-3.2-1B-ish: 16 layers, 8 KV heads, head_dim 64, bf16.
        // per token = 2 * 16 * 8 * 64 * 2 = 32768 bytes = 32 KiB
        // per block = 32 KiB * 32 = 1 MiB
        let blocks = blocks_for_budget(1 << 30, 16, 8, 64, DType::Bf16, BLOCK_SIZE);
        assert_eq!(blocks, 1024, "1 GiB should hold 1024 blocks of 1 MiB");
    }

    #[test]
    fn f32_halves_the_block_count() {
        let bf16 = blocks_for_budget(1 << 30, 16, 8, 64, DType::Bf16, BLOCK_SIZE);
        let f32 = blocks_for_budget(1 << 30, 16, 8, 64, DType::F32, BLOCK_SIZE);
        assert_eq!(bf16, f32 * 2);
    }

    #[test]
    fn a_budget_too_small_for_one_block_yields_none() {
        assert_eq!(
            blocks_for_budget(1024, 16, 8, 64, DType::Bf16, BLOCK_SIZE),
            0
        );
    }
}
