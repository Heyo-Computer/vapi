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
pub mod backend;
#[cfg(feature = "candle")]
pub mod cache;
#[cfg(feature = "candle")]
pub mod encoder;
#[cfg(feature = "candle")]
pub mod fused;
#[cfg(feature = "candle")]
pub mod models;
#[cfg(feature = "candle")]
pub mod rows;

#[cfg(feature = "candle")]
pub use backend::{CandleBackend, LoadOptions};
#[cfg(feature = "candle")]
pub use encoder::{CandleEncoder, EncoderLoadOptions};

use std::path::Path;

/// A stable fingerprint of the weights in a model directory, for the cache
/// namespace.
///
/// Hashes `config.json`, the safetensors index, every `*.safetensors` header
/// (tensor names, dtypes, shapes and offsets) together with its file size, and
/// `tokenizer.json`. That changes whenever the architecture, the checkpoint
/// layout or the vocabulary changes, which is what decides whether KV computed
/// earlier is still the same tensor — without reading gigabytes of weights on
/// every start. Available without the `candle` feature so the worker can
/// namespace a mock run that borrows a real tokenizer.
///
/// Returns `"unversioned"` when the directory holds none of those files.
pub fn weights_fingerprint(dir: &Path) -> String {
    let mut hasher = blake3::Hasher::new();
    let mut saw_anything = false;
    let mut feed = |name: &str, bytes: &[u8]| {
        hasher.update(name.as_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        saw_anything = true;
    };

    for name in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ] {
        if let Ok(bytes) = std::fs::read(dir.join(name)) {
            feed(name, &bytes);
        }
    }

    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    for path in shards {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Ok(mut f) = std::fs::File::open(&path) else {
            continue;
        };
        use std::io::Read;
        let size = f.metadata().map(|m| m.len()).unwrap_or(0);
        let mut len = [0u8; 8];
        if f.read_exact(&mut len).is_err() {
            continue;
        }
        let header_len = u64::from_le_bytes(len).min(64 << 20) as usize;
        let mut header = vec![0u8; header_len];
        if f.read_exact(&mut header).is_err() {
            continue;
        }
        feed(name, &size.to_le_bytes());
        feed(name, &header);
    }

    if !saw_anything {
        return "unversioned".into();
    }
    hasher.finalize().to_hex()[..16].to_string()
}

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
    fn the_fingerprint_tracks_the_files_that_define_the_weights() {
        let dir = std::env::temp_dir().join(format!("vapi-fp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(weights_fingerprint(&dir), "unversioned");

        std::fs::write(dir.join("config.json"), b"{\"hidden_size\": 8}").unwrap();
        let a = weights_fingerprint(&dir);
        assert_ne!(a, "unversioned");
        assert_eq!(a, weights_fingerprint(&dir), "must be stable across calls");

        // A safetensors file: 8-byte little-endian header length, then the
        // JSON header, then the data.
        let header = b"{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&[0u8; 8]);
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();
        let b = weights_fingerprint(&dir);
        assert_ne!(a, b, "adding a checkpoint changes the fingerprint");

        std::fs::write(dir.join("config.json"), b"{\"hidden_size\": 16}").unwrap();
        let c = weights_fingerprint(&dir);
        assert_ne!(b, c, "a config change changes the fingerprint");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_budget_too_small_for_one_block_yields_none() {
        assert_eq!(
            blocks_for_budget(1024, 16, 8, 64, DType::Bf16, BLOCK_SIZE),
            0
        );
    }
}
