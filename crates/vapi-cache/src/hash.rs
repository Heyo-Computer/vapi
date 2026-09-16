use std::fmt;

use vapi_core::config::DType;

/// Content address of a KV block *and its entire prefix*.
///
/// The chain is what makes this work: a block's hash mixes in its parent's
/// hash, so two sequences share a block only when every token before it is
/// also identical. A bare hash of the block's own 32 tokens would happily
/// match the same sentence appearing after two different system prompts, and
/// would serve one user attention state computed from another user's context.
///
/// 256 bits, not a truncated 64-bit hash. A collision here does not cause a
/// miss, it silently serves the wrong KV — the output is subtly wrong with no
/// error anywhere — so the extra 24 bytes per block are worth it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash([u8; 32]);

impl BlockHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Short form for logs and metric labels.
    pub fn short(&self) -> String {
        let mut s = String::with_capacity(12);
        for b in &self.0[..6] {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

impl fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockHash({})", self.short())
    }
}

/// Everything besides the tokens that must match for two blocks to be
/// interchangeable.
///
/// Omitting any of these is a silent-corruption bug rather than a performance
/// bug, which is why they are a required constructor argument rather than
/// optional fields: KV computed by a different model, at a different
/// precision, or with a different adapter is simply not the same tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheNamespace {
    root: BlockHash,
    label: String,
}

impl CacheNamespace {
    pub fn new(
        model_id: &str,
        weights_fingerprint: &str,
        dtype: DType,
        lora_id: Option<&str>,
        tenant: &str,
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"vapi-cache-ns-v1");
        for part in [model_id, weights_fingerprint, lora_id.unwrap_or(""), tenant] {
            // Length-prefix each field so ("ab","c") cannot hash the same as
            // ("a","bc").
            h.update(&(part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
        h.update(&[dtype as u8]);
        Self {
            root: BlockHash(*h.finalize().as_bytes()),
            label: tenant.to_string(),
        }
    }

    /// Hash of the empty prefix; the parent of a sequence's first block.
    pub fn root(&self) -> BlockHash {
        self.root
    }

    /// Tenant label, for metrics and logging.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Hash one block given its parent's hash and the tokens it holds.
///
/// Callers must only hash *full* blocks. A partially-filled block will receive
/// more tokens, so publishing its hash would let another sequence match a
/// prefix that does not exist yet.
pub fn hash_block(parent: BlockHash, tokens: &[u32]) -> BlockHash {
    let mut h = blake3::Hasher::new();
    h.update(b"vapi-block-v1");
    h.update(parent.as_bytes());
    h.update(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    BlockHash(*h.finalize().as_bytes())
}

/// Hash every full block of a token sequence, in order.
///
/// A trailing partial block is deliberately not hashed; the returned vector is
/// therefore `tokens.len() / block_size` long.
pub fn hash_block_chain(ns: &CacheNamespace, tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    let mut out = Vec::with_capacity(tokens.len() / block_size);
    let mut parent = ns.root();
    for chunk in tokens.chunks_exact(block_size) {
        parent = hash_block(parent, chunk);
        out.push(parent);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(tenant: &str) -> CacheNamespace {
        CacheNamespace::new("llama", "abc123", DType::F32, None, tenant)
    }

    #[test]
    fn identical_prefixes_hash_identically() {
        let a = hash_block_chain(&ns("global"), &[1, 2, 3, 4], 2);
        let b = hash_block_chain(&ns("global"), &[1, 2, 3, 4], 2);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn a_shared_block_after_different_prefixes_does_not_match() {
        // This is the entire point of chaining. Block [9,9] appears in both,
        // but preceded by different context, so its KV state differs.
        let a = hash_block_chain(&ns("global"), &[1, 1, 9, 9], 2);
        let b = hash_block_chain(&ns("global"), &[2, 2, 9, 9], 2);
        assert_ne!(a[1], b[1]);
    }

    #[test]
    fn common_prefix_matches_then_diverges() {
        let a = hash_block_chain(&ns("global"), &[1, 2, 3, 4, 5, 6], 2);
        let b = hash_block_chain(&ns("global"), &[1, 2, 3, 4, 7, 8], 2);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
    }

    #[test]
    fn tenants_are_isolated_when_they_ask_to_be() {
        let a = hash_block_chain(&ns("acme"), &[1, 2], 2);
        let b = hash_block_chain(&ns("globex"), &[1, 2], 2);
        assert_ne!(a, b, "separate namespaces must not share blocks");
    }

    #[test]
    fn model_dtype_and_adapter_all_partition_the_cache() {
        let toks = [1u32, 2];
        let base = CacheNamespace::new("llama", "w1", DType::F32, None, "g");
        let other_model = CacheNamespace::new("qwen", "w1", DType::F32, None, "g");
        let other_weights = CacheNamespace::new("llama", "w2", DType::F32, None, "g");
        let other_dtype = CacheNamespace::new("llama", "w1", DType::Bf16, None, "g");
        let with_lora = CacheNamespace::new("llama", "w1", DType::F32, Some("l1"), "g");

        let h = |n: &CacheNamespace| hash_block_chain(n, &toks, 2);
        let b = h(&base);
        for other in [&other_model, &other_weights, &other_dtype, &with_lora] {
            assert_ne!(b, h(other));
        }
    }

    #[test]
    fn field_boundaries_cannot_be_smeared() {
        // Without length-prefixing, ("ab","c") and ("a","bc") would collide.
        let x = CacheNamespace::new("ab", "c", DType::F32, None, "g");
        let y = CacheNamespace::new("a", "bc", DType::F32, None, "g");
        assert_ne!(x.root(), y.root());
    }

    #[test]
    fn partial_trailing_block_is_not_hashed() {
        // 5 tokens at block size 2 -> two full blocks, one token left over.
        let h = hash_block_chain(&ns("global"), &[1, 2, 3, 4, 5], 2);
        assert_eq!(h.len(), 2, "a partial block must never be published");
    }
}
