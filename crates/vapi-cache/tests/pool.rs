//! Behavioural tests for the shared block pool.
//!
//! These run with no model, no device and no runtime, which is what makes the
//! cache developable on a laptop with no GPU.

use proptest::prelude::*;
use vapi_cache::{BlockPool, CacheNamespace, hash_block_chain};
use vapi_core::config::DType;

const BS: usize = 4; // small block size keeps the expectations readable

fn ns() -> CacheNamespace {
    CacheNamespace::new("test-model", "fingerprint", DType::F32, None, "global")
}

fn pool(n: usize) -> BlockPool {
    BlockPool::new(n, BS, 0)
}

#[test]
fn a_fresh_pool_is_entirely_free() {
    let p = pool(8);
    assert_eq!(p.num_free(), 8);
    assert_eq!(p.stats().utilization(), 0.0);
    p.check_invariants();
}

#[test]
fn allocation_and_release_round_trips() {
    let mut p = pool(4);
    let a = p.allocate().unwrap();
    let b = p.allocate().unwrap();
    assert_ne!(a, b, "the same block must never be handed out twice");
    assert_eq!(p.num_free(), 2);
    p.check_invariants();

    p.free(a);
    p.free(b);
    assert_eq!(p.num_free(), 4);
    p.check_invariants();
}

#[test]
fn exhaustion_is_an_error_not_a_panic() {
    let mut p = pool(2);
    p.allocate().unwrap();
    p.allocate().unwrap();
    assert!(p.allocate().is_err());
    p.check_invariants();
}

#[test]
fn refcounts_keep_a_shared_block_alive() {
    let mut p = pool(4);
    let a = p.allocate().unwrap();
    p.incref(a);
    assert_eq!(p.refcount(a), 2);

    p.free(a);
    assert_eq!(p.num_free(), 3, "still held by the second reference");
    p.free(a);
    assert_eq!(p.num_free(), 4);
    p.check_invariants();
}

#[test]
fn the_watermark_stops_admission_before_the_pool_empties() {
    // Reserve 2 of 4 blocks. Guarded allocation must refuse while plain
    // allocation (used for already-admitted sequences) still succeeds.
    let mut p = BlockPool::new(4, BS, 2);
    assert!(p.can_allocate(2));
    assert!(!p.can_allocate(3));

    p.allocate_guarded().unwrap();
    p.allocate_guarded().unwrap();
    assert!(p.allocate_guarded().is_err(), "must not breach the reserve");
    assert!(
        p.allocate().is_ok(),
        "a running sequence may still grow into it"
    );
    p.check_invariants();
}

// ------------------------------------------------------------ prefix cache

#[test]
fn a_second_request_with_the_same_prompt_hits_entirely() {
    let ns = ns();
    let mut p = pool(16);
    let tokens: Vec<u32> = (0..8).collect(); // exactly two blocks
    let hashes = hash_block_chain(&ns, &tokens, BS);

    // First request: nothing cached.
    let m = p.match_prefix(&hashes);
    assert_eq!(m.tokens, 0);

    // It computes and publishes its blocks.
    let mut blocks = Vec::new();
    for h in &hashes {
        let b = p.allocate().unwrap();
        blocks.push(p.publish(b, *h));
    }
    p.free_all(&blocks);
    p.check_invariants();

    // Second request with the identical prompt pays nothing.
    let m = p.match_prefix(&hashes);
    assert_eq!(m.tokens, 8, "whole prompt should come from cache");
    assert_eq!(m.blocks, blocks);
    p.check_invariants();
}

#[test]
fn two_users_sharing_a_system_prompt_share_its_blocks() {
    // The headline behaviour: user B's request reuses KV computed for user A.
    let ns = ns();
    let mut p = pool(16);
    let system: Vec<u32> = (100..108).collect(); // two blocks of shared prefix

    let mut a_tokens = system.clone();
    a_tokens.extend([1, 2, 3, 4]);
    let mut b_tokens = system.clone();
    b_tokens.extend([9, 9, 9, 9]);

    let a_hashes = hash_block_chain(&ns, &a_tokens, BS);
    let b_hashes = hash_block_chain(&ns, &b_tokens, BS);

    // User A runs first and publishes everything.
    let a_blocks: Vec<_> = a_hashes
        .iter()
        .map(|h| {
            let b = p.allocate().unwrap();
            p.publish(b, *h)
        })
        .collect();
    p.free_all(&a_blocks);

    // User B matches the shared system prompt and stops at the divergence.
    let m = p.match_prefix(&b_hashes);
    assert_eq!(m.tokens, system.len(), "shared system prompt is reused");
    assert_eq!(m.blocks, a_blocks[..2], "and it is literally A's blocks");
    p.check_invariants();
}

#[test]
fn matching_stops_at_the_first_miss() {
    // Prefix caching is only valid for a contiguous prefix: block k's state
    // depends on blocks 0..k, so a later match cannot be used after a gap.
    let ns = ns();
    let mut p = pool(16);
    let tokens: Vec<u32> = (0..12).collect(); // three blocks
    let hashes = hash_block_chain(&ns, &tokens, BS);

    // Publish only blocks 0 and 2, leaving a hole at 1.
    for h in [hashes[0], hashes[2]] {
        let b = p.allocate().unwrap();
        let _ = p.publish(b, h);
    }

    let m = p.match_prefix(&hashes);
    assert_eq!(m.blocks.len(), 1, "must not jump the gap to block 2");
    p.check_invariants();
}

#[test]
fn a_referenced_cached_block_is_never_evicted() {
    let ns = ns();
    let mut p = pool(2);
    let hashes = hash_block_chain(&ns, &[1, 2, 3, 4], BS);

    let held = p.allocate().unwrap();
    let held = p.publish(held, hashes[0]);

    // Drain the only other block, then demand more.
    let _other = p.allocate().unwrap();
    assert!(p.allocate().is_err(), "held blocks must not be stolen");
    assert_eq!(p.lookup(&hashes[0]), Some(held));
    p.check_invariants();
}

#[test]
fn unreferenced_cached_blocks_are_evicted_only_under_pressure() {
    let ns = ns();
    let mut p = pool(2);
    let h = hash_block_chain(&ns, &[1, 2, 3, 4], BS);

    let b = p.allocate().unwrap();
    let b = p.publish(b, h[0]);
    p.free(b);

    // Still cached while there is room elsewhere.
    assert!(p.lookup(&h[0]).is_some());
    p.free(b);

    // Force eviction by consuming the whole pool.
    let _x = p.allocate().unwrap();
    let _y = p.allocate().unwrap();
    assert_eq!(p.lookup(&h[0]), None, "evicted under pressure");
    assert_eq!(p.stats().evictions, 1);
    p.check_invariants();
}

#[test]
fn empty_blocks_are_reclaimed_before_cached_ones() {
    // A block holding a useful prefix should outlive a block holding nothing.
    let ns = ns();
    let mut p = pool(2);
    let h = hash_block_chain(&ns, &[1, 2, 3, 4], BS);

    let cached = p.allocate().unwrap();
    let cached = p.publish(cached, h[0]);
    let empty = p.allocate().unwrap();

    p.free(cached);
    p.free(empty);

    let taken = p.allocate().unwrap();
    assert_eq!(taken, empty, "the valueless block should go first");
    assert!(p.lookup(&h[0]).is_some(), "the cached prefix survived");
    p.check_invariants();
}

#[test]
fn concurrent_publishers_of_identical_content_converge() {
    // Two sequences prefill the same text before either publishes. The pool
    // must pick one winner rather than leaving two blocks claiming one hash.
    let ns = ns();
    let mut p = pool(8);
    let h = hash_block_chain(&ns, &[1, 2, 3, 4], BS);

    let first = p.allocate().unwrap();
    let second = p.allocate().unwrap();

    let canon_a = p.publish(first, h[0]);
    let canon_b = p.publish(second, h[0]);
    assert_eq!(canon_a, first);
    assert_eq!(canon_b, first, "second publisher adopts the winner's block");

    // The loser frees its now-redundant block.
    p.free(second);
    p.check_invariants();
    assert_eq!(p.stats().cached_blocks, 1);
}

#[test]
fn hit_rate_reflects_actual_reuse() {
    let ns = ns();
    let mut p = pool(16);
    let hashes = hash_block_chain(&ns, &(0..8).collect::<Vec<u32>>(), BS);

    p.match_prefix(&hashes); // 8 queried, 0 hit
    let blocks: Vec<_> = hashes
        .iter()
        .map(|h| {
            let b = p.allocate().unwrap();
            p.publish(b, *h)
        })
        .collect();
    p.free_all(&blocks);
    p.match_prefix(&hashes); // 8 queried, 8 hit

    assert_eq!(p.stats().hit_rate(), 0.5);
}

// ------------------------------------------------------------- properties

#[derive(Debug, Clone)]
enum Op {
    Alloc,
    Free(usize),
    Publish(usize, u8),
    Lookup(u8),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => Just(Op::Alloc),
        3 => (0usize..8).prop_map(Op::Free),
        2 => (0usize..8, 0u8..6).prop_map(|(i, h)| Op::Publish(i, h)),
        2 => (0u8..6).prop_map(Op::Lookup),
    ]
}

proptest! {
    /// The pool must never lose or duplicate a block, whatever order it is
    /// driven in. Leaked blocks are the failure mode that shows up as a server
    /// that slowly stops accepting traffic after days of uptime.
    #[test]
    fn pool_invariants_hold_under_arbitrary_traces(ops in prop::collection::vec(op_strategy(), 0..200)) {
        let ns = ns();
        let hashes = hash_block_chain(&ns, &(0..24).collect::<Vec<u32>>(), BS);
        let mut p = BlockPool::new(8, BS, 0);
        let mut live: Vec<vapi_cache::BlockId> = Vec::new();

        for op in ops {
            match op {
                Op::Alloc => {
                    if let Ok(b) = p.allocate() {
                        live.push(b);
                    }
                }
                Op::Free(i) => {
                    if !live.is_empty() {
                        let b = live.remove(i % live.len());
                        p.free(b);
                    }
                }
                Op::Publish(i, h) => {
                    if !live.is_empty() {
                        let idx = i % live.len();
                        let b = live[idx];
                        let canon = p.publish(b, hashes[h as usize % hashes.len()]);
                        if canon != b {
                            // Adopted someone else's block: we now hold a
                            // reference to `canon` and must release `b`.
                            live[idx] = canon;
                            p.free(b);
                        }
                    }
                }
                Op::Lookup(h) => {
                    if let Some(b) = p.lookup(&hashes[h as usize % hashes.len()]) {
                        live.push(b);
                    }
                }
            }
            p.check_invariants();
        }

        // Releasing everything must return the pool to pristine.
        for b in live {
            p.free(b);
        }
        p.check_invariants();
        prop_assert_eq!(p.num_free(), p.num_blocks());
    }
}
