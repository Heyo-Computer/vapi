use std::collections::HashMap;

use crate::hash::BlockHash;

/// Index into the device-side KV cache tensors. A block id `b` addresses
/// `[b * BLOCK_SIZE .. (b + 1) * BLOCK_SIZE)` slots of every layer's K and V
/// buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AllocError {
    #[error("no free KV blocks")]
    Exhausted,
    #[error("allocation of {want} blocks would breach the {watermark}-block reserve")]
    Watermark { want: usize, watermark: usize },
}

/// Result of matching a prompt against the shared cache.
///
/// The returned blocks have already been referenced on the caller's behalf. If
/// the caller then fails to admit the sequence it **must** release them, or
/// they leak and the pool slowly starves.
#[derive(Debug, Clone, Default)]
pub struct PrefixMatch {
    pub blocks: Vec<BlockId>,
    /// Prompt tokens covered, i.e. `blocks.len() * BLOCK_SIZE`. These need no
    /// prefill compute at all.
    pub tokens: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    pub num_blocks: usize,
    pub num_free: usize,
    pub cached_blocks: usize,
    pub hit_tokens: u64,
    pub queried_tokens: u64,
    pub evictions: u64,
}

impl PoolStats {
    /// Share of queried prompt tokens served from cache, 0.0..=1.0.
    pub fn hit_rate(&self) -> f32 {
        if self.queried_tokens == 0 {
            return 0.0;
        }
        self.hit_tokens as f32 / self.queried_tokens as f32
    }

    pub fn utilization(&self) -> f32 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.num_free as f32 / self.num_blocks as f32)
    }
}

const NONE: u32 = u32::MAX;

/// Refcounted pool of paged KV blocks with a content-addressed index over
/// them.
///
/// The allocator and the prefix index are one structure on purpose. They share
/// an invariant — a cached block is reusable exactly while it is unreferenced
/// but not yet handed out — and splitting them into two types with two locks
/// is how that invariant gets violated under preemption.
///
/// Free blocks live in an intrusive doubly-linked LRU list, so reviving a
/// cache hit from the middle of the list is O(1). Blocks freed with no cached
/// content go to the *front* (evicted first, they hold nothing of value);
/// blocks holding cached KV go to the *back*, so genuinely useful prefixes
/// survive a burst of churn.
pub struct BlockPool {
    block_size: usize,
    watermark: usize,
    refcount: Vec<u32>,
    hash_of: Vec<Option<BlockHash>>,
    by_hash: HashMap<BlockHash, BlockId>,
    prev: Vec<u32>,
    next: Vec<u32>,
    lru_head: u32,
    lru_tail: u32,
    num_free: usize,
    hit_tokens: u64,
    queried_tokens: u64,
    evictions: u64,
}

impl BlockPool {
    pub fn new(num_blocks: usize, block_size: usize, watermark: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        let n = num_blocks;
        let mut pool = Self {
            block_size,
            watermark: watermark.min(n),
            refcount: vec![0; n],
            hash_of: vec![None; n],
            by_hash: HashMap::new(),
            prev: vec![NONE; n],
            next: vec![NONE; n],
            lru_head: NONE,
            lru_tail: NONE,
            num_free: 0,
            hit_tokens: 0,
            queried_tokens: 0,
            evictions: 0,
        };
        // Seed the free list in ascending order so a fresh pool hands out
        // blocks predictably, which keeps test expectations readable.
        for i in (0..n).rev() {
            pool.push_front(i as u32);
            pool.num_free += 1;
        }
        pool
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.refcount.len()
    }

    pub fn num_free(&self) -> usize {
        self.num_free
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            num_blocks: self.num_blocks(),
            num_free: self.num_free,
            cached_blocks: self.by_hash.len(),
            hit_tokens: self.hit_tokens,
            queried_tokens: self.queried_tokens,
            evictions: self.evictions,
        }
    }

    /// Blocks needed to hold `tokens`, including a partial trailing block.
    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }

    /// Whether `want` blocks can be admitted without eating into the reserve.
    ///
    /// The reserve exists so the scheduler stops admitting *before* the pool
    /// is empty: a running sequence that cannot grow by one block must be
    /// preempted, and preemption is far more expensive than making a waiting
    /// sequence wait.
    pub fn can_allocate(&self, want: usize) -> bool {
        self.num_free >= want + self.watermark
    }

    /// Take a block with no cached content, evicting the least useful free
    /// block if necessary.
    pub fn allocate(&mut self) -> Result<BlockId, AllocError> {
        if self.num_free == 0 {
            return Err(AllocError::Exhausted);
        }
        let id = self.lru_head;
        debug_assert_ne!(id, NONE, "num_free > 0 implies a non-empty LRU list");
        self.unlink(id);
        self.num_free -= 1;
        if let Some(h) = self.hash_of[id as usize].take() {
            self.by_hash.remove(&h);
            self.evictions += 1;
        }
        self.refcount[id as usize] = 1;
        Ok(BlockId(id))
    }

    /// Like [`allocate`](Self::allocate) but refuses to breach the reserve.
    pub fn allocate_guarded(&mut self) -> Result<BlockId, AllocError> {
        if !self.can_allocate(1) {
            return Err(AllocError::Watermark {
                want: 1,
                watermark: self.watermark,
            });
        }
        self.allocate()
    }

    pub fn incref(&mut self, id: BlockId) {
        let i = id.index();
        debug_assert!(self.refcount[i] > 0, "incref on a free block {id:?}");
        self.refcount[i] += 1;
    }

    /// Release one reference. When the last goes, the block becomes reusable:
    /// either as cached content (if it has a published hash) or as raw space.
    pub fn free(&mut self, id: BlockId) {
        let i = id.index();
        debug_assert!(self.refcount[i] > 0, "double free of block {id:?}");
        self.refcount[i] -= 1;
        if self.refcount[i] == 0 {
            // Cached blocks are worth keeping, so they go to the eviction
            // tail; empty ones are reclaimed first.
            if self.hash_of[i].is_some() {
                self.push_back(id.0);
            } else {
                self.push_front(id.0);
            }
            self.num_free += 1;
        }
    }

    pub fn refcount(&self, id: BlockId) -> u32 {
        self.refcount[id.index()]
    }

    /// Publish a filled block's content hash so other sequences can find it.
    ///
    /// Only call this for a block filled to exactly `block_size` tokens — a
    /// partial block will receive more tokens, and publishing it would let
    /// another sequence match a prefix that does not exist yet.
    ///
    /// Returns the canonical block for this content. If two sequences prefill
    /// identical content concurrently, the first publisher wins and this
    /// returns *its* block; the caller should switch to the returned id and
    /// free the one it passed in.
    #[must_use = "the returned block may differ from the one passed in"]
    pub fn publish(&mut self, id: BlockId, hash: BlockHash) -> BlockId {
        if let Some(&existing) = self.by_hash.get(&hash)
            && existing != id
        {
            // Someone got here first. Adopt theirs.
            self.reference(existing);
            return existing;
        }
        // A block carries one hash for the life of its contents. Re-publishing
        // a different hash onto it means the block was refilled, so retire the
        // old index entry — leaving it would point the index at a block whose
        // contents no longer match, which is silent corruption rather than a
        // miss.
        if let Some(stale) = self.hash_of[id.index()].replace(hash)
            && stale != hash
        {
            self.by_hash.remove(&stale);
        }
        self.by_hash.insert(hash, id);
        id
    }

    /// Take a reference to a block that may currently be free-but-cached,
    /// reviving it out of the LRU list if so.
    fn reference(&mut self, id: BlockId) {
        let i = id.index();
        if self.refcount[i] == 0 {
            self.unlink(id.0);
            self.num_free -= 1;
        }
        self.refcount[i] += 1;
    }

    /// Look up one cached block by hash, referencing it on success.
    pub fn lookup(&mut self, hash: &BlockHash) -> Option<BlockId> {
        let id = *self.by_hash.get(hash)?;
        self.reference(id);
        Some(id)
    }

    /// Match the longest cached prefix of a hashed token sequence.
    ///
    /// Stops at the first miss: prefix caching is only valid for a *contiguous*
    /// prefix, since block `k`'s attention state depends on blocks `0..k`.
    /// Every returned block is referenced; release them if admission fails.
    pub fn match_prefix(&mut self, hashes: &[BlockHash]) -> PrefixMatch {
        self.queried_tokens += (hashes.len() * self.block_size) as u64;
        let mut blocks = Vec::new();
        for h in hashes {
            match self.lookup(h) {
                Some(id) => blocks.push(id),
                None => break,
            }
        }
        self.hit_tokens += (blocks.len() * self.block_size) as u64;
        PrefixMatch {
            tokens: blocks.len() * self.block_size,
            blocks,
        }
    }

    /// Release a batch of blocks, e.g. when admission fails after a match.
    pub fn free_all(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            self.free(b);
        }
    }

    // ------------------------------------------------------- LRU plumbing

    fn push_front(&mut self, id: u32) {
        let i = id as usize;
        self.prev[i] = NONE;
        self.next[i] = self.lru_head;
        if self.lru_head != NONE {
            self.prev[self.lru_head as usize] = id;
        } else {
            self.lru_tail = id;
        }
        self.lru_head = id;
    }

    fn push_back(&mut self, id: u32) {
        let i = id as usize;
        self.next[i] = NONE;
        self.prev[i] = self.lru_tail;
        if self.lru_tail != NONE {
            self.next[self.lru_tail as usize] = id;
        } else {
            self.lru_head = id;
        }
        self.lru_tail = id;
    }

    fn unlink(&mut self, id: u32) {
        let i = id as usize;
        let (p, n) = (self.prev[i], self.next[i]);
        if p != NONE {
            self.next[p as usize] = n;
        } else {
            self.lru_head = n;
        }
        if n != NONE {
            self.prev[n as usize] = p;
        } else {
            self.lru_tail = p;
        }
        self.prev[i] = NONE;
        self.next[i] = NONE;
    }

    /// Structural invariants, asserted by tests after every operation.
    #[doc(hidden)]
    pub fn check_invariants(&self) {
        let mut seen = vec![false; self.num_blocks()];
        let mut walked = 0usize;
        let mut cur = self.lru_head;
        let mut last = NONE;
        while cur != NONE {
            let i = cur as usize;
            assert!(!seen[i], "block {i} appears twice in the free list");
            seen[i] = true;
            assert_eq!(
                self.refcount[i], 0,
                "referenced block {i} is in the free list"
            );
            assert_eq!(self.prev[i], last, "broken prev link at {i}");
            last = cur;
            cur = self.next[i];
            walked += 1;
        }
        assert_eq!(self.lru_tail, last, "tail does not match the walked list");
        assert_eq!(
            walked, self.num_free,
            "num_free disagrees with the list length"
        );

        for (i, &rc) in self.refcount.iter().enumerate() {
            assert_eq!(rc == 0, seen[i], "block {i} refcount/free-list mismatch");
        }
        for (h, &id) in &self.by_hash {
            assert_eq!(
                self.hash_of[id.index()].as_ref(),
                Some(h),
                "index points at block {id:?} which does not carry that hash"
            );
        }
        let carried = self.hash_of.iter().filter(|h| h.is_some()).count();
        assert_eq!(carried, self.by_hash.len(), "orphaned hash on a block");
    }
}
