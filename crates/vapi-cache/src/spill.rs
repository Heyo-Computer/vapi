//! Tier 3: evicted KV blocks kept outside the device.
//!
//! When the pool evicts a cached block the tokens behind it are still in
//! somebody's conversation, and the next turn will ask for them again. This
//! keeps the block's bytes in host memory, and past a cap on disk, so the
//! next request can copy them back instead of recomputing the prefill.
//!
//! **Whether that is a win is a measurement, not an assumption.** Copying a
//! block back costs a host-to-device transfer of `block_bytes`; recomputing
//! it costs a prefill of `block_size` tokens. On a small model with a fast
//! GPU the prefill can be cheaper, in which case this tier should stay off.
//! `vapi-worker` records `vapi_spill_import_ms` against
//! `vapi_step_forward_ms` so the answer is visible on a running system, and
//! `benchmark/README.md` carries the measurement for LFM2.5 on a 5060 Ti.
//!
//! Entries are keyed by the same content hash the prefix cache uses, which
//! already folds in the model id, its weight fingerprint and the tenant
//! namespace. Two models therefore cannot collide, and a file left on disk
//! by a previous run of the same model is still valid, which is what makes
//! the disk tier survive a restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::hash::BlockHash;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillStats {
    pub ram_blocks: usize,
    pub ram_bytes: u64,
    pub disk_blocks: usize,
    pub disk_bytes: u64,
    /// Blocks handed back to the device.
    pub hits: u64,
    /// Blocks asked for that were not held.
    pub misses: u64,
    pub writes: u64,
    /// Blocks dropped entirely because both tiers were full.
    pub drops: u64,
    pub errors: u64,
}

/// Host-memory tier with an optional disk tier behind it, both LRU.
pub struct SpillStore {
    ram: HashMap<BlockHash, Vec<u8>>,
    /// Least recently used first.
    ram_lru: Vec<BlockHash>,
    ram_max: u64,
    ram_bytes: u64,
    dir: Option<PathBuf>,
    disk: HashMap<BlockHash, u64>,
    disk_lru: Vec<BlockHash>,
    disk_max: u64,
    disk_bytes: u64,
    stats: SpillStats,
}

impl SpillStore {
    /// `dir` is created if missing; any blocks already in it are indexed and
    /// remain usable, since a hash pins the model and its weights.
    pub fn open(dir: Option<&Path>, ram_max: u64, disk_max: u64) -> std::io::Result<Self> {
        let mut store = Self {
            ram: HashMap::new(),
            ram_lru: Vec::new(),
            ram_max,
            ram_bytes: 0,
            dir: None,
            disk: HashMap::new(),
            disk_lru: Vec::new(),
            disk_max,
            disk_bytes: 0,
            stats: SpillStats::default(),
        };
        if let Some(dir) = dir.filter(|_| disk_max > 0) {
            std::fs::create_dir_all(dir)?;
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                let Some(hash) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(BlockHash::from_hex)
                else {
                    continue;
                };
                let len = entry.metadata()?.len();
                store.disk.insert(hash, len);
                store.disk_lru.push(hash);
                store.disk_bytes += len;
            }
            store.dir = Some(dir.to_path_buf());
            store.evict_disk();
        }
        Ok(store)
    }

    pub fn stats(&self) -> SpillStats {
        SpillStats {
            ram_blocks: self.ram.len(),
            ram_bytes: self.ram_bytes,
            disk_blocks: self.disk.len(),
            disk_bytes: self.disk_bytes,
            ..self.stats
        }
    }

    pub fn contains(&self, hash: &BlockHash) -> bool {
        self.ram.contains_key(hash) || self.disk.contains_key(hash)
    }

    /// Store one block's bytes. Silently drops it when it cannot fit
    /// anywhere: this tier is an optimisation and must never fail a request.
    pub fn put(&mut self, hash: BlockHash, bytes: Vec<u8>) {
        if self.ram.contains_key(&hash) || self.disk.contains_key(&hash) {
            return;
        }
        let len = bytes.len() as u64;
        if len > self.ram_max && len > self.disk_max {
            self.stats.drops += 1;
            return;
        }
        self.stats.writes += 1;
        self.ram_bytes += len;
        self.ram.insert(hash, bytes);
        self.ram_lru.push(hash);
        self.evict_ram();
    }

    /// Take a block's bytes back, if held. A disk hit is not promoted into
    /// host memory: the caller is about to put it on the device anyway.
    pub fn get(&mut self, hash: &BlockHash) -> Option<Vec<u8>> {
        if let Some(bytes) = self.ram.get(hash).cloned() {
            self.touch_ram(hash);
            self.stats.hits += 1;
            return Some(bytes);
        }
        if self.disk.contains_key(hash) {
            match std::fs::read(self.path(hash)) {
                Ok(bytes) => {
                    self.touch_disk(hash);
                    self.stats.hits += 1;
                    return Some(bytes);
                }
                Err(_) => {
                    // Deleted under us, or unreadable. Forget it.
                    self.forget_disk(hash);
                    self.stats.errors += 1;
                }
            }
        }
        self.stats.misses += 1;
        None
    }

    fn path(&self, hash: &BlockHash) -> PathBuf {
        self.dir
            .as_ref()
            .expect("a disk entry implies a directory")
            .join(format!("{}.kv", hash.to_hex()))
    }

    /// Push host-memory entries out to disk until the cap is met.
    fn evict_ram(&mut self) {
        while self.ram_bytes > self.ram_max && !self.ram_lru.is_empty() {
            let hash = self.ram_lru.remove(0);
            let Some(bytes) = self.ram.remove(&hash) else {
                continue;
            };
            self.ram_bytes -= bytes.len() as u64;
            if self.dir.is_none() || bytes.len() as u64 > self.disk_max {
                self.stats.drops += 1;
                continue;
            }
            match std::fs::write(self.path(&hash), &bytes) {
                Ok(()) => {
                    self.disk_bytes += bytes.len() as u64;
                    self.disk.insert(hash, bytes.len() as u64);
                    self.disk_lru.push(hash);
                    self.evict_disk();
                }
                Err(_) => self.stats.errors += 1,
            }
        }
    }

    fn evict_disk(&mut self) {
        while self.disk_bytes > self.disk_max && !self.disk_lru.is_empty() {
            let hash = self.disk_lru.remove(0);
            let _ = std::fs::remove_file(self.path(&hash));
            if let Some(len) = self.disk.remove(&hash) {
                self.disk_bytes -= len;
            }
        }
    }

    fn forget_disk(&mut self, hash: &BlockHash) {
        if let Some(len) = self.disk.remove(hash) {
            self.disk_bytes -= len;
        }
        self.disk_lru.retain(|h| h != hash);
    }

    fn touch_ram(&mut self, hash: &BlockHash) {
        self.ram_lru.retain(|h| h != hash);
        self.ram_lru.push(*hash);
    }

    fn touch_disk(&mut self, hash: &BlockHash) {
        self.disk_lru.retain(|h| h != hash);
        self.disk_lru.push(*hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{CacheNamespace, hash_block_chain};

    fn hashes(n: usize) -> Vec<BlockHash> {
        let ns = CacheNamespace::new("m", "fp", vapi_core::config::DType::F32, None, "global");
        let tokens: Vec<u32> = (0..n as u32 * 2).collect();
        hash_block_chain(&ns, &tokens, 2)
    }

    #[test]
    fn host_memory_only_keeps_the_most_recently_used() {
        let h = hashes(4);
        // Room for two 10-byte blocks, no disk.
        let mut s = SpillStore::open(None, 20, 0).unwrap();
        s.put(h[0], vec![0u8; 10]);
        s.put(h[1], vec![1u8; 10]);
        assert_eq!(s.get(&h[0]).unwrap()[0], 0);
        // h[0] is now the most recent, so h[1] is dropped for h[2].
        s.put(h[2], vec![2u8; 10]);
        assert!(s.get(&h[0]).is_some());
        assert!(s.get(&h[1]).is_none());
        assert!(s.get(&h[2]).is_some());
        let st = s.stats();
        assert_eq!(st.ram_blocks, 2);
        assert_eq!(st.drops, 1, "no disk tier, so the evicted block is dropped");
    }

    #[test]
    fn blocks_pushed_out_of_memory_are_readable_from_disk_and_survive_reopening() {
        let dir = std::env::temp_dir().join(format!("vapi-spill-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = hashes(4);
        {
            let mut s = SpillStore::open(Some(&dir), 10, 1 << 20).unwrap();
            s.put(h[0], vec![7u8; 10]);
            s.put(h[1], vec![8u8; 10]);
            // h[0] was pushed to disk, h[1] is in memory.
            assert_eq!(s.stats().ram_blocks, 1);
            assert_eq!(s.stats().disk_blocks, 1);
            assert_eq!(s.get(&h[0]).unwrap(), vec![7u8; 10]);
        }
        // A fresh store indexes what is on disk: hot prefixes survive a
        // restart, which is the point of the tier.
        let mut s = SpillStore::open(Some(&dir), 10, 1 << 20).unwrap();
        assert!(s.contains(&h[0]));
        assert_eq!(s.get(&h[0]).unwrap(), vec![7u8; 10]);
        assert!(s.get(&h[3]).is_none());
        assert_eq!(s.stats().misses, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_disk_cap_is_enforced() {
        let dir = std::env::temp_dir().join(format!("vapi-spill-cap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = hashes(4);
        let mut s = SpillStore::open(Some(&dir), 0, 25).unwrap();
        for (i, hash) in h.iter().take(3).enumerate() {
            s.put(*hash, vec![i as u8; 10]);
        }
        let st = s.stats();
        assert!(st.disk_bytes <= 25, "{st:?}");
        assert_eq!(st.disk_blocks, 2);
        assert!(s.get(&h[0]).is_none(), "the oldest was deleted");
        assert!(s.get(&h[2]).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
