use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Tokens per KV cache block.
///
/// This is 32, not vLLM's 16, and the reason is a hard constraint rather than
/// a tuning choice: `candle-flash-attn`'s paged kernel rejects any
/// `page_block_size` that is not a multiple of 32 (see `flash_attn_varlen_paged_windowed`).
/// Fixing it here — before any GPU is in play — keeps the CPU reference path
/// and the CUDA path on identical block geometry, so the cache-equivalence
/// tests written against CPU stay meaningful on the GPU box.
pub const BLOCK_SIZE: usize = 32;

const _: () = assert!(
    BLOCK_SIZE.is_multiple_of(32),
    "candle-flash-attn requires page_block_size % 32 == 0"
);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub nats: NatsConfig,
    pub gateway: GatewayConfig,
    pub worker: WorkerConfig,
    pub cache: CacheConfig,
    pub model: ModelConfig,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| Error::Config(e.to_string()))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    /// Load from `$VAPI_CONFIG` if set, else the given default path if it
    /// exists, else built-in defaults. Missing config is not an error — the
    /// defaults are a working local setup.
    pub fn load_or_default(default_path: impl AsRef<Path>) -> Result<Self> {
        if let Ok(p) = std::env::var("VAPI_CONFIG") {
            return Self::load(p);
        }
        let p = default_path.as_ref();
        if p.exists() {
            Self::load(p)
        } else {
            Ok(Self::default())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NatsConfig {
    pub url: String,
    /// JetStream stream holding queued prompts.
    pub jobs_stream: String,
    /// KV bucket for the tier-2 response cache.
    pub response_cache_bucket: String,
    /// KV bucket for the worker registry.
    pub workers_bucket: String,
    /// Number of job partitions for cache-affinity routing. 1 keeps the
    /// single shared queue, which is what a one-worker deployment wants.
    /// Above that the gateway routes by a hash of the conversation's first
    /// block and each worker serves a subset, so a chat's later turns reach
    /// the worker that already holds its earlier ones.
    pub job_partitions: u32,
    /// How long a job may sit unclaimed before JetStream drops it.
    pub job_max_age_secs: u64,
    /// JetStream publish-deduplication window, keyed on `Nats-Msg-Id`.
    pub dedupe_window_secs: u64,
}

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            url: "nats://127.0.0.1:4222".into(),
            jobs_stream: "VAPI_JOBS".into(),
            response_cache_bucket: "VAPI_RESP_CACHE".into(),
            workers_bucket: "VAPI_WORKERS".into(),
            job_partitions: 1,
            job_max_age_secs: 3600,
            dedupe_window_secs: 120,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatewayConfig {
    pub bind: String,
    /// Give up if no worker has produced a first token in this long.
    pub first_token_timeout_secs: u64,
    /// Give up if a stream stalls mid-generation for this long.
    pub stream_idle_timeout_secs: u64,
    /// Requests this gateway has published that no worker has started yet.
    /// Beyond this the gateway answers 503 with `Retry-After` instead of
    /// letting the queue grow without bound. 0 disables the check.
    pub max_queued_requests: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8080".into(),
            first_token_timeout_secs: 120,
            stream_idle_timeout_secs: 60,
            max_queued_requests: 256,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// Stable identity across restarts; defaults to a random id per process.
    pub id: Option<String>,
    /// Maximum sequences resident in the engine at once. This is also the
    /// JetStream consumer's `max_ack_pending`, which is what actually applies
    /// backpressure: NATS stops delivering when the worker is full, so the
    /// queue absorbs bursts instead of the worker's memory.
    pub max_concurrent_seqs: usize,
    /// Token budget for one scheduler step, across prefill and decode.
    pub max_batched_tokens: usize,
    /// Cap on prefill tokens admitted per step, so one long prompt cannot
    /// stall every in-flight decode.
    pub prefill_chunk_tokens: usize,
    /// How long JetStream waits for an ack before redelivering. The worker
    /// sends `AckKind::Progress` at a third of this while generating.
    pub ack_wait_secs: u64,
    pub metrics_bind: Option<String>,
    /// Artificial per-step delay for the mock backend, so a laptop can
    /// simulate GPU step latency. Ignored by real backends.
    pub mock_step_delay_ms: u64,
    /// Job partitions this worker serves. Empty means all of them, which is
    /// the right answer for a single worker and for a pool that shares one
    /// queue. Two workers splitting `nats.job_partitions = 2` would take
    /// `[0]` and `[1]`.
    pub partitions: Vec<u32>,
    /// A failed engine step fails every in-flight request (their clients
    /// get an error instead of a hang) and the engine carries on. After
    /// this many failures in a row the worker exits instead, so a wedged
    /// device is restarted rather than failing every job it is handed.
    pub max_step_failures: usize,
    /// On SIGTERM or Ctrl-C the worker stops taking jobs and lets what is
    /// running finish. Anything still running after this long is failed
    /// so the process can exit.
    pub drain_timeout_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            id: None,
            max_concurrent_seqs: 64,
            max_batched_tokens: 8192,
            prefill_chunk_tokens: 2048,
            ack_wait_secs: 30,
            metrics_bind: Some("0.0.0.0:9090".into()),
            mock_step_delay_ms: 0,
            max_step_failures: 3,
            drain_timeout_secs: 30,
            partitions: Vec::new(),
        }
    }
}

impl WorkerConfig {
    /// The partitions this worker should consume, given how many exist.
    pub fn owned_partitions(&self, total: u32) -> Vec<u32> {
        let total = total.max(1);
        if self.partitions.is_empty() {
            return (0..total).collect();
        }
        let mut owned: Vec<u32> = self
            .partitions
            .iter()
            .copied()
            .filter(|p| *p < total)
            .collect();
        owned.sort_unstable();
        owned.dedup();
        owned
    }

    pub fn ack_wait(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.ack_wait_secs)
    }

    /// Heartbeat interval for in-flight generations. A third of `ack_wait`
    /// leaves room for two missed heartbeats before JetStream redelivers —
    /// and a redelivery mid-generation would duplicate a user's completion.
    pub fn ack_progress_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs((self.ack_wait_secs / 3).max(1))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    /// Tier 1: share KV blocks across requests, including across users.
    pub prefix_cache: bool,
    /// Tier 2: exact-match completion cache for deterministic requests.
    pub response_cache: bool,
    pub response_cache_ttl_secs: u64,
    /// Tier 3: spill evicted blocks to disk so hot prefixes survive restart.
    pub spill: bool,
    pub spill_dir: PathBuf,
    /// Cap on the disk tier. 0 keeps the spill tier in host memory only.
    pub spill_max_bytes: u64,
    /// Cap on the host-memory tier, which sits in front of the disk one.
    pub spill_ram_bytes: u64,
    /// Fraction of blocks held back from admission so the allocator does not
    /// thrash at the boundary.
    pub watermark: f32,
    /// Default tenant namespace folded into every block hash.
    ///
    /// A single shared namespace gives the best hit rate and is the default,
    /// but note that a cache shared across users is a timing side channel:
    /// time-to-first-token reveals whether *someone* recently submitted a
    /// given prefix. Tenants who care can be pinned to their own namespace.
    pub default_namespace: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            prefix_cache: true,
            response_cache: true,
            response_cache_ttl_secs: 3600,
            spill: false,
            spill_dir: PathBuf::from(".data/spill"),
            spill_max_bytes: 16 << 30,
            spill_ram_bytes: 2 << 30,
            watermark: 0.01,
            default_namespace: "global".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Hugging Face repo id, e.g. `meta-llama/Llama-3.2-1B-Instruct`.
    pub id: String,
    /// Local directory holding weights + tokenizer, bypassing the hub.
    pub path: Option<PathBuf>,
    pub dtype: DType,
    pub device: DeviceKind,
    pub max_context: usize,
    /// Fraction of device memory the KV cache may claim.
    pub kv_cache_fraction: f32,
    /// Fixed block count, overriding memory-based sizing. Required on CPU,
    /// where there is no VRAM figure to profile against.
    pub num_blocks: Option<usize>,
    /// Capture pure-decode steps into CUDA graphs and replay them, one per
    /// batch-size bucket. Removes the host-side kernel issue cost per step.
    /// Only affects the `cuda` build; falls back to eager execution if a
    /// capture fails.
    pub cuda_graphs: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: "meta-llama/Llama-3.2-1B-Instruct".into(),
            path: None,
            dtype: DType::Auto,
            device: DeviceKind::Auto,
            max_context: 8192,
            kv_cache_fraction: 0.85,
            num_blocks: Some(2048),
            cuda_graphs: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// bf16 on CUDA, f32 on CPU.
    #[default]
    Auto,
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// CUDA if the binary was built with the `cuda` feature and a device is
    /// present, else CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_a_working_local_setup() {
        let c = Config::default();
        assert_eq!(c.nats.url, "nats://127.0.0.1:4222");
        assert!(c.cache.prefix_cache);
        assert_eq!(BLOCK_SIZE, 32);
    }

    #[test]
    fn partial_toml_fills_in_defaults() {
        let c = Config::from_toml_str(
            r#"
            [gateway]
            bind = "127.0.0.1:9999"
            "#,
        )
        .unwrap();
        assert_eq!(c.gateway.bind, "127.0.0.1:9999");
        // Untouched sections still get their defaults.
        assert_eq!(c.nats.jobs_stream, "VAPI_JOBS");
    }

    #[test]
    fn unknown_keys_are_rejected_rather_than_silently_ignored() {
        let err = Config::from_toml_str(
            r#"
            [gateway]
            bnid = "typo"
            "#,
        );
        assert!(
            err.is_err(),
            "a typo'd config key must not be silently dropped"
        );
    }

    #[test]
    fn ack_progress_leaves_room_for_missed_heartbeats() {
        let w = WorkerConfig {
            ack_wait_secs: 30,
            ..Default::default()
        };
        assert_eq!(w.ack_progress_interval().as_secs(), 10);
        // Never zero, even with a pathological ack_wait.
        let w = WorkerConfig {
            ack_wait_secs: 1,
            ..Default::default()
        };
        assert_eq!(w.ack_progress_interval().as_secs(), 1);
    }
}
