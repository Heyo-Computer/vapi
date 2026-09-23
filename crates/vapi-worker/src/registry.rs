//! Who is serving what, in a NATS KV bucket.
//!
//! Routing does not read this: the gateway partitions from config, so a
//! registry outage cannot misroute a request. It exists so an operator (or
//! a future autoscaler) can see which workers are up, which partitions each
//! one serves, and how loaded they are, without scraping every worker's
//! metrics port.
//!
//! Entries are refreshed on the worker's ack heartbeat and deleted on a
//! clean drain. A worker that dies leaves a stale entry behind, so readers
//! should treat `uptime_secs` and the bucket's own timestamps as a
//! liveness signal rather than assuming everything listed is alive.

use async_nats::jetstream::{self, kv};
use vapi_core::Config;
use vapi_proto::WorkerStats;

/// What one worker publishes about itself; the gateway's dashboard reads
/// the same type.
pub use vapi_proto::WorkerEntry as Entry;

pub struct Registry {
    store: kv::Store,
}

impl Registry {
    pub async fn open(js: &jetstream::Context, cfg: &Config) -> Option<Self> {
        let bucket = cfg.nats.workers_bucket.clone();
        // Entries outlive a few missed heartbeats and no longer, so a dead
        // worker disappears on its own.
        let ttl = cfg.worker.ack_progress_interval() * 4;
        match js
            .create_key_value(kv::Config {
                bucket,
                max_age: ttl,
                ..Default::default()
            })
            .await
        {
            Ok(store) => Some(Self { store }),
            Err(e) => {
                tracing::warn!(error = %e, "worker registry bucket unavailable");
                None
            }
        }
    }

    pub async fn publish(&self, stats: WorkerStats, partitions: &[u32]) {
        let key = key_for(&stats.worker_id);
        let entry = Entry {
            stats,
            partitions: partitions.to_vec(),
        };
        match serde_json::to_vec(&entry) {
            Ok(bytes) => {
                if let Err(e) = self.store.put(key, bytes.into()).await {
                    tracing::debug!(error = %e, "registry update failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "registry encode failed"),
        }
    }

    pub async fn remove(&self, worker_id: &str) {
        let _ = self.store.delete(key_for(worker_id)).await;
    }
}

/// KV keys allow no dots, which worker ids may contain.
fn key_for(worker_id: &str) -> String {
    worker_id.replace(['.', ' '], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_legal_and_entries_carry_their_partitions() {
        assert_eq!(key_for("w.1 a"), "w_1_a");
        let entry = Entry {
            stats: WorkerStats {
                worker_id: "w1".into(),
                model: vapi_core::ModelId("m".into()),
                running: 2,
                waiting: 1,
                max_concurrent: 64,
                block_utilization: 0.5,
                prefix_hit_rate: 0.25,
                uptime_secs: 10,
            },
            partitions: vec![0, 2],
        };
        let v: serde_json::Value = serde_json::to_value(&entry).unwrap();
        // Flattened, so a reader sees one object rather than a nested one.
        assert_eq!(v["worker_id"], "w1");
        assert_eq!(v["partitions"], serde_json::json!([0, 2]));
        let back: Entry = serde_json::from_value(v).unwrap();
        assert_eq!(back.partitions, vec![0, 2]);
        assert_eq!(back.stats.running, 2);
    }
}
