use vapi_core::{ModelId, RequestId};

/// Subject and stream naming, in one place so the gateway and the worker
/// cannot drift apart.
pub struct Subjects;

impl Subjects {
    /// Durable work queue for a model. Prompts are published here.
    pub fn jobs(model: &ModelId) -> String {
        format!("vapi.jobs.{}", model.subject_token())
    }

    /// Partitioned variant, used once cache-affinity routing is enabled: the
    /// gateway picks the partition from a hash of the conversation prefix so
    /// that turn *k* of a chat lands on the worker that already holds turns
    /// `1..k-1` in its prefix cache.
    pub fn jobs_partition(model: &ModelId, partition: u32) -> String {
        format!("vapi.jobs.{}.p{partition}", model.subject_token())
    }

    /// Which partition a prompt belongs to.
    ///
    /// The hash covers the prompt's **first block only**, so every turn of a
    /// conversation routes the same way as it grows: turn three shares its
    /// first block with turn one, and the worker that served turn one is the
    /// one whose prefix cache holds the most of it. Hashing the whole prompt
    /// would send every turn somewhere new, which is the opposite of what
    /// affinity is for.
    ///
    /// FNV-1a rather than `DefaultHasher`: this value has to agree between
    /// the gateway and the worker, and across releases.
    pub fn partition_for(prompt_tokens: &[u32], block_size: usize, partitions: u32) -> u32 {
        if partitions <= 1 {
            return 0;
        }
        let head = &prompt_tokens[..prompt_tokens.len().min(block_size)];
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in head {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        }
        // FNV's low bits carry little of the input, and a modulo reads
        // exactly those, so finish with a round of mixing. Without it every
        // conversation in a test of 200 landed on one partition.
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        (h % partitions as u64) as u32
    }

    /// The subject a job goes to: the shared queue when there is one
    /// partition, otherwise its partition's.
    pub fn jobs_for(
        model: &ModelId,
        prompt_tokens: &[u32],
        block_size: usize,
        partitions: u32,
    ) -> String {
        if partitions <= 1 {
            return Self::jobs(model);
        }
        Self::jobs_partition(
            model,
            Self::partition_for(prompt_tokens, block_size, partitions),
        )
    }

    /// Everything the jobs stream captures.
    pub const JOBS_WILDCARD: &'static str = "vapi.jobs.>";

    /// Per-request token stream. Core NATS, no persistence.
    pub fn stream(id: &RequestId) -> String {
        format!("vapi.stream.{id}")
    }

    /// Client went away. Workers hold a wildcard subscription on this.
    pub fn cancel(id: &RequestId) -> String {
        format!("vapi.cancel.{id}")
    }

    pub const CANCEL_WILDCARD: &'static str = "vapi.cancel.>";

    pub fn worker_ctl(worker_id: &str) -> String {
        format!("vapi.ctl.worker.{worker_id}")
    }

    pub const WORKER_CTL_WILDCARD: &'static str = "vapi.ctl.worker.>";

    /// Durable consumer name for a worker pool serving one model. All workers
    /// for a model share this consumer, which is what makes them compete for
    /// work rather than each receiving every job.
    pub fn consumer_name(model: &ModelId) -> String {
        format!("vapi-workers-{}", model.subject_token())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_are_flattened_into_legal_subject_tokens() {
        // `/` and `.` would otherwise split into extra subject tokens and
        // silently widen what a wildcard subscription matches.
        let m = ModelId("meta-llama/Llama-3.2-1B-Instruct".into());
        let s = Subjects::jobs(&m);
        assert_eq!(s, "vapi.jobs.meta-llama_Llama-3_2-1B-Instruct");
        assert_eq!(s.matches('.').count(), 2);
    }

    #[test]
    fn jobs_wildcard_covers_plain_and_partitioned_subjects() {
        let m = ModelId("m".into());
        for s in [Subjects::jobs(&m), Subjects::jobs_partition(&m, 3)] {
            assert!(
                s.starts_with("vapi.jobs."),
                "{s} must be captured by {}",
                Subjects::JOBS_WILDCARD
            );
        }
    }

    #[test]
    fn request_subjects_are_disjoint_from_the_jobs_stream() {
        // The jobs stream captures `vapi.jobs.>`; if token deltas or cancels
        // fell under that prefix they would be persisted to disk per token.
        let id = RequestId::parse("abc123").unwrap();
        assert!(!Subjects::stream(&id).starts_with("vapi.jobs."));
        assert!(!Subjects::cancel(&id).starts_with("vapi.jobs."));
    }
}

#[cfg(test)]
mod routing_tests {
    use super::*;

    fn model() -> ModelId {
        ModelId("m".into())
    }

    #[test]
    fn a_conversation_keeps_its_partition_as_it_grows() {
        // Turn 1 is the first 40 tokens; later turns append to it.
        let turn1: Vec<u32> = (0..40).collect();
        let turn2: Vec<u32> = (0..90).collect();
        let turn3: Vec<u32> = (0..210).collect();
        let p = |t: &[u32]| Subjects::partition_for(t, 32, 8);
        assert_eq!(p(&turn1), p(&turn2));
        assert_eq!(p(&turn1), p(&turn3));
        // And it is the subject the gateway publishes to.
        assert_eq!(
            Subjects::jobs_for(&model(), &turn3, 32, 8),
            Subjects::jobs_partition(&model(), p(&turn1))
        );
    }

    #[test]
    fn one_partition_keeps_the_shared_queue() {
        let t: Vec<u32> = (0..40).collect();
        assert_eq!(
            Subjects::jobs_for(&model(), &t, 32, 1),
            Subjects::jobs(&model())
        );
        assert_eq!(Subjects::partition_for(&t, 32, 1), 0);
        // A prompt shorter than a block still routes.
        assert_eq!(
            Subjects::jobs_for(&model(), &[7], 32, 1),
            Subjects::jobs(&model())
        );
    }

    #[test]
    fn different_conversations_spread_over_the_partitions() {
        let mut seen = std::collections::HashMap::new();
        for c in 0..200u32 {
            let tokens: Vec<u32> = (0..40).map(|i| c * 1000 + i).collect();
            *seen
                .entry(Subjects::partition_for(&tokens, 32, 4))
                .or_insert(0) += 1;
        }
        assert_eq!(seen.len(), 4, "every partition is used: {seen:?}");
        // Nothing like uniformity is promised, but nothing should be a
        // rounding error either.
        assert!(seen.values().all(|&n| n >= 20), "{seen:?}");
    }

    /// Recorded from the implementation, so a change to the hash shows up
    /// as a failing test rather than as a silent cache-affinity loss.
    const PINNED: u32 = 14;

    #[test]
    fn the_partition_is_stable_across_builds() {
        // A gateway and a worker on different releases must agree, so this
        // value is pinned rather than derived from the hasher of the day.
        let tokens: Vec<u32> = (0..32).collect();
        let got = Subjects::partition_for(&tokens, 32, 16);
        assert_eq!(got, PINNED, "the routing hash changed");
    }
}
