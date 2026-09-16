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
