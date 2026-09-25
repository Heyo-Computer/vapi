use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use vapi_core::{Config, ModelId};
use vapi_tokenize::Tokenization;

use crate::nats::Transport;

/// Settings the dashboard can change while the gateway runs.
///
/// Only what is genuinely safe to change in this process: everything the
/// workers act on (cache sizes, batch limits, the model itself) is read at
/// their start and is shown read-only.
pub struct RuntimeSettings {
    /// 0 disables the check, as in the config file.
    pub max_queued_requests: AtomicUsize,
    /// Tier 2 lookups and writes, when a cache was opened at start.
    pub response_cache: AtomicBool,
    pub first_token_timeout_secs: AtomicUsize,
    pub stream_idle_timeout_secs: AtomicUsize,
}

impl RuntimeSettings {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            max_queued_requests: AtomicUsize::new(cfg.gateway.max_queued_requests),
            response_cache: AtomicBool::new(cfg.cache.response_cache),
            first_token_timeout_secs: AtomicUsize::new(
                cfg.gateway.first_token_timeout_secs as usize,
            ),
            stream_idle_timeout_secs: AtomicUsize::new(
                cfg.gateway.stream_idle_timeout_secs as usize,
            ),
        }
    }

    pub fn first_token_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.first_token_timeout_secs.load(Ordering::Relaxed) as u64)
    }

    pub fn stream_idle_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.stream_idle_timeout_secs.load(Ordering::Relaxed) as u64)
    }
}

/// One finished request, for the dashboard's recent list.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RequestRecord {
    pub id: String,
    pub kind: &'static str,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub choices: usize,
    pub duration_ms: u64,
    pub finish: String,
    /// Answered from the response cache without reaching a worker.
    pub cached: bool,
    pub streamed: bool,
}

/// Counters the dashboard shows. Kept here rather than scraped back out of
/// the metrics exporter, so the numbers on the page are exact.
#[derive(Default)]
pub struct GatewayStats {
    pub started: AtomicU64,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub refused: AtomicU64,
    pub cache_hits: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    /// Most recent finished requests, newest last.
    pub recent: std::sync::Mutex<std::collections::VecDeque<RequestRecord>>,
}

/// How many finished requests the dashboard remembers.
pub const RECENT_REQUESTS: usize = 25;

impl GatewayStats {
    pub fn record(&self, r: RequestRecord) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(r.prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(r.completion_tokens as u64, Ordering::Relaxed);
        if r.cached {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        if let Ok(mut q) = self.recent.lock() {
            if q.len() >= RECENT_REQUESTS {
                q.pop_front();
            }
            q.push_back(r);
        }
    }

    pub fn recent(&self) -> Vec<RequestRecord> {
        self.recent
            .lock()
            .map(|q| q.iter().rev().cloned().collect())
            .unwrap_or_default()
    }
}

pub struct AppState {
    pub cfg: Config,
    pub transport: Transport,
    pub tokenizer: Tokenization,
    pub model: ModelId,
    /// Requests published but not yet started by a worker; see
    /// [`AppState::queue_slot`].
    pub queued: AtomicUsize,
    /// Tier 2, when `cache.response_cache` is on.
    pub response_cache: Option<crate::response_cache::ResponseCache>,
    /// Weight fingerprint of the served model, part of every cache key.
    pub fingerprint: String,
    /// The model's structured-output markers, when it has them.
    pub output_format: Option<crate::output::OutputFormat>,
    /// Set when the served model answers decisions rather than generating:
    /// how to build a question's sequence, and the temperatures that
    /// calibrate what comes back.
    pub decision: Option<vapi_tokenize::DecisionFormat>,
    /// Set when the served model transcribes audio: the sample rate its
    /// frontend wants, which is all the gateway needs to know about it.
    pub speech: Option<usize>,
    /// What the dashboard may change at runtime.
    pub settings: RuntimeSettings,
    pub stats: GatewayStats,
    /// Started at, for uptime on the dashboard.
    pub started: std::time::Instant,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    /// Take a place in the queue, or refuse with `Overloaded` when
    /// `gateway.max_queued_requests` are already waiting. The slot is given
    /// back when a worker starts the request, or when the request ends
    /// without ever starting.
    pub fn queue_slot(self: &Arc<Self>) -> Result<QueueSlot, vapi_core::Error> {
        let max = self.settings.max_queued_requests.load(Ordering::Relaxed);
        let mut cur = self.queued.load(Ordering::Relaxed);
        loop {
            if max > 0 && cur >= max {
                metrics::counter!("vapi_gateway_overloaded_total").increment(1);
                self.stats.refused.fetch_add(1, Ordering::Relaxed);
                return Err(vapi_core::Error::Overloaded);
            }
            match self.queued.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        metrics::gauge!("vapi_gateway_queued").set((cur + 1) as f64);
        Ok(QueueSlot {
            state: Some(self.clone()),
        })
    }
}

/// A place in the gateway's queue; released on drop or [`QueueSlot::release`].
pub struct QueueSlot {
    state: Option<Arc<AppState>>,
}

impl QueueSlot {
    pub fn release(&mut self) {
        if let Some(st) = self.state.take() {
            let now = st.queued.fetch_sub(1, Ordering::Relaxed) - 1;
            metrics::gauge!("vapi_gateway_queued").set(now as f64);
        }
    }
}

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.release();
    }
}
