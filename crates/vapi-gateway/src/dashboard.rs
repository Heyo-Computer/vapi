//! A server-rendered dashboard for watching and tuning a running vapi.
//!
//! Everything here is ordinary HTML: the page is rendered on the server,
//! the forms are plain `POST`s that redirect back, and the only script is
//! a dozen lines that re-fetch the numbers so the page does not have to be
//! reloaded to stay current. That keeps it usable with JavaScript off, and
//! it means the dashboard has no build step of its own.
//!
//! What it can change is deliberately narrow. The gateway's own admission
//! and timeout settings live in this process and are safe to move while it
//! runs; everything a worker acts on (cache size, batch limits, the model)
//! is read at the worker's start, so it is shown read-only. A dashboard
//! that appears to change a setting it cannot is worse than one that
//! admits the boundary.

use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::Form;
use axum::extract::State;
use axum::response::{Html, IntoResponse, Redirect, Response};
use serde::{Deserialize, Serialize};
use vapi_openai::{ChatCompletionRequest, ChatMessage};
use vapi_proto::WorkerEntry;

use crate::state::SharedState;

const TEMPLATE: &str = include_str!("../templates/dashboard.html");

/// Everything the page needs, in one shape so the template and the live
/// refresh cannot drift apart.
#[derive(Debug, Serialize)]
pub struct Snapshot {
    pub model: String,
    pub uptime: String,
    pub queued: usize,
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
    pub refused: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub workers: Vec<WorkerRow>,
    pub recent: Vec<crate::state::RequestRecord>,
    pub settings: SettingsView,
    pub fixed: Vec<(String, String)>,
    pub features: Vec<(String, String)>,
}

#[derive(Debug, Serialize)]
pub struct WorkerRow {
    pub id: String,
    pub model: String,
    pub running: usize,
    pub waiting: usize,
    pub max_concurrent: usize,
    pub load: String,
    pub kv: String,
    pub kv_pct: u32,
    pub prefix_hit: String,
    pub uptime: String,
    pub partitions: String,
}

#[derive(Debug, Serialize)]
pub struct SettingsView {
    pub max_queued_requests: usize,
    pub response_cache: bool,
    pub response_cache_available: bool,
    pub first_token_timeout_secs: usize,
    pub stream_idle_timeout_secs: usize,
}

/// `h:mm:ss`, or `NNd h:mm:ss` past a day.
fn duration(secs: u64) -> String {
    let (d, h, m, s) = (
        secs / 86400,
        (secs % 86400) / 3600,
        (secs % 3600) / 60,
        secs % 60,
    );
    if d > 0 {
        format!("{d}d {h}:{m:02}:{s:02}")
    } else {
        format!("{h}:{m:02}:{s:02}")
    }
}

fn percent(x: f32) -> String {
    format!("{:.0}%", x * 100.0)
}

/// Read every worker's self-published entry from the registry bucket.
///
/// Missing or unreadable is not an error: the bucket only exists once a
/// worker has started, and a dashboard that refuses to render because no
/// worker is up is useless exactly when it is needed.
async fn workers(st: &SharedState) -> Vec<WorkerRow> {
    let Ok(store) = st
        .transport
        .js
        .get_key_value(&st.cfg.nats.workers_bucket)
        .await
    else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let Ok(mut keys) = store.keys().await else {
        return rows;
    };
    use futures::StreamExt;
    while let Some(Ok(key)) = keys.next().await {
        let Ok(Some(bytes)) = store.get(&key).await else {
            continue;
        };
        let Ok(entry) = serde_json::from_slice::<WorkerEntry>(&bytes) else {
            continue;
        };
        let s = entry.stats;
        rows.push(WorkerRow {
            id: s.worker_id,
            model: s.model.0,
            running: s.running,
            waiting: s.waiting,
            max_concurrent: s.max_concurrent,
            load: if s.max_concurrent == 0 {
                "0%".into()
            } else {
                percent(s.running as f32 / s.max_concurrent as f32)
            },
            kv: percent(s.block_utilization),
            kv_pct: (s.block_utilization * 100.0).clamp(0.0, 100.0) as u32,
            prefix_hit: percent(s.prefix_hit_rate),
            uptime: duration(s.uptime_secs),
            partitions: if entry.partitions.is_empty() {
                "all".into()
            } else {
                entry
                    .partitions
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    rows
}

pub async fn snapshot(st: &SharedState) -> Snapshot {
    let s = &st.stats;
    let completed = s.completed.load(Ordering::Relaxed);
    let hits = s.cache_hits.load(Ordering::Relaxed);
    let cfg = &st.cfg;
    let yes_no = |b: bool| if b { "on" } else { "off" }.to_string();
    Snapshot {
        model: st.model.0.clone(),
        uptime: duration(st.started.elapsed().as_secs()),
        queued: st.queued.load(Ordering::Relaxed),
        started: s.started.load(Ordering::Relaxed),
        completed,
        failed: s.failed.load(Ordering::Relaxed),
        refused: s.refused.load(Ordering::Relaxed),
        cache_hits: hits,
        cache_hit_rate: if completed == 0 {
            "—".into()
        } else {
            percent(hits as f32 / completed as f32)
        },
        prompt_tokens: s.prompt_tokens.load(Ordering::Relaxed),
        completion_tokens: s.completion_tokens.load(Ordering::Relaxed),
        workers: workers(st).await,
        recent: s.recent(),
        settings: SettingsView {
            max_queued_requests: st.settings.max_queued_requests.load(Ordering::Relaxed),
            response_cache: st.settings.response_cache.load(Ordering::Relaxed),
            response_cache_available: st.response_cache.is_some(),
            first_token_timeout_secs: st.settings.first_token_timeout_secs.load(Ordering::Relaxed),
            stream_idle_timeout_secs: st.settings.stream_idle_timeout_secs.load(Ordering::Relaxed),
        },
        // Read at a worker's start, so changing them needs a restart.
        fixed: vec![
            (
                "model.path".into(),
                cfg.model
                    .path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none (byte tokenizer)".into()),
            ),
            (
                "model.max_context".into(),
                cfg.model.max_context.to_string(),
            ),
            (
                "model.num_blocks".into(),
                cfg.model
                    .num_blocks
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "from free VRAM".into()),
            ),
            (
                "model.dtype / device".into(),
                format!("{:?} / {:?}", cfg.model.dtype, cfg.model.device),
            ),
            ("model.cuda_graphs".into(), yes_no(cfg.model.cuda_graphs)),
            (
                "worker.max_concurrent_seqs".into(),
                cfg.worker.max_concurrent_seqs.to_string(),
            ),
            (
                "worker.max_batched_tokens".into(),
                cfg.worker.max_batched_tokens.to_string(),
            ),
            (
                "worker.prefill_chunk_tokens".into(),
                cfg.worker.prefill_chunk_tokens.to_string(),
            ),
            (
                "worker.drain_timeout_secs".into(),
                cfg.worker.drain_timeout_secs.to_string(),
            ),
            (
                "nats.job_partitions".into(),
                cfg.nats.job_partitions.to_string(),
            ),
            ("cache.prefix_cache".into(), yes_no(cfg.cache.prefix_cache)),
            ("cache.spill".into(), yes_no(cfg.cache.spill)),
        ],
        features: vec![
            (
                "tool calls".into(),
                match st.output_format.as_ref().map(|f| f.syntax) {
                    Some(crate::output::ToolSyntax::PythonCall) => {
                        "python call syntax, in this model's markers".into()
                    }
                    Some(crate::output::ToolSyntax::Json) => {
                        "JSON objects, in this model's markers".into()
                    }
                    None => "not supported by this model".into(),
                },
            ),
            (
                "reasoning".into(),
                match st.output_format.as_ref().and_then(|f| f.think_close) {
                    Some(m) => format!("split on {m}"),
                    None => "none".into(),
                },
            ),
            (
                "structured output".into(),
                "json_object, json_schema (subset)".into(),
            ),
            ("weight fingerprint".into(), st.fingerprint.clone()),
        ],
    }
}

fn env() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.set_keep_trailing_newline(true);
    // Registered as `.html` so minijinja escapes by default: anything the
    // page shows came from a request, a worker or a form field.
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::Html);
    env.add_template("dashboard.html", TEMPLATE)
        .expect("the dashboard template is valid");
    env
}

/// The page itself.
pub async fn page(State(st): State<SharedState>) -> Response {
    let snap = snapshot(&st).await;
    render(&snap, None)
}

fn render(snap: &Snapshot, flash: Option<&str>) -> Response {
    let env = env();
    let tmpl = env
        .get_template("dashboard.html")
        .expect("registered above");
    match tmpl.render(minijinja::context! { s => snap, flash => flash }) {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "dashboard render failed");
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("dashboard render failed: {e}"),
            )
                .into_response()
        }
    }
}

/// The same numbers as JSON, for the page's periodic refresh.
pub async fn stats(State(st): State<SharedState>) -> axum::Json<Snapshot> {
    axum::Json(snapshot(&st).await)
}

#[derive(Debug, Deserialize)]
pub struct SettingsForm {
    pub max_queued_requests: usize,
    pub first_token_timeout_secs: usize,
    pub stream_idle_timeout_secs: usize,
    /// A checkbox is absent when unticked.
    #[serde(default)]
    pub response_cache: Option<String>,
}

/// Apply the settings form. A plain redirect back, so the browser's back
/// button and reload behave.
pub async fn settings(State(st): State<SharedState>, Form(f): Form<SettingsForm>) -> Redirect {
    let clamp = |v: usize| v.clamp(1, 3600);
    st.settings
        .max_queued_requests
        .store(f.max_queued_requests, Ordering::Relaxed);
    st.settings
        .first_token_timeout_secs
        .store(clamp(f.first_token_timeout_secs), Ordering::Relaxed);
    st.settings
        .stream_idle_timeout_secs
        .store(clamp(f.stream_idle_timeout_secs), Ordering::Relaxed);
    // The form disables the box when no bucket was opened at start; a
    // direct POST must not be able to turn on a cache that is not there.
    st.settings.response_cache.store(
        f.response_cache.is_some() && st.response_cache.is_some(),
        Ordering::Relaxed,
    );
    tracing::info!(
        max_queued = f.max_queued_requests,
        response_cache = f.response_cache.is_some(),
        "settings changed from the dashboard"
    );
    Redirect::to("/dashboard?saved=1")
}

#[derive(Debug, Deserialize)]
pub struct TryForm {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub json_mode: Option<String>,
}

/// Send one request through the ordinary chat path and show the answer.
///
/// Deliberately the same code a client hits, so what the page shows is
/// what a caller would get, including the tool-call and reasoning split.
pub async fn try_it(State(st): State<SharedState>, Form(f): Form<TryForm>) -> Response {
    let mut body = serde_json::json!({
        "model": st.model.0,
        "messages": [{"role": "user", "content": f.prompt}],
        "max_tokens": f.max_tokens.unwrap_or(256).clamp(1, 4096),
        "temperature": f.temperature.unwrap_or(0.0).clamp(0.0, 2.0),
    });
    if f.json_mode.is_some() {
        body["response_format"] = serde_json::json!({"type": "json_object"});
    }
    let req: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return render(&snapshot(&st).await, Some(&format!("bad request: {e}"))),
    };
    let _ = ChatMessage::user("");
    let t = Instant::now();
    let answer = match crate::api::chat_completions(State(st.clone()), axum::Json(req)).await {
        Ok(resp) => match axum::body::to_bytes(resp.into_body(), 1 << 20).await {
            Ok(bytes) => summarise(&bytes, t),
            Err(e) => format!("could not read the response: {e}"),
        },
        Err(e) => format!("{}", e.0),
    };
    render(&snapshot(&st).await, Some(&answer))
}

/// Turn a chat response into the line the page shows.
fn summarise(bytes: &[u8], started: Instant) -> String {
    let ms = started.elapsed().as_millis();
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return format!("unreadable response after {ms} ms");
    };
    if let Some(err) = v.get("error").and_then(|e| e.get("message")) {
        return format!("{} ({ms} ms)", err.as_str().unwrap_or("error"));
    }
    let choice = &v["choices"][0];
    let msg = &choice["message"];
    let mut out = String::new();
    if let Some(r) = msg.get("reasoning_content").and_then(|r| r.as_str())
        && !r.is_empty()
    {
        out.push_str(&format!("[thought for {} characters]\n", r.len()));
    }
    if let Some(calls) = msg.get("tool_calls")
        && !calls.is_null()
    {
        out.push_str(&format!("tool calls: {calls}\n"));
    }
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    if content.is_empty() && choice["finish_reason"] == "length" {
        out.push_str("(no answer: the budget ran out before the model finished)");
    } else {
        out.push_str(content);
    }
    let usage = &v["usage"];
    format!(
        "{out}\n\n— {} prompt + {} completion tokens, {} ms, finished by {}",
        usage["prompt_tokens"],
        usage["completion_tokens"],
        ms,
        choice["finish_reason"].as_str().unwrap_or("?")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> Snapshot {
        Snapshot {
            model: "m".into(),
            uptime: "0:01:00".into(),
            queued: 2,
            started: 10,
            completed: 8,
            failed: 1,
            refused: 1,
            cache_hits: 3,
            cache_hit_rate: "38%".into(),
            prompt_tokens: 100,
            completion_tokens: 200,
            workers: vec![WorkerRow {
                id: "w-1".into(),
                model: "m".into(),
                running: 2,
                waiting: 1,
                max_concurrent: 64,
                load: "3%".into(),
                kv: "12%".into(),
                kv_pct: 12,
                prefix_hit: "50%".into(),
                uptime: "0:02:00".into(),
                partitions: "all".into(),
            }],
            recent: vec![crate::state::RequestRecord {
                id: "abc".into(),
                kind: "chat",
                prompt_tokens: 10,
                completion_tokens: 20,
                choices: 1,
                duration_ms: 30,
                finish: "stop".into(),
                cached: false,
                streamed: true,
            }],
            settings: SettingsView {
                max_queued_requests: 256,
                response_cache: true,
                response_cache_available: true,
                first_token_timeout_secs: 120,
                stream_idle_timeout_secs: 60,
            },
            fixed: vec![("model.max_context".into(), "8192".into())],
            features: vec![("tool calls".into(), "json".into())],
        }
    }

    #[test]
    fn the_page_renders_every_section() {
        let env = env();
        let html = env
            .get_template("dashboard.html")
            .unwrap()
            .render(minijinja::context! { s => snap(), flash => None::<String> })
            .expect("renders");
        for needle in [
            "vapi",
            "w-1",               // the worker row
            "model.max_context", // fixed configuration
            "8192",
            "256", // the editable setting's current value
            "tool calls",
            "stop", // the recent request
            "/dashboard/settings",
            "/dashboard/try",
        ] {
            assert!(html.contains(needle), "the page should mention {needle:?}");
        }
        assert!(html.starts_with("<!doctype html>"));
    }

    #[test]
    fn a_flash_message_is_escaped_not_injected() {
        let env = env();
        let html = env
            .get_template("dashboard.html")
            .unwrap()
            .render(minijinja::context! {
                s => snap(),
                flash => Some("<script>alert(1)</script>".to_string()),
            })
            .expect("renders");
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn a_truncated_answer_says_so_rather_than_showing_nothing() {
        let body = serde_json::json!({
            "choices": [{
                "message": {"content": "", "reasoning_content": "thinking..."},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 80}
        });
        let text = summarise(&serde_json::to_vec(&body).unwrap(), Instant::now());
        assert!(text.contains("budget ran out"), "{text}");
        assert!(text.contains("thought for"), "{text}");
    }

    #[test]
    fn durations_read_as_clocks() {
        assert_eq!(duration(0), "0:00:00");
        assert_eq!(duration(61), "0:01:01");
        assert_eq!(duration(3661), "1:01:01");
        assert_eq!(duration(90061), "1d 1:01:01");
    }
}
