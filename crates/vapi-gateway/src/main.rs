mod api;
mod auth;
mod dashboard;
mod nats;
mod output;
mod response_cache;
mod state;
mod stream;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use metrics_exporter_prometheus::PrometheusBuilder;
use vapi_core::{Config, ModelId};
use vapi_tokenize::Tokenization;

use crate::nats::Transport;
use crate::state::AppState;

const USAGE: &str = "\
USAGE:
    vapi-gateway [--config <path>]

The HTTP gateway: the OpenAI-compatible API, the dashboard at /dashboard, and
the tokenizer. It creates the JetStream stream that workers consume from, so
start it before them.

OPTIONS:
    -c, --config <path>    Configuration file. Overrides $VAPI_CONFIG, which
                           in turn overrides ./vapi.toml.
    -h, --help             Print this.
";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,vapi_gateway=debug".into()),
        )
        .init();

    let args = vapi_core::cli::parse(std::env::args_os().skip(1))?;
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }
    let cfg = Config::resolve(args.config, "vapi.toml")?;
    let model = ModelId(cfg.model.id.clone());

    // A model directory gives the real tokenizer and chat template; without
    // one the gateway still runs on the byte-level fallback, which is what
    // makes the whole pipeline demoable with no download.
    let tokenizer = match &cfg.model.path {
        Some(dir) => {
            tracing::info!(dir = %dir.display(), "loading tokenizer");
            Tokenization::from_dir(dir)?
        }
        None => {
            tracing::warn!("no model.path configured; using the byte-level tokenizer");
            Tokenization::bytes()?
        }
    };

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder).ok();

    let transport = Transport::connect(&cfg).await?;
    transport.ensure_streams(&cfg).await?;
    tracing::info!(stream = transport.jobs_stream(), "jetstream ready");

    let response_cache = if cfg.cache.response_cache {
        match response_cache::ResponseCache::open(
            &transport.js,
            &cfg.nats.response_cache_bucket,
            std::time::Duration::from_secs(cfg.cache.response_cache_ttl_secs),
        )
        .await
        {
            Ok(c) => {
                tracing::info!(bucket = %cfg.nats.response_cache_bucket, "response cache ready");
                Some(c)
            }
            Err(e) => {
                tracing::warn!(error = %e, "response cache unavailable; running without it");
                None
            }
        }
    } else {
        None
    };
    // The same fingerprint the worker uses for its prefix-cache namespace,
    // so a cached answer is never served across a weight change.
    let fingerprint = cfg
        .model
        .path
        .as_deref()
        .map(vapi_backend_candle::weights_fingerprint)
        .unwrap_or_else(|| "unversioned".into());

    let output_format = output::OutputFormat::detect(&tokenizer);
    if let Some(f) = &output_format {
        tracing::info!(
            reasoning = f.think_close.is_some(),
            "model output format: tool calls parsed at the gateway"
        );
    }

    // A decision checkpoint is recognised by its layout, the same way the
    // worker recognises it, so the two cannot disagree about which kind of
    // model this deployment serves.
    let decision = cfg.model.path.as_deref().and_then(|dir| {
        if !dir.join("rl_agent_config.json").exists() {
            return None;
        }
        match vapi_tokenize::open_decision_model(dir) {
            Ok((_, format)) => {
                tracing::info!(
                    max_len = format.config.max_len,
                    head_max_len = format.config.head_max_len,
                    "serving decisions at /v1/decisions"
                );
                Some(format)
            }
            Err(e) => {
                tracing::error!(error = %e, "decision checkpoint could not be opened");
                None
            }
        }
    });

    // Recognised by layout, the same way the worker recognises it.
    let speech = cfg.model.path.as_deref().and_then(|dir| {
        let text = std::fs::read_to_string(dir.join("processor_config.json")).ok()?;
        let settings = vapi_audio::MelSettings::from_json(&text).ok()?;
        tracing::info!(
            sample_rate = settings.sampling_rate,
            "serving transcription at /v1/audio/transcriptions"
        );
        Some(settings.sampling_rate)
    });

    for problem in cfg.auth.problems() {
        tracing::warn!(%problem, "api key configuration");
    }
    if cfg.auth.enabled() {
        tracing::info!(
            keys = cfg.auth.keys.len(),
            "api key required on /v1 and /dashboard; each key has its own cache namespace"
        );
    } else {
        tracing::warn!(
            "no api keys configured: /v1 and /dashboard are open, and every caller shares \
             one prefix-cache namespace"
        );
    }

    let bind = cfg.gateway.bind.clone();
    let cfg_for_settings = cfg.clone();
    let st = Arc::new(AppState {
        cfg,
        transport,
        tokenizer,
        model,
        queued: std::sync::atomic::AtomicUsize::new(0),
        settings: crate::state::RuntimeSettings::from_config(&cfg_for_settings),
        stats: Default::default(),
        started: std::time::Instant::now(),
        response_cache,
        fingerprint,
        output_format,
        decision,
        speech,
    });

    // Everything a caller can reach that runs work or reveals it. `/health`
    // stays open so a liveness probe needs no credential, and `/metrics`
    // stays open because that is what a scraper expects; neither exposes a
    // prompt or an answer.
    let protected = Router::new()
        .route("/v1/models", get(api::list_models))
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/completions", post(api::completions))
        .route("/v1/decisions", post(api::decisions))
        .route(
            "/v1/audio/transcriptions",
            post(api::transcriptions)
                // Audio is bigger than a JSON request by orders of magnitude,
                // and axum's 2 MB default rejects a one-minute WAV with a
                // multipart parse error that says nothing about size. The
                // real limit is the clip's *duration*, checked in the handler
                // where it can say so; this only has to be loose enough to
                // let a legal request through — 150 seconds of 48 kHz stereo
                // is about 29 MB.
                .layer(axum::extract::DefaultBodyLimit::max(api::MAX_UPLOAD_BYTES)),
        )
        // The dashboard is behind the same keys: its "Try it" box runs
        // inference, so leaving it open would make authentication decorative.
        .route("/dashboard", get(dashboard::page))
        .route("/dashboard/stats", get(dashboard::stats))
        .route("/dashboard/settings", post(dashboard::settings))
        .route("/dashboard/try", post(dashboard::try_it))
        .layer(axum::middleware::from_fn_with_state(
            st.clone(),
            auth::require_key,
        ));

    let app = Router::new()
        .route("/health", get(api::health))
        .merge(protected)
        .route(
            "/metrics",
            get(move || {
                let handle = handle.clone();
                async move { handle.render() }
            }),
        )
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(st);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(%bind, dashboard = %format!("http://{bind}/dashboard"), "gateway listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
