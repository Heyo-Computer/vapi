mod api;
mod nats;
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,vapi_gateway=debug".into()),
        )
        .init();

    let cfg = Config::load_or_default("vapi.toml")?;
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

    let bind = cfg.gateway.bind.clone();
    let st = Arc::new(AppState {
        cfg,
        transport,
        tokenizer,
        model,
    });

    let app = Router::new()
        .route("/health", get(api::health))
        .route("/v1/models", get(api::list_models))
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/completions", post(api::completions))
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
    tracing::info!(%bind, "gateway listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
