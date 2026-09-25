# vapi task runner. `just --list` to see everything.

default:
    @just --list

# Start NATS with JetStream.
nats:
    docker compose up -d nats

nats-stop:
    docker compose down

# Run the gateway and a worker together on one config.
#
# They have to agree: `model.id` is part of the NATS subject, so a gateway on
# vapi.toml and a worker on configs/qwen3-0.6b.toml publish to and listen on
# different queues, and every request times out while the dashboard shows a
# healthy worker. Taking one config for both is the point of this recipe.
#
#   just up                                  # mock backend, vapi.toml
#   just up configs/qwen3-0.6b.toml cuda     # real weights on the GPU
#
# Ctrl-C stops both.
up config="vapi.toml" features="":
    #!/usr/bin/env bash
    set -euo pipefail
    config="{{ config }}"
    features="{{ features }}"
    if [ -n "$features" ]; then
        profile=release
        build=(--release --features "$features")
    else
        profile=debug
        build=()
    fi

    # Build before starting anything: a compile error should not leave a
    # half-started stack behind.
    cargo build "${build[@]}" -p vapi-gateway -p vapi-worker
    docker compose up -d nats >/dev/null

    # The binaries directly, not `cargo run`: cargo wraps them in a shell, so
    # killing it leaves the server running and the next start fails to bind.
    gateway="target/$profile/vapi-gateway"
    worker="target/$profile/vapi-worker"

    # The gateway creates the JetStream stream the worker waits for, so it
    # goes first. The worker retries for 15s, but ordering makes the logs
    # readable.
    "$gateway" --config "$config" 2>&1 | sed -u 's/^/[gw] /' &
    "$worker"  --config "$config" 2>&1 | sed -u 's/^/[wk] /' &

    trap 'kill 0' INT TERM EXIT
    wait -n
    echo "one of them exited; stopping the other"

# Stop anything `just up` left behind, and NATS with it.
down:
    -pkill -x vapi-gateway
    -pkill -x vapi-worker
    docker compose down

# Run the HTTP gateway. Pass a config: `just gw configs/laya.toml`.
gw config="":
    cargo run -p vapi-gateway {{ if config == "" { "" } else { "-- --config " + config } }}

# Run an engine worker (mock backend unless built with a model feature).
worker config="":
    cargo run -p vapi-worker {{ if config == "" { "" } else { "-- --config " + config } }}

# Run a worker on real weights, on the CPU. Needs model.path in vapi.toml.
worker-candle:
    cargo run --release -p vapi-worker --features candle

test:
    cargo test --workspace
    cargo test -p vapi-backend-candle --features candle

# Golden-token tests against a real model (skips if none is at
# ~/models/smollm2-135m-instruct or $VAPI_TEST_MODEL_DIR).
test-model:
    cargo test -p vapi-backend-candle --features candle --test real_model -- --nocapture

check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy -p vapi-backend-candle -p vapi-worker --all-targets --features candle -- -D warnings

# End-to-end smoke test against a running gateway.
smoke:
    @echo "--- models ---"
    @curl -s localhost:8080/v1/models | python3 -m json.tool
    @echo "--- streaming ---"
    @curl -sN localhost:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":8,"temperature":0}'

# Needs `just nats` and the checkpoints in ~/models; each test skips with a
# reason rather than failing when something is missing.
#
# End-to-end: the real binaries against a real NATS and a real model.
test-e2e:
    cargo build --release --features cuda -p vapi-worker -p vapi-gateway
    cargo test -p vapi-gateway --test e2e_decisions --test e2e_transcriptions -- --nocapture

# Worker metrics, including prefix-cache hit rate.
metrics:
    @curl -s localhost:9091/metrics | grep -E '^vapi_' | grep -v '^#'

# Build with CUDA. Only on a machine with nvcc: this compiles FlashAttention
# from source once and takes tens of minutes. Set CUDA_COMPUTE_CAP for your
# card and CANDLE_FLASH_ATTN_BUILD_DIR (an existing directory) to cache it.
build-cuda:
    cargo build --release -p vapi-worker --features cuda

# Run a worker on real weights, on the GPU. `just worker-cuda configs/laya.toml`.
worker-cuda config="":
    cargo run --release -p vapi-worker --features cuda {{ if config == "" { "" } else { "-- --config " + config } }}

# GPU tests: the kernel spike and the goldens in bf16.
test-cuda:
    cargo test -p vapi-backend-candle --features cuda -- --nocapture --test-threads=1
