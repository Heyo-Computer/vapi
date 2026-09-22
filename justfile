# vapi task runner. `just --list` to see everything.

default:
    @just --list

# Start NATS with JetStream.
nats:
    docker compose up -d nats

nats-stop:
    docker compose down

# Run the HTTP gateway.
gw:
    cargo run -p vapi-gateway

# Run an engine worker (mock backend unless built with a model feature).
worker:
    cargo run -p vapi-worker

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

# Worker metrics, including prefix-cache hit rate.
metrics:
    @curl -s localhost:9090/metrics | grep -E '^vapi_' | grep -v '^#'

# Build with CUDA. Only on a machine with nvcc: this compiles FlashAttention
# from source once and takes tens of minutes. Set CUDA_COMPUTE_CAP for your
# card and CANDLE_FLASH_ATTN_BUILD_DIR (an existing directory) to cache it.
build-cuda:
    cargo build --release -p vapi-worker --features cuda

# Run a worker on real weights, on the GPU.
worker-cuda:
    cargo run --release -p vapi-worker --features cuda

# GPU tests: the kernel spike and the goldens in bf16.
test-cuda:
    cargo test -p vapi-backend-candle --features cuda -- --nocapture --test-threads=1
