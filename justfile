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

# Run an engine worker.
worker:
    cargo run -p vapi-worker

test:
    cargo test --workspace

check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings

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

# Build with CUDA. Only on a machine with nvcc: this compiles
# FlashAttention from source and takes tens of minutes.
build-cuda:
    cargo build --release -p vapi-worker --features cuda
