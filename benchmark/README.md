# vapi vs vLLM

Two models, same GPU and client. The LFM2.5-2.6B numbers are the ones that
matter (it is the target model); the SmolLM2 numbers are kept as the first
measurement.

## LFM2.5-2.6B (bf16), 2026-09-21

`LiquidAI/LFM2.5-2.6B`, 22 gated short-conv layers and 8 attention layers.
vLLM 0.29.1rc1 with `--max-model-len 8192 --gpu-memory-utilization 0.7`,
CUDA graphs on; vapi release build, 1,024 blocks. 64 distinct prompts,
128 tokens each, greedy, streaming, one warm-up request.

| Concurrent | TTFT p50 vapi / vLLM | Inter-token p50 vapi / vLLM | Throughput vapi / vLLM |
|---|---|---|---|
| 1 | 16 / 22 ms | 13.9 / 13.5 ms | 71 / 73 tok/s (1.03x) |
| 8 | 48 / 319 ms | 14.4 / 14.1 ms | 539 / 483 tok/s (0.90x) |
| 32 | 129 / 129 ms | 15.8 / 14.5 ms | 1,896 / 2,067 tok/s (1.09x) |
| 64 | 175 / 147 ms | 17.8 / 15.5 ms | 3,224 / 3,711 tok/s (1.15x) |

Final run, both servers restarted back to back, response cache off.
vapi leads on time to first token everywhere but batch 64, and by a lot
at batch 8 (48 ms against 319). Throughput is level at batch 1, ahead at
8, and 1.15x behind at 64, from 6.6x at the start of phase 4. Sampling
costs nothing: temperature 0.7 is within noise of greedy. What remains
is GPU time in the forward, candle's unfused per-layer kernels against
vLLM's.

## Qwen3-0.6B (bf16), after phase 7

Phase 7 carried the LFM2 work across: the fused FFN, and CUDA graph
capture, which needed the step split into prepared inputs and device-only
work with capture-safe row kernels in place of `index_select` and the
cache writes.

| Concurrent | TTFT p50 vapi / vLLM | Inter-token p50 vapi / vLLM | Throughput vapi / vLLM |
|---|---|---|---|
| 1 | 9 / 11 ms | 4.2 / 3.5 ms | 235 / 279 tok/s (1.19x) |
| 8 | 48 / 23 ms | 4.6 / 3.9 ms | 1,599 / 1,976 tok/s (1.24x) |
| 32 | 39 / 48 ms | 6.0 / 4.6 ms | 4,973 / 6,269 tok/s (1.26x) |
| 64 | 61 / 68 ms | 8.2 / 5.7 ms | 7,314 / 9,972 tok/s (1.36x) |

What the port moved:

| Concurrent | before phase 7 | after | vLLM |
|---|---|---|---|
| 1 | 186 tok/s | 235 | 279 |
| 8 | 1,205 tok/s | 1,599 | 1,976 |
| 32 | 4,278 tok/s | 4,973 | 6,269 |
| 64 | 6,058 tok/s | 7,314 | 9,972 |

The gap closed from 1.5-1.65x to 1.17-1.36x. Most of it was the graphs:
on a 0.6B model a decode step is small and the ~400 kernel launches per
step dominate, which is exactly what replay removes. The fused FFN was
worth a few percent on its own here, against ~25% on the 2.6B, because
these GEMMs are small enough that cuBLAS kernel choice matters less.

## Quantised weights, 2026-09-22

GGUF Q4_K_M through candle's quantised kernels, same client and GPU.
Greedy, tier 2 off.

**Qwen3-0.6B, quantised against dense bf16** (inter-token p50 / throughput):

| Concurrent | Q4_K_M | bf16 |
|---|---|---|
| 1 | 4.5 ms / 216 tok/s | 5.3 ms / 185 tok/s |
| 8 | 6.5 ms / 1,185 | 6.4 ms / 1,186 |
| 32 | 8.5 ms / 3,558 | 7.0 ms / 4,241 |
| 64 | 10.7 ms / 5,529 | 9.7 ms / 6,083 |

Quantisation wins at batch 1, where decode is bound by reading weights,
and loses above it, where the dequantisation is extra work on top of a
matrix multiply that was already compute-bound. That is the shape of the
trade, not a defect.

**Qwen2.5-7B-Instruct Q4_K_M**, which does not fit on this card in bf16
at all (14 GB of weights against 16 GB of VRAM, before any KV cache):

| Concurrent | TTFT p50 | Inter-token p50 | Throughput |
|---|---|---|---|
| 1 | 18 ms | 12.1 ms | 81 tok/s |
| 8 | 59 ms | 42.4 ms | 188 tok/s |
| 32 | 171 ms | 23.4 ms | 1,300 tok/s |

4.4 GB of weights, 896 MiB of KV cache, 8.3 GB of VRAM in total.

**The batch-8 figure is not a typo.** candle runs a vector kernel per row
for batches up to 8 and switches to a dequantise-and-multiply path above
it, so a batch of 8 pays eight vector passes while a batch of 32 pays one
matrix pass. Decode is therefore *slower* at 8 than at 32. The fix is a
quantised GEMM for the middle range, which is candle's to make.

## Reproducing

```sh
python -m venv benchmark/.venv && benchmark/.venv/bin/pip install vllm
benchmark/.venv/bin/vllm serve ~/models/smollm2-135m-instruct --served-model-name smollm2 \
    --dtype bfloat16 --port 8001 --max-model-len 4096
python benchmark/bench.py --url http://localhost:8001 --model smollm2 --label vllm
# stop vLLM, then (NATS up, model.path set in vapi.toml):
cargo run --release -p vapi-gateway & cargo run --release -p vapi-worker --features cuda
python benchmark/bench.py --url http://localhost:8080 --model m --label vapi
```
