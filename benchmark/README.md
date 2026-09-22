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
| 1 | 18 / 24 ms | 14.7 / 13.5 ms | 68 / 74 tok/s (1.1x) |
| 8 | 78 / 316 ms | 21.7 / 13.9 ms | 361 / 490 tok/s (1.4x) |
| 32 | 191 / 138 ms | 66.7 / 14.4 ms | 472 / 2,076 tok/s (4.4x) |
| 64 | 338 / 146 ms | 109.3 / 15.3 ms | 573 / 3,761 tok/s (6.6x) |

- **At batch 1 the two are at parity**: vapi's step is 14.7 ms to vLLM's
  13.5, and vapi's time to first token is lower. The per-token math is not
  the problem.
- **vapi does not batch well on this model.** Inter-token latency grows
  almost linearly with concurrency (14.7 → 21.7 → 66.7 → 109 ms) while
  vLLM's stays flat (13.5 → 15.3 ms). The cause is the conv operator:
  every one of the 22 conv layers loops over the sequences in the batch
  and, for each, gathers the sequence's *whole* history through the block
  table to read the two rows before its chunk. That is O(sequence length)
  work per sequence per layer per step, done as several small tensor ops
  each. At 64 sequences × 22 layers that is thousands of launches and
  hundreds of megabytes of gathers per decode step.
- Second and third: the per-sequence `slice_set` cache writes (now for 30
  layers, since conv layers write too) and the full-vocab f32 logits copy
  (128,000 × 4 bytes = 512 KB per row).

### Phase 4 progress on LFM2.5-2.6B

Inter-token latency p50 / throughput, same benchmark, after each change:

| Concurrent | baseline | + batched conv | + scatter write | + device argmax, hoisted indices | vLLM |
|---|---|---|---|---|---|
| 1 | 14.7 ms / 68 | 14.8 ms / 67 | 14.8 ms / 67 | 14.7 ms / 68 | 13.5 ms / 74 |
| 8 | 21.7 ms / 361 | 17.3 ms / 452 | 17.3 ms / 452 | 15.6 ms / 500 | 13.9 ms / 490 |
| 32 | 66.7 ms / 472 | 43.6 ms / 722 | 43.5 ms / 722 | 17.9 ms / 1,685 | 14.4 ms / 2,076 |
| 64 | 109.3 ms / 573 | 77.8 ms / 803 | 76.2 ms / 814 | 25.5 ms / 2,305 | 15.3 ms / 3,761 |

- **Batched conv** (one indexed gather per tap for the whole batch instead
  of a per-sequence loop over each sequence's whole history): 109 → 78 ms
  at batch 64. Batch 1 unchanged, as expected.
- **One scatter per layer** for the cache writes instead of one copy per
  sequence: 78 → 76 ms, within noise. The writes were not the cost.
- **Profile** (per-phase step histograms, now permanent): a batch-64 step
  was 38 ms = 31 ms forward (8.5 ms issuing kernels, 25 ms waiting and
  copying 33 MB of logits) + 6 ms host argmax over 64 × 128,000 floats.
- **Device argmax for greedy rows, and the per-step index tensors built
  once instead of once per layer** (~200 synchronous host-to-device copies
  per step gone): 76 → 25.5 ms at batch 64, 2.8x the throughput. The step
  is now 25 ms of forward (6.2 ms issue, 13.3 ms wait) and 0.07 ms of
  sampling.

- **RoPE without transposes** (`rope_thd` on the projection layout) and
  a **warm-up forward at load**: greedy unchanged at 25.5 ms (the copies
  were cheap), sampled 74.5 → 58.4 ms p50 at batch 64 with a p95 of 76 ms,
  so within run-to-run variance. The warm-up moves the ~175 ms first-request
  cost to startup, where the log now reports it.

- **CUDA graphs** (`model.cuda_graphs = true`; greedy pure-decode steps
  captured once per batch-size bucket and replayed). Inter-token p50 /
  throughput:

| Concurrent | eager (device argmax) | CUDA graphs | vLLM |
|---|---|---|---|
| 1 | 14.7 ms / 68 | 14.3 ms / 70 | 13.5 ms / 74 |
| 8 | 15.6 ms / 500 | 14.7 ms / 529 | 13.9 ms / 490 |
| 32 | 17.9 ms / 1,685 | 16.6 ms / 1,793 | 14.4 ms / 2,076 |
| 64 | 25.5 ms / 2,305 | 23.9 ms / 2,456 | 15.3 ms / 3,761 |

  A real gain at every size but a modest one: 25.5 → 23.9 ms at batch 64.
  The step's host-side kernel issue was largely overlapped with GPU
  execution already, so replay recovers ~6%, not the 25% estimated; the
  remaining 23 ms per batch-64 step is GPU work (weight reads plus the
  per-layer kernels). Getting it required capture-safe row gather/scatter
  kernels (`rows.rs`), because candle's `index_select` and `scatter_set`
  upload index metadata from a temporary host buffer at launch. The 2.6B
  goldens pass through graph replay (`VAPI_TEST_CUDA_GRAPHS=1`).

- **Fused kernels, guided by a kernel profile.** Nsight Systems on a
  standalone decode driver (`examples/decode_profile.rs`; nsys 2025.1 is
  needed for this driver/GPU) showed one cuBLAS kernel taking 12 of the
  23 ms at batch 64: the FFN gate and up projections, 60 GEMMs per step,
  each running at 217 GB/s on a 16×16-tile kernel cuBLAS picks for M=64,
  while the same-sized down projection ran at 376 GB/s.
  `examples/gemm_bench.rs` reproduced it and showed one GEMM of doubled
  width runs at 371 GB/s. So `w1` and `w3` are stacked into one weight at
  load and split by a fused SwiGLU kernel (`fused.rs`, nvrtc, all
  arguments by value so it is capture-safe). The conv layers got the same
  treatment: one `in_proj` GEMM instead of three, and two kernels (state
  write, then taps + gate) replacing ~15 element-wise launches per layer.
  Inter-token p50 / throughput:

| Concurrent | CUDA graphs | + fused FFN | + fused conv | vLLM |
|---|---|---|---|---|
| 1 | 14.3 ms / 70 | 14.2 ms / 70 | 13.9 ms / 71 | 13.5 ms / 74 |
| 8 | 14.7 ms / 529 | 14.7 ms / 529 | 14.4 ms / 540 | 13.9 ms / 490 |
| 32 | 16.6 ms / 1,793 | 16.2 ms / 1,850 | 15.8 ms / 1,888 | 14.4 ms / 2,076 |
| 64 | 23.9 ms / 2,456 | 18.6 ms / 3,105 | 17.8 ms / 3,240 | 15.3 ms / 3,761 |

  The batch-64 step went 23.9 → 18.6 → 17.8 ms. The weights are 5.2 GB
  and stream at ~400 GB/s on this card, a floor near 13 ms; the two FFN
  GEMMs now account for ~10.6 ms of the step and are within 10% of that
  bandwidth, so what is left is the smaller projections and the ~1 ms of
  norms, residual adds and attention. Goldens pass through graph replay
  with both fusions.

- **Device-side candidate selection for the sampled path.** With the
  fusions in, sampled decode had become 4x slower than greedy at batch 64:
  33 MB of logits copied to the host per step and two O(vocab) host passes
  per row. Now one kernel per step (`fused.rs::select_candidates`, one
  block per row) finds each row's maximum, bins every token by its
  temperature-scaled gap below it in whole nats, and compacts the deepest
  prefix of bins that fits in 8,192 entries, along with the row's full
  softmax denominator. The host sampler then works on those entries with
  exact normalisation. A row is served exactly when its whole 30-nat
  window came back, or when it has a top-k and at least `k + distinct
  observed tokens` came back (a penalty can only lower a logit, so nothing
  outside the raw top `k + observed` can enter the penalised top k; greedy
  is k = 1). Any other row gets its full logits copied, on its own, and
  goes through the old path. `vapi_engine`'s sampler tests draw 200 tokens
  per case from both paths with the same seed and require identical
  tokens. Inter-token p50 / throughput:

| Concurrent | greedy | sampled, host (temp 0.1, top-k 50, rep 1.1) | sampled, device candidates (same) | temp 0.7, no top-k, candidates only | temp 0.7 / 1.0, no top-k, Gumbel draw |
|---|---|---|---|---|---|
| 1 | 13.9 ms / 71 | 14.1 ms / 70 | 14.0 ms / 71 | 14.1 ms / 69 | 14.1 / 14.1 ms |
| 8 | 14.4 ms / 540 | 16.0 ms / 489 | 14.5 ms / 535 | 16.1 ms / 476 | 14.6 / 14.6 ms |
| 32 | 15.8 ms / 1,888 | 32.1 ms / 968 | 15.9 ms / 1,885 | 25.8 ms / 1,211 | 16.0 / 16.0 ms |
| 64 | 17.8 ms / 3,240 | 69.6 ms / 899 | 17.7 ms / 3,242 | 35.8 ms / 1,687 | 17.8 / 17.9 ms |

  The model's recommended settings now run at greedy speed. Without a
  top-k the candidate window holds thousands of tokens on many rows (p50
  351, p90 5,500 at temperature 0.7 on real prompts; most of the
  vocabulary at 1.0), so those rows are instead **drawn on the device**:
  when nothing but temperature and penalties applies (no top-k, top-p 1,
  min-p 0) the token is `argmax_v(logit_v / T + g_v)` with `g_v` standard
  Gumbel noise from a hash of `(seed, draw index, v)`, an exact sample
  from the tempered softmax (`fused.rs::gumbel_draw`; penalties are
  applied in place on the device first). A seeded request reproduces its
  tokens run to run; the draws differ from what the host sampler would
  have produced for the same seed, which is the accepted trade. Greedy
  parity at every temperature; the candidates-only column is kept to show
  what the draw replaced.

**Where it stands against vLLM after phase 4 so far** (greedy, graphs on):

| Concurrent | TTFT p50 vapi / vLLM | Inter-token p50 vapi / vLLM | Throughput vapi / vLLM |
|---|---|---|---|
| 1 | 16 / 24 ms | 13.9 / 13.5 ms | 71 / 74 tok/s (1.0x) |
| 8 | 48 / 316 ms | 14.4 / 13.9 ms | 540 / 490 tok/s (0.9x) |
| 32 | 109 / 138 ms | 15.8 / 14.4 ms | 1,888 / 2,076 tok/s (1.1x) |
| 64 | 222 / 146 ms | 17.8 / 15.3 ms | 3,240 / 3,761 tok/s (1.2x) |

Parity at batch 1 and 8, 1.1x behind at 32, 1.2x at 64, from 6.6x at the
start of the phase. The remaining 2.5 ms at batch 64 is spread over the
small projections and the element-wise kernels between them; the sampled
path (above) is now the larger problem.

### The sampled path (Liquid's recommended settings)

`temperature 0.1, top_k 50, repetition_penalty 1.1`, which is what the
model card recommends and what a real client sends. Inter-token latency
p50 / throughput:

| Concurrent | sampled, before | sampled, after | greedy | vLLM (greedy) |
|---|---|---|---|---|
| 1 | 18.9 ms / 52 | 14.8 ms / 67 | 14.7 ms / 68 | 13.5 ms / 74 |
| 8 | — | 17.1 ms / 456 | 15.6 ms / 500 | 13.9 ms / 490 |
| 32 | — | 24.7 ms / 1,230 | 17.9 ms / 1,685 | 14.4 ms / 2,076 |
| 64 | 334.9 ms / 194 | 74.5 ms / 927 | 25.5 ms / 2,305 | 15.3 ms / 3,761 |

Before, the host sampler built and sorted 128,000 (token, probability)
pairs per row: 268 ms per batch-64 step, 4 ms a row, thirteen times the
greedy step. It now finds the row maximum without allocating, gathers only
tokens within 30 nats of it (the rest carry less mass than f32 can
represent in the sum; a test checks the surviving prefix against the old
implementation), and sorts that short list: 7 ms per step. Rows are
sampled in place instead of being copied first. What is left on this path
is the 33 MB full-vocab logits copy (~8 ms per step at batch 64) and two
O(vocab) host passes; an exact device-side top-k needs a custom kernel,
since candle's GPU sort cannot handle a 128K-wide row.

## SmolLM2-135M-Instruct (bf16), 2026-09-21


Same model, same GPU, same client, servers run one at a time so they never
share the card.

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB, compute 12.0, driver 595, CUDA 12.8
- Model: HuggingFaceTB/SmolLM2-135M-Instruct, bf16
- vLLM 0.29.1rc1 (torch 2.13 cu130), `vllm serve --dtype bfloat16 --max-model-len 4096`, CUDA graphs on
- vapi release build with `--features cuda`, default `vapi.toml` limits (64 concurrent, 4096 batched tokens)
- Client: `bench.py`, N threads, distinct ~30-token prompts, 128 greedy tokens each, streaming, one warm-up request first
- Date: 2026-09-21

| Concurrent | TTFT p50 vapi / vLLM | Inter-token p50 vapi / vLLM | Throughput vapi / vLLM |
|---|---|---|---|
| 1 | 7 / 15 ms | 4.5 / 1.5 ms | 216 / 610 tok/s (2.8x) |
| 8 | 16 / 24 ms | 7.7 / 1.6 ms | 902 / 3,749 tok/s (4.2x) |
| 32 | 23 / 76 ms | 13.8 / 2.2 ms | 2,057 / 9,159 tok/s (4.5x) |
| 64 | 33 / 59 ms | 22.0 / 3.3 ms | 2,660 / 13,705 tok/s (5.2x) |

## Reading it

- **Time to first token** is lower on vapi at every batch size. Prefill of a
  30-token prompt is cheap for both; the difference is request overhead, and
  vapi's path (axum, one JetStream publish, one core-NATS subscription) is
  shorter than vLLM's Python front end.
- **Decode** is where vLLM wins: 3x at batch 1 and 6.7x per token at batch
  64. vapi's inter-token latency grows almost linearly with batch (4.5 →
  22 ms) while vLLM's barely moves (1.5 → 3.3 ms). For a 135M model the
  matmuls are nearly free, so vapi's step is dominated by per-sequence
  overhead:
  1. `write_kv_to_cache` issues one `slice_set` copy per sequence per layer
     during decode: 64 × 30 = 1,920 small launches per step at batch 64.
  2. `Logits` copies every sampled row (49,152 f32 = 192 KB) to the host and
     samples there.
  3. No CUDA graphs: every step re-launches ~30 layers × ~15 kernels from the
     host. vLLM captured 51 piecewise and 35 full graphs at startup.
  4. The reference-path RoPE and the flash kernel each take `.contiguous()`
     copies around transposes.
- **Throughput** follows from decode: 2,660 vs 13,705 tok/s at batch 64.

The gap will narrow on a larger model, where the matmuls dominate and the
per-sequence overhead is a smaller share of the step, but it will not close
without (1) and (2) above and probably (3).

## Tier 3: is restoring a KV block cheaper than recomputing it?

`crates/vapi-backend-candle/examples/spill_profile.rs`, LFM2.5-2.6B bf16 on
the 5060 Ti. One block is 32 tokens of KV across all 30 layers.

| Operation | Per block |
|---|---|
| Export, device to host | 2.66 ms (1.3 GB/s) |
| Import, host to device | 2.36 ms (1.4 GB/s) |
| Recompute by prefill, batch 1 | 4.54 ms |
| Recompute by prefill, batch 16 | 4.14 ms |

Restoring a block costs a little over half of recomputing it, so a hit is
worth about 1.8 ms. But every eviction pays the export whether or not the
block is ever asked for again, so the tier only pays for itself when a
spilled block is restored more than about 1.5 times on average. That is a
high bar for anything but repeat traffic over a shared prefix, which is
why `cache.spill` defaults to off.

Both transfers run at 1.3 GB/s, well under what PCIe can do, because
candle copies through pageable host memory. Pinned buffers would roughly
halve the cost and move the break-even to well under one restore per
eviction. Worth doing before turning this on by default.

The first implementation exported each layer separately, which cost 52 ms
a block: sixty small transfers, each with its own synchronisation. Joining
the slices on the device and copying once is the whole difference between
a tier that helps and one that is twenty times worse than recomputing.

## Cache-affinity routing across workers

Two mock workers, eight conversations of five turns each, all in flight at
once, `nats.job_partitions = 2` against a single shared queue. The number
that matters is the prefix-cache hit rate, since that is what affinity is
for.

| Routing | Jobs per worker | Prefix hit rate | Aggregate |
|---|---|---|---|
| One shared queue | 20 / 20 | 0.40 / 0.40 | 0.40 |
| Partitioned by conversation | 15 / 25 | 0.70 / 0.37 | 0.49 |

Affinity lifts the aggregate hit rate by about a fifth and unbalances the
load, which is the trade a hash makes: it routes by content, not by who is
free. With eight conversations over two partitions the split lands 15/25;
more conversations even it out, fewer make it worse. On a real model with
long system prompts the locality is worth more than it is here, because
each turn shares more of its prefix with the last.

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
