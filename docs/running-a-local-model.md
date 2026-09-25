# Running a local model on vapi

This walks from a clean machine to an OpenAI-compatible endpoint serving a
Llama-architecture model from local weights, on the CPU or on an NVIDIA GPU.

## 1. What you need

- Rust 1.88 or newer (`rustc --version`).
- Docker, for NATS. Any NATS 2.10+ with JetStream works if you already run
  one; point `nats.url` at it.
- A model directory containing `config.json`, `tokenizer.json`,
  `tokenizer_config.json` (with the chat template), and the weights as
  `model.safetensors` or sharded files with
  `model.safetensors.index.json`. `generation_config.json` is optional and
  supplies the EOS ids. Supported `model_type`s: `llama` (Llama 2/3.x,
  SmolLM2, TinyLlama and other Llama-shaped checkpoints), `lfm2` (Liquid
  AI's LFM2 and LFM2.5 dense models, e.g. `LiquidAI/LFM2.5-2.6B`), and
  `laguna` (poolside's Laguna family in bf16/f32; the published FP8, INT4
  and NVFP4 checkpoints are not loadable yet, and Laguna-XS-2.1 needs at
  least 48 GB of GPU memory in any precision).
  Repos that ship only `tokenizer.model` need converting to
  `tokenizer.json` first.
- For the GPU path: an NVIDIA driver, and a CUDA toolkit with `nvcc` that
  supports your card (12.8 or newer for RTX 50-series). FlashAttention
  needs compute capability 8.0 or newer.

## 2. Get a model

Ungated, small, and the one the repo's golden tests use:

```sh
pip install -U huggingface_hub   # or: uv tool install huggingface_hub
hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir ~/models/smollm2-135m-instruct
```

Something more useful (gated: accept the licence on Hugging Face and run
`hf auth login` first):

```sh
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ~/models/llama-3.2-1b-instruct
```

A 1B model in bf16 needs about 2.5 GB of GPU memory for weights plus the
KV cache; 3B about 6.5 GB. On the CPU, weights load in f32, so double
those figures in RAM.

## 3. Start NATS

```sh
just nats            # docker compose up -d nats: client on :14222, monitoring on :18222
curl localhost:18222/healthz
```

The container listens on NATS' conventional 4222 and 8222 internally but is
**published on 14222 and 18222**, because a machine that already runs a NATS
service takes 4222 first and the failure is not loud: Docker publishes no
ports rather than refusing to start, so the container comes up healthy and
unreachable while the gateway connects to the *other* server and is turned
away with `authorization violation`. If you would rather use the conventional
ports, change the mapping in `docker-compose.yml` and `nats.url` in
`vapi.toml` together — they have to agree.

To check which server you actually reached:

```sh
curl -s localhost:18222/varz | grep -E '"server_name"|"version"|auth_required'
docker port vapi-nats-1        # empty output means nothing was published
```

and set `url = "nats://127.0.0.1:14222"`.

## 4. Configure

Edit `vapi.toml` (or write your own file and point `VAPI_CONFIG` at it).
Every key has a default; unknown keys are rejected rather than ignored.

```toml
[nats]
url = "nats://127.0.0.1:14222"

[gateway]
bind = "127.0.0.1:8080"          # 0.0.0.0:8080 to serve off the box
max_queued_requests = 256        # beyond this, 503 with Retry-After: 1 (0 = unlimited)

[worker]
max_concurrent_seqs = 64         # also the JetStream max_ack_pending
max_batched_tokens = 4096        # tokens per scheduler step
prefill_chunk_tokens = 1024      # long prompts are prefilled in chunks this size
metrics_bind = "127.0.0.1:9091"
max_step_failures = 3            # consecutive failed steps before the worker exits
drain_timeout_secs = 30          # on SIGTERM, how long running requests get to finish

[cache]
response_cache = true            # tier 2: exact-match cache of deterministic answers in NATS KV
response_cache_ttl_secs = 3600
spill = false                    # tier 3: keep evicted KV blocks off-device; measure before enabling
spill_dir = ".data/spill"
spill_ram_bytes = 2147483648     # host-memory tier
spill_max_bytes = 17179869184    # disk tier behind it; 0 keeps it in memory only

[model]
id = "HuggingFaceTB/SmolLM2-135M-Instruct"   # reported by /v1/models and used in NATS subjects
path = "/home/you/models/smollm2-135m-instruct"
max_context = 4096
num_blocks = 512                 # KV cache size in 32-token blocks; see sizing below
# dtype = "auto"                 # bf16 on CUDA, f32 on CPU; or f32 / f16 / bf16
# device = "auto"                # cuda when built with --features cuda and a GPU is present, else cpu
# cuda_graphs = true             # CUDA: capture pure-decode steps and replay them (LFM2 and Qwen)
```

**Sizing `num_blocks`.** Each block holds 32 tokens for every layer.
Bytes per token = 2 × layers × kv_heads × head_dim × bytes per element,
so a block is that × 32:

| Model | bf16 per block | 512 blocks | 2048 blocks |
|---|---|---|---|
| SmolLM2-135M (30 layers, 3 KV heads, 64 dim) | 0.7 MB | 0.36 GB | 1.4 GB |
| LFM2.5-2.6B (8 attention layers of 8 KV heads × 64, plus 22 conv layers storing 2048 per token) | 3.3 MB | 1.7 GB | 6.8 GB |
| Llama-3.2-1B (16 layers, 8 KV heads, 64 dim) | 1 MB | 0.5 GB | 2 GB |
| Llama-3.2-3B (28 layers, 8 KV heads, 128 dim) | 3.5 MB | 1.8 GB | 7 GB |

On the CPU the cache is f32, so double it. LFM2's conv layers keep their
input history per token in the same paged cache, which is why its blocks
are larger than an attention-only model of the same size. A request needs
`(prompt + max_tokens) / 32` blocks, rounded up; the prefix cache keeps
finished blocks around until the pool needs them, so more blocks means a
higher hit rate.

Both binaries read the same file. The gateway uses `model.path` for the
tokenizer and chat template; the worker uses it for the weights too.

## 5. Run it

`just up configs/<name>.toml cuda` starts NATS, the gateway and a worker on
one config, and Ctrl-C stops all of them. Use it: the gateway and the worker
have to agree on `model.id`, because it is part of the NATS subject, and when
they do not every request times out while the dashboard shows a healthy
worker.

Both binaries also take `--config <path>` on their own, falling back to
`$VAPI_CONFIG` and then to `./vapi.toml`; `--help` prints the rest. `configs/`
holds a worked example per model.

**On the CPU** (f32, the numerical reference; fine for a 135M model,
slow for 1B):

```sh
cargo run --release -p vapi-gateway
cargo run --release -p vapi-worker --features candle
```

**On the GPU.** The first build compiles FlashAttention from CUDA source,
which takes 20–30 minutes and eight cores; the build is cached in
`CANDLE_FLASH_ATTN_BUILD_DIR`, which must exist before you start.

```sh
mkdir -p ~/.cache/candle-flash-attn
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_COMPUTE_CAP=120                 # 80 A100 · 86 RTX 30xx · 89 RTX 40xx / L4 · 90 H100 · 120 RTX 50xx
export CANDLE_FLASH_ATTN_BUILD_DIR=$HOME/.cache/candle-flash-attn

cargo run --release -p vapi-gateway
cargo run --release -p vapi-worker --features cuda
```

The worker logs `kv cache allocated num_blocks=… dtype=BF16 device="cuda"`
and then `worker ready`. Without a `model.path` it runs a mock backend and
says so loudly; the API works but the tokens are noise.

## 6. Talk to it

Any OpenAI client works. With no `[[auth.keys]]` configured there is no
authentication and the API key can be anything, which is right on a loopback
box; see "API keys" below for locking it down.

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "m",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 32, "temperature": 0
}'
```

Streaming, with the Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="m",
    messages=[{"role": "user", "content": "Write a haiku about paging."}],
    stream=True, max_tokens=64, temperature=0.7, seed=1,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Supported request fields: `messages`, `max_tokens` (or
`max_completion_tokens`), `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stop` (string or list, honoured across token boundaries),
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `stream`,
`stream_options.include_usage`, `user`, and `logprobs` / `top_logprobs`
(up to 20; reported under the model's own distribution, before
temperature and penalties, in OpenAI's shape for both streaming and
non-streaming responses), and `tools` / `tool_choice` (`"auto"` or
`"none"`; naming a function returns 400). `n` must be 1. Unknown fields
return 400 rather than being silently dropped.
`/v1/completions` takes a raw prompt (and the legacy integer `logprobs`),
and `/v1/models` lists `model.id`.

**What happens under load and on failure.** When more than
`gateway.max_queued_requests` requests are waiting for a worker, new ones
get `503` with `Retry-After: 1` and an `overloaded` error code; the
OpenAI SDKs back off and retry on that. If a worker's forward pass fails
(a device error, say), every request it was serving gets an error frame
rather than a stalled stream, and the worker keeps going; after
`worker.max_step_failures` failures in a row it exits so a supervisor can
restart it, and JetStream hands its unfinished prompts to another worker.
On SIGTERM or Ctrl-C a worker stops taking work, finishes what it has
within `worker.drain_timeout_secs`, fails anything still running after
that, and exits.

**Quantised weights.** Point `model.path` at a directory holding a single
`.gguf` and its tokenizer files, and the weights load quantised:

```sh
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir ~/models/qwen3-q4
cp ~/models/qwen3-0.6b/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt} ~/models/qwen3-q4/
```

The config comes from the file's metadata rather than a `config.json`,
which is why none is needed. The tokenizer is not read from the GGUF: the
gateway and the worker share one tokenizer and chat template, and those
come from the directory. Qwen2 and Qwen3 GGUFs are supported today.

A 7B at Q4_K_M runs in about 8 GB including its KV cache, so it fits a 16
GB card that could not hold it in bf16. Decode is faster than bf16 at
batch 1 and slower above it, and candle has a cliff at batch 8; see
`benchmark/README.md` before sizing a deployment.

## The dashboard

The gateway serves a dashboard at `/dashboard`. It is rendered on the
server, its forms are ordinary posts, and the only script on the page
keeps the counters current; it works with JavaScript off and has no build
step.

It shows the workers that have registered themselves (running and waiting
sequences, KV cache use, prefix-cache hit rate, partitions, uptime), the
gateway's own counters, and the last 25 finished requests with their token
counts, latency, finish reason and whether the response cache answered
them. "Try it" sends a prompt through the same path a client uses, so what
the page shows is what a caller would get.

Four settings can be changed while it runs, because they live in the
gateway process: the queue limit before a 503, the two stream timeouts,
and whether the response cache is consulted. They take effect on the next
request and are **not** written back to `vapi.toml`, so a restart returns
to the file. Everything a worker acts on, the model, the cache sizes, the
batch limits, is read when the worker starts and is shown read-only. A
dashboard that appears to change a setting it cannot is worse than one
that admits the boundary.

It is behind the same API keys as `/v1` when any are configured — its "Try
it" box runs inference, so leaving it open would make authentication
decorative. A browser cannot set a header, so open `/dashboard?key=<key>`
once: the gateway exchanges the key for an `HttpOnly` cookie and redirects to
a clean URL, keeping the key out of history and referrers. With no keys
configured it is open, so bind the gateway to loopback.

**Which models run.** `model_type` in `config.json` picks the
implementation: `llama` (Llama 3.x, SmolLM2), `qwen2`, `qwen3`, `lfm2`
(LFM2.5) and `laguna`. Anything else is refused at load with a list of
what is supported, rather than loaded into the wrong shape. Qwen3-0.6B
and LFM2.5-2.6B are the two checked against Hugging Face on every change.

**Several completions.** `n` up to 8 returns that many choices. The
prompt is prefilled once and the choices fork from it, sharing its KV, so
`n` costs decode time rather than `n` prefills. Greedy requests are not
forked, since the choices would be identical; a seeded request offsets
the seed per choice so they differ and still reproduce. Budget per
choice: each one spends up to `max_tokens` of its own, and on a reasoning
model much of that goes on thinking.

**Structured output.** `response_format` makes the answer parse rather
than hoping it does:

```python
resp = client.chat.completions.create(
    model="m", max_tokens=600,
    messages=[{"role": "user", "content": "Weather in Paris? Make it up."}],
    response_format={"type": "json_schema", "json_schema": {"name": "weather", "schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "temperature_c": {"type": "integer"},
                       "conditions": {"enum": ["sunny", "cloudy", "rain", "snow"]}},
        "required": ["city", "temperature_c", "conditions"]}}},
)
```

Decoding is constrained token by token: anything that would break the
format is removed before sampling, so the result is valid by construction
rather than by luck. The supported schema subset is `object` with
`properties`, `required` and `additionalProperties`, `array` with
`items`, `string`, `number`, `integer`, `boolean`, `null`, `enum` and
`const`, and a schema with no `type` (any value). Anything else (a
`pattern`, an `anyOf`, a list of types) returns 400 rather than quietly
generating unconstrained text. Note that `additionalProperties` defaults
to **false** here, the opposite of JSON Schema: a constrained generation
with free-form extra keys is barely constrained.

On a reasoning model such as LFM2.5 the constraint does not apply to the
thinking. The model reasons first, and the moment it would end its turn
it is made to close the reasoning block instead, after which every token
is constrained. Thinking gets at most three quarters of `max_tokens`, and
the last few tokens are reserved for closing whatever the document has
open, so a constrained request comes back with something that parses even
when the budget runs out. Budget accordingly: a request that needs 200
tokens of JSON wants a `max_tokens` well above that.

**Tool calling.** With a model whose format the gateway recognises
(LFM2.5 today), `tools` are rendered into the prompt by the model's own
chat template, and what the model writes back is parsed into OpenAI's
shape: `<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>`
becomes a `tool_calls` entry with JSON arguments and a `tool_calls`
finish reason, and the model's `<think>` block becomes
`reasoning_content` rather than being served as the answer. Streaming
sends whole calls in one chunk. Feed a result back as a `tool` message
with its `tool_call_id`, as you would with OpenAI. A call the gateway
cannot parse is passed through as text rather than dropped. Asking for
tools from a model with no such format returns 400.

**Running more than one worker.** Workers of the same model share a queue
and compete for prompts, so nothing needs configuring to add one. What
that costs is prefix-cache locality: turn three of a chat may land on a
worker that never saw turns one and two. Setting `nats.job_partitions` to
the number of workers and giving each worker its own `worker.partitions`
routes every turn of a conversation to the same worker, by hashing the
prompt's first block. Measured on mock workers, that lifted the aggregate
prefix-cache hit rate from 0.40 to 0.49 and left the load split 15/25
instead of 20/20: content routing does not balance work. Both files must
agree on `nats.job_partitions`; workers register themselves in the
`VAPI_WORKERS` KV bucket, which is for looking at, not for routing.

**The spill tier (tier 3)** keeps evicted KV blocks in host memory, and
past `spill_ram_bytes` on disk, so a returning conversation copies its
prefix back instead of recomputing it. Disk entries survive a restart, and
are keyed by a hash that pins the model and its weights, so they can never
be served to the wrong model. It is off by default because on this
hardware it is close to break-even: restoring a block costs about 2.4 ms
against 4.1 ms to recompute it, but every eviction pays 2.7 ms to write,
whether or not the block is ever wanted again. Turn it on for repeat
traffic over long shared prefixes, and watch
`vapi_spill_hits_total` against `vapi_spill_writes_total`: below about 1.5
hits per write it is costing more than it saves. `benchmark/README.md` has
the measurement.

**The response cache.** A request whose answer is a pure function of its
input (`temperature: 0`, or any temperature with a `seed`) is looked up
in NATS KV before it is queued, keyed on the model's weight fingerprint,
the prompt token ids and every sampling parameter including `max_tokens`
and `stop`. A hit is answered by the gateway alone, replayed token by
token when streaming. Requests asking for `logprobs` are not cached.
`vapi_response_cache_{hits,misses,writes}_total` on the gateway's
`/metrics` show whether it is earning its keep.

`temperature: 0` is greedy and reproduces Hugging Face `transformers`
token for token on the CPU (that is what `tests/goldens/` checks). On the
GPU in bf16, expect the occasional different token at an exact tie. One
caveat when comparing by hand: HF's `generate()` silently applies the
checkpoint's `generation_config.json` defaults, and LFM2.5 ships a
repetition penalty of 1.1 and top-k 50 there, so its `generate()` is not
plain argmax even with `do_sample=False`. vapi applies only what the
request asks for; pass `repetition_penalty: 1.1` yourself to match Liquid's
recommended defaults.

## Decision models

Some checkpoints do not generate text at all. A **decision model** is a
bidirectional encoder with a small head: you give it a state and a set of
typed questions, and it answers all of them in one forward pass with
calibrated probabilities. There is nothing to parse and nothing to stream.

`convaiinnovations/laya` is the one vapi has been proven against. The repo
holds two usable checkpoints: an English ModernBERT-large at the root, and an
mmBERT-base at `multilingual/` that reads 100+ languages and runs about twice
as fast. Both pass parity against the reference.

```sh
huggingface-cli download convaiinnovations/laya --local-dir ~/models/laya
```

Nothing in the config selects it. Point `model.path` at the directory and the
worker recognises the layout — an `encoder/config.json` beside an
`rl_agent_config.json` — and loads its single-pass engine instead of the
batching decoder. The gateway does the same and serves `/v1/decisions`:

```toml
[model]
id = "convaiinnovations/laya"
path = "/home/you/models/laya"          # or .../laya/multilingual
max_context = 512                       # 1024 for the multilingual checkpoint
device = "cuda"
dtype = "bf16"
```

`max_context` is the checkpoint's own budget, not the encoder's architectural
limit. On an RTX 5060 Ti one question takes 10 ms on the English checkpoint
and 8 ms on the multilingual one; sixteen take 69 ms and 35 ms.

Three question types, and each shapes its answer differently:

| type | criteria | answer |
|---|---|---|
| `choice` | named options, with optional descriptions | the winning key, and a probability for every option |
| `score` | ordinal levels, low to high | the **expectation** over the levels — 1.44, which no single level is |
| `noul` | none, or descriptions of true and false | one probability: that the statement holds |

```sh
curl localhost:8080/v1/decisions -H 'Content-Type: application/json' -d '{
  "state": "The build has been red for three days and nobody has looked at it.",
  "questions": {
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["cosmetic", "degraded", "outage"]},
    "needs_action": {"type": "noul", "instructions": "Does this require someone to act?"}
  }}'
```

The state may be a string or any JSON document. Question order is preserved,
and so is the order of a `choice`'s options: they go into the prompt in that
order, so reordering them changes what the model reads.

**What the confidences mean.** `confidence` is one minus the normalised
entropy of the answer: 1.0 when the model is certain, 0.0 when it is spreading
evenly. It is normalised so a two-option and a twenty-option question are on
one scale. `act_probability` is the model's own read on whether to answer or
escalate.

**What they do not mean.** The published checkpoint ships overconfident — mean
ECE 0.466, and 0.081 after refitting one temperature per (question type,
option count). vapi reads that fitted table from `rl_agent_config.json` and
applies it, but the table is the publisher's, fitted on the publisher's data.
Refit it on yours before you act on a threshold. The multilingual checkpoint
ships **no** fitted table — every temperature is 1.0 — so its probabilities
are raw, at a published ECE of 0.314. And the base checkpoint is
near chance zero-shot on the benchmark it is advertised with — 0.362 against a
0.461 majority-class baseline. It is a fast base to specialise, not a
zero-shot decision engine.

**Scripts.** The English checkpoint does not fail gracefully on text it cannot
read: 0.000 accuracy at 0.952 confidence on Khmer, which means no confidence
threshold catches it. The gateway reads the dominant script of every state and
logs a warning plus `vapi_decision_unreadable_script_total` when the loaded
checkpoint cannot read it. On one Hindi ticket the English checkpoint answers
`billing` at confidence 0.06 and the multilingual one at 0.95 — the warning is
the difference between noticing that and not.

Serving both at once and dispatching per request is **not** built: the
sequence has to be built with the tokenizer of whichever checkpoint will
answer it, and the two differ in tokenizer, budget and calibration. Run two
vapi deployments if you need both today.

**Options and budget.** The question and its options share `head_max_len`
tokens (192 on the English checkpoint) and the state gets the rest. Beyond
about twenty options they start getting a few tokens each, stop being
distinguishable, and accuracy falls sharply — 0.425 against 0.870 on a
77-option benchmark. vapi logs when it had to cut an option and refuses only
when an option cannot fit at all. Raise `head_max_len` in the checkpoint's
config, or split one large choice into a coarse question and a fine one.

**What a decision worker does not have.** No KV cache and no prefix cache, so
the dashboard shows zero for both. That is correct rather than missing:
attention here is bidirectional, so a prefix's representation depends on the
state that follows it and caching one would be wrong, not merely useless. The
response cache is off for decisions too — they would cache perfectly, but the
cache stores generated text.

To measure it: `python benchmark/bench_decisions.py` for the end-to-end view,
and `cargo run --release --features cuda --example decision_bench -p
vapi-backend-candle` for the forward pass alone. On an RTX 5060 Ti a single
question is about 10 ms end to end and four are 20 ms.

## Transcribing audio

A **speech model** takes audio and returns text. `mistralai/Voxtral-Mini-4B-Realtime-2602`
is the one vapi has been proven against:

```sh
huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --include 'config.json' 'model.safetensors' 'generation_config.json' \
  'processor_config.json' 'tekken.json' --local-dir ~/models/voxtral-realtime
```

The `--include` matters: without it you also get `consolidated.safetensors`,
a second copy of the same weights in Mistral's own format that nothing here
reads, for another 8.3 GB.

Nothing in the config selects it. Point `model.path` at the directory and the
worker recognises the layout — `model_type: voxtral_realtime` — and loads its
transcription engine; the gateway serves `/v1/audio/transcriptions`.

```sh
just up configs/voxtral.toml cuda

curl localhost:8080/v1/audio/transcriptions \
  -F file=@clip.wav -F response_format=verbose_json
```
```json
{"task": "transcribe", "duration": 1.428, "text": " Front, center,",
 "processing_seconds": 0.94, "realtime_factor": 1.51}
```

`response_format` accepts `json` (the default, just `{"text": ...}`), `text`
for the bare transcript, and `verbose_json` for the duration and the realtime
factor. OpenAI's `language`, `prompt` and `temperature` fields are accepted
and ignored; `srt` and `vtt` are refused rather than silently answered as
JSON.

**What it accepts.** WAV only — PCM 8/16/24/32-bit or float32, any sample
rate, mono or stereo; the gateway resamples and mixes down. MP3 and AAC would
mean a codec dependency and a much larger attack surface for something a
caller controls, so the gateway says "send me WAV" instead of failing deep
inside a decoder.

**Clips are capped at 150 seconds.** The audio rides inside the job envelope
as base64'd 16-bit PCM, about 42 KB a second, and `nats.conf` raises NATS'
`max_payload` to 8 MB to carry it. Past the cap you get a 413 that says so.
Lifting it means putting the audio in the JetStream object store and sending
its name, which is not built.

**Speed.** One decoder position covers 80 ms of audio, so a live session
would consume 12.5 positions a second. On an RTX 5060 Ti this runs at about
50, which is **3.9x realtime** end to end — a minute of audio in fifteen
seconds. The reference implementation manages 1.54x on the same card.

**One clip at a time.** A worker transcribes sequentially: the decode is
lock-step with the audio and two clips cannot yet share a forward pass. That
is the throughput story and it needs the engine integration that live audio
also needs. Memory is not the constraint — 8.25 GB of weights and about
0.33 GB for a minute-long session — compute is.

**Live audio over a session is not built.** This is the offline path: upload
a file, get a transcript.

## API keys

Without `[[auth.keys]]` the gateway is open and every caller shares one cache
namespace. Adding keys changes both:

```toml
[[auth.keys]]
name = "alice"
key = "sk-..."

[[auth.keys]]
name = "bob"
key = "sk-..."
namespace = "shared"     # optional: pool a team onto one cache namespace
```

```sh
curl localhost:8080/v1/chat/completions -H 'Authorization: Bearer sk-...' ...
```

`x-api-key` works too. Keys are compared as digests in constant time, and the
loop does not stop at the first match, so neither the timing nor the length
of a guess tells an attacker anything.

**Each key gets its own cache namespace**, and that is the reason the two
features arrived together rather than a bonus. A shared prefix cache is a
**timing side channel**: time-to-first-token reveals whether *someone*
recently submitted a given prefix, so a gateway that can tell callers apart
should keep their cached prefixes apart. Set the same `namespace` on several
keys to let a team share a cache deliberately. The tier-2 response cache is
namespaced the same way — the same prompt does produce the same text, but a
cache is a record of what someone asked.

You can watch the isolation work. With `cache.response_cache` off, send a
long shared prefix from one key twice and then from another, and read
`vapi_cached_prefix_tokens_total`:

```
alice #1 (cold)        -> +0
alice #2 (own prefix)  -> +512
bob   #3 (same prefix) -> +0      <- no reuse across keys
bob   #4 (own prefix)  -> +512
```

**What stays open:** `/health`, so a liveness probe needs no credential, and
`/metrics`, because that is what a scraper expects. Neither exposes a prompt
or an answer. Everything under `/v1` and `/dashboard` requires a key.

**What this is not.** There are no scopes, no per-key rate limits or quotas,
and no expiry or rotation — a key is a shared secret in a config file, so
treat the file as one and restart to change it.

## 7. Watch it

```sh
just metrics          # worker on :9091: vapi_scheduler_running, vapi_kv_cache_utilization, vapi_prefix_cache_hit_rate
curl -s localhost:8080/metrics | grep vapi_   # gateway: vapi_cached_prefix_tokens_total, vapi_stream_gap_total
curl -s localhost:18222/jsz?consumers=true     # JetStream: queue depth and the consumer's max_ack_pending
```

`vapi_stream_gap_total` above zero means a token delta was lost between
worker and gateway; the request is failed rather than served with a word
missing. Send the same system prompt twice and watch
`vapi_prefix_cache_hit_rate` and `vapi_cached_prefix_tokens_total` rise.

`just test-e2e` runs the other kind of test: the real binaries against a real
NATS and a real checkpoint, over HTTP — one for decisions and one for
transcription. It is the only thing that catches the class of bug that needs
separate processes to appear: a gateway and a worker that disagree about
`model.id` and so never meet, a worker that loads the wrong engine for a
checkpoint, an answer that survives the wire but loses its question order.
Each skips with a reason when the binaries, the model or NATS are missing,
rather than passing silently.

To load-test, `python benchmark/bench.py --url http://localhost:8080
--model m --label vapi` runs 1, 8, 32 and 64 concurrent streams and
reports TTFT, inter-token latency and throughput.

## 8. When it goes wrong

| Symptom | Cause and fix |
|---|---|
| `Directory doesn't exists: …/candle-flash-attn` during the cuda build | `mkdir -p` the `CANDLE_FLASH_ATTN_BUILD_DIR` first; the build script won't create it. |
| `no cuda implementation for softmax-last-dim` (or rms-norm, rope) | The binary was built with only `candle-core/cuda`. Use `--features cuda` on the worker, which enables candle-nn and candle-transformers CUDA too. |
| `CUDA_ERROR_INVALID_VALUE` on the first forward | A zero-element kernel launch; fixed for empty logits, so a new one means a new empty tensor on the GPU path. Look for a batch that produced nothing. |
| `model_type "qwen2" is not supported` | Only `llama`, `lfm2` and `laguna` checkpoints load today. |
| `syntax error: unknown statement generation` or `map has no method named get` when loading a template | Fixed in this tree: `{% generation %}` tags are stripped and Python dict/str methods are supplied through minijinja's pycompat hook. A template using some other Jinja2 extension would show up the same way; the error names the construct. |
| `tokenizer.json not found` | The repo ships only sentencepiece `tokenizer.model`; convert it with `transformers` (`AutoTokenizer.from_pretrained(dir).save_pretrained(dir)`). |
| Worker starts but the gateway's first request hangs, then 504 | The worker is not consuming: check `worker ready` in its log, that both read the same `nats.url`, and that `model.id` is the same in both (it is part of the NATS subject). |
| Throughput plateaus and `vapi_scheduler_running` never exceeds some small number | The JetStream consumer is durable; an old run created it with a smaller `max_concurrent_seqs`. The worker now updates it on start, but if in doubt delete the consumer or the stream and restart. |
| `401 missing api key` from a client that used to work | `[[auth.keys]]` is now set. Send `Authorization: Bearer <key>`; for the dashboard, open `/dashboard?key=<key>` once. |
| Prefix-cache hit rate drops after adding API keys | Expected: each key has its own namespace, so callers no longer warm each other's cache. Give keys that should share one the same `namespace`. |
| `authorization violation` connecting to NATS | You reached a *different* NATS than the one you started — something else owns the port, and it wants credentials. `docker port vapi-nats-1` with empty output confirms the container published nothing. See section 3. |
| First request takes ~200 ms, later ones ~20 ms | One-off CUDA initialisation. Send a warm-up request after starting the worker. |
| Out of GPU memory at start | Lower `num_blocks`, or `max_concurrent_seqs`; weights plus cache must fit. |
| Gibberish output | You are on the mock backend: no `model.path`, or the worker was built without `--features candle`/`cuda`. The worker log says which. |
