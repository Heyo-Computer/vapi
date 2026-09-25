# Example configurations

One file per model, each runnable with:

```sh
just up configs/<name>.toml cuda    # NATS, gateway and worker, one config, Ctrl-C stops all
```

Or separately, which is what `just up` does for you:

```sh
just nats
cargo run --release -p vapi-gateway -- --config configs/<name>.toml
cargo run --release -p vapi-worker --features cuda -- --config configs/<name>.toml
```

The `$VAPI_CONFIG` environment variable does the same thing as `--config` and
is easier to export once for a shell; `--config` wins when both are set.

The gateway and the worker **must read the same file**: `model.id` is part of
the NATS subject, so a mismatch leaves the gateway publishing to a queue no
worker consumes, and requests time out rather than failing. Worse, the
dashboard still shows a healthy worker, because registration goes through a
separate KV bucket that does not depend on the subject matching. `just up`
takes one config for both, which is the point of it.

Every path below points into `~/models`, which on this machine is a mix of
symlinks into the Hugging Face cache and directories downloaded straight
there. Nothing requires that layout — point `model.path` wherever the weights
actually are.

| file | model | fits a 16 GB card |
| --- | --- | --- |
| `qwen3-0.6b.toml` | Qwen3-0.6B, bf16 | yes, with room to spare |
| `qwen2.5-7b-q4.toml` | Qwen2.5-7B-Instruct, GGUF Q4_K_M | yes, 8.3 GB |
| `laya.toml` | Laya decision model, English | yes, 0.9 GB |
| `laya-multilingual.toml` | Laya decision model, 100+ languages | yes, 0.7 GB |
| `laguna.toml` | Laguna-XS-2.1, FP8 | **no** — needs >=48 GB, and no FP8 loader |
| `voxtral.toml` | Voxtral transcription (`/v1/audio/transcriptions`) | yes, 8.3 GB |

The last two do not run. They are written down because the settings were
worked out and measured, and rediscovering them later costs more than a file
that says plainly why it does not work yet.

Each config above was started against a real worker and gateway, except those
two: Laya answers `/v1/decisions`, Qwen answers `/v1/chat/completions`.

## API keys

Every config here is open, which is right for a loopback development box and
wrong for anything else. Adding keys turns on authentication for `/v1` **and**
the dashboard — whose "Try it" box runs inference, so leaving it open would
make the rest decorative:

```toml
[[auth.keys]]
name = "alice"
key = "sk-..."

[[auth.keys]]
name = "bob"
key = "sk-..."
namespace = "shared"     # optional: pool a team onto one cache namespace
```

Clients send `Authorization: Bearer <key>` (or `x-api-key`). A browser cannot
set a header, so opening `/dashboard?key=<key>` once exchanges the key for an
`HttpOnly` cookie and redirects to a clean URL.

**Each key gets its own cache namespace by default**, and that is the point
rather than a side effect: a shared prefix cache is a timing side channel,
because time-to-first-token reveals whether *someone* recently submitted a
given prefix. Two keys with the same `namespace` share cached prefixes and
cached responses; two without share nothing.

`/health` and `/metrics` stay open — a liveness probe should need no
credential and a scraper expects none, and neither exposes a prompt or an
answer.
