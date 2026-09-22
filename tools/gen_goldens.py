#!/usr/bin/env python3
"""Generate golden fixtures from Hugging Face transformers.

Run by hand, not in CI. For each fixture this dumps the rendered chat
template, the prompt token ids, and the first N greedy token ids to
tests/goldens/<model>-<fixture>.json. The Rust tests in
crates/vapi-backend-candle/tests/real_model.rs then check that vapi's
template rendering, tokenization and greedy decoding match exactly.

    uv run --index https://download.pytorch.org/whl/cpu \
        --with torch --with transformers --with safetensors \
        tools/gen_goldens.py ~/models/smollm2-135m-instruct \
        --model-id HuggingFaceTB/SmolLM2-135M-Instruct

Generation runs in float32 on the CPU, which is what the CPU golden test
compares against exactly. A bf16 GPU run is expected to diverge at near-ties;
the Rust test reports where and by how much.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIXTURES = {
    "capital": [
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ],
    "system": [
        {"role": "system", "content": "You are a terse assistant. Reply in at most ten words."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    "multiturn": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada."},
        {"role": "user", "content": "What did I say my name was?"},
    ],
    "unicode": [
        {"role": "user", "content": "Écris une phrase sur les crêpes, puis une en 日本語."},
    ],
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

# Fixtures that carry a `tools` list. The model is expected to answer the
# first with a tool call, and the second (which feeds the call's result
# back) with prose. Only models whose chat template handles tools get these.
TOOL_FIXTURES = {
    "tools": (
        [{"role": "user", "content": "What is the weather in Paris right now?"}],
        [WEATHER_TOOL],
    ),
    "toolresult": (
        [
            {"role": "user", "content": "What is the weather in Paris right now?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 18, \"unit\": \"celsius\", \"sky\": \"overcast\"}"},
        ],
        [WEATHER_TOOL],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--model-id", required=True, help="HF repo id, recorded in the fixture")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("tests/goldens"))
    ap.add_argument("--only", nargs="*", help="fixture names to (re)generate; default all")
    ap.add_argument("--tools", action="store_true", help="also generate the tool fixtures")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()
    config = json.loads((args.model_dir / "config.json").read_text())
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]

    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model_id).strip("-").lower()
    args.out.mkdir(parents=True, exist_ok=True)

    fixtures = {name: (messages, None) for name, messages in FIXTURES.items()}
    if args.tools:
        fixtures.update(TOOL_FIXTURES)
    for name, (messages, tools) in fixtures.items():
        if args.only and name not in args.only:
            continue
        rendered = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tok(rendered, add_special_tokens=False).input_ids
        with torch.no_grad():
            # Pure greedy. `generate()` also applies the checkpoint's
            # generation_config defaults (LFM2.5 ships repetition_penalty 1.1,
            # top_k 50), which would make the goldens something other than
            # argmax; the Rust side compares against argmax.
            out = model.generate(
                torch.tensor([prompt_ids]),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=None,
                top_k=None,
                top_p=None,
                eos_token_id=eos_ids,
                pad_token_id=eos_ids[0],
            )
        greedy_ids = out[0][len(prompt_ids):].tolist()
        record = {
            "model_id": args.model_id,
            "fixture": name,
            "config": {
                k: config.get(k)
                for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                          "num_key_value_heads", "vocab_size")
            },
            "messages": messages,
            **({"tools": tools} if tools else {}),
            "rendered": rendered,
            "prompt_ids": prompt_ids,
            "greedy_ids": greedy_ids,
            "greedy_text": tok.decode(greedy_ids, skip_special_tokens=True),
            "eos_token_ids": eos_ids,
            "max_new_tokens": args.max_new_tokens,
        }
        path = args.out / f"{slug}-{name}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"{path}: {len(prompt_ids)} prompt ids, {len(greedy_ids)} greedy ids")
        print(f"  {record['greedy_text'][:100]!r}")


if __name__ == "__main__":
    main()
