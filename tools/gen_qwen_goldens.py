#!/usr/bin/env python3
"""Tiny Qwen3 fixture: random weights, reference logits, greedy ids.

The standing rule is that nothing lands for a real model that was not first
proven on a tiny fixture against transformers, on the same weights. This
writes a small random Qwen3 to ~/models/qwen-tiny and records what
transformers computes for a fixed token sequence.

    benchmark/.venv/bin/python tools/gen_qwen_goldens.py
"""

import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

DIR = Path.home() / "models/qwen-tiny"
OUT = Path("tests/goldens")
TOKENS = [3, 17, 42, 8, 91, 5, 60, 23, 11, 77]


def main() -> None:
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # Deliberately not hidden_size / heads: the point of the fixture is
        # to catch a loader that assumes it is.
        head_dim=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval().to(torch.float32)
    # Random init is near zero for the norms; make them non-trivial so a
    # loader that skips q_norm/k_norm cannot pass.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(torch.empty_like(p).uniform_(0.5, 1.5))

    DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DIR, safe_serialization=True)

    ids = torch.tensor([TOKENS])
    with torch.no_grad():
        full = model(ids).logits[0]
        greedy = model.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            temperature=None,
            top_k=None,
            top_p=None,
        )[0][len(TOKENS):]

    record = {
        "name": "qwen-tiny",
        "config": json.loads((DIR / "config.json").read_text()),
        "tokens": TOKENS,
        "full_logits": full.tolist(),
        "greedy_ids": greedy.tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen-tiny.json").write_text(json.dumps(record) + "\n")
    print(f"{DIR}: weights; tests/goldens/qwen-tiny.json: {len(TOKENS)} positions")
    print(f"  greedy ids {record['greedy_ids']}")


if __name__ == "__main__":
    main()
