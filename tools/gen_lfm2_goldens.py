#!/usr/bin/env python3
"""Golden fixture for the LFM2 architecture from transformers' own model.

Builds a random-initialised tiny Lfm2Config with the shapes that matter
(conv and attention layers, GQA, conv kernel 3, tied embeddings), saves it
to ~/models/lfm2-tiny, and records for fixed token ids the f32 logits at
every position of a full prompt, the logits of one decode step against the
cache, and 16 greedy ids from generate(). Same schema as the Laguna
fixtures, so crates/vapi-backend-candle reuses the test harness.

    benchmark/.venv/bin/python tools/gen_lfm2_goldens.py
"""

import json
from pathlib import Path

import torch
from transformers import Lfm2Config, Lfm2ForCausalLM

OUT = Path(__file__).resolve().parent.parent / "tests/goldens"
DIR = Path.home() / "models/lfm2-tiny"
PROMPT = [5, 17, 3, 42, 8, 9, 60, 33, 21, 7, 11, 2, 19, 44, 58, 13, 25, 31]
NEXT = 23


def main():
    torch.manual_seed(0)
    cfg = Lfm2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        layer_types=["conv", "conv", "full_attention", "conv", "full_attention"],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        norm_eps=1e-5, tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    )
    cfg._attn_implementation = "eager"
    model = Lfm2ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DIR, safe_serialization=True)
    # Reload from disk so the fixture is exactly what the Rust side loads.
    model = Lfm2ForCausalLM.from_pretrained(DIR, dtype=torch.float32)
    model.config._attn_implementation = "eager"
    model.eval()

    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        full = out.logits[0].float()
        step = model(input_ids=torch.tensor([[NEXT]]), past_key_values=out.past_key_values, use_cache=True)
        decode = step.logits[0, -1].float()
        gen = model.generate(ids, max_new_tokens=16, do_sample=False, num_beams=1,
                             eos_token_id=None, pad_token_id=0)
        greedy_ids = gen[0][len(PROMPT):].tolist()
    rec = {
        "name": "lfm2-tiny",
        "config": {k: getattr(cfg, k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                                "num_key_value_heads", "vocab_size", "conv_L_cache", "layer_types")},
        "prompt_ids": PROMPT,
        "full_logits": full.tolist(),
        "decode_token": NEXT,
        "decode_logits": decode.tolist(),
        "greedy_next": int(full[-1].argmax()),
        "greedy_ids": greedy_ids,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lfm2-tiny.json").write_text(json.dumps(rec) + "\n")
    print(f"lfm2-tiny: {len(PROMPT)} prompt ids, greedy next {rec['greedy_next']}, greedy ids {greedy_ids}")
    print("tensors:", sorted(k for k in model.state_dict())[:8], "...")


if __name__ == "__main__":
    main()
