#!/usr/bin/env python3
"""Golden fixtures for the Laguna architecture, from poolside's reference code.

Two fixtures:

  laguna-tiny      poolside/Laguna-tiny-per-element as downloaded: a real
                   checkpoint (2 layers, 8 experts, dense then sparse MLP,
                   per-element gating, q/k norms, partial rotary 0.5, full
                   attention only).
  laguna-tiny-xs   a random-initialised config carrying every feature of
                   Laguna-XS-2.1 that the checkpoint lacks: sliding-window
                   layers with their own RoPE, per-head gating, YaRN with an
                   explicit attention factor, per-layer head counts, routed
                   scaling 2.5, a non-zero expert selection bias, two EOS ids.

For each: weights are saved as safetensors next to config.json, and
tests/goldens/<name>.json records, for fixed token ids, the f32 logits at
every position of a full prompt and the logits of one decode step against
the cache. The Rust tests in crates/vapi-backend-candle compare against
these under 1e-4.

    benchmark/.venv/bin/python tools/gen_laguna_goldens.py

Needs transformers >= 5.13 (the venv has 5.17) and modeling_laguna.py +
configuration_laguna.py next to the tiny checkpoint.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HOME = Path.home()
TINY = HOME / "models/laguna-tiny"
TINY_XS = HOME / "models/laguna-tiny-xs"
OUT = Path(__file__).resolve().parent.parent / "tests/goldens"

PROMPT = [5, 17, 3, 42, 8, 9, 60, 33, 21, 7, 11, 2, 19, 44, 58, 13, 25, 31]
NEXT = 23


def load_reference(model_dir):
    """Import modeling_laguna.py as a package so its relative import works."""
    import tempfile, shutil
    pkg = Path(tempfile.mkdtemp()) / "laguna_ref"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for f in ("modeling_laguna.py", "configuration_laguna.py"):
        shutil.copy(model_dir / f, pkg / f)
    sys.path.insert(0, str(pkg.parent))
    import laguna_ref.modeling_laguna as m
    import laguna_ref.configuration_laguna as c
    Cfg = c.LagunaConfig
    # poolside's config predates transformers 5.17's rope validation: with
    # rope_parameters nested by layer type, a stray top-level
    # partial_rotary_factor float trips `validate_rope`. Drop it before
    # validating; the per-layer dicts carry their own factor.
    validators = Cfg.__class_validators__
    for i, fn in enumerate(validators):
        if getattr(fn, "__name__", "") == "validate_rope":
            orig = fn

            def validate_rope(self, _orig=orig):
                rp = getattr(self, "rope_parameters", None)
                if isinstance(rp, dict) and any(isinstance(v, dict) for v in rp.values()):
                    self.rope_parameters = {k: v for k, v in rp.items() if isinstance(v, dict)}
                return _orig(self)

            validators[i] = validate_rope
    return Cfg, m.LagunaForCausalLM


def references(model):
    model.eval()
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        full = out.logits[0].float()  # (len, vocab)
        step = model(input_ids=torch.tensor([[NEXT]]), past_key_values=out.past_key_values, use_cache=True)
        decode = step.logits[0, -1].float()
        # 16 greedy tokens from the prompt, through the reference's own
        # generation loop, so the backend-level test covers dispatch, the
        # scheduler and the cache write together.
        gen = model.generate(ids, max_new_tokens=16, do_sample=False, num_beams=1,
                             eos_token_id=None, pad_token_id=0)
        greedy_ids = gen[0][len(PROMPT):].tolist()
    return {
        "prompt_ids": PROMPT,
        "full_logits": full.tolist(),
        "decode_token": NEXT,
        "decode_logits": decode.tolist(),
        "greedy_next": int(full[-1].argmax()),
        "greedy_ids": greedy_ids,
    }


def dump(name, cfg, model, model_dir):
    cfg_dict = json.loads(cfg.to_json_string())
    rec = {"name": name, "config": {k: cfg_dict.get(k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "num_experts", "num_experts_per_tok", "vocab_size", "sliding_window",
        "gating", "layer_types", "mlp_layer_types", "num_attention_heads_per_layer")}}
    rec.update(references(model))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec) + "\n")
    print(f"{name}: {len(PROMPT)} prompt ids, vocab {cfg.vocab_size}, greedy next {rec['greedy_next']}, "
          f"model dir {model_dir}")


def main():
    LagunaConfig, LagunaForCausalLM = load_reference(TINY)

    # Fixture 1: the real tiny checkpoint.
    cfg = LagunaConfig.from_pretrained(TINY)
    cfg._attn_implementation = "eager"
    model = LagunaForCausalLM.from_pretrained(TINY, config=cfg, dtype=torch.float32)
    dump("laguna-tiny", cfg, model, TINY)

    # Fixture 2: XS-2.1's feature set at toy size, random weights.
    torch.manual_seed(0)
    xs = LagunaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=6, num_key_value_heads=2, head_dim=16,
        num_attention_heads_per_layer=[6, 8, 8, 6],
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=6, gating="per-head", max_position_embeddings=128,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, norm_topk_prob=True, mlp_only_layers=[0],
        moe_routed_scaling_factor=2.5, rms_norm_eps=1e-6, tie_word_embeddings=False,
        bos_token_id=2, eos_token_id=[2, 24], pad_token_id=9,
        rope_parameters={
            "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 32.0,
                               "original_max_position_embeddings": 64, "beta_slow": 1.0,
                               "beta_fast": 64.0, "attention_factor": 1.3465735902799727,
                               "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0,
                                  "partial_rotary_factor": 1.0},
        },
    )
    xs._attn_implementation = "eager"
    model = LagunaForCausalLM(xs)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("e_score_correction_bias"):
                p.copy_(torch.linspace(-0.3, 0.3, p.numel()))
            elif "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    TINY_XS.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, TINY_XS / "model.safetensors", metadata={"format": "pt"})
    xs.save_pretrained(TINY_XS)
    for f in ("modeling_laguna.py", "configuration_laguna.py"):
        (TINY_XS / f).write_bytes((TINY / f).read_bytes())
    # Reload from disk so the fixture is exactly what the Rust side loads.
    cfg2 = LagunaConfig.from_pretrained(TINY_XS)
    cfg2._attn_implementation = "eager"
    model2 = LagunaForCausalLM.from_pretrained(TINY_XS, config=cfg2, dtype=torch.float32)
    dump("laguna-tiny-xs", cfg2, model2, TINY_XS)
    print("state_dict keys sample:", sorted(state)[:6])


if __name__ == "__main__":
    main()
