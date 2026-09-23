#!/usr/bin/env python3
"""Reference values for the single-pass decision path.

Two fixtures, for the two things that can be wrong:

  --tiny  writes a small random ModernBERT + decision head to ~/models/laya-tiny
          and records what transformers computes for it. The standing rule is
          that nothing lands for a real model that was not first proven on a
          tiny fixture against transformers, on the same weights.

  --real  records the same values for ~/models/laya, the real checkpoint, on
          fixtures that exercise every question type and every truncation rule.

  --multilingual
          the same for the mmBERT-base checkpoint bundled in that repo, on
          fixtures in the scripts the English one cannot read. A different
          encoder (22 layers, d=768, 256k vocab), a different tokenizer
          (Metaspace rather than byte-level) and a different budget, so it
          exercises the loader rather than repeating the English run.

Both write the *sequences* as well as the logits, so a mismatch says whether
the prompt builder or the forward pass is wrong.

    uv run --index https://download.pytorch.org/whl/cpu \
        --with torch --with transformers --with safetensors --with numpy \
        python tools/gen_laya_goldens.py --tiny --real
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

LAYA = Path.home() / "models/laya"
MULTI = LAYA / "multilingual"
TINY = Path.home() / "models/laya-tiny"
OUT = Path("tests/goldens")

os.environ.setdefault("USE_TF", "0")

# Every question type, and the edges of the budget arithmetic.
FIXTURES = [
    {
        "name": "email_triage",
        "state": {
            "from": "user@acme.com",
            "subject": "Duplicate charge on invoice #4411",
            "body": "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan.",
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors",
                    "sales": "pricing, new contracts",
                    "other": "everything else",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
            },
            "churn_risk": {
                "type": "noul",
                "instructions": "Does the user threaten to cancel or leave?",
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "Does the user explicitly request a refund?",
            },
        },
    },
    {
        "name": "plain_text_state",
        "state": "The build has been red for three days and nobody has looked at it.",
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "What is the tone of this message?",
                "criteria": ["positive", "neutral", "negative"],
            },
            "needs_action": {
                "type": "noul",
                "instructions": "Does this require someone to act?",
                "criteria": {"true": "somebody must do something", "false": "no action needed"},
            },
        },
    },
    {
        "name": "long_state_truncates",
        # Far past max_len, so the state budget rule is exercised.
        "state": "The customer reports intermittent 500s on checkout. " * 80,
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this incident?",
                "criteria": ["cosmetic", "degraded", "outage"],
            },
        },
    },
    {
        "name": "mask_token_in_state",
        # A caller whose document contains the marker token must not be able
        # to mint an option position.
        "state": "The user wrote [MASK] in their ticket and asked [MASK] again.",
        "questions": {
            "is_spam": {
                "type": "noul",
                "instructions": "Is this [MASK] spam?",
            },
        },
    },
    {
        "name": "many_short_options",
        "state": "I want to dispute a transaction on my card.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Which intent is this?",
                "criteria": [
                    "dispute",
                    "balance",
                    "transfer",
                    "card_lost",
                    "card_new",
                    "pin_reset",
                    "statement",
                    "close_account",
                    "fees",
                    "limits",
                    "other",
                ],
            },
        },
    },
]

PROBE_TEXT = "the invoice was paid twice"

# The point of the multilingual checkpoint: the English one scores 0.000 on
# some of these while reporting 0.95 confidence.
MULTILINGUAL_FIXTURES = [
    {
        "name": "hindi_refund",
        "state": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors",
                    "other": "everything else",
                },
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "Does the user explicitly request a refund?",
            },
        },
    },
    {
        "name": "mixed_scripts",
        "state": {
            "subject": "重複した請求",
            "body": "Ticket #4411: мы были списаны дважды. Пожалуйста, верните деньги.",
        },
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical"],
            },
        },
    },
    {
        "name": "khmer_long_state",
        # Khmer is where the English checkpoint measures 0.000 accuracy at
        # 0.952 confidence, and long enough here to exercise the 1024 budget.
        "state": "សូមសងប្រាក់មកវិញ ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង។ " * 40,
        "questions": {
            "needs_action": {
                "type": "noul",
                "instructions": "Does this require someone to act?",
            },
        },
    },
]


def load_agent(model_dir: Path, code_dir: Path | None = None):
    """`code_dir` is where the reference implementation lives; a bundled
    subfolder checkpoint has weights and config but no Python of its own."""
    sys.path.insert(0, str(code_dir or model_dir))
    from rl_agent_api import RLAgent

    return RLAgent(str(model_dir), device="cpu")


def record(agent, fixtures) -> list:
    from rl_common import QTYPES, build_sequence, render_options

    cases = []
    for fx in fixtures:
        rows = []
        for qid, qdef in fx["questions"].items():
            q = agent._to_internal(qdef)
            seq, markers = build_sequence(
                agent.tok, fx["state"], q, agent.cfg["max_len"], agent.cfg["head_max_len"]
            )
            rows.append(
                {
                    "id": qid,
                    "qtype": q["t"],
                    "qtype_index": QTYPES[q["t"]],
                    "options": render_options(q),
                    "tokens": [int(t) for t in seq],
                    "markers": [int(m) for m in markers],
                }
            )
        result = agent.system_one(fx["state"], fx["questions"])
        # Uncalibrated logits, which is what the backend returns; the
        # temperature that turns them into the answers below is applied by
        # the caller.
        logits, act = raw_scores(agent, rows)
        for row, lg, ac in zip(rows, logits, act):
            row["logits"] = [round(float(x), 6) for x in lg]
            row["act_probability"] = round(float(ac), 6)
            row["answer"] = result["answers"][row["id"]]
        cases.append(
            {
                "name": fx["name"],
                "state": fx["state"],
                "questions": fx["questions"],
                "rows": rows,
                "usage": result["usage"],
            }
        )
    return cases


@torch.no_grad()
def raw_scores(agent, rows):
    from rl_common import collate_items

    items = [
        {
            "ids": r["tokens"],
            "markers": r["markers"],
            "qtype": r["qtype_index"],
            "target": [0.0] * len(r["markers"]),
            "label": -1,
            "episode": 0,
            "ep_step": 0,
            "ep_len": 1,
            "src": "golden",
        }
        for r in rows
    ]
    b = collate_items([items], agent.tok.pad_token_id)
    logits, act = agent.model(
        b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
    )
    act = torch.softmax(act.float(), -1)
    out = []
    for i, r in enumerate(rows):
        out.append(logits[i, : len(r["markers"])].float().tolist())
    return out, [float(act[i, 0]) for i in range(len(rows))]


@torch.no_grad()
def encoder_probe(agent, text: str = PROBE_TEXT) -> dict:
    """The encoder's own output, so a mismatch can be localised below the head."""
    ids = agent.tok(text, add_special_tokens=True)["input_ids"]
    t = torch.tensor([ids])
    mask = torch.ones_like(t)
    h = agent.model.encoder(input_ids=t, attention_mask=mask).last_hidden_state[0]
    return {
        "text": text,
        "tokens": [int(i) for i in ids],
        "hidden": [[round(float(x), 5) for x in row] for row in h],
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def build_tiny() -> None:
    """A small random ModernBERT + decision head, in the checkpoint's layout."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModel, ModernBertConfig

    sys.path.insert(0, str(LAYA))
    from rl_common import DecisionModel

    torch.manual_seed(11)
    TINY.mkdir(parents=True, exist_ok=True)

    words = (
        "[UNK] [CLS] [SEP] [MASK] [PAD] choice score noul question : which department ? "
        "billing invoices refunds technical bugs sales other level 0 1 2 not urgent soon "
        "critical the user was billed twice and wants a refund today false true yes no "
        "build red three days nobody looked at it positive neutral negative severity "
        "cosmetic degraded outage spam ticket wrote asked again intent dispute balance "
        "transfer card lost new pin reset statement close account fees limits paid invoice"
    ).split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (TINY / "tokenizer").mkdir(exist_ok=True)
    tok.save(str(TINY / "tokenizer/tokenizer.json"))
    (TINY / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "model_max_length": 256,
                "tokenizer_class": "PreTrainedTokenizerFast",
            },
            indent=1,
        )
    )

    cfg = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=6,
        intermediate_size=48,
        max_position_embeddings=256,
        local_attention=8,
        global_attn_every_n_layers=3,
        cls_token_id=vocab["[CLS]"],
        sep_token_id=vocab["[SEP]"],
        pad_token_id=vocab["[PAD]"],
        bos_token_id=vocab["[CLS]"],
        eos_token_id=vocab["[SEP]"],
    )
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=2, n_act=2)
    # Random init leaves LayerNorms at identity and biases at zero, which
    # hides transposition bugs. Give everything something to say.
    for p in model.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.08)
    model.eval()

    (TINY / "encoder").mkdir(exist_ok=True)
    enc.config.to_json_file(TINY / "encoder/config.json")
    (TINY / "rl_agent_config.json").write_text(
        json.dumps(
            {
                "encoder": "tiny",
                "head_layers": 2,
                "max_len": 96,
                "head_max_len": 40,
                "act_costs": {"escalate": 0.5},
                "amp_dtype": "bf16",
                "model_name": "laya-tiny",
                "temperature": [1.3, 1.1, 1.7],
                "temperature_by_options": {"noul:2": 1.7, "choice:3-5": 1.45},
            },
            indent=1,
        )
    )
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(TINY / "model.safetensors"))
    print(f"wrote {TINY}")


def forget_reference() -> None:
    """Drop the previous checkpoint's copy of the reference implementation.

    Each checkpoint directory ships its own, and they are imported by path.
    """
    for mod in list(sys.modules):
        if mod.startswith("rl_"):
            del sys.modules[mod]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--multilingual", action="store_true")
    args = ap.parse_args()
    if not (args.tiny or args.real or args.multilingual):
        ap.error("pass --tiny, --real, --multilingual, or several")

    if args.tiny:
        build_tiny()
        agent = load_agent(TINY)
        write(
            OUT / "laya-tiny.json",
            {
                "model": "laya-tiny",
                "dir": str(TINY),
                "config": agent.cfg,
                "encoder_probe": encoder_probe(agent),
                "cases": record(agent, FIXTURES),
            },
        )

    if args.real:
        forget_reference()
        agent = load_agent(LAYA)
        write(
            OUT / "laya.json",
            {
                "model": "convaiinnovations/laya",
                "dir": str(LAYA),
                "config": agent.cfg,
                "encoder_probe": encoder_probe(agent),
                "cases": record(agent, FIXTURES),
            },
        )

    if args.multilingual:
        forget_reference()
        agent = load_agent(MULTI, code_dir=LAYA)
        write(
            OUT / "laya-multilingual.json",
            {
                "model": "convaiinnovations/laya/multilingual",
                "dir": str(MULTI),
                "config": agent.cfg,
                "encoder_probe": encoder_probe(agent, MULTILINGUAL_FIXTURES[0]["state"]),
                "cases": record(agent, FIXTURES + MULTILINGUAL_FIXTURES),
            },
        )


if __name__ == "__main__":
    main()
