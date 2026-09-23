#!/usr/bin/env python3
"""Reference values for the speech path.

    --tokenizer  Tekken ids for a set of strings, from mistral-common.
    --mel        log-mel frames from the reference feature extractor.
    --model      the prompt, the intermediate tensors and the greedy tokens
                 for one clip, so a port can be checked stage by stage rather
                 than only at the transcript.

Both need only the small files from the repo, so they can be generated before
the 8 GB of weights have finished downloading.

    uv run --index https://download.pytorch.org/whl/cpu --with torch \
        --with 'transformers>=5.2' --with 'mistral-common>=1.9.0' --with numpy \
        python tools/gen_voxtral_goldens.py --tokenizer --mel
"""

import argparse
import json
import math
from pathlib import Path

MODEL = Path.home() / "models/voxtral-realtime"
OUT = Path("tests/goldens")

# Every language the model claims, plus the shapes that break a byte-level
# BPE: a split multi-byte character, a long digit run, repeated whitespace,
# and the spellings of the model's own control tokens.
STRINGS = [
    "Hello world",
    "Bonjour le monde !",
    "Guten Tag, wie geht es Ihnen?",
    "¿Dónde está la biblioteca?",
    "Ciao, come stai?",
    "Olá, tudo bem?",
    "Hoe gaat het met je?",
    "Привет, как дела?",
    "नमस्ते, आप कैसे हैं?",
    "مرحبا كيف حالك؟",
    "你好，最近怎么样？",
    "こんにちは、お元気ですか。",
    "안녕하세요, 잘 지내세요?",
    "  leading and   internal   spaces  ",
    "numbers 1234567890 and 3.14159",
    "emoji 🎧🎙️ and a family 👨‍👩‍👧‍👦",
    "[BEGIN_AUDIO] and [STREAMING_PAD] spelled out",
    "tabs\tand\nnewlines\r\nmixed",
    "a" * 200,
    "",
]

SPECIALS = [
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[AUDIO]",
    "[BEGIN_AUDIO]",
    "[STREAMING_PAD]",
    "[STREAMING_WORD]",
    "[TRANSCRIBE]",
]


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def tokenizer_goldens() -> None:
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tok = Tekkenizer.from_file(str(MODEL / "tekken.json"))
    cases = []
    for text in STRINGS:
        ids = tok.encode(text, bos=False, eos=False)
        cases.append(
            {
                "text": text,
                "ids": [int(i) for i in ids],
                # Round-tripped, because a byte-level vocabulary can encode
                # something it does not decode back identically.
                "decoded": tok.decode(ids),
            }
        )
    specials = {}
    for name in SPECIALS:
        try:
            specials[name] = int(tok.get_control_token(name))
        except Exception:
            pass
    write(
        OUT / "voxtral-tekken.json",
        {
            "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "dir": str(MODEL),
            "n_words": int(tok.n_words),
            "num_special_tokens": int(tok.num_special_tokens),
            "bos_id": int(tok.bos_id),
            "eos_id": int(tok.eos_id),
            "specials": specials,
            "cases": cases,
        },
    )


def tone(seconds: float, rate: int = 16000) -> list:
    """A deterministic signal with content across the spectrum.

    Two tones plus a chirp: a single sine leaves most mel bins at the floor,
    where a wrong filterbank looks right.
    """
    n = int(seconds * rate)
    out = []
    for i in range(n):
        t = i / rate
        sweep = 200.0 + 3000.0 * (i / max(1, n))
        out.append(
            0.4 * math.sin(2 * math.pi * 440.0 * t)
            + 0.3 * math.sin(2 * math.pi * 1000.0 * t)
            + 0.2 * math.sin(2 * math.pi * sweep * t)
        )
    return out


def mel_goldens() -> None:
    import numpy as np
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(MODEL))
    fe = processor.feature_extractor
    audio = np.array(tone(1.2), dtype=np.float32)
    features = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="np")
    mel = np.asarray(features["input_features"])
    while mel.ndim > 2:
        mel = mel[0]
    write(
        OUT / "voxtral-mel.json",
        {
            "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "dir": str(MODEL),
            "settings": {
                k: getattr(fe, k, None)
                for k in (
                    "sampling_rate",
                    "hop_length",
                    "n_fft",
                    "win_length",
                    "feature_size",
                    "padding_value",
                    "global_log_mel_max",
                )
            },
            "samples": [round(float(x), 7) for x in audio],
            "shape": list(mel.shape),
            "mel": [[round(float(x), 5) for x in row] for row in mel],
        },
    )


# Real recorded speech with a known transcript, shipped with alsa on most
# Linux boxes. A synthetic tone makes a weak golden: the model correctly
# answers it with nothing but padding tokens, so every argmax agrees whatever
# the port does.
SPEECH_WAV = Path("/usr/share/sounds/alsa/Front_Center.wav")


def reference_clip():
    """`(samples at 16 kHz mono, where it came from, seconds)`."""
    import numpy as np

    if SPEECH_WAV.exists():
        import wave

        with wave.open(str(SPEECH_WAV)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        assert width == 2, f"expected 16-bit PCM, got {width * 8}-bit"
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        if rate != 16000:
            # Linear, matching `vapi_audio::resample_linear`, so the Rust side
            # starts from the identical samples.
            n = int(len(pcm) * 16000 / rate)
            idx = np.arange(n) * (rate / 16000)
            left = np.floor(idx).astype(int)
            frac = (idx - left).astype(np.float32)
            right = np.minimum(left + 1, len(pcm) - 1)
            pcm = pcm[left] + (pcm[right] - pcm[left]) * frac
        return pcm.astype(np.float32), f"{SPEECH_WAV} ({rate} Hz, {channels}ch)", len(pcm) / 16000
    return np.array(tone(2.0), dtype=np.float32), "synthetic tone", 2.0


def model_goldens() -> None:
    """Stage by stage, because a transcript that is merely worse is the
    hardest kind of wrong to localise."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(MODEL))
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.float32
    ).eval()

    audio, source, seconds = reference_clip()
    inputs = processor(audio=audio, sampling_rate=16000, is_streaming=False, return_tensors="pt")
    input_ids = inputs["input_ids"]
    features = inputs["input_features"]
    delay = int(inputs.get("num_delay_tokens", model.config.default_num_delay_tokens))

    # The prompt covers fewer positions than the clip: generation extends the
    # token sequence one position per 80 ms, consuming the audio as it goes.
    # For a single forward the two have to be cut to the same length.
    per_tok = int(model.config.audio_length_per_tok)
    prompt_len = int(input_ids.shape[1])
    prompt_features = features[:, :, : prompt_len * per_tok]

    with torch.no_grad():
        audio_out = model.model.get_audio_features(input_features=prompt_features)
        encoder_hidden = audio_out.last_hidden_state
        audio_embeds = audio_out.pooler_output
        t_cond = model.model.time_embedding(
            torch.full((1,), delay, dtype=torch.float32)
        )
        out = model(
            input_ids=input_ids,
            input_features=prompt_features,
            num_delay_tokens=delay,
            use_cache=False,
        )
        logits = out.logits[0]
        generated = model.generate(
            **inputs, max_new_tokens=64, do_sample=False, temperature=None, top_p=None
        )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    def slice2(t, rows, cols=None):
        t = t.reshape(-1, t.shape[-1])
        rows = min(rows, t.shape[0])
        cols = t.shape[1] if cols is None else min(cols, t.shape[1])
        return [[round(float(x), 5) for x in t[i, :cols]] for i in range(rows)]

    steps = []
    for i in range(min(24, logits.shape[0])):
        row = logits[i]
        top = torch.topk(row, 8)
        steps.append(
            {
                "position": i,
                "argmax": int(row.argmax()),
                "top_ids": [int(x) for x in top.indices],
                "top_logits": [round(float(x), 4) for x in top.values],
            }
        )

    write(
        OUT / "voxtral-model.json",
        {
            "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "dir": str(MODEL),
            "num_delay_tokens": delay,
            "audio_source": source,
            "seconds": round(seconds, 4),
            "samples": [round(float(x), 7) for x in audio],
            "input_ids": [int(i) for i in input_ids[0]],
            "audio_length_per_tok": per_tok,
            "mel_shape": list(features.shape),
            "prompt_mel_shape": list(prompt_features.shape),
            "generated_ids": [int(i) for i in generated[0]],
            "text": text,
            "encoder_shape": list(encoder_hidden.shape),
            "audio_embeds_shape": list(audio_embeds.shape),
            # A few full rows of each stage: enough to catch a transposed
            # projection or an off-by-one in the frame grouping.
            "encoder_hidden": slice2(encoder_hidden, 4),
            "audio_embeds": slice2(audio_embeds, 4),
            "t_cond": [round(float(x), 6) for x in t_cond.reshape(-1)[:64]],
            "t_cond_shape": list(t_cond.shape),
            "logits_shape": list(logits.shape),
            "steps": steps,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true")
    ap.add_argument("--mel", action="store_true")
    ap.add_argument("--model", action="store_true")
    args = ap.parse_args()
    if not (args.tokenizer or args.mel or args.model):
        ap.error("pass --tokenizer, --mel, --model, or several")
    if args.tokenizer:
        tokenizer_goldens()
    if args.mel:
        mel_goldens()
    if args.model:
        model_goldens()


if __name__ == "__main__":
    main()
