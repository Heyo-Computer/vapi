#!/usr/bin/env python3
"""Reference baseline for the speech path: transformers on the GPU.

This is what vapi's port has to match for correctness and beat for speed, so
it is worth having before the port exists. It measures the thing that decides
whether the model is usable rather than tokens per second: **the realtime
factor**, audio seconds transcribed per wall second. Below 1.0 a live session
falls behind the speaker.

    benchmark/.venv/bin/python benchmark/bench_transcription.py
"""

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

MODEL = Path.home() / "models/voxtral-realtime"
SPEECH = Path("/usr/share/sounds/alsa/Front_Center.wav")
LENGTHS = [1.4, 10.0, 30.0, 60.0]


def clip(seconds: float) -> np.ndarray:
    """Real speech, repeated to the requested length."""
    with wave.open(str(SPEECH)) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    pcm /= 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(len(pcm) * 16000 / rate)
        idx = np.arange(n) * (rate / 16000)
        left = np.floor(idx).astype(int)
        right = np.minimum(left + 1, len(pcm) - 1)
        pcm = pcm[left] + (pcm[right] - pcm[left]) * (idx - left).astype(np.float32)
    want = int(seconds * 16000)
    return np.tile(pcm, int(want / len(pcm)) + 1)[:want].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(MODEL))
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.bfloat16
    ).to("cuda").eval()
    load_s = time.perf_counter() - started
    weights_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"load {load_s:.1f} s, weights {weights_gib:.2f} GiB\n")
    print(f"{'audio s':>8} {'wall s':>8} {'realtime':>9} {'steps/s':>8} {'peak GiB':>9}  transcript")

    rows = []
    for seconds in LENGTHS:
        audio = clip(seconds)
        inputs = processor(
            audio=audio, sampling_rate=16000, is_streaming=False, return_tensors="pt"
        ).to("cuda", dtype=torch.bfloat16)
        # One position per 80 ms, and the prompt already covers the delay.
        steps = int(seconds * 12.5)
        torch.cuda.reset_peak_memory_stats()

        walls = []
        text = ""
        for i in range(args.iters + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=steps, do_sample=False,
                    temperature=None, top_p=None,
                )
            torch.cuda.synchronize()
            if i:  # the first pass is warm-up
                walls.append(time.perf_counter() - t0)
            text = processor.batch_decode(out, skip_special_tokens=True)[0]

        wall = statistics.median(walls)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append(
            {
                "audio_seconds": seconds,
                "wall_seconds": round(wall, 3),
                "realtime_factor": round(seconds / wall, 2),
                "steps_per_second": round(steps / wall, 1),
                "peak_gib": round(peak, 2),
                "text": text,
            }
        )
        print(
            f"{seconds:>8.1f} {wall:>8.2f} {seconds / wall:>8.2f}x {steps / wall:>8.1f} "
            f"{peak:>9.2f}  {text.strip()[:44]!r}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                    "runtime": "transformers (reference)",
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "dtype": "bfloat16",
                    "load_seconds": round(load_s, 2),
                    "weights_gib": round(weights_gib, 2),
                    "rows": rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
