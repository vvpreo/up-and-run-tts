"""VoxCPM2 (2B, 48 kHz): cloning with transcript ("ultimate cloning") and voice design from a description."""
import argparse
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _sf_load(path, *a, **k):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T.copy()), sr


torchaudio.load = _sf_load

from voxcpm import VoxCPM  # noqa: E402

from common import REFS, TEXTS, Recorder, Timer, cap_gpu_memory, sync

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/voxcpm2")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "voxcpm2")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = model.tts_model.sample_rate
DESIGN = "(Молодая женщина, тёплый спокойный голос, естественная разговорная интонация)"


def run(variant, kw, prefix=""):
    model.generate(text=prefix + TEXTS[0]["text"], cfg_value=2.0, inference_timesteps=10, **kw)  # warm-up
    for t in TEXTS:
        if rec.done(variant, t["id"]):
            continue
        torch.manual_seed(42)
        with Timer() as tm:
            wav = model.generate(text=prefix + t["text"], cfg_value=2.0, inference_timesteps=10, **kw)
        # time to first audio chunk with the streaming API (whole text known)
        sync()
        t0 = time.perf_counter()
        ttfa = None
        for chunk in model.generate_streaming(text=prefix + t["text"], cfg_value=2.0, inference_timesteps=10, **kw):
            ttfa = time.perf_counter() - t0
            break
        rec.save(variant, t["id"], np.asarray(wav), SR, tm.s, ttfa_s=round(ttfa, 3) if ttfa else None)


for voice, r in REFS.items():
    run(f"clone-{voice}", dict(prompt_wav_path=r["wav"], prompt_text=r["text"], reference_wav_path=r["wav"]))
run("design-female", {}, DESIGN)
rec.flush()
