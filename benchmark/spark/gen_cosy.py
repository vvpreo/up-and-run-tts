"""Fun-CosyVoice 3 (0.5B): zero-shot cloning, whole text vs. incremental text (generator input, "bistream")."""
import argparse
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/proj/.src/CosyVoice")
sys.path.insert(0, "/proj/.src/CosyVoice/third_party/Matcha-TTS")


def _sf_load(path, *a, **k):
    """torchaudio 2.11 routes load() through torchcodec; read with soundfile instead."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T.copy()), sr


torchaudio.load = _sf_load

from huggingface_hub import snapshot_download  # noqa: E402

from common import REFS, TEXTS, Recorder, cap_gpu_memory, sync  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/cosyvoice3")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "fun-cosyvoice3-0.5b")
model_dir = snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir="/proj/.models/Fun-CosyVoice3-0.5B")
cosy = AutoModel(model_dir=model_dir)
SR = cosy.sample_rate
SYS = "You are a helpful assistant.<|endofprompt|>"


def deltas(text, chars=4, delay=0.025):
    for i in range(0, len(text), chars):
        if i:
            time.sleep(delay)
        yield text[i:i + chars]


def synth(text_or_gen, r, stream):
    chunks, ttfa = [], None
    sync()
    t0 = time.perf_counter()
    for out in cosy.inference_zero_shot(text_or_gen, SYS + r["text"], r["wav"], stream=stream, text_frontend=False):
        if ttfa is None:
            sync()
            ttfa = time.perf_counter() - t0
        chunks.append(out["tts_speech"].reshape(-1).cpu().numpy())
    sync()
    return np.concatenate(chunks), time.perf_counter() - t0, ttfa


for voice, r in REFS.items():
    synth(TEXTS[0]["text"], r, False)  # warm-up
    synth(TEXTS[0]["text"], r, True)
    for t in TEXTS:
        if not rec.done(f"full-{voice}", t["id"]):
            wav, s, ttfa = synth(t["text"], r, False)
            rec.save(f"full-{voice}", t["id"], wav, SR, s)
        if not rec.done(f"fullstream-{voice}", t["id"]):  # whole text, audio streamed out in chunks
            wav, s, ttfa = synth(t["text"], r, True)
            rec.save(f"fullstream-{voice}", t["id"], wav, SR, s, ttfa_s=round(ttfa, 3))
        if not rec.done(f"stream-{voice}", t["id"]):  # text arrives gradually
            wav, s, ttfa = synth(deltas(t["text"]), r, True)
            rec.save(f"stream-{voice}", t["id"], wav, SR, s, ttfa_s=round(ttfa, 3), text_feed_s=round(len(t["text"]) / 4 * 0.025, 2))
rec.flush()
