"""Chatterbox Multilingual V3 (0.5B): cloning, Russian."""
import argparse

import soundfile as sf
import torch
import torchaudio


def _sf_load(path, *a, **k):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T.copy()), sr


torchaudio.load = _sf_load

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

from common import REFS, TEXTS, Recorder, Timer, cap_gpu_memory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/chatterbox-v3")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "chatterbox-multilingual-v3")
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
for voice, r in REFS.items():
    model.generate(TEXTS[0]["text"], language_id="ru", audio_prompt_path=r["wav"])  # warm-up
    for t in TEXTS:
        if rec.done(f"clone-{voice}", t["id"]):
            continue
        with Timer() as tm:
            wav = model.generate(t["text"], language_id="ru", audio_prompt_path=r["wav"])
        rec.save(f"clone-{voice}", t["id"], wav.squeeze().cpu().numpy(), model.sr, tm.s)
rec.flush()
