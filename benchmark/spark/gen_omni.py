"""OmniVoice (k2-fsa): zero-shot cloning."""
import argparse

import torch
from omnivoice import OmniVoice

from common import REFS, TEXTS, Recorder, Timer, cap_gpu_memory

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/omnivoice")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "omnivoice")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
for voice, r in REFS.items():
    prompt = model.create_voice_clone_prompt(ref_audio=r["wav"], ref_text=r["text"])
    model.generate(text=TEXTS[0]["text"], voice_clone_prompt=prompt)  # warm-up
    for t in TEXTS:
        if rec.done(f"clone-{voice}", t["id"]):
            continue
        with Timer() as tm:
            audio = model.generate(text=t["text"], voice_clone_prompt=prompt)
        rec.save(f"clone-{voice}", t["id"], audio[0], 24000, tm.s)
rec.flush()
