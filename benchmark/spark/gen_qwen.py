"""Qwen3-TTS 1.7B: voice clone (Base), voice design, preset voices (CustomVoice)."""
import argparse
import gc
import time

import torch
from qwen_tts import Qwen3TTSModel

from common import REFS, TEXTS, Recorder, Timer, cap_gpu_memory, clause_chunks, concat

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/qwen3-1.7b")
ap.add_argument("--only", default="clone,design,custom")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "qwen3-tts-1.7b")
DESIGN = ("Спокойный, тёплый женский голос лет тридцати. Естественная разговорная интонация, "
          "чистое русское произношение без акцента, средний темп.")


def load(name):
    m = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{name}", device_map="cuda:0", dtype=torch.bfloat16)
    return m


def run(variant, fn):
    fn(TEXTS[0]["text"])  # warm-up, not recorded
    for t in TEXTS:
        if rec.done(variant, t["id"]):
            continue
        with Timer() as tm:
            wavs, sr = fn(t["text"])
        rec.save(variant, t["id"], wavs[0], sr, tm.s)


if "clone" in args.only:
    m = load("Base")
    for voice, r in REFS.items():
        prompt = m.create_voice_clone_prompt(ref_audio=r["wav"], ref_text=r["text"], x_vector_only_mode=False)
        run(f"clone-{voice}", lambda s: m.generate_voice_clone(text=s, language="Russian", voice_clone_prompt=prompt))
        if voice == "natasha":  # same voice, text delivered clause by clause (what an LLM stream gives us)
            for t in TEXTS:
                if rec.done("chunked-natasha", t["id"]):
                    continue
                wavs, first = [], None
                with Timer() as tm:
                    for c in clause_chunks(t["text"]):
                        w, sr = m.generate_voice_clone(text=c, language="Russian", voice_clone_prompt=prompt)
                        wavs.append(w[0])
                        first = first or (time.perf_counter() - tm.t0)
                rec.save("chunked-natasha", t["id"], concat(wavs, sr), sr, tm.s, ttfa_s=round(first, 3), chunks=len(wavs))
    del m; gc.collect(); torch.cuda.empty_cache()
if "design" in args.only:
    m = load("VoiceDesign")
    run("design-female", lambda s: m.generate_voice_design(text=s, language="Russian", instruct=DESIGN))
    del m; gc.collect(); torch.cuda.empty_cache()
if "custom" in args.only:
    m = load("CustomVoice")
    for spk in ("Serena", "Ryan"):
        run(f"preset-{spk.lower()}", lambda s: m.generate_custom_voice(text=s, language="Russian", speaker=spk))
rec.flush()
