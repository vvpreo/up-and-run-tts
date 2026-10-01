"""F5-TTS Russian fine-tunes (voice cloning) with RUAccent stress marks ('+' before the stressed vowel)."""
import argparse

from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent

import json
import time

from common import REFS, ROOT, TEXTS, Recorder, Timer, cap_gpu_memory, clause_chunks, concat

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/f5-ru")
ap.add_argument("--only", default="v2,v4,espeech")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "f5-tts-ru")

accent = RUAccent()
accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
marked = {t["id"]: accent.process_all(t["text"]) for t in TEXTS}
rec.note(ruaccent={k: v for k, v in marked.items()})

M = "Misha24-10/F5-TTS_RUSSIAN"
CKPTS = {
    "v2": (M, "F5TTS_v1_Base_v2/model_last_inference.safetensors", M, "F5TTS_v1_Base/vocab.txt"),
    "v4": (M, "F5TTS_v1_Base_v4_winter/model_212000.safetensors", M, "F5TTS_v1_Base/vocab.txt"),
    "espeech": ("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt", "ESpeech/ESpeech-TTS-1_RL-V2", "vocab.txt"),
}
for key in args.only.split(","):
    repo, ckpt, vrepo, vocab = CKPTS[key]
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=hf_hub_download(repo, ckpt), vocab_file=hf_hub_download(vrepo, vocab))
    for voice, r in REFS.items():
        variant = f"{key}-{voice}"

        def synth(text):
            wav, sr, _ = tts.infer(ref_file=r["wav"], ref_text=r["text_stressed"], gen_text=text,
                                   nfe_step=32, seed=1234, show_info=lambda *a, **k: None)
            return wav, sr

        synth(marked["t01"])  # warm-up
        for t in TEXTS:
            if rec.done(variant, t["id"]):
                continue
            with Timer() as tm:
                wav, sr = synth(marked[t["id"]])
            rec.save(variant, t["id"], wav, sr, tm.s)
    if key == "v4":
        r = REFS["natasha"]
        # hand-corrected stress marks: what the model does when the accentor is right
        manual = dict(marked, **json.loads((ROOT / "benchmark/refs/stress_manual.json").read_text()))
        for t in TEXTS:
            if not rec.done("v4manual-natasha", t["id"]):
                with Timer() as tm:
                    wav, sr = synth(manual[t["id"]])
                rec.save("v4manual-natasha", t["id"], wav, sr, tm.s)
            if not rec.done("v4chunked-natasha", t["id"]):  # text delivered clause by clause
                wavs, first = [], None
                with Timer() as tm:
                    for c in clause_chunks(manual[t["id"]]):
                        w, sr = synth(c)
                        wavs.append(w)
                        first = first or (time.perf_counter() - tm.t0)
                rec.save("v4chunked-natasha", t["id"], concat(wavs, sr), sr, tm.s, ttfa_s=round(first, 3), chunks=len(wavs))
    del tts
rec.flush()
