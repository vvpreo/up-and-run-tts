"""Silero v5_5_ru baseline on the dev box CPU (no GPU involved)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero import silero_tts

ROOT = Path(__file__).resolve().parents[1]
TEXTS = json.loads((ROOT / "benchmark/texts.json").read_text())
out = ROOT / ".tmp/out/silero-v5"
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.set_num_threads(threads)
t0 = time.perf_counter()
res = silero_tts(language="ru", speaker="v5_5_ru")
model = res[0] if isinstance(res, tuple) else res
load_s = time.perf_counter() - t0
SR = 48000
metrics = {"model": "silero-v5_5_ru", "runs": {}, "notes": {"device": "cpu i7-8750H", "threads": threads, "load_s": round(load_s, 2)}}
for spk in ("xenia", "kseniya", "eugene"):
    model.apply_tts(text="Проверка связи.", speaker=spk, sample_rate=SR)  # warm-up
    for t in TEXTS:
        t1 = time.perf_counter()
        try:
            wav = model.apply_tts(text=t["text"], speaker=spk, sample_rate=SR)
        except Exception as e:  # noqa: BLE001 - record and continue, this is a survey
            print("FAIL", spk, t["id"], repr(e)[:200])
            metrics["runs"][f"{spk}/{t['id']}"] = {"error": repr(e)[:200]}
            continue
        s = time.perf_counter() - t1
        wav = np.asarray(wav, dtype=np.float32)
        d = out / spk
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{t['id']}.wav", wav, SR, subtype="PCM_16")
        a = len(wav) / SR
        metrics["runs"][f"{spk}/{t['id']}"] = {"synth_s": round(s, 3), "audio_s": round(a, 3), "rtf": round(s / a, 3), "sr": SR}
        print(spk, t["id"], f"{a:.1f}s audio in {s:.2f}s")
(out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
