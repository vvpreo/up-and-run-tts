"""ASR round-trip for generated samples: transcribe with the local up-and-run-stt (GigaAM) and compare to the text.

Measures intelligibility only (wrong/garbled/skipped words). It does NOT measure stress or intonation.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".tmp/out"
TEXTS = {t["id"]: t for t in json.loads((ROOT / "benchmark/texts.json").read_text())}
STT = "http://localhost:9007/v1/audio/transcriptions"
env = subprocess.run(["docker", "inspect", "up-and-run-stt-cuda12", "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
                     capture_output=True, text=True).stdout
TOKEN = next(l.split("=", 1)[1] for l in env.splitlines() if l.startswith("AUTH_TOKEN="))


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def transcribe(path: Path) -> str:
    with open(path, "rb") as f:
        r = requests.post(STT, headers={"Authorization": f"Bearer {TOKEN}"}, files={"file": (path.name, f, "audio/wav")},
                          data={"response_format": "text", "model": "v3_e2e_rnnt", "language": "ru"}, timeout=300)
    r.raise_for_status()
    return r.text.strip()


_mos = None


def utmos(path: Path) -> float:
    """UTMOS22 naturalness predictor. Trained on English: a weak, relative signal for Russian."""
    global _mos
    import librosa
    import torch

    if _mos is None:
        _mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    w, sr = librosa.load(path, sr=16000, mono=True)
    with torch.no_grad():
        return round(float(_mos(torch.from_numpy(w[: 16000 * 40]).unsqueeze(0), sr)), 2)


only = sys.argv[1:]
cache_path = OUT / "eval.json"
res = json.loads(cache_path.read_text()) if cache_path.exists() else {}
for mdir in sorted(p for p in OUT.iterdir() if p.is_dir()):
    if only and mdir.name not in only:
        continue
    for wav in sorted(mdir.glob("*/*.wav")):
        key = f"{mdir.name}/{wav.parent.name}/{wav.stem}"
        if key in res and res[key].get("mtime") == wav.stat().st_mtime:
            if "utmos" not in res[key]:
                res[key]["utmos"] = utmos(wav)
            continue
        t = TEXTS[wav.stem]
        # GigaAM writes numbers as digits, so number texts carry a second, digit-form reference.
        refs = [norm(t.get("spoken", t["text"]))] + ([norm(t["asr_ref"])] if "asr_ref" in t else [])
        hyp_raw = transcribe(wav)
        hyp = norm(hyp_raw)
        res[key] = {"hyp": hyp_raw, "cer": min(round(lev(r, hyp) / len(r), 4) for r in refs),
                    "wer": min(round(lev(r.split(), hyp.split()) / len(r.split()), 4) for r in refs),
                    "mtime": wav.stat().st_mtime, "utmos": utmos(wav)}
        print(key, res[key]["cer"], "|", hyp_raw[:110])
cache_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
