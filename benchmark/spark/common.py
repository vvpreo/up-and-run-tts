"""Shared helpers for the per-model generation scripts (run inside the bench container)."""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
TEXTS = json.loads((ROOT / "benchmark/texts.json").read_text())
REFS = json.loads((ROOT / "benchmark/refs/refs.json").read_text())
for _r in REFS.values():
    _r["wav"] = str(ROOT / _r["wav"])


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return -1.0


class HostMemWatch:
    """Unified memory: GPU allocations show up as a drop of host MemAvailable."""

    def __init__(self):
        self.start = mem_available_gb()
        self.low = self.start
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            self.low = min(self.low, mem_available_gb())
            time.sleep(0.5)

    def used_gb(self) -> float:
        self.low = min(self.low, mem_available_gb())
        return round(self.start - self.low, 2)


def cap_gpu_memory():
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TTS_MEM_FRACTION", "0.25")))


class Recorder:
    """Writes wavs to <out>/<variant>/<text id>.wav and timings to <out>/metrics.json."""

    def __init__(self, out: str, model: str):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.path = self.out / "metrics.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"model": model, "runs": {}}
        self.watch = HostMemWatch()

    def done(self, variant: str, tid: str) -> bool:
        return f"{variant}/{tid}" in self.data["runs"] and (self.out / variant / f"{tid}.wav").exists()

    def save(self, variant: str, tid: str, wav, sr: int, synth_s: float, **extra):
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        d = self.out / variant
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{tid}.wav", wav, sr, subtype="PCM_16")
        audio_s = len(wav) / sr
        rec = {"synth_s": round(synth_s, 3), "audio_s": round(audio_s, 3),
               "rtf": round(synth_s / max(audio_s, 1e-6), 3), "sr": sr, **extra}
        self.data["runs"][f"{variant}/{tid}"] = rec
        self.flush()
        print(f"[{self.model}] {variant}/{tid}: {audio_s:.1f}s audio in {synth_s:.2f}s (rtf {rec['rtf']}) {extra}", flush=True)

    def note(self, **kv):
        self.data.setdefault("notes", {}).update(kv)
        self.flush()

    def flush(self):
        import torch

        self.data["host_mem_used_gb"] = max(self.data.get("host_mem_used_gb", 0), self.watch.used_gb())
        # host_mem_used_gb is polluted by neighbours on the shared box; process RSS is ours alone
        import resource
        self.data["rss_peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        if torch.cuda.is_available():
            self.data["cuda_peak_alloc_gb"] = max(self.data.get("cuda_peak_alloc_gb", 0),
                                                  round(torch.cuda.max_memory_allocated() / 2**30, 2))
            self.data["cuda_peak_reserved_gb"] = max(self.data.get("cuda_peak_reserved_gb", 0),
                                                     round(torch.cuda.max_memory_reserved() / 2**30, 2))
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        sync()
        self.s = time.perf_counter() - self.t0


def clause_chunks(text: str, min_len: int = 25, min_tail: int = 15):
    """Cut text the way an incremental splitter would: always at sentence ends, at clause
    punctuation only when the piece is long enough. Models that need the whole text get these one by one."""
    import re

    out, cur = [], ""
    # punctuation between digits (dates, decimals, times) is not a boundary
    guard = {".": "\ue000", ",": "\ue001", ":": "\ue002"}
    text = re.sub(r"(?<=\d)[.,:](?=\d)", lambda m: guard[m.group()], text)
    unguard = {v: k for k, v in guard.items()}
    for piece in re.findall(r"[^.!?…;:,—]+[.!?…;:,—]*\s*", text):
        cur += "".join(unguard.get(ch, ch) for ch in piece)
        rest = len(text) - sum(map(len, out)) - len(cur)
        hard = bool(re.search(r"[.!?…]\s*$", cur))
        if hard or (len(cur.strip()) >= min_len and rest >= min_tail):
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return [c.strip() for c in out if c.strip()]


def concat(wavs, sr, gap_s=0.08):
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    parts = []
    for w in wavs:
        parts += [np.asarray(w, dtype=np.float32).squeeze(), gap]
    return np.concatenate(parts[:-1])
