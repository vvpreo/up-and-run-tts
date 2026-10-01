"""Движок Silero: модели — torch.package-архивы с models.silero.ai, хранятся на томе.

Пакет `silero` с PyPI не используется: он кладёт веса внутрь site-packages и
тянет omegaconf. Загрузка здесь повторяет его код в три строки."""

import json
import logging
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from src.config import DATA_DIR, SILERO_BASE_URL, TORCH_THREADS
from src.engines.base import SynthResult, TTSEngine, Voice
from src.engines.catalog import MODELS

log = logging.getLogger(__name__)


def _download(url: str, dst: Path) -> None:
    """Скачивание через .part со сверкой размера: обрезанный файл не должен
    закэшироваться (так уже падал up-and-run-stt)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
        expected = int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    if expected and part.stat().st_size != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download of {url} truncated: {part.stat().st_size if part.exists() else 0} != {expected}")
    part.replace(dst)


# Голоса, у которых основной тон обманывает (высокий мужской): пол по имени
_GENDER_OVERRIDES = {"ru_igor": "male"}


def _median_f0(a: np.ndarray, sr: int) -> float:
    """Медианный основной тон по автокорреляции озвученных окон, Гц."""
    fs = []
    for i in range(0, len(a) - 2048, 1024):
        w = a[i:i + 2048] * np.hanning(2048)
        if np.sqrt((w ** 2).mean()) < 0.02:
            continue
        ac = np.correlate(w, w, "full")[2047:]
        lo, hi = sr // 400, sr // 70
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.4 * ac[0]:
            fs.append(sr / k)
    return float(np.median(fs)) if fs else 0.0


class SileroEngine(TTSEngine):
    def __init__(self):
        torch.set_num_threads(TORCH_THREADS)
        self._models: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {m: threading.Lock() for m in MODELS}
        self._load_lock = threading.Lock()

    # --- загрузка -----------------------------------------------------------------
    def model_path(self, model: str) -> Path:
        return DATA_DIR / "silero" / MODELS[model].file

    def is_downloaded(self, model: str) -> bool:
        return self.model_path(model).exists()

    def is_loaded(self, model: str) -> bool:
        return model in self._models

    def load(self, model: str):
        if model not in MODELS:
            raise KeyError(f"unknown model {model!r}")
        if model in self._models:
            return self._models[model]
        with self._load_lock:
            if model in self._models:
                return self._models[model]
            path = self.model_path(model)
            if not path.exists():
                _download(f"{SILERO_BASE_URL}/{MODELS[model].file}", path)
            t0 = time.perf_counter()
            imp = torch.package.PackageImporter(str(path))
            m = imp.load_pickle("tts_models", "model")
            log.info("loaded %s in %.1fs (%d speakers)", model, time.perf_counter() - t0, len(m.speakers))
            self._models[model] = m
            return m

    def unload(self, model: str) -> None:
        with self._load_lock:
            self._models.pop(model, None)

    # --- интерфейс движка ---------------------------------------------------------
    def voices(self, model: str) -> list[Voice]:
        info = MODELS[model]
        m = self.load(model)
        genders = self._genders(model)
        out = []
        for s in m.speakers:
            if info.voice_prefix and not s.startswith(info.voice_prefix):
                continue
            if s == "random":
                continue
            out.append(Voice(id=s, model=model, title=s.removeprefix("ru_").capitalize(), gender=genders.get(s, "")))
        return out

    # --- пол голоса ---------------------------------------------------------------
    # У Silero нет метаданных о голосах; пол определяется один раз по основному тону
    # короткой фразы и кэшируется на томе рядом с весами.
    def _genders(self, model: str) -> dict[str, str]:
        path = DATA_DIR / "silero" / f"{MODELS[model].file}.voices.json"
        if path.exists():
            return {k: v["gender"] for k, v in json.loads(path.read_text()).items()}
        info = MODELS[model]
        m = self._models[model]
        result = {}
        for s in m.speakers:
            if (info.voice_prefix and not s.startswith(info.voice_prefix)) or s == "random":
                continue
            try:
                with self._locks[model], torch.inference_mode():
                    wav = m.apply_tts(text="Здравствуйте, я говорю по-русски.", speaker=s, sample_rate=24000)
                f0 = _median_f0(np.asarray(wav, dtype=np.float32), 24000)
                result[s] = {"f0": round(f0), "gender": _GENDER_OVERRIDES.get(s, "male" if f0 < 165 else "female")}
            except Exception as e:  # noqa: BLE001
                log.warning("gender probe failed for %s: %s", s, e)
                result[s] = {"f0": 0, "gender": ""}
        genders = {k: v["gender"] for k, v in result.items()}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        log.info("voice genders of %s: %s", model, genders)
        return genders

    def synth(self, model: str, voice: str, *, text=None, ssml=None, sample_rate=24000,
              put_accent=True, put_yo=True, intensity=3, **_) -> SynthResult:
        info = MODELS[model]
        m = self.load(model)
        kwargs = dict(speaker=voice, sample_rate=sample_rate, put_accent=put_accent, put_stress_homo=put_accent,
                      put_yo=put_yo, put_yo_homo=put_yo)
        if info.intonation:
            kwargs["intensity"] = intensity
        t0 = time.perf_counter()
        # Модель не рассчитана на параллельные вызовы — по одному на модель.
        with self._locks[model], torch.inference_mode():
            if ssml is not None:
                wav = m.apply_tts(ssml_text=ssml, **kwargs)
            else:
                wav = m.apply_tts(text=text, **kwargs)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthResult(audio=audio, sample_rate=sample_rate, synth_s=time.perf_counter() - t0)
