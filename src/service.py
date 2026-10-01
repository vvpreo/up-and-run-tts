"""Связка настроек, движка и конвейера текста: единственная точка, через которую
API (HTTP, WebSocket, админка) синтезируют речь."""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from src.config import MAX_INPUT_CHARS, MAX_PENDING_REQUESTS, PRELOAD_MODELS
from src.engines.base import SynthResult
from src.engines.catalog import MODELS, ModelInfo
from src.engines.silero import SileroEngine
from src.settings import Settings
from src.text.pipeline import Prepared, TextPipeline, is_ssml, rate_from_speed
from src.text.stress import accentor

log = logging.getLogger(__name__)

OPENAI_MODEL_ALIASES = {"tts-1", "tts-1-hd", "gpt-4o-mini-tts", "default", ""}


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Request:
    model: str
    voice: str
    sample_rate: int
    intensity: int
    put_yo: bool
    pipeline: TextPipeline
    info: ModelInfo


class TTSService:
    def __init__(self, settings: Settings, engine: SileroEngine | None = None):
        self.settings = settings
        self.engine = engine or SileroEngine()
        self._sem = threading.BoundedSemaphore(MAX_PENDING_REQUESTS) if MAX_PENDING_REQUESTS > 0 else None
        self.started = time.time()
        self.stats = {"requests": 0, "fragments": 0, "audio_s": 0.0, "synth_s": 0.0}
        self._stats_lock = threading.Lock()

    # --- запуск ---------------------------------------------------------------------
    def preload(self) -> None:
        d = self.settings.section("defaults")
        enabled = [m for m in MODELS if self.settings.model_enabled(m)]
        for m in dict.fromkeys([d["model"], *PRELOAD_MODELS, *enabled]):
            if m in MODELS and self.settings.model_enabled(m):
                try:
                    self.engine.load(m)
                except Exception:  # noqa: BLE001 — сервис должен подняться и без весов
                    log.exception("cannot preload %s", m)
        if not MODELS[d["model"]].builtin_stress:
            try:
                accentor.load()
            except Exception:  # noqa: BLE001
                log.exception("cannot load silero-stress")

    # --- разбор запроса -------------------------------------------------------------
    def resolve(self, model: str | None = None, voice: str | None = None, sample_rate: int | None = None,
                speed: float | None = None, rate: str | None = None, pitch: str | None = None,
                intensity: int | None = None, dictionary: dict[str, str] | None = None) -> Request:
        d = self.settings.section("defaults")
        model = (model or "").strip()
        if model in OPENAI_MODEL_ALIASES:
            model = d["model"]
        if model not in MODELS:
            raise RequestError(f"unknown model {model!r}; available: {', '.join(MODELS)}")
        if not self.settings.model_enabled(model):
            raise RequestError(f"model {model!r} is disabled in settings", 403)
        info = MODELS[model]
        voice = (voice or "").strip()
        if voice in ("", "alloy", "echo", "fable", "onyx", "nova", "shimmer", "default"):
            voice = d["voice"] if model == d["model"] else self.engine.voices(model)[0].id
        known = {v.id for v in self.engine.voices(model)}
        if voice not in known:
            raise RequestError(f"unknown voice {voice!r} for model {model!r}")
        if not self.settings.voice_enabled(model, voice):
            raise RequestError(f"voice {voice!r} is disabled in settings", 403)
        # Словарь запроса дописывает и перекрывает словарь из настроек
        words = self.settings.section("dictionary")
        if dictionary:
            words.update({str(k).strip().lower(): str(v).strip() for k, v in dictionary.items() if str(k).strip()})
        sr = int(sample_rate or d["sample_rate"])
        if sr not in info.sample_rates:
            raise RequestError(f"sample_rate must be one of {list(info.sample_rates)}")
        pipeline = TextPipeline(self.settings.section("text"), words, info,
                                rate=rate or rate_from_speed(speed, d["rate"]), pitch=pitch or d["pitch"],
                                put_yo=bool(d.get("put_yo", True)))
        return Request(model=model, voice=voice, sample_rate=sr, intensity=int(intensity or d["intensity"]),
                       put_yo=bool(d.get("put_yo", True)), pipeline=pipeline, info=info)

    def prepare(self, req: Request, text: str) -> list[Prepared]:
        if len(text) > MAX_INPUT_CHARS:
            raise RequestError(f"input is longer than {MAX_INPUT_CHARS} characters", 413)
        if not text.strip():
            raise RequestError("input is empty")
        if is_ssml(text):
            return [Prepared(ssml=text.strip(), source=text.strip())]
        return req.pipeline.prepare(text)

    # --- синтез -----------------------------------------------------------------------
    def synth_one(self, req: Request, prepared: Prepared) -> SynthResult:
        if self._sem and not self._sem.acquire(timeout=0.01):
            raise RequestError("too many concurrent requests", 429)
        try:
            res = self.engine.synth(req.model, req.voice, text=prepared.text, ssml=prepared.ssml,
                                    sample_rate=req.sample_rate, put_accent=True, put_yo=req.put_yo,
                                    intensity=req.intensity)
        finally:
            if self._sem:
                self._sem.release()
        with self._stats_lock:
            self.stats["fragments"] += 1
            self.stats["audio_s"] += len(res.audio) / res.sample_rate
            self.stats["synth_s"] += res.synth_s
        return res

    def count_request(self) -> None:
        with self._stats_lock:
            self.stats["requests"] += 1

    def synth_all(self, req: Request, fragments: list[Prepared]) -> Iterator[SynthResult]:
        self.count_request()
        for p in fragments:
            yield self.synth_one(req, p)

    async def synth_one_async(self, req: Request, prepared: Prepared) -> SynthResult:
        return await asyncio.to_thread(self.synth_one, req, prepared)
