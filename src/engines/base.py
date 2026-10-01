"""Интерфейс движка синтеза. Конвейер текста и API знают только его, поэтому
второй движок (GPU-модель, другой образ) подключается без переделки API."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Voice:
    id: str
    model: str
    title: str
    language: str = "ru"
    gender: str = ""  # male | female | "" (неизвестно)


@dataclass
class SynthResult:
    audio: np.ndarray  # float32, моно, [-1, 1]
    sample_rate: int
    synth_s: float


class TTSEngine:
    """Один движок может обслуживать несколько моделей (у Silero — несколько файлов)."""

    def voices(self, model: str) -> list[Voice]:
        raise NotImplementedError

    def synth(self, model: str, voice: str, *, text: str | None = None, ssml: str | None = None,
              sample_rate: int = 24000, **params) -> SynthResult:
        """Либо text (уже подготовленный конвейером), либо ssml — не оба сразу."""
        raise NotImplementedError

    def is_loaded(self, model: str) -> bool:
        raise NotImplementedError
