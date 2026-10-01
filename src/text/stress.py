"""Словарь произношений и расстановка ударений.

Порядок: сначала словарь (пользовательский поверх встроенного) заменяет слова
на их произношение с «+», затем silero-stress расставляет ударения в остальных
словах — уже проставленные «+» он не трогает."""

import logging
import re
import threading

from src.text.latin_words import BUILTIN

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’\-]*")


def apply_dictionary(text: str, user: dict[str, str]) -> str:
    """Замена слов по словарю без учёта регистра; многословные ключи — сначала."""
    merged = {**BUILTIN, **{k.lower(): v for k, v in user.items()}}
    multi = [k for k in merged if " " in k]
    for k in sorted(multi, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(k) + r"(?!\w)", merged[k], text, flags=re.I)

    def repl(m: re.Match) -> str:
        return merged.get(m.group(0).lower(), m.group(0))

    return _WORD.sub(repl, text)


class Accentor:
    """silero-stress, загружается лениво и один раз (около 50 МБ, 1–2 с)."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import silero_stress

                    self._model = silero_stress.load_accentor()
                    log.info("silero-stress loaded")
        return self._model

    def __call__(self, text: str, put_yo: bool = True) -> str:
        acc = self.load()
        with self._lock:
            return acc(text, put_stress=True, put_stress_homo=True, put_yo=put_yo, put_yo_homo=put_yo)


accentor = Accentor()
