"""Настройки, которые меняются из админки без перезапуска. Один JSON на томе,
атомарная запись, чтение под локом. Значения по умолчанию здесь же."""

import copy
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR
from src.engines.catalog import DEFAULT_MODEL, MODELS

log = logging.getLogger(__name__)

DEFAULTS = {
    "defaults": {
        "model": DEFAULT_MODEL,
        "voice": "ru_zhadyra",
        "sample_rate": 24000,
        # Значения SSML prosody у Silero: x-slow|slow|medium|fast|x-fast или проценты ("80%")
        "rate": "medium",
        # x-low|low|medium|high|x-high
        "pitch": "medium",
        # Сила логического ударения (*слово) у новых ru-моделей, 1–5
        "intensity": 3,
        "put_yo": True,
    },
    # Какие модели разрешены к использованию. Некоммерческие выключены, пока
    # пользователь явно не включит их в админке.
    "models": {m.id: {"enabled": m.commercial} for m in MODELS.values()},
    # Выключенные голоса: "<модель>/<голос>" -> false. Всё, чего здесь нет, включено.
    "voices": {},
    "text": {
        "strip_markdown": True,
        "normalize_numbers": True,
        "normalize_latin": True,
        "put_stress": True,
        # Нарезка для потоковой отдачи: не резать по запятым раньше этой длины
        "min_fragment_chars": 25,
    },
    # Словарь: слово -> как произносить. Ключ сравнивается без регистра; значение
    # может содержать ударения (+ перед гласной) и быть на кириллице для латиницы.
    "dictionary": {},
    # Ключи API, созданные в админке (в дополнение к AUTH_TOKEN из окружения)
    "api_keys": [],
}


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "settings.json"
        self._lock = threading.RLock()
        self._data = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    stored = json.loads(self.path.read_text())
                except json.JSONDecodeError as e:
                    log.error("settings.json is corrupt (%s); using defaults", e)
                    return
                self._data = self._merge(copy.deepcopy(DEFAULTS), stored)

    @staticmethod
    def _merge(base: dict, over: dict) -> dict:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "dictionary":
                base[k] = Settings._merge(base[k], v)
            else:
                base[k] = v
        return base

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1))
            tmp.replace(self.path)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def section(self, name: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._data[name])

    def update(self, name: str, values: dict) -> dict:
        """Частичное обновление раздела (defaults / text / models)."""
        with self._lock:
            if name == "dictionary":
                self._data["dictionary"] = {k.strip().lower(): v.strip() for k, v in values.items() if k.strip()}
            else:
                self._data[name] = self._merge(self._data[name], values)
            self.save()
            return copy.deepcopy(self._data[name])

    # --- ключи API ----------------------------------------------------------------
    def api_keys(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._data["api_keys"])

    def create_key(self, name: str) -> dict:
        key = {"id": secrets.token_hex(4), "name": name.strip() or "key", "key": "tts-" + secrets.token_urlsafe(24),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self._lock:
            self._data["api_keys"].append(key)
            self.save()
        return key

    def delete_key(self, key_id: str) -> bool:
        with self._lock:
            before = len(self._data["api_keys"])
            self._data["api_keys"] = [k for k in self._data["api_keys"] if k["id"] != key_id]
            self.save()
            return len(self._data["api_keys"]) < before

    def valid_keys(self) -> set[str]:
        with self._lock:
            return {k["key"] for k in self._data["api_keys"]}

    def voice_enabled(self, model: str, voice: str) -> bool:
        with self._lock:
            return self._data["voices"].get(f"{model}/{voice}", True) is not False

    def model_enabled(self, model: str) -> bool:
        with self._lock:
            return bool(self._data["models"].get(model, {}).get("enabled"))
