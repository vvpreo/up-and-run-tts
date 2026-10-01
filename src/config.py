"""Переменные окружения сервиса. Всё, что меняется на ходу (голос по умолчанию,
словари, ключи), живёт не здесь, а в src/settings.py на томе с данными."""

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9008"))

# Том с данными: веса моделей, settings.json, кэш silero-stress
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

# Bearer-токен из окружения; пустой = только ключи из админки (а если и их нет —
# доступ открыт, как у up-and-run-stt)
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()

# Админка: basic-auth. Пустой пароль = админка выключена целиком.
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# Какие модели загружать при старте (через запятую), остальные — лениво при
# первом запросе. Пусто = только модель по умолчанию из настроек.
PRELOAD_MODELS = [m.strip() for m in os.getenv("PRELOAD_MODELS", "").split(",") if m.strip()]

# Потоки torch на синтез
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))

# Защита от перегрузки
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

# CORS: origin'ы через запятую, '*' = все, пусто = выключен
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

ENABLE_DOCS = _bool("ENABLE_DOCS", True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Стенд — влияет только на иконку (рамка на непродовых стендах)
APP_ENV = os.getenv("APP_ENV", "prod")

# Откуда брать веса Silero (зеркало на случай недоступности)
SILERO_BASE_URL = os.getenv("SILERO_BASE_URL", "https://models.silero.ai/models/tts/ru")
