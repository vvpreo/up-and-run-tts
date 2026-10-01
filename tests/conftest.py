"""Интеграционные тесты ходят в уже запущенный сервис: TTS_URL (по умолчанию
http://localhost:9008), TTS_TOKEN, TTS_ADMIN_USER / TTS_ADMIN_PASSWORD. Если
переменных нет, значения берутся из .env проекта (тот же файл, что читает
docker compose). Сервис недоступен — интеграционные тесты пропускаются,
юнит-тесты конвейера работают всегда."""

import os
from pathlib import Path

import httpx
import pytest


def _dotenv() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


_ENV = _dotenv()
URL = os.getenv("TTS_URL", "http://localhost:9008")
TOKEN = os.getenv("TTS_TOKEN", _ENV.get("AUTH_TOKEN", ""))
ADMIN = (os.getenv("TTS_ADMIN_USER", _ENV.get("ADMIN_USER", "admin")),
         os.getenv("TTS_ADMIN_PASSWORD", _ENV.get("ADMIN_PASSWORD", "")))


@pytest.fixture(scope="session")
def api():
    try:
        r = httpx.get(f"{URL}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"service not reachable at {URL}: {e}")
    return httpx.Client(base_url=URL, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120)


@pytest.fixture(scope="session")
def admin(api):
    return httpx.Client(base_url=URL, auth=ADMIN, timeout=120)
