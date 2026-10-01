"""Bearer-авторизация API (AUTH_TOKEN из окружения + ключи из админки) и
basic-auth админки."""

import base64
import secrets

from fastapi import HTTPException, Request

from src.config import ADMIN_PASSWORD, ADMIN_USER, AUTH_TOKEN


def auth_enabled(settings) -> bool:
    return bool(AUTH_TOKEN) or bool(settings.valid_keys())


def check_bearer(request: Request, settings) -> None:
    if not auth_enabled(settings):
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else request.query_params.get("token", "")
    if not token:
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    if (AUTH_TOKEN and secrets.compare_digest(token, AUTH_TOKEN)) or token in settings.valid_keys():
        return
    raise HTTPException(401, "invalid token", headers={"WWW-Authenticate": "Bearer"})


def check_admin(request: Request) -> None:
    if not ADMIN_PASSWORD:
        raise HTTPException(404, "admin is disabled (set ADMIN_PASSWORD)")
    header = request.headers.get("authorization", "")
    ok = False
    if header.lower().startswith("basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode().partition(":")
            ok = secrets.compare_digest(user, ADMIN_USER) and secrets.compare_digest(pwd, ADMIN_PASSWORD)
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        raise HTTPException(401, "admin auth required", headers={"WWW-Authenticate": 'Basic realm="up-and-run-tts admin"'})
