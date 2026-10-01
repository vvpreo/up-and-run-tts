"""Админка: статическая страница /admin и её JSON API под basic-auth."""

import asyncio
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.audio.encode import encode_file
from src.auth import check_admin
from src.config import ADMIN_PASSWORD, AUTH_TOKEN, APP_ENV
from src.engines.catalog import MODELS
from src.service import RequestError

log = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(check_admin)])
STATIC = Path(__file__).resolve().parents[1] / "static" / "admin"


def _svc(request: Request):
    return request.app.state.service


def _st(request: Request):
    return request.app.state.settings


@router.get("/admin", include_in_schema=False)
async def page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/admin/api/state")
async def state(request: Request):
    svc, st = _svc(request), _st(request)
    data = st.get()
    data["api_keys"] = [{**k, "key": k["key"][:8] + "…"} for k in data["api_keys"]]
    models = []
    for m in MODELS.values():
        entry = {"id": m.id, "title": m.title, "license": m.license, "commercial": m.commercial,
                 "builtin_stress": m.builtin_stress, "intonation": m.intonation, "notes": m.notes,
                 "sample_rates": list(m.sample_rates), "enabled": st.model_enabled(m.id),
                 "downloaded": svc.engine.is_downloaded(m.id), "loaded": svc.engine.is_loaded(m.id), "voices": []}
        if entry["loaded"]:
            entry["voices"] = [{"id": v.id, "gender": v.gender, "enabled": st.voice_enabled(m.id, v.id)}
                               for v in svc.engine.voices(m.id)]
        models.append(entry)
    return {"settings": data, "models": models, "stats": svc.stats, "app_env": APP_ENV,
            "env_token": bool(AUTH_TOKEN), "admin_enabled": bool(ADMIN_PASSWORD)}


class Section(BaseModel):
    values: dict


@router.put("/admin/api/defaults")
async def put_defaults(body: Section, request: Request):
    st = _st(request)
    v = body.values
    if "model" in v and v["model"] not in MODELS:
        raise HTTPException(400, "unknown model")
    if "sample_rate" in v and int(v["sample_rate"]) not in (8000, 24000, 48000):
        raise HTTPException(400, "sample_rate must be 8000, 24000 or 48000")
    return st.update("defaults", v)


@router.put("/admin/api/text")
async def put_text(body: Section, request: Request):
    return _st(request).update("text", body.values)


@router.get("/admin/api/dictionary")
async def get_dictionary(request: Request):
    return _st(request).section("dictionary")


@router.put("/admin/api/dictionary")
async def put_dictionary(body: Section, request: Request):
    return _st(request).update("dictionary", {str(k): str(v) for k, v in body.values.items()})


class ModelToggle(BaseModel):
    enabled: bool


@router.put("/admin/api/models/{model_id}")
async def toggle_model(model_id: str, body: ModelToggle, request: Request):
    if model_id not in MODELS:
        raise HTTPException(404, "unknown model")
    st = _st(request)
    st.update("models", {model_id: {"enabled": body.enabled}})
    if not body.enabled:
        _svc(request).engine.unload(model_id)
    return {"id": model_id, "enabled": body.enabled}


@router.put("/admin/api/models/{model_id}/voices/{voice_id}")
async def toggle_voice(model_id: str, voice_id: str, body: ModelToggle, request: Request):
    if model_id not in MODELS:
        raise HTTPException(404, "unknown model")
    _st(request).update("voices", {f"{model_id}/{voice_id}": body.enabled})
    return {"model": model_id, "voice": voice_id, "enabled": body.enabled}


@router.post("/admin/api/models/{model_id}/load")
async def load_model(model_id: str, request: Request):
    if model_id not in MODELS:
        raise HTTPException(404, "unknown model")
    svc = _svc(request)
    try:
        await asyncio.to_thread(svc.engine.load, model_id)
    except Exception as e:  # noqa: BLE001
        log.exception("load %s failed", model_id)
        raise HTTPException(500, f"load failed: {e}")
    return {"id": model_id, "loaded": True, "voices": [v.id for v in svc.engine.voices(model_id)]}


@router.post("/admin/api/models/{model_id}/unload")
async def unload_model(model_id: str, request: Request):
    _svc(request).engine.unload(model_id)
    return {"id": model_id, "loaded": False}


class KeyCreate(BaseModel):
    name: str = ""


@router.get("/admin/api/keys")
async def keys(request: Request):
    return {"keys": [{**k, "key": k["key"][:8] + "…"} for k in _st(request).api_keys()], "env_token": bool(AUTH_TOKEN)}


@router.post("/admin/api/keys")
async def create_key(body: KeyCreate, request: Request):
    return _st(request).create_key(body.name)  # полный ключ показывается один раз


@router.delete("/admin/api/keys/{key_id}")
async def delete_key(key_id: str, request: Request):
    if not _st(request).delete_key(key_id):
        raise HTTPException(404, "no such key")
    return {"deleted": key_id}


class Speak(BaseModel):
    text: str
    model: str | None = None
    voice: str | None = None
    sample_rate: int | None = None
    rate: str | None = None
    pitch: str | None = None
    intensity: int | None = None
    dictionary: dict[str, str] | None = None
    format: str = "wav"


@router.post("/admin/api/prepare")
async def prepare(body: Speak, request: Request):
    svc = _svc(request)
    try:
        req = svc.resolve(body.model, body.voice, body.sample_rate, None, body.rate, body.pitch, body.intensity,
                          body.dictionary)
        frags = await asyncio.to_thread(svc.prepare, req, body.text)
    except RequestError as e:
        raise HTTPException(e.status, str(e))
    return {"model": req.model, "voice": req.voice, "fragments": [{"source": f.source, "prepared": f.shown} for f in frags]}


@router.post("/admin/api/speak")
async def speak(body: Speak, request: Request):
    svc = _svc(request)
    try:
        req = svc.resolve(body.model, body.voice, body.sample_rate, None, body.rate, body.pitch, body.intensity,
                          body.dictionary)
        frags = await asyncio.to_thread(svc.prepare, req, body.text)
        parts = [(await svc.synth_one_async(req, p)).audio for p in frags]
    except RequestError as e:
        raise HTTPException(e.status, str(e))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    fmt = body.format if body.format in ("wav", "mp3") else "wav"
    return Response(encode_file(audio, req.sample_rate, fmt), media_type="audio/wav" if fmt == "wav" else "audio/mpeg",
                    headers={"X-Fragments": str(len(frags)), "Cache-Control": "no-store"})
