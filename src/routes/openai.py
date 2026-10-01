"""OpenAI-совместимый API: POST /v1/audio/speech, GET /v1/models, GET /v1/voices."""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.audio.encode import FORMATS, STREAMABLE, encode_file, to_int16, wav_header
from src.auth import check_bearer
from src.engines.catalog import MODELS
from src.service import RequestError, TTSService

log = logging.getLogger(__name__)
router = APIRouter()


class SpeechRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = ""
    response_format: str = Field("mp3", pattern="^(wav|pcm|mp3|opus|flac)$")
    speed: float | None = Field(None, ge=0.25, le=4.0)
    # Расширения сверх OpenAI
    stream: bool | None = None
    sample_rate: int | None = None
    rate: str | None = None  # SSML rate: x-slow … x-fast или "80%"
    pitch: str | None = None  # SSML pitch: x-low … x-high
    intensity: int | None = Field(None, ge=1, le=5)
    # Словарь произношений на один запрос: дописывает и перекрывает словарь сервиса
    dictionary: dict[str, str] | None = None


def service(request: Request) -> TTSService:
    return request.app.state.service


def settings(request: Request):
    return request.app.state.settings


def require_auth(request: Request):
    check_bearer(request, request.app.state.settings)


@router.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def speech(body: SpeechRequest, request: Request):
    svc = service(request)
    try:
        req = svc.resolve(body.model, body.voice, body.sample_rate, body.speed, body.rate, body.pitch, body.intensity,
                          body.dictionary)
        fragments = svc.prepare(req, body.input)
    except RequestError as e:
        raise HTTPException(e.status, str(e))
    svc.count_request()
    fmt = body.response_format
    stream = body.stream if body.stream is not None else fmt in STREAMABLE
    headers = {"X-Sample-Rate": str(req.sample_rate), "X-Fragments": str(len(fragments)),
               "X-Model": req.model, "X-Voice": req.voice}
    t0 = time.perf_counter()

    if stream and fmt in STREAMABLE | {"mp3"}:
        async def gen():
            if fmt == "wav":
                yield wav_header(req.sample_rate)
            try:
                for i, p in enumerate(fragments):
                    res = await svc.synth_one_async(req, p)
                    if i == 0:
                        log.info("first audio in %.2fs (%s/%s, %d fragments)", time.perf_counter() - t0, req.model,
                                 req.voice, len(fragments))
                    yield to_int16(res.audio) if fmt != "mp3" else encode_file(res.audio, res.sample_rate, "mp3")
            except RequestError as e:
                log.warning("stream aborted: %s", e)
            except asyncio.CancelledError:
                raise

        return StreamingResponse(gen(), media_type=FORMATS[fmt], headers={**headers, "Cache-Control": "no-store"})

    # Целиком: синтезируем все фрагменты, склеиваем, кодируем
    import numpy as np

    parts = []
    try:
        for p in fragments:
            parts.append((await svc.synth_one_async(req, p)).audio)
    except RequestError as e:
        raise HTTPException(e.status, str(e))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    data = encode_file(audio, req.sample_rate, fmt)
    headers["X-Audio-Seconds"] = f"{len(audio) / req.sample_rate:.2f}"
    headers["X-Synth-Seconds"] = f"{time.perf_counter() - t0:.2f}"
    return Response(data, media_type=FORMATS[fmt], headers=headers)


@router.get("/v1/models", dependencies=[Depends(require_auth)])
async def models(request: Request):
    st = settings(request)
    d = st.section("defaults")
    now = int(time.time())
    data = [{"id": "tts-1", "object": "model", "created": now, "owned_by": "up-and-run-tts", "alias_of": d["model"]}]
    for m in MODELS.values():
        if st.model_enabled(m.id):
            data.append({"id": m.id, "object": "model", "created": now, "owned_by": "silero", "license": m.license,
                         "commercial": m.commercial, "title": m.title})
    return {"object": "list", "data": data}


@router.get("/v1/voices", dependencies=[Depends(require_auth)])
async def voices(request: Request, model: str | None = None):
    svc = service(request)
    st = settings(request)
    d = st.section("defaults")
    out = []
    for m in MODELS.values():
        if model and m.id != model:
            continue
        if not st.model_enabled(m.id):
            continue
        try:
            vs = await asyncio.to_thread(svc.engine.voices, m.id)
        except Exception as e:  # noqa: BLE001 — модель не скачалась: не валить весь список
            log.warning("voices of %s unavailable: %s", m.id, e)
            continue
        for v in vs:
            if not st.voice_enabled(m.id, v.id):
                continue
            out.append({"id": v.id, "model": m.id, "title": v.title, "language": v.language, "gender": v.gender,
                        "license": m.license, "commercial": m.commercial,
                        "default": v.id == d["voice"] and m.id == d["model"]})
    return {"voices": out, "default": {"model": d["model"], "voice": d["voice"]}}
