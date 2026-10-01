import time

import psutil
from fastapi import APIRouter, Request

from src import __version__
from src.config import APP_ENV
from src.engines.catalog import MODELS

router = APIRouter()
VERSION = __version__


@router.get("/health")
async def health(request: Request):
    svc = request.app.state.service
    st = request.app.state.settings
    d = st.section("defaults")
    proc = psutil.Process()
    return {
        "status": "healthy",
        "version": VERSION,
        "engine": "silero",
        "app_env": APP_ENV,
        "default": {"model": d["model"], "voice": d["voice"], "sample_rate": d["sample_rate"]},
        "models": {m: {"enabled": st.model_enabled(m), "downloaded": svc.engine.is_downloaded(m),
                       "loaded": svc.engine.is_loaded(m)} for m in MODELS},
        "model_loaded": svc.engine.is_loaded(d["model"]),
        "auth_required": bool(request.app.state.auth_required()),
        "uptime_s": round(time.time() - svc.started, 1),
        "stats": svc.stats,
        "rss_mb": round(proc.memory_info().rss / 2**20),
    }
