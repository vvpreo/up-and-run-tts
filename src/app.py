import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

from src import config
from src.auth import auth_enabled
from src.routes import admin, health, openai, stream
from src.service import TTSService
from src.settings import Settings

log = logging.getLogger(__name__)
ICONS = Path(__file__).resolve().parent / "static" / "icons"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.service = TTSService(app.state.settings)
    app.state.auth_required = lambda: auth_enabled(app.state.settings)
    import asyncio

    await asyncio.to_thread(app.state.service.preload)
    log.info("ready on %s:%s (admin %s)", config.HOST, config.PORT, "on" if config.ADMIN_PASSWORD else "off")
    yield


def create_app() -> FastAPI:
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="up-and-run-tts", version=health.VERSION, lifespan=lifespan,
                  docs_url="/docs" if config.ENABLE_DOCS else None, redoc_url=None)
    if config.CORS_ORIGINS:
        app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"],
                           allow_headers=["*"], expose_headers=["*"])
    app.include_router(health.router)
    app.include_router(openai.router)
    app.include_router(stream.router)
    app.include_router(admin.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/admin" if config.ADMIN_PASSWORD else "/health")

    # Иконка приложения: на непродовых стендах — с цветной рамкой (см. src/icons.py)
    from src.icons import badged_svg, normalize_stand

    stand = normalize_stand(config.APP_ENV)
    icon_headers = {"Cache-Control": "public, max-age=3600"}

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon_svg():
        return Response(badged_svg(stand), media_type="image/svg+xml", headers=icon_headers)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_ico(request: Request):
        ico = ICONS / "generated" / stand / "favicon.ico"
        if ico.exists():
            return Response(ico.read_bytes(), media_type="image/x-icon", headers=icon_headers)
        return RedirectResponse("/favicon.svg")

    return app
