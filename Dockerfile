# up-and-run-tts — русский TTS на Silero (CPU) с OpenAI-совместимым API и админкой.
# Веса моделей в образ не запекаются: скачиваются при первом обращении в том /app/data.
# Сборка многостадийная: зависимости ставит uv, в рантайм переезжает только venv.
# Собирается под linux/amd64 и linux/arm64.

# ============================ builder =================================
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: ровно версии из лока; --no-dev: без pytest; группа bench не нужна
RUN uv sync --frozen --no-dev --no-group bench --no-install-project

# Иконки всех стендов из прод-исходника (рендерер SVG нужен только здесь)
COPY scripts/gen_icons.py scripts/gen_icons.py
COPY src/icons.py src/icons.py
COPY src/static/icons/favicon.svg src/static/icons/favicon.svg
RUN uv run --no-project --with resvg-py==0.5.0 python scripts/gen_icons.py --out /app/icons

# ============================ runtime =================================
FROM python:3.12-slim-bookworm

ARG VERSION=dev
ARG REVISION=unknown
ARG CREATED=""
LABEL org.opencontainers.image.title="up-and-run-tts" \
      org.opencontainers.image.description="Local Russian text-to-speech service (Silero, CPU-only) with an OpenAI-compatible API and an admin page" \
      org.opencontainers.image.source="https://github.com/vvpreo/up-and-run-tts" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data PATH="/app/.venv/bin:$PATH"

# Непривилегированный пользователь создаётся до COPY, владелец ставится в COPY --chown
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app main.py /app/
COPY --from=builder --chown=app:app /app/icons /app/src/static/icons/generated
RUN mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 9008
# Первый старт скачивает модель (90–150 МБ) и грузит её — щедрый start-period
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9008/health')" || exit 1
CMD ["python", "main.py"]
