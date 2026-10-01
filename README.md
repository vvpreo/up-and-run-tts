# up-and-run-tts — local Russian TTS service (Silero, CPU) with an OpenAI-compatible API

A self-contained Docker service that turns Russian text into speech with
**Silero TTS v5** models on a plain CPU, exposes an **OpenAI-compatible API**
(`POST /v1/audio/speech`), streams audio sentence by sentence, accepts text
arriving in pieces from an LLM over a WebSocket, and has an **admin page** for
everything Silero lets you control.

Sister project: [`up-and-run-stt`](https://github.com/vvpreo/up-and-run-stt)
(speech-to-text, same conventions).

> Model choice and measurements behind this project live in
> [`docs/research/`](docs/research/README.md). Silero was chosen because it runs
> 17× faster than real time on a laptop CPU and needs no GPU; GPU models were
> compared on a DGX Spark in [`docs/research/spark-quality-benchmark.md`](docs/research/spark-quality-benchmark.md).

---

## Quick start

```bash
cp .env.example .env        # set AUTH_TOKEN and ADMIN_PASSWORD
docker compose up -d --build
docker compose logs -f
curl -s http://localhost:9008/health
```

Or straight from Docker Hub, without cloning:

```bash
docker run -d --name up-and-run-tts -p 9008:9008 \
  -v up-and-run-tts-data:/app/data \
  -e AUTH_TOKEN=change-me -e ADMIN_PASSWORD=change-me-too \
  --restart unless-stopped vvpreo/up-and-run-tts:latest
```

First start downloads the default model (`v5_cis_base`, ~90 MB) into the named
volume `up-and-run-tts-data`; later starts are instant. Then:

```bash
curl -s http://localhost:9008/v1/audio/speech \
  -H "Authorization: Bearer change-me" -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Привет! Встреча назначена на 12.05.2026 в 14:30.","response_format":"wav"}' \
  -o out.wav
```

Admin page: `http://localhost:9008/admin` (basic auth, see `ADMIN_PASSWORD`).

Secrets live in `.env` next to `docker-compose.yml` (git-ignored). With an empty
`AUTH_TOKEN` and no keys created in the admin page the API is open; with an
empty `ADMIN_PASSWORD` the admin page is disabled.

## Models and licences

| Model | Voices | Stress marks | Licence | Enabled by default |
|---|---|---|---|---|
| `v5_cis_base` | 29 Russian-speaking voices (`ru_*`, speakers of CIS languages — an accent is audible) | external, `silero-stress` (MIT) | **MIT** | yes |
| `v5_5_ru`, `v5_4_ru`, `v5_3_ru`, `v4_ru` | `aidar`, `baya`, `kseniya`, `xenia`, `eugene` | built in (plus homographs and question intonation in v5.4+) | CC BY-NC 4.0 — **non-commercial** | no — switch on in the admin page |

Weights are downloaded from `models.silero.ai` on first use and kept in the
volume. Silero models are closed `torch.package` archives: there is no voice
cloning and no fine-tuning; "managing voices" means choosing among the shipped
ones and shaping them with rate, pitch, pauses and stress marks.

## What the service does to the text

Silero silently skips digits and Latin letters, so the service prepares the
text before synthesis — the same pipeline for a whole request and for text
arriving in pieces:

1. **Markdown removal** — code blocks, links, emoji, headings, list bullets, tables.
2. **Splitting** into sentences, and into clauses when a fragment is long enough
   (`min_fragment_chars`). Punctuation inside dates, times and decimals is not a
   boundary; common Russian abbreviations (`т.е.`, `см.`, `руб.`) do not end a sentence.
3. **Pronunciation dictionary** — your words first (admin page), then a built-in
   list of tech terms (`API`, `GitHub`, `Python`, `Docker`, …). Values may carry
   stress marks: `+` before the stressed vowel (`мука → мук+а`).
4. **Numbers to words** — integers, decimals, dates (`12.05.2026`), times
   (`14:30`), years (`в 2024 году`), ordinals (`3-я`), percents, money
   (`руб.`, `₽`, `$`, `€`, `тыс.`/`млн`/`млрд`), common units (`кг`, `км`, `мин`, `ГБ`…)
   with case agreement of the unit.
5. **Latin to Cyrillic** by rough English reading rules for anything the
   dictionary did not cover; short all-caps words are spelled letter by letter.
6. **Stress and «ё»** — `silero-stress` for `v5_cis_base`; the `v5_*_ru` models
   do it themselves. Marks you put in the text (`+`) are always respected.
7. **SSML wrapping** when rate or pitch differ from `medium`.

Input that starts with `<speak>` is passed to Silero as SSML untouched (Silero
supports `<break>`, `<prosody rate pitch>`, `<p>`, `<s>`). In `v5_5_ru`
`*слово` puts logical emphasis on a word.

Every step can be switched off in the admin page; the "Тест" tab shows
what the text turned into before it is spoken.

## API

### `POST /v1/audio/speech`

OpenAI-compatible. Fields:

| Field | Meaning |
|---|---|
| `input` | text or SSML (`<speak>…</speak>`); up to `MAX_INPUT_CHARS` |
| `model` | `tts-1` (alias of the default) or a Silero model id (`v5_cis_base`, `v5_5_ru`, …) |
| `voice` | a voice of that model (`GET /v1/voices`); empty or an OpenAI name → default voice |
| `response_format` | `wav` (default for streaming clients: send it explicitly, OpenAI's default is `mp3`), `pcm` (16-bit mono), `mp3`, `opus`, `flac` |
| `speed` | 0.25–4.0, mapped to SSML rate |
| `stream` | extension; default `true` for `wav`/`pcm`/`mp3`: audio is sent sentence by sentence as it is synthesised |
| `sample_rate` | extension; 8000, 24000 (default) or 48000 |
| `rate`, `pitch` | extension; SSML values `x-slow … x-fast`, `x-low … x-high`, or `"80%"` for rate |
| `intensity` | extension; strength of `*word` emphasis, 1–5 (ru models) |
| `dictionary` | extension; pronunciation dictionary for this request (`{"мука": "мук+а"}`), added on top of the service dictionary and overriding it |

Response headers: `X-Sample-Rate`, `X-Fragments`, `X-Model`, `X-Voice`
(plus `X-Audio-Seconds`, `X-Synth-Seconds` for non-streamed responses).

### `WS /v1/audio/speech/stream` — text arriving in pieces

Send JSON messages, receive JSON and binary PCM16 frames:

```
→ {"type":"session.config","model":"tts-1","voice":"ru_zhadyra","sample_rate":24000,"dictionary":{...}}   (optional)
← {"type":"session.ready", ...}
→ {"type":"input.text","text":"Чтобы сварить яйцо всмятку, "}      (any number of deltas)
← {"type":"audio.start","generation":0,"index":0,"text":"Чтобы сварить яйцо всмятку,","prepared":"Чт+обы ..."}
← <binary PCM16 LE mono frames>
← {"type":"audio.done","generation":0,"index":0,"seconds":2.05,"synth_s":0.3}
→ {"type":"input.done"}                → ← {"type":"done","generation":0,"fragments":4}
→ {"type":"input.cancel"}              → ← {"type":"cancelled","generation":1}   (barge-in: queue dropped)
```

A fragment is synthesised as soon as its sentence or clause boundary is
confirmed; with an LLM producing ~160 characters per second the first audio
arrives about half a second after the first word.

### Other endpoints

- `GET /v1/voices[?model=…]` — enabled voices of the enabled models with gender, licence flags and the default.
- `GET /v1/models` — enabled models (`tts-1` alias first).
- `GET /health` — status, loaded models, counters, RSS.
- `GET /docs` — Swagger UI (`ENABLE_DOCS`).

Authorization: `Authorization: Bearer <token>` where the token is `AUTH_TOKEN`
from the environment or a key created in the admin page. With neither set,
the API is open.

## Admin page `/admin`

Basic auth with `ADMIN_USER` / `ADMIN_PASSWORD` (empty password = admin disabled).

- **Настройки** — default model, voice, sample rate, rate, pitch, emphasis
  strength, «ё»; text processing switches; minimum fragment length.
- **Голоса и модели** — models on/off (non-commercial ones ask for confirmation),
  download/load/unload; voices of each model grouped by gender (detected from
  pitch once and cached), each with a play button and an on/off switch —
  switched-off voices disappear from `/v1/voices`.
- **Словарь** — pronunciation dictionary (word → how to say it).
- **Тест** — synthesise any text with any parameters and see the prepared text.
- **Ключи** — create and revoke API keys (the key is shown once).

Settings are stored in `settings.json` on the data volume and apply immediately.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `9008` / `0.0.0.0` | listen address inside the container |
| `DATA_DIR` | `/app/data` | weights, `settings.json` |
| `AUTH_TOKEN` | empty | Bearer token for the API |
| `ADMIN_USER` / `ADMIN_PASSWORD` | `admin` / empty | admin page credentials; empty password disables it |
| `PRELOAD_MODELS` | empty | models to load at start besides the default one |
| `TORCH_THREADS` | `4` | CPU threads for synthesis |
| `MAX_INPUT_CHARS` | `20000` | request text limit (413 above it) |
| `MAX_PENDING_REQUESTS` | `8` | concurrent syntheses (429 above it) |
| `CORS_ORIGINS` | empty | comma-separated origins, `*` for all |
| `ENABLE_DOCS` | `true` | Swagger UI |
| `LOG_LEVEL` | `INFO` | |
| `APP_ENV` | `prod` | `dev` / `test` / `uat` put a coloured frame on the app icon |
| `SILERO_BASE_URL` | `https://models.silero.ai/models/tts/ru` | mirror for weights |

## Performance

Measured on an Intel i7-8750H (laptop, 4 threads): synthesis takes about 6 % of
the audio duration — a 4-second sentence in 0.2–0.4 s. Time to first audio for
a streamed request is the synthesis of its first fragment, typically under
half a second. RSS with two models loaded is about 1 GB.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install uv
.venv/bin/uv sync --all-groups
DATA_DIR=.tmp/data ADMIN_PASSWORD=x AUTH_TOKEN=x .venv/bin/python main.py
.venv/bin/python -m pytest          # unit tests + integration tests against a running service
```

Layout: `src/engines/` (engine interface + Silero), `src/text/` (the pipeline),
`src/audio/` (encoding), `src/routes/` (OpenAI API, WebSocket, admin, health),
`src/static/admin/` (one page, no build step), `benchmark/` (the model
comparison harness), `docs/` (research, plans, Spark notes).

## Licence

Code: MIT. Model weights follow their own licences (see the table above).
