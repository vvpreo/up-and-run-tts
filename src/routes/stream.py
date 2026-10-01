"""WebSocket /v1/audio/speech/stream — текст приходит кусками (например, дельты
ответа LLM), звук уходит по фрагментам, как только граница фразы подтверждена.

Сообщения клиента (JSON):
  {"type": "session.config", "model", "voice", "sample_rate", "speed", "rate", "pitch", "dictionary"}  — необязательно
  {"type": "input.text", "text": "..."}        — очередная дельта
  {"type": "input.done"}                        — текст закончился, дослать остаток
  {"type": "input.cancel"}                      — перебили: бросить очередь, начать заново
Сообщения сервера:
  {"type": "session.ready", "model", "voice", "sample_rate"}
  {"type": "audio.start", "generation", "index", "text"}  → бинарные кадры PCM16 LE моно → {"type": "audio.done", ...}
  {"type": "done", "generation", "fragments"}   — после input.done всё отдано
  {"type": "error", "message"}
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.audio.encode import to_int16
from src.auth import check_bearer
from src.service import RequestError
from src.text.markdown import strip_markdown
from src.text.splitter import IncrementalSplitter

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(ws: WebSocket):
    app = ws.app
    svc, settings = app.state.service, app.state.settings
    try:
        check_bearer(ws, settings)
    except Exception as e:  # noqa: BLE001
        await ws.accept()
        await ws.send_json({"type": "error", "message": getattr(e, "detail", str(e))})
        await ws.close(code=4401)
        return
    await ws.accept()

    state = {"gen": 0, "req": None, "splitter": None, "queue": asyncio.Queue(), "done": False}

    def configure(cfg: dict):
        state["req"] = svc.resolve(cfg.get("model"), cfg.get("voice"), cfg.get("sample_rate"), cfg.get("speed"),
                                   cfg.get("rate"), cfg.get("pitch"), cfg.get("intensity"), cfg.get("dictionary"))
        text_cfg = settings.section("text")
        state["splitter"] = IncrementalSplitter(int(text_cfg.get("min_fragment_chars", 25)))

    async def worker():
        index = 0
        while True:
            gen, item = await state["queue"].get()
            if gen != state["gen"]:
                continue  # отменённое поколение
            if item is None:
                await ws.send_json({"type": "done", "generation": gen, "fragments": index})
                index = 0
                continue
            req = state["req"]
            if settings.section("text").get("strip_markdown", True):
                item = strip_markdown(item) or item
            try:
                prepared = req.pipeline.prepare_fragment(item)
                res = await svc.synth_one_async(req, prepared)
            except RequestError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            if gen != state["gen"]:
                continue
            await ws.send_json({"type": "audio.start", "generation": gen, "index": index, "text": item,
                               "prepared": prepared.shown})
            pcm = to_int16(res.audio)
            for i in range(0, len(pcm), 48000):
                await ws.send_bytes(pcm[i:i + 48000])
            await ws.send_json({"type": "audio.done", "generation": gen, "index": index,
                               "seconds": round(len(res.audio) / res.sample_rate, 3), "synth_s": round(res.synth_s, 3)})
            index += 1

    task = asyncio.create_task(worker())
    try:
        configure({})
        await ws.send_json({"type": "session.ready", "model": state["req"].model, "voice": state["req"].voice,
                           "sample_rate": state["req"].sample_rate})
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "expected JSON"})
                continue
            t = msg.get("type")
            try:
                if t == "session.config":
                    configure(msg)
                    await ws.send_json({"type": "session.ready", "model": state["req"].model,
                                       "voice": state["req"].voice, "sample_rate": state["req"].sample_rate})
                elif t == "input.text":
                    # Разметка чистится по фрагментам в worker: дельта может резать
                    # маркер пополам, а strip() склеил бы слова на стыке дельт.
                    for frag in state["splitter"].push(msg.get("text", "")):
                        await state["queue"].put((state["gen"], frag))
                elif t == "input.done":
                    for frag in state["splitter"].flush():
                        await state["queue"].put((state["gen"], frag))
                    await state["queue"].put((state["gen"], None))
                elif t == "input.cancel":
                    state["gen"] += 1
                    state["splitter"].buf = ""
                    while not state["queue"].empty():
                        state["queue"].get_nowait()
                    await ws.send_json({"type": "cancelled", "generation": state["gen"]})
                else:
                    await ws.send_json({"type": "error", "message": f"unknown message type {t!r}"})
            except RequestError as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
