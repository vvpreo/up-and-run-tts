import io

import soundfile as sf


def test_health(api):
    d = api.get("/health").json()
    assert d["status"] == "healthy" and d["engine"] == "silero"


def test_models_and_voices(api):
    ids = [m["id"] for m in api.get("/v1/models").json()["data"]]
    assert "tts-1" in ids and "v5_cis_base" in ids
    v = api.get("/v1/voices").json()
    assert any(x["id"].startswith("ru_") for x in v["voices"])


def test_speech_wav_streamed(api):
    r = api.post("/v1/audio/speech", json={"model": "tts-1", "input": "Привет! Это проверка. Цена 5 руб.",
                                            "response_format": "wav"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    assert int(r.headers["x-fragments"]) == 3
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and len(data) / sr > 2


def test_speech_formats(api):
    for fmt in ("pcm", "mp3", "opus", "flac"):
        r = api.post("/v1/audio/speech", json={"input": "Проверка формата.", "response_format": fmt, "sample_rate": 48000})
        assert r.status_code == 200, fmt
        assert len(r.content) > 1000


def test_ssml_passthrough(api):
    r = api.post("/v1/audio/speech", json={"input": "<speak>Пауза <break time=\"300ms\"/> и дальше.</speak>",
                                            "response_format": "pcm"})
    assert r.status_code == 200 and int(r.headers["x-fragments"]) == 1


def test_errors(api):
    assert api.post("/v1/audio/speech", json={"input": "x", "voice": "nope"}).status_code == 400
    assert api.post("/v1/audio/speech", json={"input": "x", "model": "nope"}).status_code == 400
    assert api.post("/v1/audio/speech", json={"input": "   "}).status_code == 400


def test_auth_required(api):
    import httpx

    r = httpx.post(f"{api.base_url}/v1/audio/speech", json={"input": "x"})
    assert r.status_code == 401


def test_admin_prepare_and_dictionary(admin):
    saved = admin.get("/admin/api/dictionary").json()
    try:
        admin.put("/admin/api/dictionary", json={"values": {**saved, "тестслово": "тестсл+ово"}})
        r = admin.post("/admin/api/prepare", json={"text": "Тестслово 12.05.2026"}).json()
        assert "тестсл+ово" in r["fragments"][0]["prepared"]
        assert "двенадцатое мая" in r["fragments"][0]["prepared"].replace("+", "")
    finally:
        admin.put("/admin/api/dictionary", json={"values": saved})


def test_request_dictionary_overrides(admin):
    r = admin.post("/admin/api/prepare", json={"text": "Тестслово", "dictionary": {"тестслово": "перекр+ыто"}}).json()
    assert "перекр+ыто" in r["fragments"][0]["prepared"]


def test_voices_have_gender(api):
    v = api.get("/v1/voices").json()["voices"]
    assert all(x["gender"] in ("male", "female", "") for x in v)
    assert sum(x["gender"] == "male" for x in v) >= 3 and sum(x["gender"] == "female" for x in v) >= 3
