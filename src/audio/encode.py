"""Кодирование PCM в форматы ответа. wav и pcm отдаются потоково (по фрагментам),
mp3 / opus / flac — одним файлом через libsndfile."""

import io
import struct

import numpy as np
import soundfile as sf

FORMATS = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
}
STREAMABLE = {"wav", "pcm"}


def to_int16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_header(sample_rate: int, data_bytes: int | None = None) -> bytes:
    """RIFF-заголовок; без известной длины — «бесконечный» поток (0xFFFFFFFF),
    который понимают браузеры и ffmpeg."""
    size = 0xFFFFFFFF if data_bytes is None else data_bytes
    riff = 0xFFFFFFFF if data_bytes is None else 36 + data_bytes
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, 1, sample_rate,
                       sample_rate * 2, 2, 16, b"data", size)


def encode_file(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return to_int16(audio)
    if fmt == "wav":
        pcm = to_int16(audio)
        return wav_header(sample_rate, len(pcm)) + pcm
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, audio, sample_rate, format="MP3")
    elif fmt == "opus":
        sf.write(buf, audio, sample_rate, format="OGG", subtype="OPUS")
    elif fmt == "flac":
        sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    else:
        raise ValueError(f"unsupported format {fmt!r}")
    return buf.getvalue()
