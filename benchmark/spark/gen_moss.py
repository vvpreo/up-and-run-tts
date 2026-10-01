"""MOSS-TTS-Realtime: native incremental text input (push_text). Built on the upstream streaming example."""
import argparse
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/proj/.src/MOSS-TTS/moss_tts_realtime")
import example_llm_stream_to_tts as ex  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import REFS, TEXTS, Recorder, cap_gpu_memory, sync  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/moss-realtime")
ap.add_argument("--attn", default="sdpa")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "moss-tts-realtime")
SR = 24000
MAX_AUDIO_S = 60
MODEL, CODEC = "OpenMOSS-Team/MOSS-TTS-Realtime", "OpenMOSS-Team/MOSS-Audio-Tokenizer"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
processor = ex.MossTTSRealtimeProcessor(tokenizer)
model = ex.MossTTSRealtime.from_pretrained(MODEL, attn_implementation=args.attn, torch_dtype=torch.bfloat16).to(device).eval()
codec = ex._load_codec(CODEC, device)


def prompt_tokens_for(path):
    wav, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(wav).reshape(1, -1)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    with torch.inference_mode():
        res = codec.encode(wav.unsqueeze(0).to(device), chunk_duration=0.24)
    return ex._extract_codes(res).cpu().numpy().squeeze(1)


def deltas(text, chars, delay):
    """Simulated LLM stream: `chars` characters every `delay` seconds."""
    for i in range(0, len(text), chars):
        if delay and i:
            time.sleep(delay)
        yield text[i:i + chars]


def synth(prompt_tokens, text, chars, delay):
    inferencer = ex.MossTTSRealtimeInference(model, tokenizer, max_length=1200)
    inferencer.reset_generation_state(keep_cache=False)
    session = ex.MossTTSRealtimeStreamingSession(
        inferencer, processor, codec=codec, codec_sample_rate=SR, codec_encode_kwargs={"chunk_duration": 0.24},
        prefill_text_len=processor.delay_tokens_len, temperature=0.8, top_p=0.6, top_k=30, do_sample=True,
        repetition_penalty=1.1, repetition_window=50)
    session.set_voice_prompt_tokens(prompt_tokens)
    system_prompt = processor.make_ensemble(prompt_tokens)
    ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    prefix = np.full((len(ids), system_prompt.shape[1]), fill_value=processor.audio_channel_pad, dtype=np.int64)
    prefix[:, 0] = ids
    session.reset_turn(input_ids=np.concatenate([system_prompt, prefix], axis=0), include_system_prompt=False, reset_cache=True)
    decoder = ex.AudioStreamDecoder(codec, chunk_frames=3, overlap_frames=0, decode_kwargs={"chunk_duration": -1}, device=device)
    chunks, ttfa, total, runaway = [], None, 0, False
    sync()
    t0 = time.perf_counter()
    import contextlib, io
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        for c in ex.run_streaming_tts(session=session, codec=codec, decoder=decoder, text_deltas=deltas(text, chars, delay)):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(c.astype(np.float32).reshape(-1))
            total += len(chunks[-1])
            if total > MAX_AUDIO_S * SR:  # runaway generation (seen on digits: 327 s of audio for one sentence)
                runaway = True
                break
    sync()
    return np.concatenate(chunks), time.perf_counter() - t0, ttfa, runaway


torch.manual_seed(1234)
for voice, r in REFS.items():
    pt = prompt_tokens_for(r["wav"])
    synth(pt, TEXTS[0]["text"], 10**6, 0)  # warm-up
    for t in TEXTS:
        # whole phrase known up front
        if not rec.done(f"full-{voice}", t["id"]):
            wav, s, ttfa, runaway = synth(pt, t["text"], 10**6, 0)
            rec.save(f"full-{voice}", t["id"], wav, SR, s, ttfa_s=round(ttfa, 3), runaway=runaway)
        # text arrives gradually: 4 characters every 25 ms (~160 chars/s, a fast local LLM)
        if not rec.done(f"stream-{voice}", t["id"]):
            wav, s, ttfa, runaway = synth(pt, t["text"], 4, 0.025)
            rec.save(f"stream-{voice}", t["id"], wav, SR, s, ttfa_s=round(ttfa, 3), runaway=runaway,
                     text_feed_s=round(len(t["text"]) / 4 * 0.025, 2))
rec.flush()
