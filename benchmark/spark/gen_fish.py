"""Fish Audio S2 Pro: cloning through the upstream text2semantic + DAC codec pipeline."""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def _sf_load(path, *a, **k):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T.copy()), sr


torchaudio.load = _sf_load
sys.path.insert(0, "/proj/.src/fish-speech")

from huggingface_hub import snapshot_download  # noqa: E402

import fish_speech.models.text2semantic.inference as fi  # noqa: E402
from common import REFS, TEXTS, Recorder, Timer, cap_gpu_memory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".tmp/out/fish-s2-pro")
args = ap.parse_args()
cap_gpu_memory()
rec = Recorder(args.out, "fish-s2-pro")
ckpt = Path(snapshot_download("fishaudio/s2-pro", local_dir="/proj/.models/s2-pro"))
device, precision = "cuda", torch.bfloat16
model, decode_one_token = fi.init_model(ckpt, device, precision, compile=False)
with torch.device(device):
    model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, dtype=next(model.parameters()).dtype)
codec = fi.load_codec_model(ckpt / "codec.pth", device, precision)


def synth(text, r, tokens):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    codes = []
    for resp in fi.generate_long(model=model, device=device, decode_one_token=decode_one_token, text=text, num_samples=1,
                                 top_p=0.9, top_k=30, temperature=1.0, iterative_prompt=True, chunk_length=300,
                                 prompt_text=[r["text"]], prompt_tokens=[tokens]):
        if resp.action == "sample":
            codes.append(resp.codes)
    audio = fi.decode_to_audio(torch.cat(codes, dim=1).to(device), codec)
    return audio.cpu().float().numpy()


for voice, r in REFS.items():
    tokens = fi.encode_audio(r["wav"], codec, device).cpu()
    synth(TEXTS[0]["text"], r, tokens)  # warm-up
    for t in TEXTS:
        if rec.done(f"clone-{voice}", t["id"]):
            continue
        with Timer() as tm:
            wav = synth(t["text"], r, tokens)
        rec.save(f"clone-{voice}", t["id"], wav, codec.sample_rate, tm.s)
rec.flush()
