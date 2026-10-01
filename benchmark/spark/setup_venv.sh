#!/usr/bin/env bash
# Create a per-model venv inside the bench container: setup_venv.sh <qwen|f5|moss|cosy>
set -euo pipefail
M=${1:?model}
V=/proj/.venvs/$M
[ -d "$V" ] || python -m venv --system-site-packages "$V"
source "$V/bin/activate"
mkdir -p /proj/.src && cd /proj/.src
case "$M" in
  qwen)
    pip install "qwen-tts==0.1.1" "huggingface-hub<1.0" ;;
  f5)
    pip install f5-tts ruaccent ;;
  moss)
    [ -d MOSS-TTS ] || git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git
    pip install -e MOSS-TTS ;;
  cosy)
    [ -d CosyVoice ] || git clone --depth 1 --recursive --shallow-submodules https://github.com/FunAudioLLM/CosyVoice.git
    pip install conformer diffusers hydra-core HyperPyYAML inflect librosa lightning modelscope omegaconf onnx \
        onnxruntime openai-whisper pyworld transformers x-transformers wetext pyarrow pydantic rich gdown wget \
        matplotlib ;;
  voxcpm)
    pip install voxcpm ;;
  omni)
    pip install omnivoice ;;
  chatter)
    # upstream pins torch/numpy for py3.11; keep our validated torch and let the rest resolve
    [ -d chatterbox ] || git clone --depth 1 https://github.com/resemble-ai/chatterbox.git
    pip install --no-deps -e chatterbox
    pip install librosa s3tokenizer transformers diffusers resemble-perth conformer safetensors spacy-pkuseg pykakasi pyloudnorm omegaconf einops ;;
  fish)
    [ -d fish-speech ] || git clone --depth 1 https://github.com/fishaudio/fish-speech.git
    # upstream pins torch==2.8.0 (no GB10 build); install without deps and add the rest by hand
    pip install --no-deps -e fish-speech
    pip install "transformers<=4.57.3" lightning hydra-core natsort einops librosa rich loguru loralib pyrootutils resampy \
        "einx[torch]==0.2.2" zstandard pydub ormsgpack tiktoken pydantic cachetools descript-audio-codec safetensors \
        silero-vad opencc-python-reimplemented kui click ;;
esac
python -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())"
pip list 2>/dev/null | grep -i -E "^(torch|torchaudio|transformers|numpy|onnxruntime|f5-tts|qwen-tts|ruaccent|torchcodec) "
echo "SETUP_OK $M"
