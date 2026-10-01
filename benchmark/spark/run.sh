#!/usr/bin/env bash
# Run a command inside the benchmark base image on DGX Spark.
#   run.sh <job> <venv|-> <gpu|cpu> <command...>
# gpu jobs take the machine-wide GPU lock and get a memory guard next to them.
set -euo pipefail
JOB=${1:?job name}; VENV=${2:?venv name or -}; MODE=${3:?gpu|cpu}; shift 3
PROJ=/home/sparky/code/up-and-run-tts
NAME=up-and-run-tts-${JOB}-bench
LOCK=/home/sparky/.cody/gpu.lock
ACT=""; [ "$VENV" != "-" ] && ACT="source /proj/.venvs/${VENV}/bin/activate && "
mkdir -p "$PROJ/.tmp" "$PROJ/.hf-cache" "$PROJ/.venvs"
ARGS=(--rm --name "$NAME" --ipc=host --memory 40g --memory-swap 40g
      -v "$PROJ:/proj" -w /proj
      -e HF_HOME=/proj/.hf-cache -e PYTHONUNBUFFERED=1 -e PIP_CACHE_DIR=/proj/.pip-cache -e PIP_NO_CACHE_DIR=0
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TTS_MEM_FRACTION="${TTS_MEM_FRACTION:-0.25}")
if [ "$MODE" = gpu ]; then
  ( for _ in $(seq 1800); do docker ps -q --filter "name=$NAME" | grep -q . && break; sleep 2; done; bash /home/sparky/code/vvpreo-jindr/ml/train/memguard.sh 12 "name=$NAME" \
      >> "$PROJ/.tmp/memguard_${JOB}.log" 2>&1 ) &
  exec flock -w 3600 "$LOCK" docker run "${ARGS[@]}" --gpus all up-and-run-tts-bench-base bash -c "${ACT}$*"
else
  exec docker run "${ARGS[@]}" up-and-run-tts-bench-base bash -c "${ACT}$*"
fi
