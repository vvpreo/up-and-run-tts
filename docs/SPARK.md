# DGX Spark в этом проекте

Общие правила работы на Spark — в скилле `cody-use-spark` (лок GPU, капы памяти,
восстановление). Здесь — проектная специфика и история граблей.

## Что и как запускается

- Код: `rsync -a --exclude __pycache__ benchmark/ sparky@192.168.1.188:code/up-and-run-tts/benchmark/`.
  На Spark код не правят.
- Базовый образ: `docker build -f benchmark/spark/Dockerfile.base -t up-and-run-tts-bench-base benchmark/spark`
  (сборка на самом Spark, ~3 мин, 6,3 ГБ). `nvidia/cuda:13.0.2-base-ubuntu24.04` +
  `torch==2.12.1`, `torchaudio==2.11.0` с индекса `cu130` — связка, проверенная на GB10.
  Версии закреплены через `PIP_CONSTRAINT`, чтобы зависимости моделей не заменили torch.
- Окружения моделей: venv с `--system-site-packages` в `~/code/up-and-run-tts/.venvs/<модель>`,
  создаются `benchmark/spark/setup_venv.sh <модель>` внутри контейнера. Исходники моделей,
  которых нет в PyPI, — в `.src/`, веса — в `.hf-cache/` и `.models/`.
- Запуск: `benchmark/spark/run.sh <задача> <venv|-> <gpu|cpu> <команда>`. Режим `gpu` берёт общий
  лок `/home/sparky/.cody/gpu.lock`, ставит рядом `memguard.sh`, контейнер —
  `up-and-run-tts-<задача>-bench`, `--memory 40g`. Скрипты генерации ограничивают себя
  `torch.cuda.set_per_process_memory_fraction(0.25)`.
- Результаты: `~/code/up-and-run-tts/.tmp/out/<модель>/` (wav и `metrics.json`), логи — `.tmp/*.log`.
  Файлы из контейнера принадлежат root; на девбокс забираются `rsync`-ом на чтение.

## Грабли

1. **`torchaudio.load` в 2.11 идёт через torchcodec**, которого в окружениях нет. В скриптах
   `torchaudio.load` подменён чтением через `soundfile`.
2. **CosyVoice 3 не работает с transformers 5.x** (падает в `forward_one_step` и следом в
   вокодере на пустом входе). Нужен `transformers==4.51.3`, как в их `requirements.txt`.
   Их закрепления torch 2.3.1 и onnxruntime-gpu не нужны: работает на torch 2.12.1 и CPU-сборке
   onnxruntime.
3. **MOSS-TTS: `pip install -e .` не ставит transformers** — он в необязательной группе
   `torch-runtime` вместе с torch cu128. Ставить вручную `transformers==5.0.0 accelerate`.
4. **fish-speech закрепляет `torch==2.8.0`**, сборки под GB10 для него нет. Ставится
   `pip install --no-deps -e fish-speech` и список зависимостей руками.
5. **Chatterbox: необязательный расстановщик ударений `russian-text-stresser` не собирается**
   (тянет spacy 3.6, который не компилируется под Python 3.12). Модель проверена без него.
6. **MOSS-TTS-Realtime уходит в бесконечную генерацию на цифрах**: фраза с датой и суммой дала
   327 секунд звука. В `gen_moss.py` стоит отсечка 60 секунд.
7. **`MemAvailable` хоста как мера памяти модели не годится**: сосед (ollama) в это же время
   грузит модели по 12–32 ГБ. В `metrics.json` пишутся пик `torch.cuda.max_memory_reserved` и
   RSS процесса.
8. **`__pycache__` из контейнера принадлежит root** и ломает следующий `rsync` — исключать.
9. **`ssh host 'cd dir && setsid nohup cmd > log &'` не возвращает управление**: фоновая
   подоболочка держит stdout ssh. Оборачивать в скобки: `(setsid nohup cmd > log 2>&1 < /dev/null &)`.
10. Первый вызов MOSS и VoxCPM2 компилирует граф (`torch.compile`): 3–6 минут при 100 % одного
    ядра и 0 % GPU — это не зависание.
