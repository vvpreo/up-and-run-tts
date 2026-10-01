# GPU TTS models with native incremental (streaming) TEXT input — state as of 2026-09-30

Scope: open-weights, self-hosted, GPU-class TTS where the model itself consumes text incrementally (LLM token stream) and emits audio before the sentence is complete. Target language: Russian. Target machines: (a) GTX 1050 Ti 4 GB (Pascal, x86_64), (b) DGX Spark (GB10, aarch64, CUDA 13, 121 GB unified memory).

Reading conventions used below:
- "VERIFIED" = stated on a primary source I fetched (repo README, model card, paper, issue). "COMMUNITY" = blog/forum/Habr/vc.ru claim, not independently verified. "SNIPPET" = seen only in a search-result summary, page not opened — treat as weak.
- All page fetches were done on 2026-09-30. Fetched pages were summarised by a small model, so exact wording of quotes may be slightly off; numbers were taken as reported.
- RTF convention differs between sources: Microsoft/Kyutai/MOSS/vllm-omni use RTF = compute time / audio time (lower is better, <1 = real time); faster-qwen3-tts uses the INVERSE ("RTF > 1.0 = faster than real-time"). Marked per item.

Quick classification (details and sources in the sections below):

| Model | Native streaming TEXT input? | Exposed in official code? | Russian official? |
|---|---|---|---|
| CosyVoice 2 (0.5B) | Yes (interleaved text/speech tokens, "bistream") | Yes — Python generator as `tts_text` in `inference_zero_shot`; no official network protocol for it | No (zh/en/ja/ko + dialects) — see Gaps |
| Fun-CosyVoice 3 (0.5B-2512) | Yes (claimed "text-in streaming and audio-out streaming") | Claimed in README/model card; the generator example in `example.py` is shown for CosyVoice2 only | YES (one of 9 languages) |
| Kyutai TTS 1.6B (DSM) | Yes (delayed streams, word-level) | Yes — Rust websocket server (moshi-server), `tts_pytorch_streaming.py`; used in Unmute | No (en/fr only) |
| Kyutai Unmute | It is a system (STT + LLM + Kyutai TTS), not a model | Yes, docker compose | No (en/fr) |
| Kyutai Pocket TTS (100M) | CPU model — out of scope (does not require GPU) | — | No (en, fr, de, es, pt, it [+nl per kyutai.org]) |
| VibeVoice-Realtime-0.5B | Yes by design (interleaved windowed text encoding) | Partially: websocket demo exists, but the doc still carries a TODO "Implement streaming text input function to feed new tokens while audio is still being generated" | No (en + experimental de/fr/it/ja/ko/nl/pl/pt/es) |
| VibeVoice-1.5B / Large(7B) | No (long-form full-script TTS); official TTS code removed | — | No (en/zh) |
| Qwen3-TTS (0.6B/1.7B, 12Hz) | Yes in architecture/paper (Dual-Track); in open-source code only as "feed full text progressively" — NO official incremental-text API | No (issue closed "not planned"; vllm-omni RFC for token-level text streaming opened Aug 2026) | YES (one of 10 languages) |
| MOSS-TTS-Realtime (1.7B, OpenMOSS) | Yes (hierarchical text/speech inputs, `push_text()` with LLM deltas) | Yes — official streaming API + Gradio/FastAPI demo; vLLM-Omni support | YES (one of 20 languages) |
| Dia2 (1B/2B, Nari Labs) | Yes (Kyutai-style, starts from the first words) | CLI/API; "real streaming" server listed as upcoming | No (English only) |
| Voxtral TTS (Mistral, 4B) | NO text-streaming (audio-out streaming only) | — | No (9 languages, no Russian); CC BY-NC |
| Sesame CSM-1B | NO native text-streaming (utterance-level; context-conditioned) | — | No |
| Orpheus TTS (3B) | NO (full prompt in, audio tokens streamed out) | — | No official; community RU fine-tune exists |
| FlashTTS (arXiv 2606.09141) | Yes per paper ("lagged multi-track streaming") | Code/weights promised, no release located | Unknown |
| Qwen-Audio-3.0-TTS (Alibaba) | n/a — cloud API only, no weights | — | yes at API level |

---

## Q1. Russian: official support, reported quality, community fine-tunes

### Takeaway
Only three candidates with native text-streaming architecture officially list Russian: Fun-CosyVoice3-0.5B-2512, Qwen3-TTS, and MOSS-TTS-Realtime. For both CosyVoice 3 and Qwen3-TTS the dominant community complaint is the same: lexical stress (ударения) is wrong often enough to matter and neither model was trained with stress marks; Qwen3-TTS additionally gets "Asian accent"/weak intonation and number-reading complaints. Kyutai TTS, VibeVoice-Realtime, Dia2, Voxtral, CSM and Orpheus have no official Russian; I found no Russian fine-tune of any of the truly text-streaming non-Russian models.

### Cited Findings

**Fun-CosyVoice 3**
- VERIFIED: README lists "9 common languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)" plus 18+ Chinese dialects. — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice) (repo now resolves under `QwenAudio/CosyVoice`; older links use `FunAudioLLM/CosyVoice`), [HF model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- VERIFIED: the published benchmark table on the model card covers only test-zh (CER 1.21%), test-en (WER 2.24%) and test-hard (CER 6.71%); the RL variant `Fun-CosyVoice3-0.5B-2512_RL` gets 0.81 / 1.68 / 5.44. No Russian metric is reported on the card. — [HF model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- COMMUNITY (vc.ru, author "xVibeNot", dated 8 January [2026 by context]): the model accepts stress as Unicode `́` or as `+` before the stressed vowel (auto-converted to acute after the vowel); "модель явно не обучалась с ударениями на русском языке, а генерализировала их по примерам из других языков"; "если расставлять ударения прям в каждом слове, то становится хуже, чем без ударений совсем"; recommendation: "рекомендую ставить ударения только там, где модель сама не справляется". — [vc.ru: CosyVoice TTS ударения](https://vc.ru/ai/2681383-cosyvoice-tts-podderzhka-udareniy-v-russkom-yazyke)
- COMMUNITY (vc.ru, same author, 2025-12-17): overview of CosyVoice3 "с поддержкой русского языка" focusing on prompt-based style control and special tokens (cough, laugh, sighs); no measured quality/speed — "Что касается скорости и реалтайма - будет отдельный пост." — [vc.ru: обзор CosyVoice3](https://vc.ru/ai/2653802-obzor-cosyvoice3-novaya-tts-s-podderzhkoy-russkogo-yazyka)
- COMMUNITY: a Russian YouTube review is titled "CosyVoice 3.0 - ИМХО лучшая TTS-модель для ..." (title only; content not reviewed). — [YouTube](https://www.youtube.com/watch?v=1BMN-sm1o54)
- SNIPPET: one English guide says Russian "currently requires debugging" while en/zh are "particularly impressive" (source among search results not identified precisely; likely the stable-learn guide). — [stable-learn CosyVoice3 guide](https://stable-learn.com/en/cosyvoice3-tech-guide/)

**CosyVoice 2**
- I did not fetch an explicit CosyVoice2 language list. The CosyVoice 2 paper describes a multilingual zh/en/ja/ko model; Russian is only advertised for CosyVoice 3. — [CosyVoice 2 paper](https://arxiv.org/html/2412.10117v1) (language list not re-verified in this session — see Gaps)

**Qwen3-TTS**
- VERIFIED: supported languages "Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian". — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- VERIFIED (vendor benchmark): Qwen3-TTS-12Hz-1.7B-Base WER on Russian = 3.212 vs MiniMax 4.281 in the multilingual content-consistency table. — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- COMMUNITY/primary discussion (opened 2026-02-06, updates through April–May 2026, no maintainer reply): "Qwen-tts was trained without using stress marks, so...every word is pronounced incorrectly" (complaint is about stress placement, e.g. за́мок/замо́к); user Cupuyc1989 tried 180 variants of injecting acute marks converted from `+` notation — "Результат получается хуже"; partial workaround from user alimax21: mark stress in the reference (prompt) text so the model "learns" to differentiate. — [QwenLM/Qwen3-TTS Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185)
- COMMUNITY (Habr, 27 January [2026 by context; Qwen3-TTS was released 2026-01-22]): "у всех голосов есть азиатский акцент) Конечно, это относительно большой недостаток, так как голоса слабо адаптируются под новый язык"; for cloned voices: "даты и числа. Почему то их генерация у клонированных находится на уровне автоматов пятерочки"; problems with words containing iotated letters; pauses imitated imperfectly. — [Habr 989244](https://habr.com/ru/articles/989244/)
- COMMUNITY (Habr, Raft company blog; fetched summary dated it "3 февраля 2025" but it reviews QwenTTS, so it must be February 2026): QwenTTS (VoiceDesign) naturalness 4.5/5 for Russian, 15 s on GPU / 40.5 s on CPU for the test text; same score as F5-TTS Russian fine-tune and ESpeech TTS-1 RL-V2 (4.5/5), above Silero v5_ru (4/5), XTTS-v2 (3/5), HiggsAudio (1/5, "с сильным акцентом"). GPU model not captured. — [Habr Raft: Обзор Open Source моделей для задачи TTS](https://habr.com/ru/companies/raft/articles/991844/)
- SNIPPET (Russian search summary across Habr comments): "Средненькое качество с интонациями, которые часто не в тему"; "русская интонация оставляет желать лучшего"; abbreviations read incorrectly. — [Habr 988110 comments](https://habr.com/ru/companies/bothub/news/988110/comments/), [Habr 989244 comments](https://habr.com/ru/articles/989244/comments/)
- Community Russian fine-tune: `sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf` — Qwen3-TTS-12Hz-1.7B-Base "дообученная слушаться знака ударения в тексте"; data: Russian LibriSpeech (OpenSLR 96, 11.9 h, 5 speakers) + 3000 phrases with shifted stress voiced by ESpeech-TTS-1 RL-V2; stress mark = U+0301 after the stressed vowel; reported effect: stress obeyed on marked non-standard syllables in 43% (of 215) vs 26% (of 134) for the base model (p = 0.0008) — i.e. an improvement, but far from reliable; Apache-2.0. (Fetched summary gave card date "May 16, 2025", which is impossible for a Qwen3-TTS derivative; presumably May 2026.) — [HF sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf](https://huggingface.co/sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf)

**MOSS-TTS-Realtime (OpenMOSS / Fudan-SII)**
- VERIFIED: 20 languages listed, including Russian (also zh, en, de, es, fr, ja, it, hu, ko, fa, ar, pl, pt, cs, da, sv, el, tr). — [MOSS-TTS-Realtime model card (GitHub)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)
- No Russian-language quality review of MOSS-TTS-Realtime was found (see Gaps).

**Kyutai TTS 1.6B / Unmute / Pocket TTS**
- VERIFIED: `kyutai/tts-1.6b-en_fr` supports "English and French" only. — [HF kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- VERIFIED: fine-tuning is not supported — tracking issue "Fine-tuning Kyutai Text-To-Speech" (#64); Kyutai was "discussing internally how/if" to provide it (status as returned by search; page not opened). — [delayed-streams-modeling issue #64](https://github.com/kyutai-labs/delayed-streams-modeling/issues/64)
- VERIFIED: Pocket TTS (100M, CPU) went multilingual in April/May 2026: English, French, German, Spanish, Portuguese, Italian (kyutai.org additionally lists Dutch); no Russian. — [Kyutai on X, 2026-05-04](https://x.com/kyutai_labs/status/2051317316713894368), [pocket-tts issue #118](https://github.com/kyutai-labs/pocket-tts/issues/118), [kyutai.org/tts](https://kyutai.org/tts/)

**VibeVoice**
- VERIFIED: Realtime-0.5B — English primary; experimental voices for "German, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese, and Spanish"; "Transcripts in languages other than English may result in unexpected audio outputs." Russian not mentioned. — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- VERIFIED: for the Realtime model "Voice prompts are provided in an embedded format"; custom voices — "please reach out to our team" (no open voice cloning). — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- Community LoRAs exist for VibeVoice 1.5B/7B (e.g. `vibevoice-community/*`, Hungarian `Cseti/VibeVoice_7B_Diffusion-head-LoRA_Hungarian-CV17`), but a search found no Russian VibeVoice LoRA/fine-tune, and these target the non-streaming long-form models. — [HF vibevoice-community](https://huggingface.co/vibevoice-community/models), [HF Cseti Hungarian LoRA](https://huggingface.co/Cseti/VibeVoice_7B_Diffusion-head-LoRA_Hungarian-CV17)

**Voxtral TTS**
- VERIFIED: 9 languages — "English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi"; no Russian. — [HF mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)

**Sesame CSM-1B**
- VERIFIED (FAQ): "The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well." — [SesameAILabs/csm](https://github.com/SesameAILabs/csm)

**Orpheus TTS**
- VERIFIED: multilingual "research preview" family released April 2025 ("7 pairs of pretrained and finetuned models"); the fetched README text did not enumerate languages, Russian not confirmed as official. — [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- Community Russian fine-tune: `papacliff/orpheus-3b-0.1-ft-ru` (seen in search results; card not opened, quality unknown). — [HF papacliff/orpheus-3b-0.1-ft-ru](https://huggingface.co/papacliff/orpheus-3b-0.1-ft-ru)

**Dia2**
- VERIFIED: "The model only supports up to 2 minutes of generation in English." — [nari-labs/dia2](https://github.com/nari-labs/dia2)

**Qwen-Audio-3.0-TTS (not open)**
- COMMUNITY (Habr news): Flash/Plus cloud models on Alibaba Cloud Model Studio, Russian supported at model level, WebSocket streaming, but "Links to downloadable weights or a separate open-source repository were not found in release materials" — cloud-only, out of scope. — [Habr news 1062380](https://habr.com/ru/news/1062380/)

### Inferences
- For a Russian service, the realistic shortlist with native text-streaming is {Fun-CosyVoice3, MOSS-TTS-Realtime, Qwen3-TTS}, and of those only CosyVoice (generator input) and MOSS-TTS-Realtime (`push_text`) actually expose incremental text in official code.
- Stress is the systematic weak point of the multilingual Chinese-origin models for Russian: neither CosyVoice 3 nor Qwen3-TTS can be reliably steered by a stress dictionary/accentor (ruaccent, silero-stress). CosyVoice 3 at least accepts sparse stress marks as hints; Qwen3-TTS base ignores/garbles them and even a dedicated fine-tune reaches only ~43% obedience. A production Russian pipeline would therefore have homograph/stress errors it cannot deterministically fix — and with token-level text streaming there is even less room for a text-normalisation/accentuation front-end (the vllm-omni RFC makes the same point for numbers, see Q7).
- Vendor WER for Russian (Qwen3-TTS 3.2%) measures ASR-intelligibility, not stress/accent correctness, so it does not contradict the community complaints.

### Gaps
- No quantitative Russian benchmark (WER/MOS/stress accuracy) for Fun-CosyVoice3 was found on primary sources; the CosyVoice 3 paper ([arXiv 2505.17589](https://arxiv.org/pdf/2505.17589)) may contain a multilingual table but was not opened.
- No Russian quality reports at all for MOSS-TTS-Realtime.
- CosyVoice 2's exact official language list was not re-verified in this session; whether it can speak Russian cross-lingually is unknown.
- Reddit r/LocalLLaMA threads on Russian quality were not retrieved (search tool returned none).
- Orpheus multilingual language list and the quality of `papacliff/orpheus-3b-0.1-ft-ru` not verified.
- No Russian fine-tunes found for Kyutai TTS, VibeVoice-Realtime, Dia2, CSM (absence of evidence from a handful of searches, not proof).

---

## Q2. How exactly does streaming text input work, and is it exposed in the official repo/server?

### Takeaway
"Native text streaming" is truly usable out of the box in four places: Kyutai TTS (word-level websocket protocol, production-proven in Unmute), CosyVoice 2 (Python generator → interleaved text/speech tokens), MOSS-TTS-Realtime (`push_text()` deltas) and Dia2 (CLI/API). VibeVoice-Realtime and Qwen3-TTS have the architecture but the official open-source code does not (fully) expose incremental text; community wrappers for both fall back to sentence-level or full-text input. Voxtral TTS, CSM-1B and Orpheus do NOT have native text-streaming.

### Cited Findings

**CosyVoice 2 / Fun-CosyVoice 3**
- VERIFIED: README claims "Support both text-in streaming and audio-out streaming" with "latency as low as 150ms". — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice), [HF Fun-CosyVoice3](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- VERIFIED: official `example.py` shows bistream for CosyVoice2 by passing a Python generator as the text argument — comment "you can use generator as input, this is useful when using text llm model as input": `def text_generator(): yield '收到好友从远方寄来的生日礼物，' ...` then `cosyvoice.inference_zero_shot(text_generator(), ...)`. In the fetched file the CosyVoice3 examples use plain strings (generator usage not demonstrated for v3). — [example.py](https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/example.py)
- Mechanism (paper-level, via secondary summary): backbone Qwen2.5-0.5B autoregressively predicts supervised semantic speech tokens; text and speech tokens are interleaved with special markers (start, turn-to-speech, end and streaming "fill" tokens) so one model serves streaming and non-streaming; a chunk-aware causal flow-matching decoder turns tokens into mel, then vocoder. (The CosyVoice 2 paper's fixed text:speech interleave ratio N:M = 5:15 is from my prior knowledge of the paper and was not re-verified in this session.) — [CosyVoice 2 paper](https://arxiv.org/html/2412.10117v1), [emergentmind summary](https://www.emergentmind.com/topics/cosyvoice2-tts-model)
- VERIFIED: the shipped servers/UI do not expose token-level text streaming — issue #1509 (2025-08-05, stale, no maintainer answer): "the input text (tts_text) must still be a complete sentence, and the user must press the 'generate' button to begin synthesis". — [CosyVoice issue #1509](https://github.com/FunAudioLLM/CosyVoice/issues/1509)
- VERIFIED: open bug since 2025-02-12 (stale, no maintainer reply): `inference_bistream` "may repeat the last sentence in the prompt audio" — the model continues speaking the prompt text because prompt text and prompt audio tokens are interleaved with a language-dependent ratio. — [CosyVoice issue #967](https://github.com/FunAudioLLM/CosyVoice/issues/967)
- VERIFIED: the vllm-omni CosyVoice 2/3 acceleration RFC (2026-08-31) does not mention streaming text input at all. — [vllm-omni issue #6870](https://github.com/vllm-project/vllm-omni/issues/6870)

**Kyutai TTS 1.6B (DSM) / Unmute**
- VERIFIED: "delayed streams modeling" — text stream and audio stream are time-aligned and the audio is delayed relative to text; model card: audio is shifted by 1.28 s (16 steps at 12.5 Hz) relative to the text, 32 audio tokens (Mimi codebooks) per frame, acoustic/semantic delay of 2. Text is therefore consumed word by word; the model needs only a short look-ahead of upcoming words. — [HF kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- VERIFIED (Kyutai statement): "Kyutai TTS is the first text-to-speech model that is also streaming in text. You can pipe in text as it's being generated by an LLM". — [kyutai.org/next/tts](https://kyutai.org/next/tts), quoted in [issue #101](https://github.com/kyutai-labs/delayed-streams-modeling/issues/101)
- VERIFIED: exposed in code — Rust server gives "streaming access to the model over websockets"; PyTorch script `tts_pytorch_streaming.py` is "a fully streaming implementation"; MLX implementation for Apple silicon. — [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/)
- VERIFIED: Unmute backend — "As the response is being generated, the backend feeds it to the text-to-speech server to read it out loud" (LLM output piped word-wise into the TTS websocket). — [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- VERIFIED: documentation of the streaming-input protocol is thin — issue #101 (2025-07-31) asking how to feed LLM tokens progressively was open without maintainer answer in the fetched view; Unmute's source is the de-facto reference. — [issue #101](https://github.com/kyutai-labs/delayed-streams-modeling/issues/101)

**VibeVoice-Realtime-0.5B**
- VERIFIED: design — "interleaved, windowed design: it incrementally encodes incoming text chunks while, in parallel, continuing diffusion-based acoustic latent generation from prior context"; based on Qwen2.5-0.5B, 7.5 Hz acoustic tokenizer, 8K context, up to ~10 min; acoustic-only tokenizer (no semantic tokenizer) in the realtime variant. — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md), [secondary confirmation of specs](https://github.com/microsoft/VibeVoice)
- VERIFIED: the official doc still lists the TODO "Implement streaming text input function to feed new tokens while audio is still being generated" — i.e. the shipped demo (`python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B`, websocket) streams audio out, and text is windowed internally, but a public push-tokens-while-generating API was not yet delivered as of the fetched doc. (Text-window size — commonly cited as 5 text tokens per window — not verified in this session.) — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- VERIFIED: the public DGX Spark pipeline on VibeVoice-Realtime uses SENTENCE-level streaming, not token-level: "we buffer tokens until a sentence boundary (. ! ?), then immediately stream that sentence to TTS". — [Logos-Flux/spark-voice-pipeline](https://github.com/Logos-Flux/spark-voice-pipeline)
- VERIFIED: VibeVoice-1.5B / Large are long-form multi-speaker TTS (no text streaming); "we have removed the VibeVoice-TTS code from this repository" after misuse; only Realtime-0.5B offers "streaming text input". — [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)

**Qwen3-TTS**
- VERIFIED (paper/README claim): "Dual-Track hybrid streaming generation architecture… It can output the first audio packet immediately after a single character is input"; 12 Hz tokenizer (Qwen3-TTS-Tokenizer-12Hz) with lightweight causal decoder. — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS), [Technical Report arXiv 2601.15621](https://arxiv.org/html/2601.15621v1)
- VERIFIED: official open-source release does not ship the streaming service: README for vLLM-Omni says "Now only offline inference is supported. Online serving will be supported later."; issue #77 "Streaming support inference" (2026-01-25) was closed as "not planned", labelled inactive, no maintainer response. — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS), [issue #77](https://github.com/QwenLM/Qwen3-TTS/issues/77)
- VERIFIED: what the model code does offer is a generation mode where the (already complete) text is fed progressively alongside audio frames: "The original Qwen3TTS implementation supports two mode of generation. It either takes the full input text and prepares the utterance, or it feeds the text progressively" (`non_streaming_mode` flag). This confirms the dual-track mechanism is in the weights, but the API still takes the whole text up front. — [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- VERIFIED: community streaming forks add AUDIO-output streaming only, e.g. `rekuenkdr/Qwen3-TTS-streaming` — API `stream_generate_voice_clone(text="…")` requires complete text; also `dffdeeq/Qwen3-TTS-streaming` (~6x speedup claim), `CloudWells/qwen3-tts-realtime-streaming`, `xmillogx-cmd/Qwen3-tts-streaming`, Rust port `TrevorS/qwen3-tts-rs` (not individually opened except rekuenkdr). — [rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming), [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
- VERIFIED: vllm-omni RFC "Token-level streaming text input for TTS: commitment policy, …" opened 2026-08-22 (feedback until 2026-09-05): "vLLM-Omni TTS today accepts either a complete prompt or, on the omni path, chunked tokens forwarded from an upstream stage"; Qwen3-TTS is "the natural target"; wire format undecided ("Incremental text over the existing speech endpoint, or a separate streaming session?"). So as of September 2026 token-level text streaming for Qwen3-TTS in the main serving stack is a proposal, not a feature. — [vllm-omni issue #6496](https://github.com/vllm-project/vllm-omni/issues/6496)

**MOSS-TTS-Realtime**
- VERIFIED: 1.7B "context-aware, multi-turn streaming TTS foundation model designed for real-time voice agents"; hierarchical input where text tokens and speech tokens are processed at different levels → streaming text in + streaming audio out; API: "provide a streaming text_deltas source that yields incremental text chunks", `push_text()` accepting LLM deltas; conditions on multi-turn dialogue history. — [MOSS-TTS-Realtime model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md), [moss_tts_realtime README](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_realtime/README.md)

**Dia2 (Nari Labs)**
- VERIFIED: "does not need the entire text to produce the audio, and can start generating as the first few words are given as input"; exposed via CLI and Python API; "Dia2 TTS Server: Real streaming support" listed as upcoming; prefix-audio conditioning (Whisper transcribes the prefix). — [nari-labs/dia2](https://github.com/nari-labs/dia2)

**NOT native text-streaming (explicit)**
- Voxtral TTS (Mistral): model card speaks of "streaming and batch inference" for audio output; "No mention of incremental text input streaming". — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- Sesame CSM-1B: official repo documents per-utterance generation conditioned on text+audio context; no streaming input/output documented. A community fork adds audio streaming (`davidbrowne17/csm-streaming`). A HF discussion claims fine-tuning CSM for input/output streaming "works pretty well" — community claim, not an official capability. — [SesameAILabs/csm](https://github.com/SesameAILabs/csm), [davidbrowne17/csm-streaming](https://github.com/davidbrowne17/csm-streaming), [HF csm-1b discussions](https://huggingface.co/sesame/csm-1b/discussions/52)
- Orpheus TTS: README advertises "~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming", but the released package takes a complete prompt and streams audio (SNAC) tokens out via vLLM; no incremental-text API in the repo. — [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- "Moshi-derived TTS": the Moshi-derived TTS is precisely Kyutai TTS 1.6B (DSM "pioneered with Moshi") and, architecturally, Dia2 and CSM (Mimi codec + depth transformer). No additional Moshi-derived open TTS with text streaming was found. — [kyutai.org/next/tts](https://kyutai.org/next/tts)

**Other models surfaced (2026)**
- FlashTTS (arXiv 2606.09141, submitted 2026-06-08, Interspeech 2026): Qwen2-0.5B-based, "lagged multi-track streaming" + multi-token prediction, streaming text and speech inputs natively, first-packet latency 325 ms; "The model code and checkpoints will be released as open source" — no repo/weights located. — [arXiv 2606.09141](https://arxiv.org/abs/2606.09141)
- "Prosodic Boundary-Aware Streaming Generation for LLM-Based TTS" (arXiv 2603.06444) and "VoiceChat-TTS: A Low-Latency Continuous Speech Synthesis Model for Interactive Agents" (arXiv 2608.13831) appeared in search results; not opened, availability of weights unknown. — [arXiv 2603.06444](https://arxiv.org/html/2603.06444), [arXiv 2608.13831](https://arxiv.org/pdf/2608.13831)
- GLM-TTS (Zhipu, Dec 2025), IndexTTS2, Ming Flash Omni are mentioned in the vllm-omni RFC only as having text front-ends; none is stated to have token-level text streaming. — [vllm-omni issue #6496](https://github.com/vllm-project/vllm-omni/issues/6496)

### Inferences
- Maturity ranking of the text-streaming INTERFACE (not language fit): Kyutai TTS (production websocket, used in Unmute) > MOSS-TTS-Realtime (documented `push_text` API) ≈ CosyVoice 2 (in-process generator, you must write your own network layer; open bistream bug) > Dia2 (no server yet) > VibeVoice-Realtime (TODO in docs) > Qwen3-TTS (not exposed; RFC stage).
- For CosyVoice 3 the README claim of bidirectional streaming is inherited from the shared code base (same `inference_zero_shot` path); that the generator input works identically with Fun-CosyVoice3-0.5B-2512 is likely but was not confirmed by an official example in this session.
- With the fixed text:speech interleave used by CosyVoice 2, audio generation can start only after the first block of text tokens has arrived — "token-by-token" is in practice "after the first few tokens", similar to Kyutai's ~1.28 s text look-ahead expressed in words.

### Gaps
- Exact text window/chunk sizes for VibeVoice-Realtime and CosyVoice 2's interleave ratio were not re-verified from primary text this session.
- Whether Fun-CosyVoice3 generator input works through the vLLM / Triton-TRT-LLM runtimes (vs only the PyTorch path) is not documented in what I fetched.
- Whether Microsoft has since implemented the TODO (token push API) in code while leaving the doc stale — not checked in source.
- MOSS-TTS-Realtime streaming in vLLM-Omni: whether the vLLM-Omni path preserves incremental text input or only the native torch API does — not verified.

---

## Q3. Streaming audio output: TTFA / first-packet latency, RTF, measurement hardware

### Takeaway
Published first-audio latencies cluster at 100–400 ms on datacenter/high-end GPUs; none of the vendors reports numbers on anything close to a GTX 1050 Ti. DGX Spark numbers exist for Qwen3-TTS (280–400 ms TTFA) and VibeVoice-Realtime (~300 ms first chunk, RTF 0.48).

### Cited Findings
- Kyutai TTS 1.6B: "latency of 220ms from receiving the first text token to generating the first chunk of audio"; in the Unmute deployment with batching of up to 32 simultaneous requests, 350 ms on an L40S (July 2025 figures). Model card: "throughput of 75x generated audio per compute unit of time" (batched). — [kyutai.org/next/tts](https://kyutai.org/next/tts), [HF kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- Unmute (self-hosted): "The TTS latency decreases from ~750ms when running everything on a single L40S GPU to around ~450ms" with separate GPUs per service. — [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- CosyVoice 2/3: vendor claim "latency as low as 150ms" (hardware not stated). — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice)
- CosyVoice 2/3 in vllm-omni, measured 2026-08-31 on RTX 5080, single stream: TTFA median 0.357 s, RTF 0.214 (4.7x real time; lower-is-better convention); at 4x concurrency RTF 0.385 and TTFA 0.813 s; per-chunk cost grows with prefix ("per_chunk(n) ≈ 106 ms + 0.50 ms/token × n"). — [vllm-omni issue #6870](https://github.com/vllm-project/vllm-omni/issues/6870)
- CosyVoice: "Using TensorRT-LLM to accelerate cosyvoice2 llm could give 4x acceleration" (vs HF transformers), Triton+TRT-LLM runtime added 2025/08. — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice)
- VibeVoice-Realtime-0.5B: first audible speech "~200 milliseconds" in the doc (README says ~300 ms), hardware-dependent; "NVIDIA T4 / Mac M4 Pro achieve real-time performance in our tests; other devices with weaker inference capability may require further testing". — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md), [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- VibeVoice-Realtime on DGX Spark (community, 2026-01-04): RTF 0.48 on GB10 (53 s of audio in 26 s), first chunk ≈300 ms in streaming mode; full Whisper+Ollama+VibeVoice pipeline ≈766 ms to first audio. — [HF discussion #23](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/discussions/23), [Logos-Flux/spark-voice-pipeline](https://github.com/Logos-Flux/spark-voice-pipeline)
- Qwen3-TTS (vendor, paper Jan 2026): first-packet latency 97 ms (0.6B: 93 ms LM + 4 ms tokenizer) and 101 ms (1.7B) on internal vLLM (V0 backend) with torch.compile + CUDA Graph on "a single typical computational resource" (GPU model not named in the retrieved summary); RTF ≤ 0.5 at six-way concurrency is the API SLA. These numbers are for Alibaba's internal engine, not reproducible with the open `qwen-tts` package. — [arXiv 2601.15621](https://arxiv.org/html/2601.15621v1)
- Qwen3-TTS via faster-qwen3-tts (community, CUDA graphs; NOTE inverse RTF = audio/compute, higher is better): 0.6B — RTX 4090: RTF 4.78, TTFA 156 ms; Jetson AGX Orin 64GB: 1.307, 597 ms; DGX Spark (GB10): 2.56, 280 ms. 1.7B — RTX 4090: 4.22, 174 ms; Jetson AGX Orin: 1.089, 693 ms; DGX Spark: 1.87, 400 ms. — [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- Qwen3-TTS streaming fork (rekuenkdr): first chunk 208 ms ("two-phase", 2.75x faster than baseline), RTF 0.39 (lower-is-better); GPU not captured. — [rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
- MOSS-TTS-Realtime: TTFB 180 ms "(After warm up)", RTF 0.51 on a single NVIDIA L20; LLM-first-sentence + TTS TTFB = 377 ms. — [MOSS-TTS-Realtime model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)
- Voxtral TTS (not text-streaming; for reference): 70 ms latency, RTF 0.103 at concurrency 1 on a single NVIDIA H200 via vllm-omni. — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- FlashTTS (paper): first-packet latency 325 ms; hardware not captured. — [arXiv 2606.09141](https://arxiv.org/abs/2606.09141)
- Orpheus: "~200ms streaming latency… reducible to ~100ms with input streaming" (vendor claim, hardware unspecified). — [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- Dia2: no latency/RTF published in the repo. — [nari-labs/dia2](https://github.com/nari-labs/dia2)

### Inferences
- MOSS-TTS-Realtime's RTF 0.51 on an L20 (a 48 GB datacenter card) leaves only 2x headroom; on DGX Spark (which ran Qwen3-TTS 1.7B at ~1.87x real time and VibeVoice-Realtime at ~2x) a 1.7B MOSS model would likely sit close to real time — must be measured.
- All TTFA figures measure "first text token/request → first audio chunk" on a warm model; in a text-streaming setup perceived latency additionally includes the model's text look-ahead (Kyutai: needs the next few words; CosyVoice: first interleave block), which depends on LLM token rate rather than GPU speed.

### Gaps
- No TTFA/RTF measurement for any candidate on GTX 1050 Ti or comparable Pascal 4 GB cards.
- No DGX Spark numbers for CosyVoice 2/3, Kyutai TTS, MOSS-TTS-Realtime, Dia2.
- Hardware behind CosyVoice's "150 ms" and Qwen3-TTS's "97 ms" claims is not specified in the sources fetched.

---

## Q4. VRAM, fit in 4 GB (GTX 1050 Ti), aarch64 CUDA (DGX Spark / Jetson)

### Takeaway
No source reports any of these models running on a 4 GB Pascal card; by stated requirements, Kyutai TTS (5.3 GB), MOSS-TTS-Realtime, Voxtral (≥16 GB), Dia2-2B and CosyVoice (community ~7–8 GB) do not fit, and only the 0.5–0.6B models (VibeVoice-Realtime-0.5B, Qwen3-TTS-0.6B, possibly CosyVoice with quantised LLM) are even candidates for 4 GB — with real-time speed on a 1050 Ti being doubtful. On DGX Spark, Qwen3-TTS and VibeVoice-Realtime are confirmed working (PyTorch cu130 aarch64 wheels); Kyutai's Unmute officially requires x86_64.

### Cited Findings
- GTX 1050 Ti platform constraint (VERIFIED): PyTorch cu128/cu129 wheels dropped Maxwell/Pascal; cu126 is the last official wheel line supporting Pascal (sm_61) and there is an RFC to deprecate CUDA 12.6 in PyTorch 2.15; CUDA 13.x does not support pre-Turing GPUs at all. — [PyTorch dev-discuss announcement](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128), [pytorch issue #157517](https://github.com/pytorch/pytorch/issues/157517), [pytorch RFC #190385](https://github.com/pytorch/pytorch/issues/190385), [Whisper-WebUI issue #653](https://github.com/jhj0517/Whisper-WebUI/issues/653)
- Kyutai TTS in Unmute: "a GPU with CUDA support and at least 16 GB VRAM. Architecture must be x86_64"; per service: STT 2.5 GB, TTS 5.3 GB, LLM 6.1 GB. PyTorch scripts offer `--quantize 8` / `--quantize 4` "if the model is not fast enough to keep with real-time" (that flag is documented for the MLX implementation context; applicability to CUDA not verified). — [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute), [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/)
- CosyVoice 2/3: no official VRAM figure in README/model card. SNIPPET-level community figures: "CosyVoice2 on Google Colab with fp16 uses ~7.1GB VRAM"; "0.5B models require ~8GB VRAM minimum"; default flow precision fp16 for CosyVoice2, fp32 for CosyVoice3. — [shinshin86/local-tts-on-google-colab](https://github.com/shinshin86/local-tts-on-google-colab) (snippet, not opened)
- CosyVoice 3 low-VRAM route: PR adding an optional llama-cpp-python backend for the CosyVoice3 LLM using GGUF quantised models for "CPU and low-VRAM inference" (PR status not checked); ONNX export exists (`ayousanz/cosy-voice3-onnx`), Rust bindings `SpenserCai/cosyvoice3.rs`. — [CosyVoice PR #1872](https://github.com/FunAudioLLM/CosyVoice/pull/1872), [HF ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx)
- CosyVoice 3 TensorRT caveat: "CosyVoice3 load_trt=True + fp16=True produces non-finite (NaN)" audio; fp32 TRT is the supported config. — [CosyVoice issue #1930](https://github.com/QwenAudio/CosyVoice/issues/1930)
- Qwen3-TTS on DGX Spark (VERIFIED community, May 2026): runs as Docker containers with OpenAI-compatible API; reported GPU memory "faster-qwen3-tts 2126MiB / 121.7GiB" (which model size this refers to is not stated in the captured text); earlier thread was about "torchaudio installation failure on ARM64", resolved by prebuilt Docker images (v5+). — [NVIDIA forum: faster-qwen3-tts for DGX Spark](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530), [NVIDIA forum: Qwen3-TTS support on DGX Spark, torchaudio failure](https://forums.developer.nvidia.com/t/support-for-qwen3-tts-on-dgx-spark-gb10-torchaudio-installation-failure-on-arm64/359663/7), [mARTin-B78/dgx-spark-faster-qwen3-tts](https://github.com/mARTin-B78/dgx-spark-faster-qwen3-tts)
- Qwen3-TTS on Jetson AGX Orin (aarch64): benchmarked in faster-qwen3-tts (see Q3). — [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- VibeVoice-Realtime on DGX Spark (VERIFIED community, 2026-01-04): works after installing PyTorch from `https://download.pytorch.org/whl/cu130` (default install gives `CUDA available: False`); "Flash Attention not needed - SDPA fallback works fine". — [HF discussion #23](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/discussions/23)
- MOSS-TTS-Realtime: no official VRAM number in the model card; SNIPPET third-party guide: minimum 6 GB, recommended 8 GB, optimal 12+ GB for the 1.7B realtime model. The family has a torch-free llama.cpp path with ONNX/TensorRT audio tokenizer; README states the "8B model fits onto 8GB GPUs" with optimisation. No aarch64/Jetson notes. — [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), [third-party guide (snippet)](https://qwen-image-2512.com/blog/moss-tts-complete-guide-en)
- Voxtral TTS: "can run on a single GPU with >= 16GB memory". — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- Dia2: no official VRAM; SNIPPET community figures: "bfloat16 uses ~4-5GB VRAM", 2B realistic at ≥12 GB, Dia2-1B for tight VRAM. — [nari-labs/dia2](https://github.com/nari-labs/dia2), [HF nari-labs/Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B)
- DGX Spark ecosystem: a German-language TTS serving repo for Spark (`MvdB/southbyte-tts`) and curated lists (`bidual/awesome-dgx-spark`) exist; contents not reviewed. — [MvdB/southbyte-tts](https://github.com/MvdB/southbyte-tts), [bidual/awesome-dgx-spark](https://github.com/bidual/awesome-dgx-spark)

### Inferences
- GTX 1050 Ti (4 GB) verdict per model — all INFERRED, none measured:
  - Kyutai TTS 1.6B: does not fit (5.3 GB stated) unless quantised; irrelevant for Russian anyway.
  - MOSS-TTS-Realtime 1.7B, Qwen3-TTS 1.7B, Dia2, Voxtral, Orpheus 3B, CosyVoice (stock PyTorch): do not fit or do not run in real time.
  - Qwen3-TTS-0.6B: memory plausibly fits (≈2.1 GB observed on Spark for an unspecified size), but DGX Spark itself reaches only ~2.56x real time and Jetson AGX Orin ~1.3x; a 1050 Ti is far slower than GB10 and lacks fast FP16/BF16, so sub-real-time is the expected outcome.
  - VibeVoice-Realtime-0.5B: memory likely fits; Microsoft says a T4 is the tested real-time floor and "weaker devices may require further testing" — a 1050 Ti is weaker than a T4.
  - The faster-qwen3-tts / vLLM / CUDA-graph / TensorRT accelerations assume recent CUDA stacks; the 1050 Ti is locked to cu126-era PyTorch, which may block newer serving stacks (vLLM, vllm-omni) outright.
  - Practical conclusion: the 4 GB Pascal card should be treated as unsuitable for GPU-class text-streaming TTS; it is a dev/STT box. DGX Spark is the only one of the two machines where these models are viable.
- DGX Spark: anything that is pure PyTorch works with cu130 aarch64 wheels (confirmed for Qwen3-TTS, VibeVoice). Risk items on aarch64 are native dependencies: torchaudio builds, TensorRT/Triton images (CosyVoice's Triton-TRT-LLM compose is built for x86_64), Kyutai's Rust moshi-server (Unmute says x86_64 only), onnxruntime-gpu aarch64 wheels (CosyVoice front-end uses ONNX models for speech tokenizer/campplus).

### Gaps
- No report found of CosyVoice 2/3 on DGX Spark or Jetson (search returned nothing specific).
- No report of Kyutai TTS on aarch64 CUDA; the "x86_64 only" statement is for Unmute as a whole.
- No authoritative VRAM numbers for CosyVoice 2/3, VibeVoice-Realtime, MOSS-TTS-Realtime, Dia2.
- No 1050 Ti/4 GB data points for any model.

---

## Q5. Licenses (code and weights), commercial use, voice-cloning restrictions

### Takeaway
Apache-2.0 for both code and weights: CosyVoice 2/3, Qwen3-TTS, MOSS-TTS family, Dia2, CSM-1B, Orpheus — commercial use allowed. Kyutai TTS weights are CC-BY 4.0 (commercial use allowed with attribution) but voice cloning is deliberately withheld. VibeVoice is MIT but explicitly "not recommended for commercial or real-world applications" and has no open voice cloning for the Realtime model. Voxtral TTS weights are CC BY-NC 4.0 (non-commercial).

### Cited Findings
- CosyVoice (code + models): Apache 2.0. Zero-shot voice cloning from prompt audio is part of the open release. — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice), [HF Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- Qwen3-TTS: Apache-2.0; Base models do voice cloning, CustomVoice = preset speakers, VoiceDesign = text-described voices. — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- faster-qwen3-tts wrapper: MIT. — [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- MOSS-TTS family: "Models in MOSS-TTS Family are licensed under the Apache License 2.0."; Realtime model advertises "High-Fidelity Voice Cloning with Multi-Turn Consistency". — [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), [model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)
- Kyutai TTS: code "MIT license for the Python parts, and Apache license for the Rust backend"; weights CC-BY 4.0; voice conditioning restricted to pre-computed embeddings from `kyutai/tts-voices` — the speaker-embedding model is not released; no watermarking. — [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/), [HF kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- Unmute: MIT. — [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- VibeVoice: MIT license on the repo; model doc: "Not recommended for commercial or real-world applications without further testing", "research and development purposes only"; Realtime voice prompts only "in an embedded format", custom voices by contacting the team; TTS (1.5B) code removed for misuse. — [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice), [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- Voxtral TTS: CC BY-NC 4.0; open weights ship with 20 preset voices; commercial use via Mistral API / separate licence. — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603), [Mistral announcement](https://mistral.ai/news/voxtral-tts/)
- Dia2: Apache 2.0. — [nari-labs/dia2](https://github.com/nari-labs/dia2)
- CSM-1B: Apache-2.0. — [SesameAILabs/csm](https://github.com/SesameAILabs/csm)
- Orpheus: Apache-2.0. — [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- `sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf`: Apache-2.0 (inherited). — [HF](https://huggingface.co/sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf)

### Inferences
- For a commercial Russian service the licence-clean options with Russian are Fun-CosyVoice3, Qwen3-TTS and MOSS-TTS-Realtime (all Apache-2.0, all with open zero-shot cloning). Voice-cloning consent is left to the deployer by these licences.

### Gaps
- FlashTTS licence unknown (no release). Licence of `papacliff/orpheus-3b-0.1-ft-ru` not checked.

---

## Q6. Maturity: 2026 activity, ready-made servers and deployments

### Takeaway
CosyVoice has the broadest deployment tooling (FastAPI/gRPC Docker, vLLM, TensorRT-LLM + Triton, vllm-omni work in Aug 2026) but none of it exposes text streaming over the wire. Kyutai ships the only production-grade text-streaming websocket server. Qwen3-TTS has the largest 2026 community ecosystem (OpenAI-compatible servers, DGX Spark Docker images) but without incremental text. MOSS-TTS is the most actively released family in mid-2026 and is supported in vLLM-Omni.

### Cited Findings
- CosyVoice: roadmap — 2024/08 streaming inference mode; 2025/05 CosyVoice2-0.5B vLLM support; 2025/07 Fun-CosyVoice 3.0 eval set; 2025/08 Triton TRT-LLM runtime; 2025/12 "release Fun-CosyVoice3-0.5B-2512 base model, rl model and its training/inference script". "CosyVoice2/3 now supports vLLM 0.11.x+ (V1 engine) and vLLM 0.9.0 (legacy)". FastAPI and gRPC server examples with Docker; `docker compose up -d` in `runtime/triton_trtllm`. No roadmap entries for 2026 were present in the fetched README. — [CosyVoice GitHub](https://github.com/QwenAudio/CosyVoice)
- CosyVoice in vllm-omni: active performance RFC dated 2026-08-31 with multiple workstreams and merged PRs. — [vllm-omni issue #6870](https://github.com/vllm-project/vllm-omni/issues/6870)
- CosyVoice issue hygiene: streaming-input questions/bugs (#967 from 2025-02, #1509 from 2025-08) are stale without maintainer replies. — [issue #967](https://github.com/FunAudioLLM/CosyVoice/issues/967), [issue #1509](https://github.com/FunAudioLLM/CosyVoice/issues/1509)
- Third-party CosyVoice serving: ModelTC LightTTS ("lightweight TTS inference" framework) appeared in search; not reviewed. — [ModelTC/LightTTS](https://github.com/ModelTC/LightTTS)
- Kyutai: TTS 1.6B released July 2025; Rust websocket server + PyTorch + MLX; Unmute with Docker Compose (136 commits). 2026 Kyutai activity went to Pocket TTS (Jan 2026; multilingual Apr–May 2026), Hibiki-Zero (2026-02-12), Invincible Voice (2026-02-24) — no new large multilingual DSM TTS was announced. — [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/), [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute), [kyutai.org/tts](https://kyutai.org/tts/), [kyutai blog](https://kyutai.org/blog/)
- VibeVoice: repo now centred on ASR (VibeVoice-ASR-7B, ASR-Streaming, ASR-BitNet; vLLM for ASR); TTS-1.5B code removed; Realtime-0.5B available with a websocket demo; community wrappers: `pinokiofactory/vibevoice-realtime`, `neosun100/VibeVoice` (WebSocket `ws://localhost:8765/stream` per search snippet), `Logos-Flux/spark-voice-pipeline`. — [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice), [Logos-Flux/spark-voice-pipeline](https://github.com/Logos-Flux/spark-voice-pipeline)
- Qwen3-TTS: released 2026-01-22 (0.6B/1.7B; Base, CustomVoice, VoiceDesign, Tokenizer-12Hz); official online serving "will be supported later"; community: faster-qwen3-tts (353 commits, `examples/openai_server.py` exposing `POST /v1/audio/speech`), DGX Spark Docker images with OpenAI-compatible API on ports 8020–8023 incl. a chunk-streaming variant, several streaming forks, Rust port. — [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS), [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), [NVIDIA forum thread](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530)
- Alibaba's newer TTS generation (Qwen-Audio-3.0-TTS Flash/Plus) is cloud-only, suggesting the open Qwen3-TTS line may not receive the streaming service code. — [Habr news 1062380](https://habr.com/ru/news/1062380/)
- MOSS-TTS family: 2026-05-26 MOSS-TTS-v1.5 and SoundEffect-v2.0; 2026-06-02 "vLLM-Omni added full MOSS-TTS series support" (architectures `MossTTSDelay`, `MossTTSRealtime`, `MossTTSNano`); 2026-06-18 Local-Transformer-v1.5; also SGLang-Omni, llama.cpp torch-free path; Realtime has Gradio streaming demo and a FastAPI server limited to batch size 1. Technical report arXiv 2603.18090. — [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), [model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md), [arXiv 2603.18090](https://arxiv.org/pdf/2603.18090)
- Voxtral TTS: released March 2026 (model id `Voxtral-4B-TTS-2603`), served via "vllm-omni (recommended)" with vLLM >= 0.18.0. — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- Dia2: small repo (19 commits), HF Space demo, streaming server "upcoming". — [nari-labs/dia2](https://github.com/nari-labs/dia2)
- CSM: last notable update 2025-05-20 (Transformers integration). Orpheus: last notable updates April–May 2025; pinned to `vllm==0.7.3` due to a buggy later release. — [SesameAILabs/csm](https://github.com/SesameAILabs/csm), [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- vllm-omni as a convergence point: as of Aug 2026 it serves Qwen3-TTS, CosyVoice 2/3, MOSS-TTS, Voxtral TTS, GLM-TTS, IndexTTS2, but token-level text streaming is only an RFC (#6496) with wire format undecided; an earlier request (#1766) asked for streaming input without specifying a format. — [vllm-omni issue #6496](https://github.com/vllm-project/vllm-omni/issues/6496)

### Inferences
- No candidate offers an off-the-shelf server that (a) speaks Russian and (b) accepts token-level text over WebSocket. For Russian, a text-streaming endpoint would have to be self-built on top of CosyVoice's generator API or MOSS-TTS-Realtime's `push_text` API; OpenAI-compatible `/v1/audio/speech` is inherently full-text-in and cannot carry incremental text.
- CSM and Orpheus look stagnant in 2026 (no 2026 entries surfaced); CosyVoice's own repo is quiet in 2026 while optimisation work moved to vllm-omni.

### Gaps
- Exact last-commit dates for the repos were not captured (GitHub pages fetched did not expose them).
- Whether official Docker images for CosyVoice/MOSS are multi-arch (arm64) was not checked.

---

## Q7. Known weaknesses

### Takeaway
The recurring weaknesses relevant to a Russian real-time service are: uncontrollable stress in Russian (CosyVoice 3, Qwen3-TTS), accent/intonation and number reading (Qwen3-TTS), bistream-specific artefacts (CosyVoice prompt-repeat bug), English/French-only coverage and no cloning (Kyutai), English-only/2-minute cap (Dia2), and the inherent problem that token-level text streaming removes the look-ahead needed for text normalisation.

### Cited Findings
- Inherent to token-level text streaming (vllm-omni RFC, 2026-08-22): "Text on a screen can be silently rewritten; a spoken syllable cannot be recalled."; numbers are ambiguous until later tokens arrive (the "2026" example: year vs amount); fixed segmentation ramps; "audible timbre and prosody jumps" at cross-segment seams. — [vllm-omni issue #6496](https://github.com/vllm-project/vllm-omni/issues/6496)
- CosyVoice: `inference_bistream` may speak the tail of the prompt transcript (open since 2025-02); CosyVoice3 TRT fp16 NaN audio; over-marking Russian stress degrades output; O(N²)-like per-chunk cost growth and poor concurrency scaling in vllm-omni (2.2x throughput at 4x concurrency). — [issue #967](https://github.com/FunAudioLLM/CosyVoice/issues/967), [issue #1930](https://github.com/QwenAudio/CosyVoice/issues/1930), [vc.ru](https://vc.ru/ai/2681383-cosyvoice-tts-podderzhka-udareniy-v-russkom-yazyke), [vllm-omni #6870](https://github.com/vllm-project/vllm-omni/issues/6870)
- CosyVoice 3 "hard" test set CER is 6.71% (5.44% RL) vs ~1–2% on normal sets — hard cases (tongue twisters, repetitions) remain error-prone. — [HF Fun-CosyVoice3](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- Qwen3-TTS (Russian): wrong stress, stress marks not understood, "азиатский акцент", dates/numbers read badly with cloned voices, issues with iotated letters, abbreviations; no official streaming code; maintainers unresponsive on both the streaming issue and the Russian discussion. — [Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185), [Habr 989244](https://habr.com/ru/articles/989244/), [issue #77](https://github.com/QwenLM/Qwen3-TTS/issues/77)
- Kyutai TTS: English/French only; no fine-tuning path; no user voice cloning (embeddings only); 1.28 s text→audio delay means the model needs upcoming words before speaking; Unmute x86_64-only and 16 GB VRAM. — [HF kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr), [issue #64](https://github.com/kyutai-labs/delayed-streams-modeling/issues/64), [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- VibeVoice-Realtime: English-centric ("other languages may produce unpredictable results"); single speaker; no open voice cloning; speech only (no music/background); not recommended for commercial use; token-push API still a TODO. — [VibeVoice realtime doc](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- MOSS-TTS-Realtime: official FastAPI server handles batch size 1 only; RTF 0.51 on L20 (limited headroom). — [model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)
- Dia2: English only, ≤2 minutes per generation (1500 steps at ~12.5 Hz), no streaming server yet. — [nari-labs/dia2](https://github.com/nari-labs/dia2)
- Voxtral TTS: non-commercial licence, ≥16 GB GPU, no Russian, no text streaming. — [HF Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- CSM-1B: non-English "likely won't do well". — [SesameAILabs/csm](https://github.com/SesameAILabs/csm)
- HiggsAudio on Russian (not text-streaming; for context): "Синтезированный голос получился с сильным акцентом", 1/5. — [Habr Raft](https://habr.com/ru/companies/raft/articles/991844/)

### Inferences
- For Russian, true token-level text streaming conflicts with the mitigation every Russian TTS deployment relies on (pre-normalisation of numbers/abbreviations and explicit stress marks via an accentor): that front-end needs at least clause-level context. A hybrid — clause/sentence-level chunking in front of a bistream-capable model (CosyVoice 3 or MOSS-TTS-Realtime) — is the realistic compromise; it keeps prosodic continuity across chunks (the model's own context) while allowing normalisation per chunk.
- Long-form stability data for the Russian-capable streaming models was not found; CosyVoice and Qwen3-TTS are utterance-oriented, MOSS-TTS-Realtime is explicitly multi-turn context-aware, VibeVoice-Realtime claims ~10 min long-form.

### Gaps
- No systematic hallucination/long-form failure statistics for Russian on any candidate.
- No independent (non-vendor) evaluation of MOSS-TTS-Realtime found.
- Reddit r/LocalLLaMA and Russian Telegram community reports were not reachable through the search tool in this session; community evidence here is limited to Habr, vc.ru, GitHub discussions, NVIDIA forum and HF discussions.
