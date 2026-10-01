# GPU TTS models without native incremental text input, used via sentence/clause chunking — Russian focus (state: 30 Sep 2026)

Scope: XTTS-v2, F5-TTS + Russian fine-tunes (Misha24-10, ESpeech), Fish Speech / OpenAudio S1-mini / Fish Audio S2, Chatterbox Multilingual (V3) / Turbo, IndexTTS-2 / 2.5, Higgs Audio v2 / v3, Dia / Dia2, Zonos, Spark-TTS, MaskGCT, OuteTTS, Parler-TTS, MeloTTS, StyleTTS2-RU, Silero TTS v5, Russian-origin models (Sber GigaTTS, Yandex, T-Bank, TeraTTS), plus new 2026 models with Russian (MOSS-TTS-Realtime). Skipped as assigned to another researcher: CosyVoice, Kyutai, VibeVoice, Qwen3-TTS, Voxtral, Sesame CSM, Orpheus.

Reading conventions used below:
- "(primary)" = I fetched the official repo / model card / article itself. "(search snippet)" = the claim comes only from a search-result summary of that URL and was NOT opened — treat as unverified.
- Date caveat for Habr: the fetch tool reported several Habr article dates with year 2025 where the content and the article-ID order imply 2026 (e.g. article 991844 reviews Qwen3-TTS, which was released 22 Jan 2026, and sits between article 968988 dated 24 Nov 2025 and later ones). I give the date as reported and flag the probable real year. The report writer should not quote these years without re-checking.

## 1. Russian support and Russian quality (stress, ё, normalization, accent, stability)

### Takeaway
Only four families have Russian that the Russian-speaking community rates as native-grade: Silero v5 (fixed voices, built-in stress + homographs), the F5-TTS Russian fine-tunes (Misha24-10 and ESpeech, both relying on RUAccent and `+` stress marks), and — per a single Raft review — Qwen3-TTS (out of my scope). The multilingual "official Russian" models (XTTS-v2, Chatterbox Multilingual, Higgs Audio, OuteTTS, Fish S1-mini/S2, MOSS-TTS) list Russian, but the only side-by-side Russian review found reports stress errors and/or accent for XTTS-v2, Chatterbox and Higgs v2; for Fish S2, Higgs v3, OuteTTS 1.0 and MOSS-TTS-Realtime I found no independent Russian-quality evaluation at all. Several candidates simply have no Russian: Dia/Dia2, IndexTTS-2.5, Zonos, Spark-TTS, MaskGCT, Parler-TTS, MeloTTS.

### Cited Findings

**Community comparison (the only multi-model Russian side-by-side found)**
- Raft (Habr company blog), "Обзор Open Source моделей для задачи TTS": compares HiggsAudio (Boson AI), ChatterBox, ESpeech, F5-TTS (Russian fine-tune), QwenTTS, Silero, XTTS-v2 on Russian, on RTX 5070 + Ryzen 7500F. Subjective scores (naturalness / expressiveness / third column): HiggsAudio 1 / 1 / 3; ChatterBox 3 / 3 / 3; ESpeech-TTS-1 RL-V2 4.5 / 4 / 2; F5-TTS `v1_Base_v4_winter` 4.5 / 5 / 1; QwenTTS 1.7B-VoiceDesign 4.5 / 5 / 4; Silero v5_ru 4 / 4 / 4; XTTS-v2 3 / 3 / 2 (primary) — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/). Date: fetch reported "February 3, 2025"; a mirror dates it 9 Feb 2026, which matches the content — [ai-news.ru mirror](https://ai-news.ru/2026/02/open_source_protiv_proprietarshiny_obzor_tts_modelej_dlya_russkogo_yazyka.html).
- CONFLICT in that same table: the Habr fetch labels the third column "Integration" (ease of integration); the mirror fetch labels it "Stability". A sibling Raft article uses "Integration" as its third column, so "Integration" is more likely correct — [Habr/Raft CosyVoice vs Yandex](https://habr.com/ru/companies/raft/articles/1023206/). Do not report these numbers as "stability" without re-reading the original.
- Same review, qualitative: HiggsAudio speaks Russian "с сильным акцентом" and has no explicit language parameter; ChatterBox has problems with stress placement and intonation; ESpeech "порядок с произношением, порядок с интонацией"; the F5 Russian fine-tune gives "отличная генерация"; Silero is native Russian with an integrated stress tool (primary) — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- ESpeech hosts two HF Spaces, "Open TTS Leaderboard Ru" and "Russian ASR Leaderboard" (primary, org page) — [HF ESpeech org](https://huggingface.co/ESpeech). The leaderboard Space itself would not render for me (JS app), so I could not extract a single number — [Open TTS Leaderboard Ru](https://huggingface.co/spaces/ESpeech/open_tts_leaderboard_ru).
- Older baseline (12 Jul 2024, OUTDATED but the only metric-based Russian evaluation found): alphacephei evaluated Silero v3.1/v4, Vosk-TTS 0.6, TeraTTS, UtrobinTTS, XTTS2, Piper, EdgeTTS, Yandex, Tortoise, Bark on ~1k audiobook utterances (CER, UTMOS, similarity, FAD). Conclusions: "Fastspeech2 methods still show best clarity (Silero/Yandex/EdgeTTS)"; "XTTS2 results are much worse than I expected"; Yandex and EdgeTTS CER 0.6; Piper Irina CER 1.4 / UTMOS 3.672 (primary) — [alphacephei](https://alphacephei.com/nsh/2024/07/12/russian-tts.html).

**Silero TTS v5 (Russian-origin, fixed voices)**
- Russian models `v5_ru` … `v5_5_ru`; speakers aidar, baya, kseniya, xenia, eugene; auto-stress and homographs in all v5; question intonation from v5_4; SSML; sample rates 8000/24000/48000 (primary) — [silero-models GitHub](https://github.com/snakers4/silero-models).
- v5 announcement (31 Oct 2025): automatic stress and first version of homograph resolution; text normalization named as the main remaining problem ("97% или 99%" of synthesis cases solved) (primary) — [Habr 961930](https://habr.com/ru/articles/961930/).
- Questions update: 4 question types supported (special, general, alternative, tag questions); no accuracy numbers for stress/homographs published (primary; fetch reported 27 Mar 2025, article ID implies 27 Mar 2026) — [Habr 1015942](https://habr.com/ru/articles/1015942/).
- CIS multi-language v5 models (24 Nov 2025): 20 languages, 95 voices; authors warn "без правильного ударения модели славянских языков работают плохо" (primary) — [Habr 968988](https://habr.com/ru/articles/968988/).

**F5-TTS Russian fine-tunes**
- Misha24-10/F5-TTS_RUSSIAN (author Mikhail Yakovlev): fine-tune on 5,000+ h (custom Russian 4,000 h, Common Voice RU 239 h, Common Voice EN 240 h, Sova 400 h, LibriHeavy partial 180 h). Variants: `F5TTS_v1_Base`, `F5TTS_v1_Base_accent_tune` (100% of training sentences stress-annotated), `F5TTS_v1_Base_v2` (+16 epochs, data filtering). Stress controlled by `+` before the stressed vowel; RUAccent recommended for automatic stress. 81,687 downloads in the last month at fetch time (primary) — [HF Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN). The Raft review tested a later checkpoint named `v1_Base_v4_winter` — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- ESpeech ("Ebany Speech", 5-member enthusiast team): ESpeech-TTS-1 family — SFT-95K, SFT-256K, podcaster, RL-V1, RL-V2 — all last updated 25 Aug 2025; no TTS successor ("TTS-2") visible on the org page as of fetch; 2026 uploads are only a VAE (E-VAE-44100-25hz, Jan) and a denoiser (Jun) (primary) — [HF ESpeech org](https://huggingface.co/ESpeech).
- ESpeech-TTS-1_RL-V2: F5-TTS DiT config (dim 1024, depth 22, heads 16), trained on ESpeech-webinars2 (792k samples, 339 h, 8 speakers); RUAccent built into the reference inference script; `+` for manual stress; reference audio ≤12 s plus its transcript are mandatory (primary) — [HF ESpeech-TTS-1_RL-V2](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2).
- Another F5 Russian fine-tune exists: hotstone228/F5-TTS-Russian, "813k steps on 100k hours" Russian+English (search snippet) — [HF hotstone228](https://huggingface.co/hotstone228/F5-TTS-Russian).

**XTTS-v2**
- Russian is one of 17 officially supported languages; 24 kHz; cloning from a 6-second clip (primary) — [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2).
- Community assessment: no reliable stress control and no mechanism for manual stress marks, "unreliable for professional Russian voice-over" (search snippet, 2026 review site) — [promptquorum XTTS v2 review](https://www.promptquorum.com/power-local-llm/xtts-v2-review). Raft review: 3/5 naturalness, 3/5 expressiveness — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).

**Chatterbox (Resemble AI)**
- Current lineup: Chatterbox-Turbo (350M, English), Chatterbox-Nano (110M, English), Chatterbox Multilingual V3 (500M, 23 languages incl. Russian), original Chatterbox (500M, English), plus a "Single Language Pack" of 6 dedicated fine-tunes (primary) — [GitHub resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox). Turbo/Nano are English-only, so only Multilingual is relevant for Russian.
- Model card: multilingual trained on 0.5M h, "improved speaker similarity, reduced hallucinations"; the reference clip must match the language tag or the output inherits the reference's accent (primary) — [HF ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).
- Upstream Chatterbox V3 multilingual preprocessing includes optional Russian stress insertion; a third-party ONNX port notes it omits it (search snippet, PR dated 22 Sep 2026) — [KitsuMate.Onnx PR 18](https://github.com/KitsuMate/KitsuMate.Onnx/pull/18).
- Raft review (early 2026, pre-V3 most likely): stress and intonation problems, 3/5 — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).

**Fish Speech / OpenAudio S1-mini / Fish Audio S2**
- OpenAudio S1-mini (0.5B, distilled from S1, trained on 2M+ h) lists Russian among 13 languages (search snippet of the model card) — [HF fishaudio/openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini).
- Fish Audio S2 Pro (open weights, tech report 9 Mar 2026): 80+ languages in three tiers; Russian is Tier 2 (with Korean, Spanish, Portuguese, Arabic, French, German); Tier 1 is Japanese, English, Chinese (primary) — [HF fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro); [arXiv 2603.08823](https://arxiv.org/abs/2603.08823).

**Higgs Audio (Boson AI)**
- v2 language list cited by a third party: en, zh, fr, de, es, it, ja, ko — no Russian (search snippet) — [localclaw.io](https://localclaw.io/tts/higgs-audio-v2). Consistent with the Raft finding of a strong accent and 1/5 naturalness in Russian — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- Higgs Audio v3 (2026, supersedes v2/v2.5): 4B params, "conversational TTS across 100+ languages" (primary) — [GitHub boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio). Russian is reported to be in the "<5% WER/CER production-quality" tier of 102 languages (search snippet; vendor-reported intelligibility, says nothing about stress) — [HF bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b).

**OuteTTS 1.0**
- Russian is listed among "High Training Data Languages"; model needs a speaker reference, otherwise "generates random vocal characteristics, often leading to lower-quality outputs" (primary) — [HF OuteAI/Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B).

**MOSS-TTS-Realtime (new 2026, OpenMOSS)**
- 1.7B model supporting 20 languages incl. Russian; no per-language Russian WER published (only English/Chinese aggregate) (primary) — [MOSS-TTS realtime model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md). Note: it accepts incremental text (`session.push_text(delta)`), so strictly it belongs to the "native incremental text input" category rather than this one.

**Models with NO Russian (eliminated for this use case)**
- Dia2 (Nari Labs): English only, up to 2 min (search snippet) — [HF nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B).
- IndexTTS-2.5 (published 7 Jan 2026): "Chinese, English, Japanese, Spanish and Arabic" (primary) — [HF IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).
- Zonos v0.1: English, Japanese, Chinese, French, German; not trained on Russian even though eSpeak can phonemize it (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos). A downstream issue reports garbage output for Russian/Cyrillic when espeak-ng is missing (search snippet) — [CrispASR issue 435](https://github.com/CrispStrobe/CrispASR/issues/435).
- Spark-TTS: Chinese and English (search snippet) — [GitHub SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS); new-language request is an open issue — [Spark-TTS issue 50](https://github.com/SparkAudio/Spark-TTS/issues/50).
- MaskGCT: English, Chinese, Korean, Japanese, French, German (search snippet) — [arXiv 2409.00750](https://arxiv.org/abs/2409.00750).
- Parler-TTS Mini Multilingual: 8 European languages (en, fr, es, pt, pl, de, it, nl), no Russian (search snippet) — [HF parler-tts-mini-multilingual](https://huggingface.co/parler-tts/parler-tts-mini-multilingual).
- MeloTTS: official languages English, Spanish, French, Chinese, Japanese, Korean; Russian is only a user request (search snippet) — [MeloTTS issue 57](https://github.com/myshell-ai/MeloTTS/issues/57).

**Russian-origin companies**
- Sber GigaTTS (Habr, 21 Nov 2025): LLM-based decoder-only TTS with an XCodec-based tokenizer at 12.5 tokens/s; available in GigaChat voice mode; no open release announced; voice cloning explicitly withheld: "по понятным причинам мы не хотим отдавать эту фичу в публичный доступ" (primary) — [Habr/Sber 966640](https://habr.com/ru/companies/sberbank/articles/966640/). => not locally runnable.
- Yandex SpeechKit is a proprietary cloud API (sync/async/streaming) (primary) — [Habr/Raft 1023206](https://habr.com/ru/companies/raft/articles/1023206/).
- T-Bank's open model T-one is ASR for telephony, not TTS (search snippet) — [Habr/T-Bank 929850](https://habr.com/ru/companies/tbank/articles/929850/).
- TeraTTS (Tera2Space): small Russian VITS-family models with G2P/stress and ё handling (search snippet; 2023–2024 era project) — [GitHub Tera2Space/TeraTTS](https://github.com/Tera2Space/TeraTTS).

### Inferences
- For Russian, stress is the dividing line. Models that expose an explicit stress channel (Silero v5 built-in; F5-RU/ESpeech via RUAccent + `+`; Chatterbox V3 optional stresser) let you fix errors deterministically; end-to-end multilingual models (XTTS-v2, Higgs, OuteTTS, Fish) do not, so their stress errors cannot be corrected at the text level.
- Number/abbreviation normalization is not solved inside any of these models: Silero names it as its main open problem, and the F5 fine-tunes ship only RUAccent. A separate Russian text normalizer in front of the TTS is needed regardless of model choice.
- "Supports Russian" in a language list (Fish S2 Tier 2, Higgs v3, OuteTTS, MOSS) is vendor-reported and mostly WER-based; WER does not penalize wrong stress, so it is weak evidence of native-sounding Russian.
- ESpeech appears dormant as a TTS line (last TTS upload Aug 2025), while Misha24-10's F5 fine-tune is the actively used one (82k monthly downloads, newer `v4_winter` checkpoint).

### Gaps
- No Russian-specific evaluation found for Fish Audio S2, OpenAudio S1-mini, Higgs Audio v3, OuteTTS 1.0, Chatterbox Multilingual V3 (post-stresser) or MOSS-TTS-Realtime. No Reddit r/LocalLLaMA thread on Russian TTS surfaced in search (the search tool returned only aggregator pages).
- The ESpeech "Open TTS Leaderboard Ru" could not be read (dynamic Space). It is the most likely place for current Russian metrics and should be opened in a browser.
- ё handling: no source states behaviour per model, apart from TeraTTS and a third-party Silero wrapper mentioning ёфикация. Unanswered.
- StyleTTS2 Russian fine-tunes: search found only Ukrainian and unrelated StyleTTS2 fine-tunes; no established Russian StyleTTS2 checkpoint identified.
- Exact release dates of Chatterbox Multilingual V3 and of F5-TTS_RUSSIAN `v4_winter` not found (the HF card shows "09/04" for the original Multilingual announcement, which is Sep 2025; the fetch tool's "2024" is wrong).
- Telegram-channel comparisons were not reachable via search.

## 2. Streaming audio output, time-to-first-audio, RTF, behaviour on short chunks

### Takeaway
True chunk-while-generating audio exists for XTTS-v2 (`inference_stream`, <200 ms claimed), Fish Audio S2 (~100 ms TTFA but on an H200), Higgs v3, MOSS-TTS-Realtime (180 ms TTFB on L20) and the Chatterbox streaming fork (0.47 s on a 4090, English-only fork). F5-TTS and its Russian fine-tunes are not streaming models: "streaming" means sentence-chunk batches, so TTFA equals the full generation time of the first chunk. Silero is also non-streaming but so fast (250–350 s of audio per second on a 3090) that sentence chunking gives tens of milliseconds per sentence.

### Cited Findings
- Silero v5: 300–350 s of audio per second on GPU (RTX 3090) TTS-only, 250–300 with stress+homographs; CPU 1 thread 37–42, 4 threads 100–110 (Intel i9-10940X) (primary, 31 Oct 2025) — [Habr 961930](https://habr.com/ru/articles/961930/). CIS v5 models: "до 100 секунд аудио в секунду на CPU" (primary) — [Habr 968988](https://habr.com/ru/articles/968988/).
- Raft full-utterance synthesis time for one ~10–15 s Russian phrase on RTX 5070 (GPU) / Ryzen 7500F (CPU): Silero v5_ru 0.071 s GPU / 2.62 s CPU; F5-TTS RU 2.32 s / 128.83 s; ESpeech RL-V2 3.76 s / 3.85 s; QwenTTS 15 s / 40.49 s; XTTS-v2 18.54 s / 20.65 s; ChatterBox – / 31.63 s; HiggsAudio 50.36 s / 123.34 s (primary; these are non-streaming total times, not TTFA) — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/). CONFLICT: the mirror renders the same numbers as milliseconds — [ai-news.ru mirror](https://ai-news.ru/2026/02/open_source_protiv_proprietarshiny_obzor_tts_modelej_dlya_russkogo_yazyka.html); seconds is the plausible unit. The XTTS GPU≈CPU and ESpeech GPU≈CPU figures suggest those runs were not actually GPU-accelerated — treat as a quirk of that test.
- XTTS-v2: "XTTS can stream with <200ms latency" (primary, maintained fork README) — [GitHub idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS); `inference_stream()` yields audio chunks as generated (search snippet) — [Baseten blog](https://www.baseten.co/blog/streaming-real-time-text-to-speech-with-xtts-v2/). A 2026 arXiv paper is quoted as measuring XTTS-v2 native chunk streaming at TTFT 0.30 s (search snippet, source paper not identified precisely) — [F5-TTS search context / discussion 1003](https://github.com/SWivid/F5-TTS/discussions/1003).
- F5-TTS: "The latency to the first audio is the generation time of the first batch of texts"; `socket_server.py` streams by sending per-chunk audio (search snippet) — [F5-TTS discussion 1003](https://github.com/SWivid/F5-TTS/discussions/1003). User report (21 Nov 2025): first packet ≈2 s and RTF problems in a FastAPI wrapper using `infer_batch_process` with 50–200-char chunks; closed without a documented maintainer fix (primary) — [F5-TTS issue 1225](https://github.com/SWivid/F5-TTS/issues/1225).
- F5-TTS throughput: RTF 0.0394, average latency 253 ms on one L20 with Triton + TensorRT-LLM at concurrency 2 (search snippet of upstream README numbers) — [F5-TTS discussion 1003](https://github.com/SWivid/F5-TTS/discussions/1003); RTF 0.14 on RTX 4060 Ti 16 GB, 0.10 on RTX 3090 (search snippet, third-party benchmark site) — [gigagpu](https://gigagpu.com/tts-latency-benchmarks/).
- ESpeech reference script defaults: NFE 48, cross-fade 0.15 s between chunks (primary) — [HF ESpeech-TTS-1_RL-V2](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2).
- Chatterbox: official repo documents no streaming (primary) — [GitHub resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox). Streaming fork: `generate_stream` yields audio chunks (default 50 speech tokens); first-chunk latency 0.472 s, RTF 0.499 on RTX 4090; examples English only, no multilingual mentioned (primary) — [GitHub davidbrowne17/chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming). devnen server: `stream=true` "is chunk-level, not token-level" — splits on sentence boundaries and flushes WAV bytes per synthesized chunk (primary) — [GitHub devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server). An "OpenAI-compatible multilingual server with ~0.7 s time-to-first-byte" is mentioned (search snippet, repo not identified) — [GitHub topic chatterbox](https://github.com/topics/chatterbox).
- Fish Audio S2 Pro: RTF 0.195, time-to-first-audio ~100 ms, 3,000+ acoustic tokens/s, measured on a single NVIDIA H200 with the SGLang-based streaming engine (primary) — [HF fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro).
- Higgs Audio v3: streaming supported; OpenAI-compatible API; self-hosting via SGLang-Omni; no latency numbers in the repo (primary) — [GitHub boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio); background — [LMSYS blog, 4 Jun 2026](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/).
- MOSS-TTS-Realtime: TTFB 180 ms after warm-up, RTF 0.51 on a single L20 with SDPA + torch.compile; streaming audio out and incremental text in (primary) — [MOSS-TTS realtime model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md).
- Zonos: ~2x real time on RTX 4090; streaming not documented in README (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos).
- OuteTTS 1.0: 150 tokens per second of audio with the new encoder; streaming not documented (primary) — [HF Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B).

### Inferences
- For sentence-chunked LLM voicing, TTFA = (time to collect first clause from the LLM) + (time to synthesize it). With Silero that second term is negligible even on a weak GPU or CPU; with F5-RU at NFE 32–48 it is roughly RTF × clause length (about 0.1–0.2 × a 3–5 s clause on a 3090/4060 Ti, i.e. a few hundred ms), and much more on a 1050 Ti-class card.
- Published TTFA numbers for Fish S2 (H200), MOSS (L20), Chatterbox fork (4090) are on hardware far above a GTX 1050 Ti and cannot be transferred to it.
- Short-chunk prosody: only indirect evidence. F5-family models cross-fade independently generated chunks (0.15 s default), so inter-sentence intonation is not carried over; Silero likewise synthesizes each sentence independently. No source quantified short-text artifacts for Russian.

### Gaps
- No measured TTFA/RTF for any of these models on GTX 1050 Ti or on DGX Spark (GB10).
- No first-party RTF for XTTS-v2 on a specific GPU was retrieved (only the "<200 ms" claim without hardware).
- Short-clause artifacts (hallucinated tails on very short inputs for F5/XTTS/Chatterbox) are widely discussed anecdotally, but I did not retrieve a citable source; treat as unverified.
- Whether any streaming fork of Chatterbox supports the Multilingual V3 model was not confirmed.

## 3. VRAM, GTX 1050 Ti (4 GB) fit, aarch64 CUDA (DGX Spark / Jetson)

### Takeaway
On 4 GB only Silero clearly fits with large headroom; F5-TTS-class models (336M params, 1.35 GB fp32 weights) are reported to run on a 4 GB card but with little margin; the large LLM-based models (Fish S2 Pro 5B, Higgs v3 4B, IndexTTS-2.5, Zonos) do not fit. I found no source documenting any of these models on aarch64 CUDA.

### Cited Findings
- Silero v5 Russian model is ~140 MB (primary) — [Habr 961930](https://habr.com/ru/articles/961930/); "consume 500MB of GPU" (search snippet, third-party) — [silero_openai_tts](https://github.com/ndrco/silero_openai_tts).
- F5-TTS v1 Base: 335.8M params, 1.35 GB fp32 weights; "8GB VRAM runs inference"; reported to run on a 4 GB RTX 2050 (search snippet, third-party guide) — [nexgpu F5-TTS](https://nexgpu.net/en/models/f5-tts/). ESpeech RL-V2 checkpoint file is 2.7 GB (search snippet) — [HF ESpeech-TTS-1_RL-V2](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2).
- Fish Audio S2 Pro: 5B parameters (BF16); VRAM not stated on the card (primary) — [HF fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro). OpenAudio S1-mini: 0.5B params (search snippet) — [HF openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini).
- Higgs Audio v3 4B: known-good on 1× A100 40 GB with SGLang-Omni; GGUF quantizations q4_k_m 8 GB … q8_0 12 GB, bf16 16 GB (search snippet) — [HF bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b).
- IndexTTS-2.5: "roughly 6 GB of VRAM for inference" (primary) — [HF IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).
- Zonos: "6GB+ VRAM"; hybrid variant needs a 3000-series or newer NVIDIA GPU (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos).
- MOSS-TTS-Realtime 1.7B: ~4 GB VRAM in bf16, ~2 GB in 4-bit (search snippet) — [MOSS-TTS GitHub](https://github.com/OpenMOSS/MOSS-TTS).
- Chatterbox Multilingual: 0.5B Llama backbone; VRAM not stated (primary) — [HF ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox); server supports NVIDIA CUDA, AMD ROCm, Apple MPS, CPU fallback (primary) — [devnen server](https://github.com/devnen/Chatterbox-TTS-Server).
- Higgs v3 also has an MLX-Audio path for Apple Silicon (search snippet) — [HF bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b).

### Inferences
- GTX 1050 Ti verdict by weight size: Silero — yes (trivially; also fine on CPU); F5-TTS RU / ESpeech — probably yes in fp32 (weights ~1.35 GB + vocoder + activations), but slow, since a 1050 Ti is several times slower than the 3090/4060 Ti used in published RTF figures and diffusion needs 32–48 NFE; XTTS-v2 and Chatterbox Multilingual (0.5B) — plausibly fit but unverified; S1-mini (0.5B) — plausible but unverified; MOSS-TTS-Realtime — borderline (4 GB bf16, and Pascal cards have no fast bf16/fp16 path, so fp32 would not fit; 4-bit might); Fish S2 Pro, Higgs v2/v3, IndexTTS-2.5, Zonos — no.
- The GB10 (121 GB unified memory) removes the VRAM constraint for every model here; the open question there is only software: PyTorch aarch64 CUDA wheels and each project's native dependencies (e.g. SGLang, TensorRT, espeak-ng, custom CUDA ops). Pure-PyTorch models (Silero, F5-TTS, XTTS, Chatterbox) are the lowest-risk on aarch64; this is reasoning, not a verified fact.
- If the 4 GB card is shared with the project's existing STT model, co-residency, not just fit, is the constraint — this favours Silero (≈0.5 GB or CPU).

### Gaps
- No source found for XTTS-v2, Chatterbox Multilingual V3, S1-mini VRAM figures (commonly quoted community numbers exist but were not retrieved).
- No source found for any of these models running on DGX Spark / Jetson / other aarch64 CUDA. Entirely unverified.
- No measurement on Pascal-generation GPUs for any model.

## 4. Licenses (code and weights) — commercial use

### Takeaway
Russian-capable models with weights clearly usable commercially: ESpeech-TTS-1 (Apache-2.0), Chatterbox Multilingual (MIT), MOSS-TTS family (Apache-2.0), and only the MIT "base" CIS variants of Silero. Non-commercial: Silero v5 Russian (CC BY-NC), Misha24-10 F5-TTS_RUSSIAN (CC-BY-NC-4.0), XTTS-v2 (Coqui Public Model License), OpenAudio S1-mini (CC-BY-NC-SA-4.0), Fish Audio S2 (Fish Audio Research License), Higgs v3 (research/non-commercial), OuteTTS 1.0 (CC-BY-NC-SA-4.0 components), IndexTTS-2.5 (bilibili model license).

### Cited Findings
- Silero: "All of the models are published under the main repo license (i.e. CC-NC-BY) except for the `base` cis-tts models, which are under MIT" (primary) — [silero-models GitHub](https://github.com/snakers4/silero-models). CIS release: the model built on Silero's own data is MIT, the extended one CC-NC-BY (primary) — [Habr 968988](https://habr.com/ru/articles/968988/). A third-party project describes the v5.5 Russian model as CC BY-NC-SA 4.0 (search snippet; conflicts slightly with the repo's "CC-NC-BY" wording — both are non-commercial) — [ruvoice-tts](https://github.com/kost-t-human/ruvoice-tts). Raft: free use effectively only for educational purposes; a commercial license is sold by Silero — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- Misha24-10/F5-TTS_RUSSIAN: `cc-by-nc-4.0` (primary) — [HF](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN).
- ESpeech-TTS-1_RL-V2: `apache-2.0` (primary) — [HF](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2).
- XTTS-v2 weights: "Coqui Public Model License" (primary) — [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2); code of the maintained fork: MPL-2.0 (primary) — [GitHub idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS).
- Chatterbox (all variants): MIT; every output carries Resemble's Perth perceptual watermark (primary) — [GitHub resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox).
- OpenAudio S1-mini: CC-BY-NC-SA-4.0 (search snippet) — [HF openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini). Fish Audio S2 Pro: "Fish Audio Research License. Research and non-commercial use is permitted free of charge. Commercial use requires a separate license from Fish Audio." (primary) — [HF fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro).
- Higgs Audio v3: code Apache 2.0; weights "Boson Higgs Audio v3 Research and Non-Commercial License" (primary) — [GitHub boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio).
- IndexTTS-2.5: "bilibili-model-license" (primary) — [HF IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).
- Zonos: Apache-2.0 (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos).
- OuteTTS 1.0 (1B): CC-BY-NC-SA-4.0 for custom components (primary) — [HF Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B).
- MOSS-TTS / MOSS-TTSD: Apache 2.0 (search snippet) — [MOSS-TTS LICENSE](https://github.com/OpenMOSS/MOSS-TTS/blob/main/LICENSE).

### Inferences
- The best-rated Russian options split on license: Silero v5_ru and Misha24-10's F5 are non-commercial; the commercially clean Russian-capable set is ESpeech-TTS-1 (Apache-2.0 card), Chatterbox Multilingual (MIT) and MOSS-TTS (Apache-2.0).
- ESpeech's Apache-2.0 label deserves a caveat: it is an F5-TTS-architecture model, and upstream F5-TTS pretrained weights are CC-BY-NC (stated in the task brief; not re-verified here). Whether ESpeech was trained from scratch (clean) or initialized from the NC upstream checkpoint was not determined from the card.
- For a purely private/internal assistant the NC licenses may be acceptable; for anything sold, they are not.

### Gaps
- The exact commercial-use wording of the Coqui Public Model License was not retrieved (the card excerpt did not include it); the "non-commercial" characterization comes from the task brief and general knowledge, plus the fact that Coqui (the licensor able to sell commercial licenses) is described as unmaintained/defunct.
- Upstream SWivid/F5-TTS weight license not re-fetched.
- License of hotstone228/F5-TTS-Russian and of Misha24-10 `v4_winter` specifically not checked (card-level license is CC-BY-NC-4.0).
- Exact terms of the Boson v3 and Fish Audio research licenses (e.g. internal business use) not read.

## 5. Maturity, maintenance in 2026, ready-made servers

### Takeaway
Actively developed in 2026: Silero (v5.x through at least v5_5_ru, Habr updates through spring 2026), Chatterbox (Multilingual V3, Turbo, Nano), Fish Audio (S2, Mar 2026), Boson (Higgs v3, mid-2026), OpenMOSS, and the Misha24-10 F5 fine-tune. XTTS-v2 is frozen: the original Coqui repo is unmaintained and lives on only through the idiap fork. ESpeech has not shipped a TTS model since Aug 2025. OpenAI-compatible `/v1/audio/speech` servers exist off the shelf for Chatterbox and Silero; F5-TTS ships only a socket server and a Triton/TensorRT recipe.

### Cited Findings
- XTTS / Coqui: idiap/coqui-ai-TTS is a "Fork of the original, unmaintained repository. New PyPI package: coqui-tts"; tested on Ubuntu 24.04, Python ≥3.10,<3.15, PyTorch 2.2+ (primary) — [GitHub idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS).
- Silero: release cadence v5 (31 Oct 2025) → CIS v5 (24 Nov 2025) → questions update (Mar 2026 by ID order) → v5_5_ru in the repo table (primary) — [Habr 961930](https://habr.com/ru/articles/961930/); [Habr 968988](https://habr.com/ru/articles/968988/); [Habr 1015942](https://habr.com/ru/articles/1015942/); [silero-models](https://github.com/snakers4/silero-models).
- Silero servers: `ndrco/silero_openai_tts` implements OpenAI `POST /v1/audio/speech` and an ElevenLabs-compatible API, CPU by default with optional CUDA (search snippet) — [GitHub ndrco/silero_openai_tts](https://github.com/ndrco/silero_openai_tts).
- Chatterbox servers: devnen/Chatterbox-TTS-Server v2.0.0 — hot-swappable Original / Multilingual / Turbo engines, `/v1/audio/speech` and `/v1/audio/voices`, sentence-boundary chunking, chunk-level streaming, CUDA/ROCm/MPS (primary) — [GitHub devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server). Chatterbox Multilingual is also packaged by NVIDIA as a hosted model card (search snippet) — [build.nvidia.com](https://build.nvidia.com/resembleai/chatterbox-multilingual-tts/modelcard).
- F5-TTS: `socket_server.py` chunk streaming and a Triton + TensorRT-LLM deployment contributed by Yuekai Zhang (search snippet) — [F5-TTS discussion 1003](https://github.com/SWivid/F5-TTS/discussions/1003). Raft rated F5-RU integration 1/5 and ESpeech 2/5 — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- Fish Audio S2: release includes weights, fine-tuning code and an SGLang-based streaming inference engine (continuous batching, paged KV cache, CUDA graph replay, prefix caching) (primary) — [HF fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro); announcement — [Fish Audio blog](https://fish.audio/blog/fish-audio-open-sources-s2/).
- Higgs Audio v3: repo tells users not to clone it but to use the hosted API or download weights; OpenAI-compatible API via SGLang-Omni; vLLM recipe exists (primary + search snippet) — [GitHub boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio); [vLLM recipes](https://recipes.vllm.ai/bosonai/higgs-audio-v3-tts-4b).
- Zonos: Dockerfile and docker-compose in repo; 82 commits; still v0.1 (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos).
- ESpeech: last TTS uploads 25 Aug 2025; RL-V2 had 44 downloads/month at fetch time vs 81,687 for Misha24-10's F5 (primary) — [HF ESpeech org](https://huggingface.co/ESpeech); [HF Misha24-10](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN).

### Inferences
- The lowest-integration-risk path for "voice an LLM stream in Russian locally" is Silero v5 behind an OpenAI-compatible wrapper with your own sentence splitter; the highest-quality cloning path (F5-RU) requires writing your own chunking/streaming server.
- XTTS-v2 remains usable but is a dead-end technically (no new weights since Coqui's shutdown), and its Russian stress cannot be fixed.

### Gaps
- Latest release tags/dates for idiap/coqui-ai-TTS, SWivid/F5-TTS, fishaudio/fish-speech, resemble-ai/chatterbox in 2026 were not retrieved.
- No ready-made Docker image / OpenAI-compatible server found specifically for the F5 Russian fine-tunes or ESpeech.
- WebSocket-streaming servers: none verified for any model in scope other than F5's raw socket server.
- Company status of Zyphra (Zonos) and Nari Labs in 2026 not checked (both irrelevant for Russian).

## 6. Voice cloning vs fixed voices; need for reference audio

### Takeaway
Silero is the only fixed-voice option (5 Russian speakers, no reference, no cloning). Everything else in scope is zero-shot cloning that requires a reference clip; F5-family models additionally require the reference transcript, and for multilingual models the reference must itself be Russian or the output inherits a foreign accent.

### Cited Findings
- Silero v5_ru: fixed speakers aidar, baya, kseniya, xenia, eugene (primary) — [silero-models](https://github.com/snakers4/silero-models).
- ESpeech / F5-RU: reference audio (≤12 s recommended) and its text are mandatory inputs (primary) — [HF ESpeech-TTS-1_RL-V2](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2).
- XTTS-v2: cloning from a 6-second clip, cross-language (primary) — [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2).
- Chatterbox Multilingual: zero-shot cloning via `audio_prompt_path`; "Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language" (primary) — [HF ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).
- Higgs Audio v3: preset voices and zero-shot cloning (primary) — [GitHub boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio).
- OuteTTS 1.0: designed to be used with a speaker reference; without one quality drops (primary) — [HF Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B).
- Zonos: 10–30 s speaker sample (primary) — [GitHub Zyphra/Zonos](https://github.com/Zyphra/Zonos). IndexTTS-2.5: single reference clip (primary) — [HF IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).
- Raft: QwenTTS CustomVoice preset voices have a "небольшой акцент" in Russian because the base speakers are not Russian — the same mechanism applies to preset voices of other multilingual models (primary) — [Habr/Raft](https://habr.com/ru/companies/raft/articles/991844/).
- Sber deliberately does not release voice cloning (primary) — [Habr/Sber 966640](https://habr.com/ru/companies/sberbank/articles/966640/).

### Inferences
- For an assistant voice, cloning models need one curated Russian reference clip (clean, native speaker, with exact transcript for F5). Its quality sets the ceiling for accent and prosody; this is a one-off setup cost, not a per-request one, and speaker conditioning can be cached.
- Fixed voices (Silero) give the most stable result across short chunks because there is no per-chunk re-conditioning drift in timbre.

### Gaps
- Whether Fish S2 / S1-mini ship usable built-in Russian preset voices was not determined.
- Speaker-embedding caching support per server implementation was not verified.
