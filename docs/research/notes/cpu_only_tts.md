# CPU-only local TTS for voicing a streaming LLM answer in Russian (state as of 2026-09-30)

Method note: repo metadata (licenses, push dates, releases, READMEs, PR/issue state, HF model cards, PyPI metadata) was pulled directly from the GitHub / Hugging Face / PyPI APIs on 2026-09-30 — these are marked **[verified 2026-09-30]**. Items marked **[community claim]** or **[self-reported]** come from a single third party or the model author and were not independently reproduced. No benchmark was run by the researcher; all speed figures are quoted.

Short verdict per candidate (details in the sections below):

| Candidate | Russian? | CPU real-time? | Native incremental text in? | Chunked audio out? | Commercial use |
|---|---|---|---|---|---|
| Silero v5 (`v5_*_ru`) | official, best-known | yes (36x RT on 1 thread) | no | no (whole utterance) | **No** (CC BY-NC) |
| Silero `v5_cis_base` (`ru_*` speakers) + silero-stress | official | yes | no | no | **Yes (MIT)** |
| Piper (piper1-gpl) ru_RU denis/dmitri/irina/ruslan | official voices | yes | sentence-level (Wyoming) | per sentence; intra-sentence only in open PR #302 | code GPL-3; voices unclear/mixed |
| Kokoro-82M official | **no** | — | — | — | Apache-2.0 |
| kokoro-ru (community, zaakirio) | community, new (Jul–Aug 2026) | ~9.8x RT (self-reported) | no | no | OpenRAIL weights |
| Vosk-TTS ru-0.9-multi | official | yes (numbers not found) | no | no | Apache-2.0 |
| Supertonic 3 | official (1 of 31 langs) | yes | no | no | OpenRAIL-M; **project archived Sept 2026** |
| sherpa-onnx (runtime) | via Piper ru voices, MMS-rus, Supertonic 3 | yes | no | callback while generating | Apache-2.0 runtime |
| RHVoice | official, many voices | yes (parametric) | — | — | engine LGPL/GPL; most voices NC |
| eSpeak NG | yes (robotic) | yes | — | — | GPL-3 |
| MMS-TTS rus | yes | yes | no | no | **No** (CC-BY-NC 4.0) |
| Pocket TTS, KittenTTS, NeuTTS, MeloTTS, Matcha-TTS | **no Russian** | — | — | — | — |
| Coqui VITS | no Russian single-language model | — | — | — | — |
| OmniVoice (k2-fsa, 2026) | yes (600+ langs) | "compute-heavy" on CPU | no | no | Apache-2.0 repo |

---

## 1. Russian support and quality (official vs community; stress, ё, homographs, numbers)

### Takeaway
Only a handful of CPU-class engines actually speak Russian: Silero (the only one with a purpose-built Russian stress + homograph front end), Piper (4 voices, espeak-ng phonemization with known Russian stress/phoneme weaknesses), Vosk-TTS, Supertonic 3 (stress errors acknowledged in its own discussions), RHVoice/eSpeak NG (parametric/formant), MMS (no stress modelling), plus a brand-new community Kokoro port. Pocket TTS, KittenTTS, NeuTTS, MeloTTS, Matcha-TTS and official Kokoro have **no Russian** as of Sept 2026. Every neural option needs external number/abbreviation normalization; most also benefit from an external accentor (silero-stress or RUAccent).

### Cited Findings

**Silero TTS**
- Current Russian models: `v5_ru`, `v5_2_ru`, `v5_3_ru`, `v5_4_ru`, `v5_5_ru` (speakers `aidar`, `baya`, `kseniya`, `xenia`, `eugene`; `v5_4_ru` lacks `eugene`). All have automatic stress + homograph resolution; `v5_4_ru` and `v5_5_ru` additionally "support questions" (interrogative intonation). `v4_ru` (same speakers + `random`, auto-stress yes) is still listed. [verified 2026-09-30] — [silero-models README](https://github.com/snakers4/silero-models)
- `v5_ru` was released 2025-10-30 (release v5.0: "v5_ru TTS models released with homographs, higher quality and improved speed"; legacy tools/models deprecated); v1/v2 models removed in v5.2 (2025-11-22). [verified 2026-09-30] — [silero-models releases](https://github.com/snakers4/silero-models/releases)
- `v5_cis_base` / `v5_cis_base_nostress` (released 2025-11-22, 20 languages of Russia/CIS) include Russian speakers with a `ru_` prefix. They contain **no built-in auto-stress or homographs**: `v5_cis_base` expects a `+` stress mark on every word (`к+ошка`); `_nostress` expects it only for Slavic languages (ru/bel/ukr). [verified 2026-09-30] — [silero-models README](https://github.com/snakers4/silero-models)
- Silero's Habr announcement (2025-11-24): 95 voices in 20 languages, SSML supported, separate `silero-stress` project with accentors for Russian and Ukrainian. — [Habr 968988](https://habr.com/ru/articles/968988/)
- UTMOS for Russian: `v5_cis_base` 3.04 ± 0.02, `v5_cis_base_nostress` 2.97 ± 0.02, vs 3.08 ± 0.01 for the original recordings (Silero's own measurement, undated wiki page). [self-reported] — [Silero wiki: speed and quality](https://github.com/snakers4/silero-models/wiki/%D0%9F%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D0%B0%D1%8F-%D0%B4%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F-%D0%BF%D0%BE-%D1%81%D0%BA%D0%BE%D1%80%D0%BE%D1%81%D1%82%D0%B8-%D0%B8-%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D1%83-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B)
- **silero-stress** (separate pip package `silero-stress`, last push 2026-09-15): covers ~4M Russian words/word forms "with 100% accuracy", ~2.2K homographs with F1 0.92 / 94% accuracy, 60–70% accuracy on unknown/invented words; restores `ё` (example output `Л+ёва Корол+ёв`); flags to customise stress / `ё` placement; user regex/phrases can be attached to the homograph solver. [self-reported metrics; verified text 2026-09-30] — [silero-stress README](https://github.com/snakers4/silero-stress)
- SSML (`<speak>`, `<break>`, `<prosody rate/pitch>`, `<p>`, `<s>`) is supported via the `ssml_text` parameter; manual stress with `+` inside SSML/plain text. — [Silero SSML wiki](https://github.com/snakers4/silero-models/wiki/SSML)
- Independent evaluation (older — 2024-07-12, by the Vosk author): Silero v3.1 was among the top models for clarity (ASR-measured CER) among Russian TTS, "though with compromised intonation". — [alphacephei.com, Evaluation of Russian TTS models](https://alphacephei.com/nsh/2024/07/12/russian-tts.html)
- Third-party Silero servers add their own numeral expansion and noun agreement ("21 рубль / 22 рубля / 25 рублей"), and state dates/times/abbreviations need further rules. — [ndrco/silero_openai_tts](https://github.com/ndrco/silero_openai_tts); a Home Assistant community server likewise "added morphology support for number-to-text conversion in Russian and Ukrainian" (Feb 2026) — [HA community thread](https://community.home-assistant.io/t/silero-tts-on-gpu-cpu-external-install/989464)

**Piper (rhasspy/piper → OHF-Voice/piper1-gpl)**
- Four official Russian voices, all `medium`, single-speaker, 22,050 Hz: `ru_RU-denis-medium`, `ru_RU-dmitri-medium`, `ru_RU-irina-medium`, `ru_RU-ruslan-medium`; all "finetuned from U.S. English lessac voice (medium quality)". [verified 2026-09-30] — [piper-voices ru_RU model cards](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU)
- Phonemization is done by embedded espeak-ng; no Russian-specific accentor or text normalizer ships with Piper. — [piper1-gpl README](https://github.com/OHF-Voice/piper1-gpl)
- Community project "Work on improving Piper for Russian" (mitrokun, last push 2026-09-18): ships a replacement Russian espeak-ng dictionary "with added iotated phonemes and an option to set stress via U+0301 … necessary for adequate operation of the phonemizer for piper"; the upstream espeak-ng PR #2234 was not accepted because of its impact on existing projects; a March 2026 update reworked phonemes to fix "аканье" and other defects, which "slightly broke synthesis on existing models" (needs fine-tuning). I.e. explicit stress marks only work with the replaced dictionary and models (re)trained on it. [community] — [mitrokun/espeak-ng-data](https://github.com/mitrokun/espeak-ng-data)
- 2024 evaluation (older): Piper `irina` got CER 1.4 and UTMOS 3.672 — "exceptional results … despite minimal training data". — [alphacephei.com 2024-07-12](https://alphacephei.com/nsh/2024/07/12/russian-tts.html)
- kokoro-ru author's Whisper large-v3 round-trip (Aug 2026, 79-word set — tiny): Piper's four Russian voices average 4.38% WER; "Piper's irina transcribes perfectly". [self-reported by a competitor] — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- Habr (2026-04-02), Repka-Pi 4 (ARM64 SBC): Piper rated clearly better than eSpeak NG ("robot voice") and acceptable for smart-home/kiosk use; no RTF numbers, stress not discussed. — [Habr 1016060](https://habr.com/ru/articles/1016060/)

**Kokoro-82M**
- Official model card lists `language: en` and ships voices for American/British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin — no Russian. `hexgrad/kokoro` GitHub repo's last push was 2025-08-06. [verified 2026-09-30] — [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
- Issue "Have you plans to add russian tts?" is open since 2025-02-10 with only "up!" comments through 2026-05-21; no maintainer commitment. [verified 2026-09-30] — [hexgrad/kokoro#76](https://github.com/hexgrad/kokoro/issues/76)
- **Community Russian port `zaakirio/kokoro-ru`** (created 2026-07-29, updated 2026-08-25; 5.4k downloads, 10 likes): 81.8M params, 3 voices (sveta, masha — female; dima — male, separate checkpoint), 24 kHz, fine-tuned on 29.28 h / 16 speakers. Front end `ru_g2p.py` uses RUAccent for stress, ё and homographs + Russian vowel reduction. Self-reported: 2.50% WER vs Piper 4.38% (79-word set), stress/orthoepy 15/16, homographs 93.8% (known failure "Это была настоящая мука"). Stated limitations: darker timbre (sveta −33.8 dB in 6–10 kHz), fricative artifacts on ж/ш/х, deterministic prosody. [self-reported, no independent review found] — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- The same card notes that with plain espeak-ng G2P instead of RUAccent, WER on a stress-heavy 55-word set rises (~27% vs ~22%, Whisper base). — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- `igorshmukler/kokoro-ruslan` is a Russian *training pipeline* on the Kokoro architecture (search result only; no evidence of released production weights). — [kokoro-ruslan](https://github.com/igorshmukler/kokoro-ruslan)

**Vosk-TTS (alphacep)**
- "Simple TTS based on VITS with some old ideas"; current model `vosk-model-tts-ru-0.9-multi` (bumped to 0.9 on 2025-07-10), 5 Russian voices (3 female, 2 male; speaker IDs 0–4). "We plan to add more voices and languages in the future." [verified 2026-09-30] — [alphacep/vosk-tts](https://github.com/alphacep/vosk-tts)
- HF card (checkpoint = 0.7-multi, last modified 2024-08-03): voices F01 (Tiflocomp Irina), F02 (trained on Natasha from Sova), F03/M01/M02 (artificial voices); ships a `dictionary` file (lexicon-based pronunciation); finetunable to a custom voice. — [alphacep/vosk-tts-ru-multi](https://huggingface.co/alphacep/vosk-tts-ru-multi)
- 2024 evaluation by the Vosk author himself (bias caveat): Vosk-TTS 0.6 and the GPT variant "outperformed XTTS2 significantly"; multi-voice systems "seriously suffer from fuzziness" (CER > 2.0). — [alphacephei.com 2024-07-12](https://alphacephei.com/nsh/2024/07/12/russian-tts.html)
- Older community assessment (NtechLab Habr article, late 2024, via search summary): Vosk-TTS runs on Raspberry Pi but Russian quality is "average"; Piper "about the quality of Mimic 3". [community claim, older] — [Habr ntechlab 854724](https://habr.com/ru/companies/ntechlab/articles/854724/)

**Supertonic 3 (Supertone)**
- Released 2026-04-29 with 31 languages including Russian (`ru`) and Ukrainian; Supertonic 2 (2026-01-06) had only 5 languages (no Russian). [verified 2026-09-30] — [supertonic README](https://github.com/supertone-oss-archive/supertonic)
- Vendor WER on Minimax-MLS-test, Russian: Supertonic 3 **3.99**, VoxCPM2 3.31, OmniVoice 4.53, Qwen3-TTS 4.48. [self-reported] — [supertonic README](https://github.com/supertone-oss-archive/supertonic)
- Open discussion (2026-05-28): "Russian language TTS model misplaces word stress on certain words" (multi-syllable words, homographs, loanwords, proper nouns). A user reply (2026-08-13): the model honours the combining acute accent U+0301, and a ~159-word replacement dictionary made pronunciation "consistently correct" for that user. [community claim] — [Supertone/supertonic-3 discussion #23](https://huggingface.co/Supertone/supertonic-3/discussions/23)
- й→и and ё→е confusion reported 2026-05-07; Supertone staff (2026-05-11) traced part of it to the Python package stripping combining marks after NFKD normalization and said it was fixed in the SDK. — [discussion #2](https://huggingface.co/Supertone/supertonic-3/discussions/2)
- Built-in text handling of numbers/currency/phone numbers reported as unreliable by a user even in English (2026-06-02, open). [community claim] — [discussion #24](https://huggingface.co/Supertone/supertonic-3/discussions/24)

**sherpa-onnx (runtime) — Russian-capable models it ships**
- Russian TTS sample page lists exactly: `vits-piper-ru_RU-denis-medium`, `-dmitri-medium`, `-irina-medium`, `-ruslan-medium`, and `supertonic-3-ru`. — [sherpa-onnx Russian TTS samples](https://k2-fsa.github.io/sherpa/onnx/tts/all/Russian/index.html)
- `tts-models` release assets [verified 2026-09-30] also include `vits-mms-rus.tar.bz2` (108 MB, 2023-12-14), the four `vits-piper-ru_RU-*-medium` archives (~67 MB each, re-uploaded 2025-12-02), `sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2` (129 MB), `sherpa-onnx-pocket-tts-*` (English), Kokoro `multi-lang-v1_0/v1_1` (Chinese+English etc., no Russian), KittenTTS `*-en-*` only, Matcha `zh-baker`, `en_US-ljspeech`, `zh-en`, `fa_en` only. — [sherpa-onnx tts-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
- Docs list supported families: VITS (incl. Piper, MeloTTS), Matcha, Kokoro, KittenTTS, PocketTTS, SupertonicTTS, ZipVoice, MMS. — [sherpa-onnx TTS docs](https://k2-fsa.github.io/sherpa/onnx/tts/index.html)

**RHVoice**
- Russian voices: Aleksandr, Aleksandr-hq, Anna, Arina, Artemiy, Elena, Evgeniy-rus, Irina, Mikhail, Pavel, Seva, Tatiana, Timofey, Umka, Victoria, Vitaliy, Vitaliy-ng, Vsevolod, Yuriy (+ newer repos dasha, lyudmila, michal, ryhor). Latest release 1.18.4 (2026-03-31), repo pushed 2026-09-28. [verified 2026-09-30] — [RHVoice org repos](https://github.com/orgs/RHVoice/repositories), [RHVoice releases](https://github.com/RHVoice/RHVoice/releases)

**eSpeak NG**
- Formant synthesizer; described in a 2026-04-02 Habr article as a "robot voice" compared with Piper. Repo active (pushed 2026-09-22). — [Habr 1016060](https://habr.com/ru/articles/1016060/), [espeak-ng](https://github.com/espeak-ng/espeak-ng)

**MMS-TTS (`facebook/mms-tts-rus`)**
- Per-language VITS checkpoint, available in Transformers ≥ 4.33; last modified 2023-09-01 (old, unmaintained). — [facebook/mms-tts-rus](https://huggingface.co/facebook/mms-tts-rus)

**No Russian as of Sept 2026**
- **Kyutai Pocket TTS** (v3.3.0, 2026-09-24): "Multi-language support: english, french, german, portuguese, italian, spanish. Additional languages may be added in the future." The Feb 2026 roadmap issue did not list Russian. [verified 2026-09-30] — [pocket-tts README](https://github.com/kyutai-labs/pocket-tts), [issue #118](https://github.com/kyutai-labs/pocket-tts/issues/118)
- **KittenTTS** (0.8.1, 2026-02-24): English models only; "Release multilingual TTS" is still an unchecked roadmap item. [verified 2026-09-30] — [KittenTTS README](https://github.com/KittenML/KittenTTS)
- **NeuTTS** (Air / Nano / 2E, Neuphonic): "Supported Languages: English, Spanish, German, French (model-dependent)". [verified 2026-09-30] — [neuphonic/neutts](https://github.com/neuphonic/neutts)
- **MeloTTS**: README language table has English variants, Spanish, French, Chinese, Japanese, Korean — no Russian; last push 2024-12-24 (stale). [verified 2026-09-30] — [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- **Coqui TTS** (maintained fork `idiap/coqui-ai-TTS`, pushed 2026-06-10): the `.models.json` `tts_models` catalogue has keys be, bg, uk, … but **no `ru`** single-language model; Russian only via `multilingual` models (XTTS etc.). [verified 2026-09-30] — [idiap/coqui-ai-TTS .models.json](https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/.models.json)

**Large / GPU-class Russian models (out of CPU scope, for context)**
- F5-TTS-Russian (hotstone228) and ESpeech-TTS-1 (F5-TTS DiT, dim 1024, depth 22; uses RUAccent; Apache-2.0; Aug 2025) are diffusion models; one report gives "2 minutes generation time for a 9-second sample" on CPU. [secondary sources] — [hotstone228/F5-TTS-Russian](https://huggingface.co/hotstone228/F5-TTS-Russian), [ESpeech/ESpeech-TTS-1_RL-V2](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2)
- "XTTS-2 and F5-TTS technically run on CPU but produce audio 30-50x slower than real time". [secondary, vendor blog] — [spheron.network](https://www.spheron.network/blog/self-host-voice-cloning-gpu-cloud-xtts-f5-tts-openvoice-v2/)
- **OmniVoice** (k2-fsa, repo created 2026-03-31, 14k stars): zero-shot TTS for 600+ languages incl. Russian, "RTF as low as 0.025" (hardware not stated on the README lines read — presumably GPU). An int4 ONNX CPU backend exists experimentally inside wyoming-piper, which warns: "OmniVoice is compute-heavy and best suited to a desktop/server CPU rather than low-power devices". — [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice), [wyoming-piper README](https://github.com/OHF-Voice/wyoming-piper)

**Sber / Yandex / T-Bank**
- No open-weights TTS from these vendors was found. Sber's 2025–2026 open releases are GigaChat LLMs and GigaAM-v3 (speech *recognition*), MIT-licensed. — [iol.co.za report on Sber release](https://www.iol.co.za/technology/europes-largest-open-source-ai-release-sber-unveils-breakthrough-russian-neural-networks-8608f4c2-dc33-4217-bf9e-bce3b00ae6d6)

### Inferences
- For Russian, the practical split is: **Silero = best stress/homograph handling out of the box** (v5_*_ru built in; or MIT `v5_cis_base` + MIT `silero-stress`); **Piper = weakest linguistic front end** (espeak-ng rules, no homograph resolution), good audio for `irina`; **Supertonic 3 = highest fidelity (44.1 kHz) but grapheme-based with stress errors**, fixable by feeding pre-accented text (U+0301) — so an external accentor (silero-stress/RUAccent output converted to U+0301) is a plausible combination, untested here.
- None of the neural engines reads digits, dates, abbreviations or Latin-script words reliably in Russian; an LLM-output pipeline needs its own normalizer (numbers → words with case agreement) before the accentor. The evidence is indirect (third-party Silero servers add numeral expansion; Supertonic text-handling complaint; Piper relies on espeak-ng number rules).
- `silero-stress` accuracy on unknown words (60–70%) means LLM answers with neologisms, brand names and English loanwords will still have audible stress errors.
- kokoro-ru is 2 months old with one author and tiny self-evaluation; treat as experimental.

### Gaps
- No 2026 independent, side-by-side listening test (MOS) of Silero v5 vs Piper vs Supertonic 3 vs Vosk 0.9 vs kokoro-ru for Russian was found; the only systematic comparison is the July 2024 alphacephei post (pre-v5 Silero, Vosk 0.6).
- Silero Telegram channel (`t.me/silero_news`) announcements for `v5_4_ru`/`v5_5_ru` were not retrieved; exact release dates of those two models are unknown.
- Whether Silero v5 handles digits/Latin internally was not confirmed from a primary source.
- Vosk-TTS 0.9 quality vs 0.7 and its behaviour on stress/homographs: no source found.
- TeraTTS HF repos (`TeraTTS/natasha-g2p-vits`, `glados2-g2p-vits`) exist but were last modified 2023-10 and currently have no README ("Entry not found"); status/quality unknown — [TeraTTS on HF](https://huggingface.co/TeraTTS).
- StyleTTS2-lite / Russian StyleTTS2 ONNX models other than kokoro-ru: nothing citable found.
- Matcha-TTS: no public Russian checkpoint found (only negative evidence from sherpa-onnx's model list).

---

## 2. Streaming: incremental text input, chunked audio output, TTFA and RTF on CPU

### Takeaway
No Russian-capable CPU engine accepts token-by-token text natively; all are sentence-level (or whole-utterance) synthesizers, so the LLM stream must be split into sentences/clauses by the caller. Silero is fast enough (36x real time on one x86 thread) that per-sentence synthesis lands well under 300 ms on a desktop CPU; Piper is ~5–8x real time on 1–2 ARM server cores and gets intra-sentence audio chunking only through a still-unmerged PR (137 ms TTFA). Pocket TTS — the one model designed for true streaming (~200 ms first chunk) — has no Russian.

### Cited Findings

**Silero**
- Throughput of V5 CIS models on Intel Core i9-10940X @ 3.30 GHz: **1 thread → 36.48 ± 0.54 s of audio per second; 4 threads → 85.72 ± 1.90** (1/RTF; undated Silero wiki, published with the Nov 2025 release). [self-reported] — [Silero wiki](https://github.com/snakers4/silero-models/wiki/%D0%9F%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D0%B0%D1%8F-%D0%B4%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F-%D0%BF%D0%BE-%D1%81%D0%BA%D0%BE%D1%80%D0%BE%D1%81%D1%82%D0%B8-%D0%B8-%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D1%83-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B)
- Habr (2025-11-24): "up to 100 seconds of audio per second on CPU", 20–25% faster than the previous version; CPU not specified. — [Habr 968988](https://habr.com/ru/articles/968988/). Release v5.6 notes (2026-06-04): "under ideal conditions they should work x100 RT". — [silero-models releases](https://github.com/snakers4/silero-models/releases)
- A third-party repackaging claims RTF ≈ 0.04 (24x real time) on an unspecified CPU for `v5_cis_base`. [community claim] — [Dimitrius174/silero-v5-tts-ru](https://huggingface.co/Dimitrius174/silero-v5-tts-ru)
- silero-stress cost: ~0.5 ms per ordinary word, ~30 ms per 400-character sentence with two homographs, on 1 CPU thread. [self-reported] — [silero-stress README](https://github.com/snakers4/silero-stress)
- API is `model.apply_tts(text=… | ssml_text=…, speaker, sample_rate)` returning a complete audio tensor; the README and the Habr article do not describe any streaming/chunked output. — [silero-models README](https://github.com/snakers4/silero-models), [Habr 968988](https://habr.com/ru/articles/968988/)

**Piper**
- Python API: "For streaming, use `PiperVoice.synthesize`" — a generator yielding audio chunks (`chunk.audio_int16_bytes`); CLI `--output-raw` streams raw PCM. In the released version chunks are per sentence. — [piper1-gpl API_PYTHON.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md), [CLI.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md)
- **PR #302 "Add low-latency streaming synthesis through split VITS ONNX graphs"** — opened 2026-09-10, still **open / not merged** on 2026-09-30 [verified]. Splits existing voices into encoder + decoder graphs (no retraining: `python -m piper.split voice.onnx`), decodes overlapping latent chunks; `PiperVoice.load(..., streaming=True)` + `synthesize_stream()`, CLI `--output-raw --stream`. Author's measurements: Lessac voice, ~12 s sentence, "local CPU" (unspecified): time to first audio **296 ms → 137 ms**; `en_US-kristin-medium`: first byte 0.19 s → 0.05 s (short sentences), 0.37 s → 0.09 s (longer). Limitations: needs the complete sentence text; encoder still runs over the whole text; no whole-sentence peak normalization. [PR author's numbers, English voices] — [piper1-gpl PR #302](https://github.com/OHF-Voice/piper1-gpl/pull/302)
- wyoming-piper: "streaming audio on sentence boundaries" added in 1.6.0; since 2.0.0 streaming is on by default and sentence splitting uses the `sentence-stream` library (so a Wyoming client can send text incrementally and get audio per sentence). Latest 2.5.2 (2026-09-11). [verified 2026-09-30] — [wyoming-piper CHANGELOG](https://github.com/OHF-Voice/wyoming-piper/blob/main/CHANGELOG.md)
- CPU speed, third-party (AI-authored repo, 0 stars, corrections log — low confidence; French voices; piper-tts 1.8.0, onnxruntime 1.30; 2-core ARM Neoverse-N1): `fr_FR-siwis-medium` ×8.1–8.5 real time at 2 threads, ×5.07 at 1 thread; `fr_FR-tom-medium` ×4.4–4.6 at 2 threads, ×2.70 at 1 thread; splitting into more sentences costs Piper nothing. (Sept 2026) [community claim] — [obole-ia/tts-cpu-benchmark](https://github.com/obole-ia/tts-cpu-benchmark)
- Other secondary figures (undated aggregator summaries): Piper RTF ≈ 0.19–0.2; short sentence on Raspberry Pi 5 "as low as 0.54 s"; a vendor benchmark (Picovoice, who sell a competing engine) reports a first-token-to-speech figure of 1,720 ms for Piper in an LLM pipeline. [unverified, snippets only] — [cekura.ai Piper overview](https://www.cekura.ai/discover/piper-tts), [Picovoice TTS latency benchmark](https://picovoice.ai/docs/benchmark/tts/)

**Kokoro / kokoro-ru**
- kokoro-ru: RTF 0.102 (9.8x real time) "on a laptop CPU" (model not specified), peak RAM 2.39 GB. [self-reported] — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- Third-party benchmark of (English) Kokoro-82M on 2 ARM Neoverse-N1 cores: ×0.87 real time at 2 threads, ×0.52 at 1 thread, and ~0.51 s fixed cost per call, so sentence splitting costs ~8% throughput. (Sept 2026; low-confidence source, see above) [community claim] — [obole-ia/tts-cpu-benchmark](https://github.com/obole-ia/tts-cpu-benchmark)

**Supertonic 3**
- Non-autoregressive flow-matching pipeline (duration predictor, text encoder, vector estimator, vocoder ONNX graphs) with a configurable number of steps; README says it "runs fast on CPU, even compared with larger baselines measured on A100 GPU" but gives the numbers only as an image; an e-reader demo (Onyx Boox Go 6) reports average RTF 0.3. — [supertonic README](https://github.com/supertone-oss-archive/supertonic)
- A secondary write-up quotes "1,200+ characters per second on Apple Silicon CPU". [secondary] — [mer.vin, June 2026](https://mer.vin/2026/06/supertonic-3-99m-on-device-tts-with-onnx-31-languages-and-1200-chars-sec/)
- "Streaming" discussion (2026-05-24, "Is there a plan to support streaming?") is open and unanswered. — [Supertone/supertonic-3 discussion #20](https://huggingface.co/Supertone/supertonic-3/discussions/20)

**sherpa-onnx**
- `python-api-examples/offline-tts-play.py`: "Different from ./offline-tts.py, this file plays back the generated audio while the model is still generating" — i.e. a generation callback delivers audio pieces before the whole text is done. Separate `pocket-tts-play.py`, `zipvoice-tts-play.py`, `supertonic-tts.py` examples exist. [verified 2026-09-30] — [offline-tts-play.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts-play.py)
- A third-party FastAPI project exposes sherpa-onnx TTS over WebSocket, returning PCM16 chunks with progress. — [ruzhila/voiceapi](https://github.com/ruzhila/voiceapi)

**Pocket TTS (no Russian — reference point for true streaming)**
- "Audio streaming; low latency, ~200 ms to get the first audio chunk; ~6x real-time on a CPU of MacBook Air M4; uses only 2 CPU cores; can handle infinitely long text inputs"; no GPU speed-up observed. — [pocket-tts README](https://github.com/kyutai-labs/pocket-tts)

**NeuTTS (no Russian)**
- LLM-backbone models; streaming only with GGUF builds; backbone throughput (Q4 GGUF): 119 tokens/s (Air) / 221 tokens/s (Nano) on AMD Ryzen 9 HX 370 CPU, excluding the codec. — [neuphonic/neutts](https://github.com/neuphonic/neutts)

**Reference architecture numbers for sentence-chunked LLM → TTS**
- An arXiv tutorial (March 2026) on voice agents: sentence buffer accumulates LLM tokens until a boundary, each sentence goes to TTS immediately; measured LLM TTFT 296 ms + sentence detection 143 ms + TTS 316 ms = 755 ms TTFA (cloud/GPU stack, not these engines). — [arXiv 2603.05413](https://arxiv.org/html/2603.05413v1)

### Inferences
- **Silero, sentence-level chunking, x86 desktop:** at 36x real time on 1 thread (i9-10940X), a 5-second sentence takes ≈ 140 ms plus ≈ 5–30 ms for silero-stress; at 4 threads ≈ 60 ms. So "<300 ms to first audio after the first sentence is complete" is realistic on a modern x86 CPU even without intra-sentence streaming; the dominant delay becomes waiting for the LLM to finish the first sentence (mitigate by cutting at the first clause/comma).
- **Piper, sentence-level chunking:** with ~5–8x real time on 1–2 ARM server cores, a 5-second sentence takes ≈ 0.6–1.0 s — above 300 ms unless the first chunk is a short clause (~1.5 s of audio) or PR #302's split-graph streaming is used (137 ms reported on an unspecified CPU). On a fast x86 desktop the PR author's *baseline* of 190–300 ms per sentence suggests Piper already sits near the 300 ms line there.
- **kokoro-ru:** at RTF ~0.1 on a laptop CPU a 5-second sentence is ≈ 0.5 s, and Kokoro has a sizeable fixed per-call cost, so it misses 300 ms with sentence chunking; on small ARM cores it may not reach real time at all.
- **Supertonic 3:** no streaming; if the quoted throughput holds, short sentences would synthesize in well under 300 ms on Apple-Silicon-class CPUs, but no citable per-sentence latency for x86/ARM servers was found.
- sherpa-onnx's callback most likely fires per generated sentence/batch for VITS-type models rather than mid-sentence (researcher's prior knowledge of `max_num_sentences`; not re-verified in the sources read).
- The Silero "100x" and wiki "36x / 86x" figures are compatible only if "100x" assumes more threads/ideal conditions; treat 36x (1 thread) as the planning number.

### Gaps
- No Russian-specific TTFA measurements (ms) for any engine on a named CPU were found; all latency figures above are for English/French voices or are derived from RTF.
- Silero on ARM64/aarch64: no speed numbers found.
- Vosk-TTS RTF/latency on CPU: not found (only the statement that it runs on Raspberry Pi).
- Supertonic 3 numeric RTF table for x86 CPUs: only published as an image; not extracted.
- Whether sherpa-onnx callbacks give intra-sentence chunks for Piper/Supertonic models was not confirmed.
- RHVoice/eSpeak NG latency: not researched (known to be effectively instantaneous, but no source collected).

---

## 3. Resource footprint (model size, RAM, threads, ARM64, ONNX)

### Takeaway
Everything in scope is small (25–330 MB). Piper, Supertonic 3, Kokoro(-ru), MMS and KittenTTS are ONNX and run on x86_64 and aarch64; Silero is a PyTorch (TorchScript) package requiring AVX2 on x86, with no official ONNX export found.

### Cited Findings
- **Silero**: "Minimal system requirements: a PyTorch-compatible system, a modern processor with AVX2 instruction set for x86/64 platform"; `pip install silero` (PyPI 0.5.5); `v5_2_ru` "removes numpy and scipy dependencies"; v5.1 removed torchaudio. — [silero-models README](https://github.com/snakers4/silero-models), [silero on PyPI](https://pypi.org/project/silero/)
- **silero-stress**: total package ≈ 50 MB; needs python 3.10+, 1 GB+ RAM, `torch>=1.12.0`, CPU with AVX/AVX2/AVX-512/AMX (x86-64 requirements as stated). — [silero-stress README](https://github.com/snakers4/silero-stress)
- **Piper**: each Russian `medium` voice is a single 63.2 MB ONNX file + JSON config; PyPI `piper-tts` 1.8.0 ships abi3 wheels for manylinux x86_64, **manylinux aarch64**, macOS x86_64/arm64, Windows amd64; espeak-ng is embedded; `--cuda` optional. Legacy binaries exist for amd64/arm64/armv7. [verified 2026-09-30] — [piper-voices ru_RU](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU), [piper-tts on PyPI](https://pypi.org/project/piper-tts/), [CLI.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md)
- **kokoro-ru**: 81.81M params; ONNX fp32 326 MB, q8 138 MB, fp16 164 MB (WebGPU only); peak RAM 2.39 GB; loads with stock `kokoro`, transformers.js / kokoro-js / onnxruntime. — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- **Supertonic 3**: ~99M params across four ONNX graphs (`duration_predictor`, `text_encoder`, `vector_estimator`, `vocoder`); ONNX Runtime incl. onnxruntime-web; batch inference; sherpa-onnx int8 bundle is 129 MB (tar.bz2). — [supertonic README](https://github.com/supertone-oss-archive/supertonic), [sherpa-onnx tts-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
- **sherpa-onnx**: runs on Windows/macOS/Linux and embedded boards (Raspberry Pi, Jetson, RK3588…) with bindings for 12 languages plus WebAssembly (as described in the Pocket TTS README's integration list); `pip install sherpa-onnx`. — [pocket-tts README](https://github.com/kyutai-labs/pocket-tts), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **Vosk-TTS**: `pip3 install vosk-tts` (0.3.61, pure-Python wheel, 2025-08-08); model downloaded as `vosk-model-tts-ru-0.9-multi.zip`. — [alphacep/vosk-tts](https://github.com/alphacep/vosk-tts), [vosk-tts on PyPI](https://pypi.org/project/vosk-tts/)
- **MMS rus**: sherpa-onnx ONNX bundle 108 MB. — [sherpa-onnx tts-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
- **Pocket TTS** (no Russian): 100M params, 2 CPU cores; sherpa-onnx int8 bundle 98 MB. — [pocket-tts README](https://github.com/kyutai-labs/pocket-tts)
- **KittenTTS** (no Russian): 15M–80M params, 25–80 MB, ONNX, CPU. — [KittenTTS README](https://github.com/KittenML/KittenTTS)
- **Piper on ARM64 SBC**: works on Repka-Pi 4 (ARM64, from 2 GB RAM, no GPU) per Habr 2026-04-02. — [Habr 1016060](https://habr.com/ru/articles/1016060/)

### Inferences
- For an aarch64 deployment the lowest-friction Russian options are Piper / sherpa-onnx (official aarch64 wheels, ONNX). Silero should work wherever PyTorch has aarch64 wheels, but this is not stated by Silero and its perf numbers are x86-only.
- Silero forces a PyTorch dependency (hundreds of MB of CPU torch) into the image; ONNX engines need only onnxruntime.

### Gaps
- Silero v5 model file sizes and RAM usage: not found in README/Habr article.
- Vosk-TTS 0.9 model size, sample rate, ONNX vs PyTorch runtime: not confirmed (the HF 0.7 repo contains `.pth` training checkpoints).
- RAM figures for Piper and Supertonic 3 in a long-running server: not found.

---

## 4. Licenses of code and weights; commercial use

### Takeaway
The license picture is the main discriminator: the best Russian models (Silero `v5_*_ru`) are non-commercial; the commercially clean Silero path is `v5_cis_base` (MIT) + `silero-stress` (MIT). Piper's code is GPL-3 and its Russian voices have unresolved provenance questions (lessac fine-tune; irina "Unknown"; ruslan CC BY-NC-SA). Supertonic 3 and kokoro-ru weights are OpenRAIL(-M) (commercial allowed with use restrictions); Vosk-TTS is Apache-2.0; MMS is CC-BY-NC.

### Cited Findings
- **Silero models**: "All of the models are published under the main repo license (i.e. CC-NC-BY) except for the `base` cis-tts models, which are under MIT." `v5_cis_base` / `v5_cis_base_nostress`: MIT (`LICENSE_CIS`). `v5_cis_ext`: CC-NC-BY. [verified 2026-09-30] — [silero-models README](https://github.com/snakers4/silero-models)
- Inconsistency: the README badge says "CC BY-NC 4.0", but the repository `LICENSE` file text is "Attribution-NonCommercial-**ShareAlike** 4.0 International", and GitHub reports the license as `NOASSERTION`. [verified 2026-09-30] — [silero-models LICENSE](https://github.com/snakers4/silero-models/blob/master/LICENSE)
- The `silero` pip package is classified "OSI Approved :: MIT License" (release v5.4, 2026-01-30: "Change pip-package license") — this covers the loader code, not the non-`base` model weights. — [silero on PyPI](https://pypi.org/project/silero/), [releases](https://github.com/snakers4/silero-models/releases)
- Habr (2025-11-24): the `base` model "is published under the MIT license"; the `ext` version under CC-NC-BY. — [Habr 968988](https://habr.com/ru/articles/968988/)
- **silero-stress**: MIT; "no telemetry, no keys, no registration, no built-in expiration". — [silero-stress README](https://github.com/snakers4/silero-stress)
- **Piper code**: `OHF-Voice/piper1-gpl` is GPL-3.0 (PyPI: `GPL-3.0-or-later`); the old `rhasspy/piper` (MIT) was archived (last push 2025-08-26). [verified 2026-09-30] — [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl), [rhasspy/piper](https://github.com/rhasspy/piper)
- **Piper Russian voices** (model cards): denis — dataset CC0; dmitri — dataset CC0; irina — dataset from RHVoice, "License: **Unknown**"; ruslan — RUSLAN corpus, **CC BY-NC-SA 4.0**. All four fine-tuned from `en_US-lessac-medium`. The HF repo `rhasspy/piper-voices` is tagged `mit`. [verified 2026-09-30] — [piper-voices ru_RU](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU)
- Open issue #314 (2026-09-28, no maintainer reply as of 2026-09-30): asks whether denis/dmitri may be used commercially, given that they are fine-tuned from lessac, whose Blizzard 2013 data license "excludes commercial speech-product development", while the cards say CC0 and the repo says MIT. **Unresolved.** — [piper1-gpl issue #314](https://github.com/OHF-Voice/piper1-gpl/issues/314)
- **wyoming-piper**: MIT (but depends on GPL piper-tts ≥ 1.8.0). — [wyoming-piper](https://github.com/OHF-Voice/wyoming-piper)
- **Kokoro**: code and official weights Apache-2.0. — [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). **kokoro-ru**: "Weights are OpenRAIL, code is Apache-2.0"; voices from the "Dialogs corpus (studio, consented actors, OpenRAIL)". — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- **Vosk-TTS**: repo Apache-2.0; HF model card `license: apache-2.0`. — [alphacep/vosk-tts](https://github.com/alphacep/vosk-tts), [alphacep/vosk-tts-ru-multi](https://huggingface.co/alphacep/vosk-tts-ru-multi)
- **Supertonic**: sample code MIT; model "released under the OpenRAIL-M License". A user question "Can we launch it as a paid TTS service" exists (discussion #19, 2026-05-19). — [supertonic README](https://github.com/supertone-oss-archive/supertonic), [Supertone/supertonic-3 discussions](https://huggingface.co/Supertone/supertonic-3/discussions)
- **sherpa-onnx**: Apache-2.0 runtime (model licenses follow the models). — [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **RHVoice**: library LGPL v2.1+, but with MAGE the combination is GPL v3+ (can be built without MAGE). "All voices from RHVoice Lab's site are distributed under CC BY-NC-ND 4.0"; commercial integration requires a request to the lab. Individual Russian voice repos mostly carry no license metadata (dasha-rus: CC-BY-SA-4.0). — [RHVoice License.md](https://github.com/RHVoice/RHVoice/blob/master/doc/en/License.md), [RHVoice org](https://github.com/orgs/RHVoice/repositories)
- **eSpeak NG**: GPL-3.0. — [espeak-ng](https://github.com/espeak-ng/espeak-ng)
- **MMS-TTS rus**: CC-BY-NC 4.0 (non-commercial). — [facebook/mms-tts-rus](https://huggingface.co/facebook/mms-tts-rus)
- **F5-TTS-Russian**: CC BY-NC-SA 4.0. [secondary] — [aimodels.fyi summary](https://www.aimodels.fyi/models/huggingFace/f5-tts-russian-hotstone228)
- Pocket TTS MIT code (voices have individual licenses, prohibited-use clause); KittenTTS Apache-2.0; NeuTTS-Air Apache 2.0, NeuTTS-Nano/2E "NeuTTS Open License 1.0"; MeloTTS MIT; OmniVoice repo Apache-2.0; idiap coqui-ai-TTS MPL-2.0. [verified 2026-09-30] — [pocket-tts](https://github.com/kyutai-labs/pocket-tts), [KittenTTS](https://github.com/KittenML/KittenTTS), [neutts](https://github.com/neuphonic/neutts), [MeloTTS](https://github.com/myshell-ai/MeloTTS), [OmniVoice](https://github.com/k2-fsa/OmniVoice), [coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS)

### Inferences
- Commercially usable, Russian, CPU-fast combinations as of Sept 2026: (a) Silero `v5_cis_base` + `silero-stress` (both MIT) — but without the built-in homograph model of `v5_*_ru` (silero-stress supplies homographs separately); (b) Vosk-TTS (Apache-2.0; voice F02 derives from Sova's Natasha dataset — dataset license not checked); (c) Supertonic 3 under OpenRAIL-M use restrictions; (d) Piper denis/dmitri only if the lessac-initialization question is resolved and GPL-3 for the engine is acceptable (running piper as a separate network service/process keeps GPL obligations away from the calling application's code, but distributing a Docker image containing piper triggers GPL source-offer duties — legal interpretation, not verified).
- Silero `v5_*_ru` (the ones with built-in stress/homographs/questions) are non-commercial; for internal/personal use they are fine, commercial use needs a license from Silero.
- "ShareAlike vs not" ambiguity in Silero's repo is irrelevant for the MIT models but matters for anyone redistributing the NC ones.

### Gaps
- Exact OpenRAIL-M use-restriction text of Supertonic 3 was not read; "commercial use allowed with restrictions + attribution" comes from secondary summaries ([toknow.ai](https://toknow.ai/posts/supertonic-3-on-device-tts-99m-parameters-31-languages/)).
- License of RUAccent (needed by kokoro-ru and ESpeech front ends) not checked.
- Licenses of the individual classic RHVoice Russian voices (Anna, Elena, Irina, Aleksandr…) are not stated in repo metadata; not resolved.
- Vosk-TTS training data licenses per voice not verified.

---

## 5. Maturity: repo activity in 2026, packaging, ready-made servers

### Takeaway
Silero, Piper, sherpa-onnx and Pocket TTS are actively released in 2026; Piper is explicitly "looking for maintainers"; Vosk-TTS is in slow maintenance; **Supertonic was archived in September 2026 ("development and support have ended")**; official Kokoro and MeloTTS are stale. Ready-made servers: Piper has an official HTTP server and Wyoming server; Silero has only community OpenAI-compatible/Wyoming servers; Supertonic's SDK has an OpenAI-compatible `serve`.

### Cited Findings
- **Silero**: releases v5.0 (2025-10-30), v5.2 (2025-11-22, CIS models), v5.4 (2026-01-30), v5.5 (2026-02-03), v5.6 (2026-06-04, SAPI5 for Windows screen readers); last push 2026-07-31 (added Turkic + Caucasian languages); 6.1k stars. Distribution: `pip install silero`, `torch.hub`. No official server. [verified 2026-09-30] — [silero-models releases](https://github.com/snakers4/silero-models/releases)
- Community Silero servers: `ndrco/silero_openai_tts` (MIT, 4 stars, last push 2026-02-25): OpenAI TTS API + ElevenLabs-compatible API (`/v1/text-to-speech/{voice_id}/stream` is a compat alias), wav/mp3/opus/aac/flac output, RU/EN numeral normalization, file cache — [repo](https://github.com/ndrco/silero_openai_tts); `NW15D/silero-api-server` (OpenAI-format API + Wyoming for Home Assistant, Feb 2026) — [HA community thread](https://community.home-assistant.io/t/silero-tts-on-gpu-cpu-external-install/989464)
- **silero-stress**: 161 stars, last push 2026-09-15, `pip install silero-stress`. — [silero-stress](https://github.com/snakers4/silero-stress)
- **Piper**: `OHF-Voice/piper1-gpl` releases v1.6.0 (2026-07-23, Hebrew phonemizer), v1.6.1 (2026-08-13, removed torch dependency for zh), v1.7.0 (2026-08-15, Japanese via OpenJTalk), v1.8.0 (2026-09-04, Thai); last push 2026-09-28; 5.7k stars. README banner: "**Looking for Maintainers** — The Open Home Foundation is looking for maintainers for Piper!" Packaging: `pip install piper-tts`; `python3 -m piper.http_server` (port 5000; `POST /synthesize` returns WAV; `/voices`, `/info`; web UI); C/C++ `libpiper`; training docs. [verified 2026-09-30] — [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl), [API_HTTP.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md)
- **wyoming-piper** v2.5.2 (2026-09-11): Docker images (CPU and GPU Dockerfile), Wyoming protocol with sentence-boundary streaming on by default, voice-management web UI, experimental OmniVoice backend; voices.json updated with "13 new voices, 11 updated" in 2.5.1. — [wyoming-piper CHANGELOG](https://github.com/OHF-Voice/wyoming-piper/blob/main/CHANGELOG.md)
- `rhasspy/piper-voices` HF repo last modified 2026-09-17. — [piper-voices](https://huggingface.co/rhasspy/piper-voices)
- **sherpa-onnx**: v1.13.8 (2026-09-10), releases every ~2 weeks, 15k stars; pip package, prebuilt binaries/APKs, Python examples for TTS incl. play-while-generating. No official OpenAI-compatible TTS endpoint was found in the pages read. — [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **Vosk-TTS**: GitHub releases stop at 0.3.59 (2025-05-05); PyPI 0.3.61 (2025-08-08); 2026 commits are only test/synth scripts for other engines (last push 2026-06-06); repo has a gRPC `server/` (`tts_server.py`, `tts_service.proto`); 274 stars. [verified 2026-09-30] — [alphacep/vosk-tts](https://github.com/alphacep/vosk-tts)
- **Supertonic**: README now opens with "This repository is archived. Development and support have ended … No updates, bug fixes, security patches, or support will be provided"; code and weights moved to the `supertone-oss-archive` GitHub org / HF namespace (archive HF repo created 2026-09-08; GitHub repo archived, last push 2026-09-09); "This archive does not include hosted demos or Voice Builder services"; the Python SDK is to be installed from a pinned git commit. Before archiving: PyPI `supertonic` (2025-12-10), SDK v1.3.1 (2026-05-18) added `supertonic serve` with native `/v1/tts` and **OpenAI-compatible `/v1/audio/speech`**; a `pipecat-supertonic` integration was announced by a community member. 13.8k stars. [verified 2026-09-30] — [supertonic README](https://github.com/supertone-oss-archive/supertonic), [discussions](https://huggingface.co/Supertone/supertonic-3/discussions)
- **Kokoro**: `hexgrad/kokoro` last push 2025-08-06; `remsky/Kokoro-FastAPI` (OpenAI-compatible server, Apache-2.0, 5.5k stars) last push 2026-09-10 — serves official voices (no Russian). — [hexgrad/kokoro](https://github.com/hexgrad/kokoro), [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
- **Pocket TTS** (no Russian): v3.0.2 → v3.3.0 within Aug–Sept 2026, `pocket-tts serve` HTTP server, Docker image, several community OpenAI-compatible streaming servers. — [pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- **RHVoice**: 1.18.1–1.18.4 released Feb–Mar 2026; packaged in Arch (`rhvoice-language-russian 1.18.4`), NVDA add-on, Android (F-Droid). — [RHVoice releases](https://github.com/RHVoice/RHVoice/releases), [Arch package](https://archlinux.org/packages/extra-testing/x86_64/rhvoice-language-russian/)
- **MeloTTS**: last push 2024-12-24. **MMS-TTS rus**: last modified 2023-09-01. — [MeloTTS](https://github.com/myshell-ai/MeloTTS), [facebook/mms-tts-rus](https://huggingface.co/facebook/mms-tts-rus)

### Inferences
- Supertonic 3 remains usable (weights and ONNX code are archived under their licenses, and sherpa-onnx carries an int8 build), but it is now a frozen artifact: the Russian stress issue will not be fixed upstream and the Voice Builder (custom voices) is gone. This is a material risk for building on it.
- Piper's "looking for maintainers" notice plus an unanswered licensing issue suggest limited upstream capacity, even though releases continue.
- No engine ships an official OpenAI-compatible `/v1/audio/speech` server with Russian support except the (now archived) Supertonic SDK; for Silero/Piper/Vosk a thin custom wrapper (or a community one) is needed.

### Gaps
- Why Supertone archived the project (no announcement found beyond the README notice); whether the `supertonic` PyPI package and the original `Supertone/supertonic-3` HF repo will stay available.
- Whether an official Silero Docker image or server exists (none found).
- Silero Telegram/Habr posts from 2026 were not retrieved beyond release notes.

---

## 6. Voices, sample rate, voice cloning

### Takeaway
Silero offers the widest Russian voice choice (5 classic voices in `v5_*_ru`, dozens of `ru_*` speakers in the MIT CIS model) at up to 48 kHz; Piper has 4 voices at 22.05 kHz; Vosk-TTS 5 voices; Supertonic 3 has 10 language-agnostic preset styles at 44.1 kHz. None of the fast CPU Russian engines offers local zero-shot voice cloning — custom voices mean fine-tuning (Piper, Vosk) — while the CPU cloning models (Pocket TTS, NeuTTS) lack Russian.

### Cited Findings
- **Silero `v5_*_ru`**: speakers `aidar`, `baya`, `kseniya`, `xenia`, `eugene`; sample rates 8000 / 24000 / 48000 Hz; SSML prosody (rate, pitch) and breaks. — [silero-models README](https://github.com/snakers4/silero-models), [SSML wiki](https://github.com/snakers4/silero-models/wiki/SSML)
- **Silero `v5_cis_base`**: all CIS speakers are also available as Russian-speaking voices with a `ru_` prefix (cross-lingual; Habr: voices speak non-native languages with an accent); 95 voices / 20 languages overall; 8/24/48 kHz. A third-party card counts "29 Russian voices" (e.g. `ru_eduard`, `ru_aigul`, `ru_ekaterina`, `ru_alexandr`). — [silero-models README](https://github.com/snakers4/silero-models), [Habr 968988](https://habr.com/ru/articles/968988/), [Dimitrius174/silero-v5-tts-ru](https://huggingface.co/Dimitrius174/silero-v5-tts-ru) [community claim for the count]
- **Piper**: 4 Russian voices (3 male: denis, dmitri, ruslan; 1 female: irina), single-speaker, medium quality, 22,050 Hz; synthesis knobs `length_scale`, `noise_scale`, `noise_w_scale`, volume; raw phoneme injection `[[ … ]]`; training/fine-tuning docs for new voices; no zero-shot cloning. — [piper-voices ru_RU](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU), [API_PYTHON.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md)
- A community note says Piper `irina` "has a stuttering issue". [community claim, from search summary] — [smartliving.ru Piper add-on page](https://connect.smartliving.ru/addons/category4/312.html)
- **Vosk-TTS**: 5 voices (3 F, 2 M) in one multi-speaker model; "easily finetuned to any voice"; separate `alphacep/vosk-vc-ru` voice-conversion model exists. — [alphacep/vosk-tts-ru-multi](https://huggingface.co/alphacep/vosk-tts-ru-multi), [alphacep/vosk-vc-ru](https://huggingface.co/alphacep/vosk-vc-ru)
- **kokoro-ru**: 3 fixed voices (sveta, masha, dima), 24 kHz, no cloning. — [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru)
- **Supertonic 3**: 10 preset voice styles (F1–F5, M1–M5 JSON files), 44.1 kHz 16-bit output, expression tags; "This open-weight repository focuses on fixed-voice, local TTS and does not include an official voice-cloning pipeline"; custom voices came from the hosted Voice Builder, which the archive no longer provides. — [supertonic README](https://github.com/supertone-oss-archive/supertonic), [supertone-oss-archive/supertonic-3 files](https://huggingface.co/supertone-oss-archive/supertonic-3)
- **RHVoice**: ~19+ Russian voices (see section 1). — [RHVoice org](https://github.com/orgs/RHVoice/repositories)
- **Pocket TTS** (no Russian): voice cloning from a wav; **NeuTTS** (no Russian): instant voice cloning; **OmniVoice** (Russian, heavy): zero-shot cloning from 3–10 s reference + voice design. — [pocket-tts](https://github.com/kyutai-labs/pocket-tts), [neutts](https://github.com/neuphonic/neutts), [OmniVoice](https://github.com/k2-fsa/OmniVoice)

### Inferences
- If a specific custom Russian voice is required on CPU, the realistic routes are fine-tuning Piper (GPL tooling; mitrokun's improved Russian phoneme set) or Vosk-TTS, or using the OmniVoice int4 ONNX backend on a strong desktop CPU (latency unknown).
- 48 kHz (Silero) and 44.1 kHz (Supertonic) outputs are higher fidelity than Piper's 22.05 kHz / Kokoro's 24 kHz; for telephony Silero's native 8 kHz avoids resampling.

### Gaps
- Exact list/count of `ru_*` speakers in `v5_cis_base` was not enumerated from `models.yml`.
- Vosk-TTS 0.9 output sample rate not confirmed.
- No quality comparison of Silero cross-lingual `ru_*` voices vs the native `v5_*_ru` voices other than Silero's UTMOS figure (3.04).
- OmniVoice CPU RTF/latency for Russian: no measurement found.
