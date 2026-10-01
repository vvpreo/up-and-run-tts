# Orchestration / serving layer for voicing a streaming LLM answer with local TTS, and Russian text preprocessing (state as of 2026-09-30)

Conventions used in these notes:
- **[V]** = verified by fetching the primary page on 2026-09-30 (repo README, official docs, issue page, article).
- **[S]** = seen only in a search-result snippet of the cited page on 2026-09-30; the page itself was not opened or the fetch did not show that passage. Treat as probable, not confirmed.
- **[C]** = community / vendor self-claim (blog, vendor benchmark of own product); not independently verified.
- Dates are given for every claim where the source exposes one. "Older" info is flagged explicitly.

---

## 1. Techniques: turning token-by-token LLM text into low-latency speech when the TTS only accepts complete text

### Takeaway
The de-facto standard in every open-source stack (RealtimeTTS, Pipecat, LiveKit, Wyoming/Home Assistant, vLLM-Omni) is the same: buffer LLM tokens, cut at sentence (optionally clause) boundaries with a small lookahead, synthesize each piece as a separate request, and play the results in order through a queue. The tunables that matter are a shorter/earlier *first* fragment, a minimum fragment length (~10 characters), a forced flush after N words or a timeout, and a markdown/emoji/link filter before the splitter; the first-chunk size can be derived from the engine's real-time factor.

### Cited Findings

**Boundary aggregation and lookahead**
- [V] `stream2sentence` (the splitter inside RealtimeTTS) defaults: `context_size=12` characters examined for boundary detection, `context_size_look_overhead=12`, `minimum_sentence_length=10`, `minimum_first_fragment_length=10`, `force_first_fragment_after_words=15`, `sentence_fragment_delimiters=".?!;:,\n…)]}"`, `full_sentence_delimiters=".?!\n…"`, tokenizer options `nltk`, `stanza`, `rule-based`, `nltk+rule-based`; `quick_yield_single_sentence_fragment`, `quick_yield_for_all_sentences`, `quick_yield_every_fragment` (all default False) emit a clause-level fragment early; `auto_context` yields safe boundaries before the full context delay; `cleanup_text_links` / `cleanup_text_emojis` strip links and emoji. License MIT. — [KoljaB/stream2sentence](https://github.com/KoljaB/stream2sentence)
- [V] `stream2sentence` also has a time-based strategy `generate_sentences_time_based` with `target_tps=4` (tokens/s), `lead_time=1` (seconds of buffer before output), `max_wait_for_fragments=[3, 2]`, `min_output_lengths=[2, 3, 3, 4]` (minimum words per emitted piece) — i.e. a deadline-driven flush instead of pure punctuation. — [KoljaB/stream2sentence](https://github.com/KoljaB/stream2sentence)
- [V] RealtimeTTS README states the default is "stream2sentence's nltk+rule-based consensus mode", Stanza optional. — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [V] Pipecat: `SimpleTextAggregator` "buffers text until sentence boundaries are detected" and is the default for most TTS services; `SkipTagsAggregator` avoids false boundaries inside tags; `PatternPairAggregator` flushes on a sentence boundary or a complete pattern pair (XML tags / custom delimiters). The aggregator returns `None` until a unit is ready, then an `Aggregation` with text and a `type` ("sentence", "xml", ...). `MarkdownTextFilter` converts streaming markdown into TTS-friendly plain text. — [Pipecat: Text Aggregators and Filters](https://docs.pipecat.ai/server/utilities/text/overview)
- [V] Pipecat default is `TextAggregationMode.SENTENCE`; sentence detection follows the TTS language setting (default English); `TextAggregationMode.TOKEN` passes tokens straight through for TTS services with native text streaming, with a docs claim of "end-to-end latency often under 200ms". — [Pipecat: Text to Speech](https://docs.pipecat.ai/pipecat/learn/text-to-speech)
- [S] LiveKit Agents wraps any non-streaming TTS in `tts.StreamAdapter` with a sentence tokenizer (default `tokenize.blingfire.SentenceTokenizer`, `retain_format=True`); each sentence is requested as soon as it is complete and played in order. — [LiveKit StreamAdapter reference](https://docs.livekit.io/reference/python/livekit/agents/tts/index.html), [aplisay/llm-agent PR #357](https://github.com/aplisay/llm-agent/pull/357)
- [V] Failure mode when the splitter does not know the script's terminators: in LiveKit issue #7508, an English reply was first split after word 7 of 24, a Hindi reply only at word 25 of 28, and Arabic/Urdu replies never split (whole response held). Both `blingfire` (default) and `basic` tokenizers were affected; fix proposed in PR #7509. Russian uses Latin-style `. ! ?` so is not in the affected list. — [livekit/agents issue #7508](https://github.com/livekit/agents/issues/7508)
- [V] vLLM-Omni's WebSocket endpoint exposes the policy as a parameter: `split_granularity` = `"none"` (buffer until flush) / `"sentence"` / `"clause"`, splitting on punctuation including CJK `。！？…` and Arabic `۔`. — [vLLM-Omni Speech API docs, dated 2026-09-29](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [S] A 2026 tutorial states the sentence buffer should enforce a minimum sentence length of 10 characters to avoid sending fragments to TTS. — [arXiv 2603.05413 "Building Enterprise Realtime Voice Agents from Scratch"](https://arxiv.org/pdf/2603.05413)

**First-chunk shortening**
- [V] Formula-based first split (dev.to, 2026-08-11): with real-time factor r = synthesis time / audio duration, the first chunk must be at least a fraction p ≥ r/(1+r) of the utterance so that chunk 2 finishes synthesizing before chunk 1 finishes playing (r=1.0 → split at 50%; r=0.25 → 20%). Measured on VOICEVOX 0.25.2 (Apple M4): r=0.64 for 3 characters, 0.41 for 7, 0.26 for 21, 0.23 for 40 — short chunks have worse RTF because of fixed overhead. The author's first measurement (r=1.09) was wrong by 4x due to a loaded machine. Engine is Japanese VOICEVOX, not a Russian engine. — [dev.to, renga154](https://dev.to/renga154/where-to-split-a-sentence-for-streaming-tts-is-decided-by-one-number-4o6b)
- [V] ElevenLabs' input-streaming protocol encodes the same idea server-side as `chunk_length_schedule`, default `[120, 160, 250, 290]` characters (first generation after 120 chars, then progressively larger); lowering it reduces latency at a quality cost. — [ElevenLabs realtime TTS WebSocket guide](https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts)
- [V] Cartesia buffers incoming text until there is "sufficient context" or `max_buffer_delay_ms` elapses (0–5000 ms, default 3000 ms). — [Cartesia: Stream Inputs using Continuations](https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations)

**Markdown / code / numbers**
- [V] Pipecat ships `MarkdownTextFilter`; stream2sentence can strip links and emoji. — [Pipecat text utilities](https://docs.pipecat.ai/server/utilities/text/overview), [stream2sentence](https://github.com/KoljaB/stream2sentence)
- [S] Open WebUI has had bugs of markdown content being duplicated in TTS (issue #6021) and a request to split on markdown/paragraphs (discussion #9087). — [open-webui issue #6021](https://github.com/open-webui/open-webui/issues/6021), [discussion #9087](https://github.com/open-webui/open-webui/discussions/9087)
- [V] Cartesia's rule for chunking LLM output: chunks must concatenate into a valid transcript (keep leading spaces), complete sentences must end with closing punctuation, and markup tags / decimal values must not be split across chunks. — [Cartesia continuations guide](https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations)

**Barge-in / interruption**
- [V] Pipecat: word-level timestamps (available from Cartesia, ElevenLabs, Rime) let the pipeline record exactly which words were spoken before an interruption, so the LLM context contains only the heard part. — [Pipecat: Text to Speech](https://docs.pipecat.ai/pipecat/learn/text-to-speech)
- [V] Cartesia: each generation lives in a `context_id`; SDK `ctx.push()` sets `continue: true`, `ctx.no_more_inputs()` sends `continue: false`. — [Cartesia continuations guide](https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations)
- [V] ElevenLabs: `flush: true` forces generation of buffered text without closing the socket; empty string closes; connection auto-closes after 20 s of inactivity. — [ElevenLabs realtime TTS guide](https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts)

**Queueing and gapless playback**
- [V] vLLM-Omni frames each sentence as `audio.start` → binary PCM frames → `audio.done`, in order. — [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [S] Chatterbox-TTS-Server v2.0.0 streams WAV bytes per chunk "with 20 ms crossfades" between chunks. — [devnen/Chatterbox-TTS-Server releases](https://github.com/devnen/Chatterbox-TTS-Server/releases)
- [V] RealtimeTTS 0.8.9 changelog: "avoiding artificial rebuffering at audio boundaries"; 0.8.10: continues through initial quiet audio and trims it in the stream; Voice Studio defaults to 0 ms additional browser buffering. — [RealtimeTTS releases](https://github.com/KoljaB/RealtimeTTS/releases)
- [S] Open WebUI bug: the TTS dispatch logic tracked only the last dispatched sentence, so sentences were skipped when several arrived in quick succession during streaming (issue #19861). — [open-webui issue #19861](https://github.com/open-webui/open-webui/issues/19861)

**Quality trade-off: sentence chunking vs native text-streaming models**
- [V] Cartesia: continuations on one context "maintain prosody" across inputs; separately synthesized chunks produce audible seams, hence the recommendation to use continuations for streamed LLM output. — [Cartesia continuations guide](https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations)
- [S][C] Soniox wiki: sentence-boundary streaming helps prosody but raises time-to-first-audio; token-level streaming minimizes TTFA but risks audible discontinuities at chunk boundaries; feeding incomplete syntactic units causes unnatural pacing and flat intonation. — [Soniox: Streaming TTS](https://soniox.com/wiki/streaming-tts)

### Inferences
- A local implementation for a full-text-only engine needs four pieces: (a) incremental splitter with ~10–12 characters of lookahead and a min length of ~10 chars, (b) a special rule for the first fragment (clause delimiter `,;:—` allowed, or force after ~15 words / a deadline), (c) a text filter stage *before* the splitter (markdown, code blocks, links, emoji), and (d) an ordered per-response audio queue keyed by a generation/context id that can be cancelled atomically on barge-in.
- Normalization and stress placement for Russian must run *per fragment after splitting* (they need the whole clause for case agreement and homographs) — which argues against very short first fragments in Russian: "5 кг" or a homograph cut off from its context is more likely to be read wrongly.
- The r/(1+r) rule implies that on a GPU where RTF ≈ 0.1–0.25 the first chunk can be as small as 10–20% of the first sentence; on a CPU engine with RTF close to 1 there is little benefit from an early clause cut, because the second chunk will not be ready in time.
- Splitters keyed to English (NLTK punkt with `language="en"`, blingfire) work on Russian mostly because the terminators are the same, but Russian abbreviations ("т.е.", "г.", "ул.", "им.") and decimals/dates ("12.05.2026") will produce false boundaries unless normalization of those runs first or an abbreviation list is supplied.

### Gaps
- No primary source was found that measures MOS/prosody degradation of per-sentence chunking versus native incremental-text models on Russian specifically.
- LiveKit's `min_sentence_len` / `stream_context_len` defaults and its `filter_markdown` / `filter_emoji` text transforms could not be confirmed (the StreamAdapter reference URL returned 404; the TTS overview page did not include them).
- Pipecat's internal sentence detection implementation (NLTK-based or not, and its lookahead) is not described on the fetched docs pages.
- How each framework handles code blocks specifically (skip vs read) is not documented in the pages fetched.

---

## 2. Open-source frameworks / wrappers that implement the LLM-stream → TTS pipeline

### Takeaway
RealtimeTTS (MIT) is the most directly relevant Python library and is very actively released in September 2026, now centred on a Qwen3-TTS engine; Pipecat (BSD-2) and LiveKit Agents are the two maintained full voice-agent frameworks; Vocode is effectively dormant since late 2024. Out of the box, Russian-capable local engines in these stacks are essentially Piper (4 ru_RU voices), Qwen3-TTS, Chatterbox Multilingual, XTTS (unmaintained upstream) and Silero via custom integration; Kokoro has no Russian.

### Cited Findings

**RealtimeTTS (KoljaB)**
- [V] Local engines listed in README: QwenEngine, QwenCpuEngine, InflectEngine, SystemEngine, CoquiEngine (XTTS), PiperEngine, StyleTTSEngine, ParlerEngine, KokoroEngine, OmniVoiceEngine, PocketTTSEngine, NeuTTSEngine, ZipVoiceEngine, LuxTTSEngine, ChatterboxEngine, SoproTTSEngine, SopranoEngine, MossTTSEngine. QwenEngine is "currently the recommended and preferred RealtimeTTS engine for high-quality, low-latency conversational speech". Code license MIT; engines/models/weights/voices have separate terms. 4,000+ stars, 616 commits. — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [S] Cloud/other engines also listed: azure, elevenlabs, openai, gtts, edge, camb, minimax, cartesia, modelslab, orpheus, typecast. — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [V] Latency claim (author's own, RTX 4090, Linux, Qwen engine): "approximately 35 ms engine TTFT, another 35 ms until first PCM chunk emission, predicted audible onset was 80.9 ms". [C] — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [V] Release cadence: 0.8.5 (Aug 31), 0.8.6 (Sep 20), 0.8.8 and 0.8.9 (Sep 26), 0.8.10 (Sep 27) — the releases page shows month/day; year is 2026 by context (not printed in the fetched text). 0.8.6 added "Qwen GPU and CPU server installation extras", em-dash streaming, language detection, segment controls; 0.8.8 reports a "23–26% reduction in RTF" from CPU decoder overlap. — [RealtimeTTS releases](https://github.com/KoljaB/RealtimeTTS/releases)
- [V] Russian is not specifically named in the README; QwenEngine is described as "multilingual". — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS). Qwen3-TTS itself officially lists Russian among its languages — [vLLM-Omni Speech API docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)

**Pipecat (Daily)**
- [S] License BSD 2-Clause, copyright 2024–2026 Daily. — [pipecat LICENSE](https://github.com/pipecat-ai/pipecat/blob/main/LICENSE)
- [S] Since pipecat 1.0, `TTSService` no longer accepts a `text_aggregator` argument; aggregation moved to an `LLMTextProcessor` placed upstream of TTS (it turns `LLMTextFrame`s into `AggregatedTextFrame`s). The deprecation of the TTS-constructor `text_aggregator` began around v0.0.96. — [Pipecat text utilities](https://docs.pipecat.ai/server/utilities/text/overview), [llm_text_processor reference](https://reference-server.pipecat.ai/en/latest/api/pipecat.processors.aggregators.llm_text_processor.html), [v0.0.96 notes](https://newreleases.io/project/github/pipecat-ai/pipecat/release/v0.0.96)
- [V] `PiperTTSService` (in-process, auto-downloads voices, optional `use_cuda`) and `PiperHttpTTSService` (talks to a Piper HTTP server, `piper-tts[http]`). The docs note PiperTTSService uses GPL-3.0 licensed code, and the HTTP variant avoids linking it. — [Pipecat Piper docs](https://docs.pipecat.ai/api-reference/server/services/tts/piper)
- [V] `KokoroTTSService`: local, uses `kokoro-onnx` (`pipecat-ai[kokoro]`, `kokoro-onnx>=0.5.0`), streams audio incrementally; Russian is not in its language list. — [Pipecat Kokoro docs](https://docs.pipecat.ai/server/services/tts/kokoro)
- [S] `XTTSService` is deprecated as of v1.7.0 and will be removed in v2.0.0 because the Coqui XTTS streaming server has been unmaintained since February 2024. — [Pipecat XTTS docs](https://docs.pipecat.ai/server/services/tts/xtts)
- [V] Pipecat recommends WebSocket TTS services (Cartesia, ElevenLabs, Rime) for lowest latency; HTTP services may have higher intermittent latency. — [Pipecat: Text to Speech](https://docs.pipecat.ai/pipecat/learn/text-to-speech)
- [S] NVIDIA publishes a Pipecat-based "NeMo voice agent" with its own TTS service module. — [NVIDIA NeMo voice agent docs](https://docs.nvidia.com/nemo/labs-voice-agent/nemo-voice-agent/nemo_voice_agent/pipecat/services/nemo/tts/)

**LiveKit Agents**
- [V] 40+ TTS plugins (Python and Node.js); fully custom TTS is done by implementing the TTS node in the agent. — [LiveKit TTS models overview](https://docs.livekit.io/agents/models/tts/)
- [S] Non-streaming TTS is wrapped by `tts.StreamAdapter` + sentence tokenizer (blingfire default). — [LiveKit tts reference](https://docs.livekit.io/reference/python/livekit/agents/tts/index.html)
- [S] Official docs have a Kokoro guide that points the OpenAI TTS plugin at Kokoro-FastAPI via `base_url="http://localhost:8880/v1"`; community plugins exist (`livekit-kokoro`, `livekit-streaming-tts` with Kokoro/Piper backends). — [LiveKit Kokoro guide](https://docs.livekit.io/agents/models/tts/kokoro/), [taresh18/livekit-kokoro](https://github.com/taresh18/livekit-kokoro), [livekit-streaming-tts on PyPI](https://pypi.org/project/livekit-streaming-tts/)
- [V] Tokenizer gaps for non-Latin terminators reported and fixed via PR in 2026 (issue #7508) — evidence of active maintenance. — [livekit/agents issue #7508](https://github.com/livekit/agents/issues/7508)

**Vocode**
- [S] Latest release v0.1.113, pre-releases v0.1.114a0–a2 dated 2024-08-07; repository last updated 2024-11-15; the team was looking for community maintainers. (Older info — no 2025/2026 activity found.) — [vocode-core releases](https://github.com/vocodedev/vocode-core/releases), [vocode-core](https://github.com/vocodedev/vocode-core)

**Wyoming protocol / Home Assistant**
- [S] Streaming TTS was added in Home Assistant 2025.7: new Wyoming events `SynthesizeStart`, `SynthesizeChunk`, `SynthesizeStop`, `SynthesizeStopped`; a server advertises `supports_synthesize_streaming=True` in its Info reply; text chunks are combined into sentences server-side and audio is streamed per sentence. — [kokoro-wyoming PR #13](https://github.com/nordwestt/kokoro-wyoming/pull/13), [HA community: Streaming support for Wyoming TTS](https://community.home-assistant.io/t/streaming-support-for-wyoming-tts/900708), [HA community: documentation to test streaming TTS](https://community.home-assistant.io/t/documentation-to-test-streaming-tts-feature/906308)
- [S] wyoming-piper 2.x streams by default (only a `--no-streaming` flag remains). — [HA community thread](https://community.home-assistant.io/t/documentation-to-test-streaming-tts-feature/906308)
- [S][C] Home Assistant's own measurement: without streaming, Cloud TTS and Piper took more than five seconds to respond to a long LLM request; with streaming, about half a second to start speaking. — [Home Assistant blog, Voice Chapter 11, 2025-10-22](https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11/)
- [V] Limitation: streaming TTS is only used when the conversation agent delivers text as deltas; a response set in one piece (`set_conversation_response`) is "parsed in its entirety and not chunked". Issue opened 2025-06-28, closed as not planned. — [home-assistant/core issue #147727](https://github.com/home-assistant/core/issues/147727)
- [V] A community custom integration for Wyoming TTS streaming first released 2025-06-12, updated 2026-05-20 (failover to backup server, text file reading); "not fully supported for ESP satellites yet". — [HA community thread #900708](https://community.home-assistant.io/t/streaming-support-for-wyoming-tts/900708)
- [S] Third-party Wyoming servers with streaming exist for Kokoro, Qwen3-TTS, and OpenAI-compatible backends (`wyoming-openai`). — [kokoro-wyoming PR #13](https://github.com/nordwestt/kokoro-wyoming/pull/13), [qwen3-tts-stripped-wyoming](https://github.com/allenbenz/qwen3-tts-stripped-wyoming), [wyoming-openai on PyPI](https://pypi.org/project/wyoming-openai/)
- [S] Piper: `rhasspy/piper` was archived 2025-10-06; development moved to `OHF-Voice/piper1-gpl` (GPL). Russian voices: `ru_RU` irina, denis, dmitri, ruslan (medium, 22.05 kHz, ~63 MB each). Open licensing question (issue #314): denis/dmitri list CC0 datasets but were fine-tuned from `en_US-lessac-medium`, which traces to the Blizzard 2013 research licence that excludes commercial use. — [rhasspy/piper](https://github.com/rhasspy/piper), [piper1-gpl VOICES.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md), [piper1-gpl issue #314](https://github.com/OHF-Voice/piper1-gpl/issues/314)

**Open WebUI**
- [V] As of v0.11.4 (issue dated 2026-09-23, open): with "Response Splitting" enabled, auto-playback still waits for the whole LLM answer before sending TTS requests; sentence-by-sentence dispatch during generation exists only in call mode. — [open-webui issue #30422](https://github.com/open-webui/open-webui/issues/30422)
- [S] Response Splitting modes split at punctuation (periods, `!`, `?`, newlines) or paragraphs; Punctuation is the recommended mode. — [Open WebUI TTS docs](https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/openai-tts-integration/)

**LocalAI**
- [S] Streaming TTS is implemented by the audio-cpp, crispasr, llama-cpp, magpie-tts-cpp, moss-tts-cpp, omnivoice-cpp, qwen3-tts-cpp, sherpa-onnx, supertonic, vibevoice-cpp and voxcpm backends; Piper and Kokoro backends are not in that list. LocalAI also has an OpenAI-Realtime-style API with a configurable TTS stage. — [LocalAI Text to Audio docs](https://localai.io/docs/features/text-to-audio/), [LocalAI Realtime API docs](https://localai.io/docs/features/openai-realtime/)

**TEN framework**
- [S] Apache 2.0 "with additional restrictions" for the framework; components in `packages` under Apache 2.0. — [TEN-framework/ten-framework](https://github.com/ten-framework/ten-framework)

### Inferences
- For a Python service that must voice a streamed LLM answer with a local Russian-capable engine, the lowest-effort reuse options are: (1) `stream2sentence` alone (MIT, tiny, engine-agnostic) plus own queue; (2) RealtimeTTS if Qwen3-TTS / Piper / Chatterbox / XTTS is the engine; (3) Pipecat when a full duplex agent (VAD, STT, interruption, transports) is needed.
- Russian coverage "out of the box" across these frameworks is thin: Piper (license caveat on two of four voices; GPL engine), Qwen3-TTS (officially lists Russian), Chatterbox Multilingual (ru), XTTS-v2 (upstream server unmaintained since Feb 2024). Silero TTS, the most common Russian-first local engine, is not a built-in engine in RealtimeTTS, Pipecat or LiveKit per the pages read — it needs a custom engine/service class.
- Open WebUI should not be treated as a reference implementation of streaming TTS: as of September 2026 its non-call playback is still whole-response.

### Gaps
- TEN framework: list of TTS extensions, local-engine support and activity level were not verified (only the license line from a search snippet).
- Exact date and contents of the Pipecat 1.0 release, and the current Pipecat version number, were not confirmed from release notes (docs pages mention v0.0.105 deprecations and v1.7.0/v2.0.0 for XTTS, implying 1.x is current).
- LiveKit Agents license and current version were not confirmed on the fetched page (the project is widely known as Apache-2.0, but this was not sourced here).
- RealtimeTTS: whether QwenEngine's Russian quality is usable, and how the library handles Russian in `stream2sentence` (`language` param for NLTK), is not documented in the README.
- No evidence found of a maintained, general-purpose "Silero TTS" plugin for Pipecat/LiveKit.

---

## 3. Servers: OpenAI-compatible `/v1/audio/speech` and WebSocket text-in/audio-out for local models

### Takeaway
The OpenAI speech endpoint takes the complete `input` string and only streams the *output* (raw chunked audio, or SSE with `stream_format`), so incremental text needs a different protocol. The de-facto references are ElevenLabs `stream-input` (text chunks + `flush` + `chunk_length_schedule`) and Cartesia contexts (`context_id` + `continue`); among open-source servers, vLLM-Omni's `/v1/audio/speech/stream` WebSocket is the only one found that natively accepts incremental text with server-side sentence/clause splitting — and it serves Qwen3-TTS, which supports Russian.

### Cited Findings

**OpenAI API semantics**
- [S] `/v1/audio/speech` takes `input` (full text) and optional `stream_format` = `"audio"` (raw bytes) or `"sse"` (events); `sse` is not supported for `tts-1` / `tts-1-hd`, supported by `gpt-4o-mini-tts`. The streaming concerns output delivery only; the input is not incremental. — [OpenAI API reference: Create speech](https://platform.openai.com/docs/api-reference/audio/createSpeech), [OpenAI text-to-speech guide](https://developers.openai.com/api/docs/guides/text-to-speech)

**Incremental-text WebSocket protocols (cloud, as protocol references)**
- [V] ElevenLabs: `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=...`; first message `{"text": " ", voice_settings, generation_config: {chunk_length_schedule}}`; then `{"text": "...", "flush": bool}`; `{"text": ""}` ends the stream; server sends `{"audio": base64, "alignment": ..., "isFinal": bool}`; default `chunk_length_schedule` `[120, 160, 250, 290]`; 20 s inactivity timeout; closing auto-flushes. — [ElevenLabs realtime TTS guide](https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts)
- [S] ElevenLabs `try_trigger_generation` carries a lower-quality warning and is recommended to stay false; `flush: true` is the documented end-of-turn trigger. — [hermes-talk PR #117](https://github.com/TheSmokeDev/hermes-talk/pull/117)
- [V] Cartesia: fields `context_id`, `transcript`, `continue` (true = more input coming), `flush`, `max_buffer_delay_ms` (0–5000, default 3000); inputs on the same context continue the generation with preserved prosody. — [Cartesia continuations guide](https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations), [Cartesia contexts](https://docs.cartesia.ai/api-reference/tts/working-with-web-sockets/contexts)

**vLLM-Omni**
- [V] Docs dated 2026-09-29. Models served through the speech API: Qwen3-TTS (CustomVoice, VoiceDesign, Base; 24 kHz), Fish Speech S2 Pro (44.1 kHz), Voxtral TTS, CosyVoice3, Gepard-1.0, OmniVoice, VoxCPM2, MOSS-TTS-Nano, Breeze-TTS-2. `/v1/audio/speech` supports `stream` (bool), `stream_format` (`audio` | `sse`), `response_format` (`wav`, `mp3`, `flac`, `pcm`, `opus`); streaming requires `pcm` or `wav` and `speed` 1.0. — [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [V] WebSocket `/v1/audio/speech/stream`: client sends `session.config`, `input.text` chunks, `input.done` (flush); server replies per sentence with `audio.start`, binary PCM frames, `audio.done`; `split_granularity` = `none` / `sentence` / `clause`; `stream_audio: true` gives chunked PCM within a sentence, otherwise one payload per utterance. — [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [V] Qwen3-TTS language list per these docs: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. — [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [V] PR #4185 (opened 2026-06-05) to add OpenAI-style `stream_format` was closed unmerged on 2026-09-30 for inactivity; a related PR #4490 was merged instead. — [vllm-omni PR #4185](https://github.com/vllm-project/vllm-omni/pull/4185)
- [S] A TTS development roadmap RFC for Q2 2026 exists. — [vllm-omni issue #2115](https://github.com/vllm-project/vllm-omni/issues/2115)

**Speaches (formerly faster-whisper-server)**
- [S] OpenAI-compatible server; STT via faster-whisper, TTS via Piper and Kokoro; dynamic model load/unload; `/v1/audio/speech` with streaming responses. — [speaches-ai/speaches](https://github.com/speaches-ai/speaches), [Speaches TTS docs](https://speaches.ai/usage/text-to-speech/)

**openedai-speech**
- [S] Archived by the owner on 2026-01-04 (read-only), 859 stars at archival. — [matatonic/openedai-speech](https://github.com/matatonic/openedai-speech/pkgs/container/openedai-speech)

**Kokoro-FastAPI**
- [S] Dockerized OpenAI-compatible wrapper for Kokoro-82M with streaming. Russian: requested in an August 2025 discussion ("NEED A RUSSIAN LANGUAGE"); Kokoro-82M's language codes are en-us, en-gb, es, fr, hi, it, pt-br, ja, zh — no Russian. — [remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI), [Kokoro-FastAPI discussions](https://github.com/remsky/Kokoro-FastAPI/discussions), [hexgrad/Kokoro-82M VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

**Chatterbox-TTS-Server (devnen)**
- [S] OpenAI-compatible API + Web UI for Resemble AI Chatterbox; CUDA / ROCm / CPU; v2.0.0 added opt-in `stream: true` on `/tts` returning a StreamingResponse that flushes WAV bytes per chunk with 20 ms crossfades. — [devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server), [releases](https://github.com/devnen/Chatterbox-TTS-Server/releases)

**AllTalk TTS v2**
- [S] v2 is the maintained line; "Streaming" generation mode exists; XTTS engine supports streaming, F5-TTS engine does not. A Pinokio listing shows an update on 2026-09-19. — [erew123/alltalk_tts](https://github.com/erew123/alltalk_tts), [Pinokio AllTalk v2](https://pinokio.co/apps/github-com-6morpheus6-alltalk-tts)

**Orpheus-FastAPI (Lex-au)**
- [S] OpenAI-compatible server for Orpheus; latest version found v1.3.1 (2025-07-05, older info); real-time streaming is an open feature request (issue #46). — [Lex-au/Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI), [issue #46](https://github.com/Lex-au/Orpheus-FastAPI/issues/46)

**LocalAI** — see section 2 (streaming only on specific backends; Piper/Kokoro backends not streaming). — [LocalAI Text to Audio](https://localai.io/docs/features/text-to-audio/)

**NVIDIA Riva / Speech NIM TTS**
- [V] Support matrix: Magpie TTS Multilingual and Magpie TTS Zeroshot cover en, es, fr, de, zh, vi, it, hi, ja, ko, ar, pt — no Russian. Russian (ru-RU) is available only via "Chatterbox TTS Multilingual" (voice `Chatterbox-Multilingual.ru-RU.Male`), a community model under MIT whose API use is "governed by the NVIDIA API Trial Service Terms of Use". Requirements: NVIDIA GPU with compute capability ≥ 8.0 and ≥ 16 GB VRAM, FP16. — [NVIDIA TTS NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html)

**Triton deployments**
- [S] F5-TTS ships a Triton + TensorRT-LLM runtime (`src/f5_tts/runtime/triton_trtllm`); F5-TTS is non-autoregressive, which limits true streaming. CosyVoice 2 is a unified streaming/non-streaming design, and its Triton deployment uses a model-ensemble pattern. — [F5-TTS Triton README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/README.md), [CosyVoice 2 paper](https://arxiv.org/html/2412.10117v1), [DeepWiki: CosyVoice deployment](https://deepwiki.com/FunAudioLLM/CosyVoice/7-deployment)

### Inferences
- A local server that wants drop-in compatibility has two realistic targets: (1) OpenAI `/v1/audio/speech` with chunked `pcm`/`wav` output (works with Open WebUI, LiveKit's OpenAI plugin, `wyoming-openai`, Pipecat HTTP services) — the *client* does sentence splitting; (2) a WebSocket with incremental text where the *server* splits — mimic either the vLLM-Omni message set (`session.config` / `input.text` / `input.done` → `audio.start` / PCM / `audio.done`) or ElevenLabs `stream-input` (for which Pipecat/LiveKit/RealtimeTTS clients already exist).
- The 16 GB VRAM requirement and trial-terms licensing make NVIDIA's NIM route impractical for small local GPUs; its only Russian option is a repackaged open model that can be self-hosted directly.
- With openedai-speech archived (Jan 2026) and Orpheus-FastAPI apparently stale, the actively maintained OpenAI-compatible local servers in 2026 are Speaches, Kokoro-FastAPI, Chatterbox-TTS-Server, LocalAI and vLLM-Omni — and of these only Speaches (via Piper), Chatterbox-TTS-Server and vLLM-Omni (Qwen3-TTS) cover Russian.

### Gaps
- Speaches: current version, license, and whether its `/v1/audio/speech` supports `stream_format: sse` were not verified on the primary docs.
- Kokoro-FastAPI, Chatterbox-TTS-Server, AllTalk: licenses and last-commit dates not verified; Chatterbox-TTS-Server Russian support (multilingual model) not confirmed on the repo.
- NVIDIA NIM: cost of production (NVIDIA AI Enterprise) licensing for local deployment was not found on the support-matrix page.
- No source found for a Triton deployment of CosyVoice/F5-TTS evaluated on Russian; CosyVoice3's Russian support was not verified.
- OpenAI reference claims are from search snippets of the official pages, not a full fetch.

---

## 4. Reported latency budgets and chunking best practices

### Takeaway
Practitioners in 2026 converge on an ~800 ms voice-to-voice budget, with LLM time-to-first-token (roughly 150–600 ms depending on source) and STT as the largest contributors and TTS time-to-first-audio around 100–250 ms for hosted streaming engines; starting TTS at the first sentence boundary rather than at end of response is reported to save hundreds of milliseconds to several seconds. Most published numbers are vendor or blog claims, not reproducible benchmarks on local engines.

### Cited Findings
- [V] WebRTC.ventures (Jen Oppenheimer, 2026-09-23): ~800 ms total round-trip budget before a conversation feels slow; ranks STT and LLM TTFT as the largest contributors, TTS small-to-medium, transport/jitter buffer small; declines to give per-stage milliseconds ("depend on the specific STT, LLM, and TTS providers"). Recommends streaming text to TTS as the LLM generates it. — [webrtc.ventures: The Voice AI Latency Budget](https://webrtc.ventures/2026/09/voice-ai-latency-budget/)
- [S][C] FutureAGI (2026): "a good 2026 LLM hits TTFT in 150–300ms"; streaming LLM tokens into TTS at the first sentence boundary "saves 200–500ms"; a warm per-session TTS connection makes first audio arrive 50–150 ms sooner than cold start. — [futureagi.com](https://futureagi.com/blog/how-to-optimize-voice-agent-latency-2026/)
- [S][C] Other sources put LLM TTFT at 300–600 ms, and TTS time-to-first-audio at 100–200 ms. — [AWS builder: The 800ms Rule](https://builder.aws.com/content/3JDFAfXBiuwPP5MPzgf4RUWSAIp/the-800ms-rule-budgeting-latency-for-a-real-time-voice-agent-on-aws), [Deepgram: voice agent architecture](https://deepgram.com/learn/voice-agent-architecture-stt-llm-tts-pipeline-design)
- [S][C] Gradium TTS: 214 ms median perceived time to first audio, 31 ms P25–P75 spread, on the Coval leaderboard (2026-09-08) — vendor citing a third-party leaderboard. — [gradium.ai](https://gradium.ai/content/how-to-connect-tts-to-llm-voice-pipeline)
- [S][C] Home Assistant (2025-10-22): long LLM reply, non-streaming TTS > 5 s to first audio; streaming (sentence-by-sentence) ≈ 0.5 s, for both Cloud TTS and local Piper. — [Home Assistant Voice Chapter 11](https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11/)
- [V][C] RealtimeTTS Qwen engine on RTX 4090: ~35 ms engine TTFT + ~35 ms to first PCM chunk, predicted audible onset 80.9 ms (author's own benchmark). — [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [V][C] Pipecat docs: token-mode streaming into a native text-streaming TTS gives "end-to-end latency often under 200ms" (TTS stage). — [Pipecat: Text to Speech](https://docs.pipecat.ai/pipecat/learn/text-to-speech)
- [V] LiveKit issue #7508 gives a concrete "time to first sentence" measurement in tokens: English first sentence released after word 7 of a 24-word reply. — [livekit/agents issue #7508](https://github.com/livekit/agents/issues/7508)
- [V] First-chunk sizing by RTF, p ≥ r/(1+r), and RTF worsening on very short inputs (0.64 at 3 chars vs 0.23 at 40 chars on VOICEVOX). — [dev.to, 2026-08-11](https://dev.to/renga154/where-to-split-a-sentence-for-streaming-tts-is-decided-by-one-number-4o6b)
- [S][C] "Chunking LLM output at sentence boundaries and streaming TTS against those chunks is how you achieve first-audio under 300ms"; "the strongest systems blend aggressive first-chunk delivery with sentence-aware chunking once conversation is moving". — [Soniox: Streaming TTS](https://soniox.com/wiki/streaming-tts), [smallest.ai blog](https://smallest.ai/blog/streaming-voice-api-for-real-time-speech-voice-agents-and-ai-apps)

### Inferences
- Time-to-first-audio for a chunked local pipeline ≈ LLM TTFT + (tokens in first fragment ÷ LLM tokens/s) + normalization/stress time + TTS TTFA for that fragment. With a 7–15-word first sentence and a local LLM at 30–60 tok/s, waiting for the first sentence alone costs roughly 0.2–0.7 s — comparable to or larger than the TTS stage, which is why first-fragment shortening (clause cut, `force_first_fragment_after_words`) pays off more than TTS engine tuning.
- Russian preprocessing adds to this budget: per section 5, silero-stress is ~30 ms per 400-character paragraph on CPU (negligible), whereas a T5-based normalizer (95M–860M params) on CPU is likely the slowest preprocessing step and should be bypassed when the fragment contains no digits/Latin/abbreviations.

### Gaps
- No reproducible, third-party end-to-end latency measurement of a fully local LLM → local Russian TTS pipeline was found.
- Figures from FutureAGI, AWS builder, Deepgram, Gradium, Soniox and smallest.ai are from search snippets of vendor/blog pages; methodology not examined.
- No published A/B data on how often an early comma-split first fragment damages intonation.

---

## 5. Russian preprocessing: stress, ё, normalization, English words, markdown

### Takeaway
For stress and ё, silero-stress (MIT, PyTorch-only, ~30–50 MB, single-thread CPU, v1.5 released 2026-09-15) and RUAccent (ONNX, several model sizes, COLING 2025 paper) are the two maintained options, and each vendor's benchmark favours its own tool. For normalization, the options are rule-based WFST (NeMo text processing, Russian supported) versus seq2seq T5 models (RUNorm by Den4ikAI, Apache-2.0, 95M/222M/860M; saarus72's FRED-T5 normalizer); none publishes CPU speed figures, and English-word transliteration remains the weakest link.

### Cited Findings

**silero-stress**
- [V] First public article 2025-10-09: MIT license; places stress, resolves homographs, restores ё; ~4.1 million word forms in dictionary; ~2,000 homographs; package ~50 MB (~30 MB archive); dependencies "PyTorch and standard Python library", Python 3.10+; speed: ~0.5 ms per single word, ~30 ms for a ~400-character paragraph with 2 homographs; runs on a single CPU thread, needs AVX2; generalization to unknown words 60–70% accuracy. — [Habr 955130, 2025-10-09](https://habr.com/ru/articles/955130/)
- [V] Vendor benchmark from the same article (homograph set): silero-stress F1 0.85 / word acc 0.92 / total acc 0.93; silero-stress-private 0.91 / 0.96 / 0.96; RUAccent-turbo3.1 0.64 / 0.76 / 0.84; RUAccent-tiny2.1 0.56 / 0.69 / 0.78; omogre 0.46 / 0.67 / 0.73. [C — Silero's own test set] — [Habr 955130](https://habr.com/ru/articles/955130/)
- [V] v1.5 (article 2026-09-15): 2,208 homographs (was 1,924 in v1.0; +395, −111); training data ~195M sentences (+~54M); on the 2,208-homograph set F1 0.86, word acc 0.92, total acc 0.93 (private model 0.89 / 0.94 / 0.94); ё flags `put_yo`, `put_yo_homo`; accepts custom regex patterns and phrase dictionaries; preserves punctuation/formatting; more robust to punctuation variations; still single-thread CPU; Belarusian and Ukrainian models plus dictionaries for 17 more languages; `pip install silero-stress`; repo `snakers4/silero-stress`. — [Habr 1079674, 2026-09-15](https://habr.com/ru/articles/1079674/)
- [S] Silero TTS v5 Russian models include automated stress and homograph handling inside the TTS package and support SSML. — [snakers4/silero-models](https://github.com/snakers4/silero-models), [Habr 961930](https://habr.com/ru/articles/961930/)

**RUAccent (Den4ikAI)**
- [V] README: models tiny, tiny2, tiny2.1, turbo2, turbo3, turbo3.1, turbo, big_poetry; runs on ONNX Runtime, CPU or CUDA (`onnxruntime-gpu`); `tiny_mode` disables the rule pipeline and part of the models; custom dictionary `{'слово': 'сл+ово с удар+ением'}`; minimum 512 MB RAM with tiny model; license "transitioning to MIT for v1"; 210 stars; README has no accuracy or speed numbers. — [Den4ikAI/ruaccent README](https://github.com/Den4ikAI/ruaccent/blob/main/README.md)
- [S] Search snippets disagree on the license (MIT vs a change to Apache 2.0). — [Den4ikAI/ruaccent](https://github.com/Den4ikAI/ruaccent), [PyPI ruaccent](https://pypi.org/project/ruaccent/)
- [S] COLING 2025 paper: RUAccent combines morphological analysis, context-aware neural models and a "Ё-fikator"; reports 0.96 accuracy on homographs and 0.97 on non-homograph words. [C — authors' own benchmark]; contradicted by Silero's benchmark above (RUAccent-turbo3.1 total acc 0.84 on Silero's homograph set) — [ACL Anthology 2025.coling-main.444](https://aclanthology.org/2025.coling-main.444/) vs [Habr 955130](https://habr.com/ru/articles/955130/)
- [S] Forks/derivatives exist (sh1man/ruaccent, NikiPshg/ruaccent with a fix for homographs at sentence start), and Den4ikAI also publishes `ruphon` (IPA phonemizer). — [sh1man/ruaccent](https://github.com/sh1man/ruaccent), [NikiPshg/ruaccent PR #2](https://github.com/NikiPshg/ruaccent/pull/2), [Den4ikAI/ruphon](https://github.com/Den4ikAI/ruphon)

**Other stress tools**
- [S] `ruaccent-predictor` (PyPI): character-level Transformer, claims 99.7% accuracy on its validation set [C]; described in a Habr article (992892); a community pipeline combines silero-stress with ruaccent-predictor for Qwen3-TTS. — [PyPI ruaccent-predictor](https://pypi.org/project/ruaccent-predictor/), [Habr 992892](https://habr.com/ru/articles/992892/), [sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf](https://huggingface.co/sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf)
- [S] StressRNN: LSTM-based, a modified RusStress with bug fixes (older generation tool). — [dbklim/StressRNN](https://github.com/dbklim/StressRNN)
- [S] Russian-language users report stress problems with Qwen3-TTS out of the box, motivating external stress placement or stress-aware fine-tunes. — [QwenLM/Qwen3-TTS discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185)

**Text normalization**
- [V] RUNorm (Den4ikAI): Apache 2.0; three sizes — small FRED-T5-95M (fast, popular cases), medium ruT5-base 222M, large FRED-T5-Large 860M; normalizes numbers, expands abbreviations, converts Latin to Cyrillic, spells acronyms phonetically (GPT → "джи пи ти"); PyTorch, device-configurable; `pip install runorm`; 49 stars; no quantified speed or accuracy in README. — [Den4ikAI/runorm](https://github.com/Den4ikAI/runorm)
- [S] saarus72/russian_text_normalizer: fine-tune of FRED-T5 large 820M on ficbook/librusec/pikabu sentences inverse-normalized with a modified `word_to_number_ru`; Kaggle Text Normalization Challenge data added for Latin, which "made performance worse on numbers"; usage requires wrapping numbers and Latin words in square brackets followed by T5 sentinel tokens, numbers split into groups of 3 digits. — [saarus72/russian_text_normalizer](https://huggingface.co/saarus72/russian_text_normalizer), [saarus72/text_normalization](https://github.com/saarus72/text_normalization)
- [S] NeMo text processing (`nemo_text_processing`, WFST grammars via Pynini/OpenFst): Russian (ru) is supported for both text normalization and inverse text normalization; there is a fast deterministic mode and a context-aware mode. — [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing), [NeMo WFST text normalization docs (24.12)](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html)
- [S] Other small models on HF: `intx82/byt5-textnorm-ru`, `maximxls/text-normalization-ru-terrible`. — [intx82/byt5-textnorm-ru](https://huggingface.co/intx82/byt5-textnorm-ru), [maximxls/text-normalization-ru-terrible](https://huggingface.co/maximxls/text-normalization-ru-terrible)
- [S] Silero published a Russian text normalization system in 2020 (older info), and Silero TTS v5_ru is reported to normalize numbers, fractions and dates into words. — [silero.ai: Russian text normalization (2020)](https://www.silero.ai/russian-text-normalization/), [kost-t-human/ruvoice-tts](https://github.com/kost-t-human/ruvoice-tts)
- [S] Community rule-based pipelines for Russian TTS preprocessing exist (e.g. Balamoote/tts-scripts; alphacep/awesome-russian-speech lists tools). — [Balamoote/tts-scripts](https://github.com/Balamoote/tts-scripts), [alphacep/awesome-russian-speech](https://github.com/alphacep/awesome-russian-speech)

**English words / transliteration**
- [S] Habr (2020, older info) on the Kaggle-style normalization task: non-standard transliteration was the biggest problem; mapping CMUdict phonemes to Russian failed because of dictionary coverage (134k words); the `transliterate` package gave 15% accuracy; 40+ hand rules reached 50%. — [Habr 491260](https://habr.com/ru/articles/491260/)
- [V] RUNorm covers Latin→Cyrillic conversion and letter-by-letter acronyms as part of the model. — [Den4ikAI/runorm](https://github.com/Den4ikAI/runorm)

### Inferences
- Recommended order for an LLM-answer fragment: strip markdown/code/links/emoji → split into sentence/clause → normalize (numbers, dates, abbreviations, Latin) → ё restoration + stress → engine. Stress tools must run after normalization because the words produced from numbers also need stress and case-correct forms.
- silero-stress is the better fit for a latency-sensitive CPU path (no ONNX/transformers dependency, ~30 ms per paragraph, MIT, release two weeks before this note); RUAccent remains a reasonable alternative with ONNX and a GPU option. The accuracy comparison between them is unresolved: both headline numbers come from the respective authors' own test sets.
- If the TTS engine is Silero v5_ru, stress/homographs (and reportedly number normalization) are already built in, so external stress placement is redundant there; for Qwen3-TTS, XTTS, Piper (espeak-ng phonemization) and Chatterbox, external normalization is needed, and whether those engines honour `+`/U+0301 stress marks depends on the engine and must be checked per engine.
- T5-based normalizers are heavy for a per-fragment hot path (95M is the only plausible CPU size); a hybrid — cheap regex/WFST or `num2words`-style rules for plain numbers and a neural model only for ambiguous fragments — is the pragmatic design. This is inference; no source benchmarks it.
- The cheapest mitigation for English words and markdown is at the LLM prompt: instruct the model to answer in plain spoken-style Russian without markdown, with numbers spelled out where possible — preprocessing then only handles leaks.

### Gaps
- CPU speed of RUNorm (any size), saarus72 normalizer and NeMo Russian TN: no figures found.
- Accuracy of any Russian normalizer on LLM-style output (markdown, mixed Latin, units, dates): no benchmark found.
- RUAccent: current version, last release date and final license (MIT vs Apache-2.0) not confirmed; no CPU speed numbers in README.
- `num2words` Russian quality (case/gender agreement), `russtress`, `omogre` maintenance status in 2026: not researched/confirmed beyond mentions.
- Whether silero-stress v1.5's license remained MIT was not restated in the 2026 article (MIT per the 2025 article and a 2026 search snippet).
- No independent (third-party) benchmark comparing silero-stress and RUAccent was found.
- Which stress-mark syntax each local engine accepts (Qwen3-TTS, Chatterbox, XTTS, Piper) was not verified.

---

## 6. Browser / client playback for streamed TTS audio

### Takeaway
Primary-source evidence gathered here is thin: servers in this space overwhelmingly emit raw PCM (or WAV) in chunks, and the documented browser pattern for glitch-free playback of such a stream is an AudioWorklet fed from a ring buffer. Documented pitfalls are mostly about queue management (dropped or skipped sentences, leaked audio elements, double-fired end events) rather than codecs.

### Cited Findings
- [V] vLLM-Omni's streaming paths require `response_format` `pcm` or `wav`; the WebSocket sends binary PCM frames bracketed by `audio.start` / `audio.done` JSON messages. — [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [V] ElevenLabs' WebSocket sends base64 audio inside JSON with optional alignment data and an `isFinal` flag. — [ElevenLabs realtime TTS guide](https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts)
- [S] Chrome's AudioWorklet design-pattern article: a ring buffer between the data source and `AudioWorkletProcessor` solves the buffer-size mismatch (the processor pulls fixed-size render quanta from the ring buffer). — [Chrome for Developers: Audio worklet design pattern](https://developer.chrome.com/blog/audio-worklet-design-pattern)
- [S] A ready-made library for this pattern exists: `@ain1084/audio-worklet-stream`. — [ain1084/audio-worklet-stream](https://github.com/ain1084/audio-worklet-stream)
- [S] Open WebUI pitfalls: TTS playback stops after N responses on Android Chromium because leaked audio streams hit Chrome's stream limit (issue #29969); voice mode did not start playback until the entire audio file was received (issues #16644, #16457); sentences skipped when several arrive quickly (issue #19861). — [open-webui #29969](https://github.com/open-webui/open-webui/issues/29969), [#16644](https://github.com/open-webui/open-webui/issues/16644), [#19861](https://github.com/open-webui/open-webui/issues/19861)
- [V] Playback queue must guard against simultaneous "ended" and "error" events advancing the queue twice. — [dev.to, 2026-08-11](https://dev.to/renga154/where-to-split-a-sentence-for-streaming-tts-is-decided-by-one-number-4o6b)
- [V] RealtimeTTS's browser demo ("Voice Studio") defaults to 0 ms additional browser buffering as of 0.8.10 and fixed "artificial rebuffering at audio boundaries" in 0.8.9 — i.e. per-chunk rebuffering is a known source of audible gaps. — [RealtimeTTS releases](https://github.com/KoljaB/RealtimeTTS/releases)
- [S] Chatterbox-TTS-Server applies 20 ms crossfades between streamed chunks server-side. — [Chatterbox-TTS-Server releases](https://github.com/devnen/Chatterbox-TTS-Server/releases)

### Inferences
- For a same-origin web client, PCM (16-bit or float32, 24 kHz mono — the native rate of Qwen3-TTS/CosyVoice3 per vLLM-Omni docs) over a binary WebSocket into an AudioWorklet ring buffer is the simplest gapless design: no container parsing, sample-accurate concatenation, instant flush on barge-in (clear the ring buffer), and a playback cursor that tells the server how much was actually heard. Bandwidth (~48 KB/s at 24 kHz s16) is irrelevant on LAN/SSH-tunnel setups.
- Per-sentence `<audio>`/`AudioBufferSourceNode` playback of separate WAV/MP3 files is what produces the documented pitfalls (gaps between elements, leaked elements, event-ordering bugs); if used, schedule buffers on the AudioContext clock rather than chaining on `ended`.
- Inference without a fetched source: compressed streaming (Opus in WebM/Ogg via MediaSource, or WebCodecs `AudioDecoder`) only pays off over constrained networks and adds encoder delay plus browser-compatibility work; AudioContext autoplay policy (must be resumed from a user gesture) and sample-rate mismatch between stream and `AudioContext` are standard traps.

### Gaps
- No primary sources were fetched on MediaSource Extensions + Opus/WebM for TTS streaming, Safari/iOS limitations, WebCodecs, or autoplay policy; the codec-related statements above are inferences, not cited facts.
- No measured comparison of client-side latency (AudioWorklet vs MSE vs chained `<audio>`) was found.
