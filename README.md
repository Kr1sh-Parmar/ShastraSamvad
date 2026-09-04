# Shastra Samvad

An offline, scripture-grounded Guru–Shishya conversational device.

## What it does

A student asks a question — by typing or speaking. The system retrieves the
verses of the Bhagavad Gita and Mahabharata most relevant to the question,
builds a prompt that hands those verses to a small local language model, and
streams back an answer as the Guru's voice. Before any sentence is spoken, it
is checked against the verses it claims to be citing: if a sentence names an
anchor that wasn't retrieved, drifts off-topic from what it cites, or
contradicts the verse it names, it is withheld rather than spoken. Nothing
leaves the device to answer a question — speech-to-text, retrieval,
generation, verification, and text-to-speech all run locally, so the whole
pipeline works with no network connection.

## Architecture overview

Seven stages, one turn at a time:

```
 student speech/text
        |
        v
 [1] STT               faster-whisper (dev) / whisper.cpp (device)
        |
        v
 [2] Dialogue routing   rule phrases, then embedding similarity to
        |               per-mode example centroids -> TEACH / DOUBT /
        |               DEBATE / COUNSEL  (+ harm-screen guardrail)
        v
 [3] Verse retrieval    FAISS flat index, ~16.5k chunks, per-mode
        |               strategy (sequential / anchored / two-sided /
        |               broad)
        v
 [4] Prompt building    persona + rules + mode instruction + the
        |               retrieved verses, as a system/user pair
        v
 [5] LLM generation     phi4-mini via Ollama (dev) or llama-cpp-python
        |               (device), streamed token by token
        v
 [6] Citation           per sentence: cosine similarity (invention,
     verification        drift) then NLI cross-encoder (contradiction)
        |               -> speak it, or withhold it
        v
 [7] TTS                Piper (WAV) / browser speechSynthesis fallback
        |
        v
 spoken answer + citation panel (verses consulted, which were cited)
```

Stages 3–6 repeat per sentence as the LLM streams — a sentence is verified
and can be spoken while the next sentence is still being generated, rather
than waiting for the whole answer.

## Key design decisions

- **phi4-mini, not qwen3:4b.** qwen3:4b is a reasoning model: on these
  rule-heavy prompts it spent ~1800 tokens deliberating and produced no
  answer at all, and with `think=false` it wrote that deliberation into the
  reply instead. phi4-mini answers directly in 110–220 tokens.
- **`nli-deberta-v3-small`, not the `xsmall` sibling.** xsmall reads "let the
  results go" as abandoning the duty, and scored the plainest true reading of
  Gita 2.47 as a contradiction at 0.970 — against a weakest *true*
  contradiction of 0.997, a 0.027 margin. `nli-deberta-v3-small` separates the
  same 16 test sentences by 0.661. Pinned by a regression test
  (`test_pipeline.py::test_the_plainest_reading_of_2_47_is_not_called_a_contradiction`)
  so a downgrade fails a test, not a lesson.
- **Cosine + NLI, two stages, not one.** Cosine similarity is cheap and
  catches invented anchors, off-topic drift, and unsupported claims — but it
  scores *topic*, not entailment, so it passes a sentence that negates the
  verse it cites (measured: faithful paraphrase 0.840, its exact negation
  0.777 — no threshold separates those). A cross-encoder runs behind it,
  reading the (verse, sentence) pair rather than comparing two independent
  vectors, and rejects on **contradiction only** — never requires entailment,
  since faithful teaching often reads as `neutral` to an NLI model.
- **Sentence-level verification before speech, always.** The LLM's output is
  split into sentences as it streams; each is checked independently before
  being spoken, so an unsupported or contradicting claim is never voiced —
  only shown, greyed out, with its rejection reason.
- **Dual backends for every model-loading module.** `llm.py`
  (Ollama/llama-cpp-python), `stt.py` (faster-whisper/whisper.cpp) each keep
  a laptop dev path and an embedded path side by side, gated by a config
  flag, so the same codebase runs a demo on a laptop and a kiosk on a board
  without a fork.
- **`EMBEDDED_MODE`, one flag.** Setting it forces `LLM_USE_OLLAMA=False`,
  `STT_USE_WHISPER_CPP=True`, and `VAD_USE_SILERO=True` together, so a
  deployment script doesn't need to know or repeat which flags an embedded
  build requires.

## Corpus

| | count |
|---|---|
| Bhagavad Gita verse translations | 657 |
| Gita purport (commentary) chunks | 1,322 |
| Mahabharata passages (18 parvas, 2,108 sections) | 14,572 |
| **Total chunks indexed** | **16,551** |

Plus a curated cross-reference map: 1,576 links between Gita verses and
Mahabharata passages that illustrate them narratively, kept at cosine
similarity ≥ 0.80 (`CROSSREF_MIN_SCORE`) — see `backend/config.py` for how
that floor was measured against the corpus by hand-sampling scores from 0.70
to 0.80+.

Gita anchors look like `gita_2_47`; Mahabharata anchors look like
`mbh_udyoga_33`. Sourcing, licensing notes (the Gutenberg Ganguli translation
is public domain; the BBT Gita commentary is not), and the chunking/anchoring
details are documented in `software_architecture.md` and the scripts
themselves — see `scripts/00_fetch_mahabharata.py` and
`scripts/02_chunk_and_tag.py` in particular.

## Hardware targets

| Target | LLM | STT | VAD | Status |
|---|---|---|---|---|
| Laptop (dev/demo) | Ollama (`phi4-mini`) | faster-whisper | browser RMS (client-side) | working |
| Jetson Nano 4GB | llama-cpp-python, GGUF, `LLM_GPU_LAYERS` for CUDA offload | whisper.cpp | Silero VAD (`backend/vad.py`) | backends implemented, not yet run on real hardware |
| RK3588 | rkllm (separate NPU runtime, `.rkllm` model format) | whisper.cpp | Silero VAD | **not implemented** — needs its own rkllm-backed module, see Limitations |

Memory budget for an embedded build with every model resident at once
(from `backend/config.py`'s `--- embedded deployment ---` section):

| Component | Resident memory |
|---|---|
| LLM (phi4-mini, Q4_K_M GGUF) | ~2.0 GB |
| Embedder (bge-small) | ~130 MB |
| NLI (deberta-v3-small) | ~400 MB (0 once idle, if `NLI_MODEL_LAZY`) |
| Whisper (base.en) | ~150 MB |
| Piper TTS | ~60 MB |
| FAISS index + corpus | ~50 MB |
| **Total** | **~2.8 GB** — leaves ~1.2 GB for OS/buffers on a 4GB board |

## Project structure

```
backend/
  main.py               FastAPI orchestrator: WS /chat and /mic, /stt, /tts,
                         /texts, /verse/{id}, /health; ties every stage together
  config.py              every tunable constant, with measured rationale in comments
  dialogue_manager.py     mode classification (rules + embedding centroids),
                         lesson-position bookkeeping, harm-content screening
  retriever.py            per-mode verse retrieval over the FAISS index
  crossref.py             Gita <-> Mahabharata illustrative-link lookup
  prompt_builder.py       persona, per-mode instructions, verse injection
  llm.py                  streaming LLM client: Ollama or llama-cpp-python
  citation_verifier.py    "no verse, no answer" — cosine + NLI two-stage gate
  nli.py                  NLI cross-encoder wrapper (contradiction detection)
  embed.py                shared sentence-embedding model (bge-small)
  stt.py                  speech-to-text: faster-whisper or whisper.cpp
  tts.py                  speech synthesis: Piper, or None (client falls
                         back to browser speechSynthesis)
  vad.py                  server-side Silero VAD for barge-in/endpointing
                         on a kiosk build with no browser doing it in JS
  test_pipeline.py        pytest suite: verifier, dialogue routing, retriever,
                         prompt formatting, corpus chunking, harm guardrail

frontend/
  app/page.jsx            Next.js/React dev & demo frontend
  app/layout.jsx           root layout, page metadata
  app/globals.css          the Guru's dark/warm visual theme
  kiosk.html               standalone, dependency-free HTML build for an
                         embedded touchscreen kiosk (no Node, no build step)
  package.json, next.config.mjs

scripts/
  00_fetch_mahabharata.py  download the Ganguli Mahabharata (Project Gutenberg)
  01_clean_texts.py        raw source -> clean, structure-marked text
  02_chunk_and_tag.py      verse-anchored chunking (chunk ids ARE citation anchors)
  03_build_embeddings.py   embed every chunk, build the FAISS index
  04_build_crossref.py     build the Gita <-> Mahabharata cross-reference map
  05_fetch_piper.py        download Piper + an English voice
  06_measure_turns.py      measure spoken-vs-withheld rate across whole turns
  smoke_chat.py            drive WS /chat end to end, print what happened

software_architecture.md  the fuller architecture spec these modules implement
webapp_context.md         additional project context
requirements.txt          Python dependencies (backend + data pipeline)
```

## Setup and running

**Prerequisites**: Python 3.10+, Node.js (for the Next.js frontend only —
`kiosk.html` needs nothing), and [Ollama](https://ollama.com/download) for
the laptop/dev LLM path.

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. the LLM (dev path)
ollama pull phi4-mini

# 3. build the corpus and FAISS index — NOT included in the repo, must be run
python scripts/00_fetch_mahabharata.py
python scripts/01_clean_texts.py
python scripts/02_chunk_and_tag.py
python scripts/03_build_embeddings.py   # ~17k chunks, several minutes on CPU
python scripts/04_build_crossref.py

# 4. the voice (optional — skip it and the browser speaks instead)
python scripts/05_fetch_piper.py

# 5. run the backend
python -m uvicorn backend.main:app --port 8000

# 6a. frontend option A: Next.js dev server
cd frontend && npm install && npm run dev      # UI on :3001

# 6b. frontend option B: kiosk build, no server needed
# open frontend/kiosk.html directly in a browser, or launch Chromium with
# --kiosk pointed at it (see the comment header in kiosk.html)
```

The embedder and the NLI cross-encoder download themselves from Hugging Face
on first use. `GET /health` reports whether the index, Ollama, Whisper,
Piper, VAD, and the cross-reference map are each ready.

## Configuration

Every tunable lives in `backend/config.py`, with the measurement behind each
value documented inline as a comment — the file's own rule is "retune here,
nowhere else." Highlights:

| Setting | Value | Why |
|---|---|---|
| `TOP_K` | 6 | verses per non-DEBATE retrieval |
| `TEACH_WINDOW` | 2 | verses taught per TEACH turn |
| `DEBATE_TOP_K` | 8 | verses fetched per side in DEBATE — starving this made the model argue from memory |
| `CROSSREF_MIN_SCORE` | 0.80 | hand-sampled cutoff; below it, links are mostly noise |
| `TEMPERATURE` | 0.3 | LLM sampling |
| `CITED_MIN_SIM` | 0.60 | floor: a sentence vs. the verse it explicitly cited |
| `CITATION_MIN_SIM` | 0.65 | floor: an uncited sentence vs. its best-matching retrieved verse |
| `NLI_CONTRADICTION_MAX` | 0.90 | ceiling: contradiction probability before a sentence is rejected |
| `LLM_MODEL_PATH` | `models/phi4-mini-q4.gguf` | GGUF path for the llama-cpp-python backend |
| `WHISPER_CPP_MODEL_PATH` | `models/ggml-base.en.bin` | whisper.cpp model path |

`EMBEDDED_MODE` (default `False`) is the master flag: setting it to `True`
overrides `LLM_USE_OLLAMA`, `STT_USE_WHISPER_CPP`, and `VAD_USE_SILERO` to
their embedded-appropriate values in one place, at the bottom of the file.

## Frontends

- **`frontend/app/page.jsx`** — the Next.js/React frontend used for laptop
  development and demos. WebSocket chat, source/chapter pickers, streaming
  sentence display, a citation panel, push-to-talk mic input, and
  RMS-loudness barge-in.
- **`frontend/kiosk.html`** — a standalone, zero-dependency replica of the
  same core loop (same WebSocket protocol, same audio-queue and barge-in
  logic) as a single self-contained HTML file: no React, no build step, no
  CDN. Meant for a Chromium instance launched with `--kiosk` on an embedded
  touchscreen, or opened directly as a `file://` URL.

## Ethical guardrails

- **Harmful content is screened before the LLM is ever called**
  (`dialogue_manager.screen`) — a trust boundary checked first, so a harmful
  request costs no generation and gets no model-produced text at all.
- **A dedicated crisis reply for self-harm queries**, distinct from the
  generic refusal used for requests to harm others — pointing the student
  toward a person or professional, not toward the scriptures as a substitute.
- **Citation verification means no hallucinated scripture reaches the
  student.** Every sentence must name a verse that was actually retrieved
  (an invented anchor is rejected outright), must not drift topically from
  what it cites, and must not contradict it — enforced sentence by sentence,
  before speech, not as a post-hoc check.

## What's working now

- Full seven-stage pipeline end to end on a laptop with Ollama: STT →
  dialogue routing → retrieval → prompt building → generation → citation
  verification → TTS.
- Four dialogue modes (TEACH, DOUBT, DEBATE, COUNSEL) with distinct retrieval
  and prompting strategies.
- Two-stage citation verifier (cosine + NLI) with a pytest regression suite
  covering invented anchors, drift, unsupported claims, and negation.
- Harm-content screening and a distinct crisis reply.
- Sentence-by-sentence streaming with per-sentence audio via Piper, or
  browser `speechSynthesis` fallback.
- Barge-in (client-side RMS) and mid-turn interrupt.
- Dual backends, implemented and unit-tested for import/wiring, but not yet
  run end-to-end on real embedded hardware: `llm.py` (llama-cpp-python),
  `stt.py` (whisper.cpp), `vad.py` (Silero VAD) plus the `/mic` WebSocket
  endpoint for server-side barge-in/endpointing on a kiosk build.
- `frontend/kiosk.html`, a standalone touchscreen-ready frontend requiring no
  Node.js dev server.

## Known limitations / future scope

- **The FAISS index is not in the repo.** `scripts/00`–`04` must be run once
  to fetch the source texts and build `data/index/corpus.faiss` before the
  backend can start; `retriever.py` raises clearly if it's missing.
- **Sanskrit language support** — showing the Devanagari shloka alongside its
  English translation is a feasible addition (the source data already
  carries verse-level structure); full Sanskrit speech input/output is not
  practical with the current STT/TTS models, which are English-only.
- **RK3588 NPU integration is not implemented.** `llm.py`'s embedded path
  covers Jetson Nano and any generic Linux board via llama-cpp-python; RK3588
  needs a separate module built against rkllm's own Python bindings and
  `.rkllm` model format, which nothing in this repo currently provides.
- **The `/mic` endpoint's transcription step needs a WAV header.** It
  concatenates raw PCM16LE chunks and passes them to `stt.transcribe()`,
  which expects an encoded container (webm/ogg/wav) and decodes via PyAV —
  as written, the raw PCM will fail to decode. This needs either a minimal
  WAV header wrapped around the buffer before transcription, or a dedicated
  raw-PCM decode path.
- **No end-to-end testing on actual embedded hardware yet.** The
  llama-cpp-python, whisper.cpp, and Silero VAD backends are implemented and
  gated behind config flags, but have not been run on a Jetson Nano or any
  other board — memory and latency numbers in this README and in
  `config.py` are measured/estimated, not device-verified.

## Team

Kr1sh-Parmar and contributors.

## License

No license file is currently present in this repository. *(Add one — e.g.
MIT, Apache-2.0 — before treating this as open for reuse; the Mahabharata
translation and Gita commentary each carry their own separate licensing
terms, documented in `software_architecture.md`.)*
