# Shastra Samvad — Web App Build Context
*For building the full software pipeline as a web application on your laptop (prototyping & testing)*

> **How to use this document.** This is a complete context/spec you can build from directly, or paste into an AI coding assistant as the source of truth. It targets a **laptop web app** for fast prototyping. Everything here maps cleanly to the eventual on-device build — the only differences are that on the laptop you can use bigger models and a browser front-end, and you can start with easier tools (e.g. Ollama, browser speech APIs) before swapping in the device-grade ones.

---

## 1. Goal of the prototype

A local web application where you:
1. Open a browser, pick a text (Gita / Mahabharata) and a chapter.
2. Choose or auto-detect a mode (Teach / Doubt / Debate / Counsel).
3. Speak or type a question.
4. Get a scripture-grounded, streamed answer — as text and as speech — with the cited verses shown.

Runs entirely on your laptop. No cloud APIs.

---

## 2. Architecture (web version)

```
Browser (React front-end)
  |  - mic capture, playback, chat UI, chapter/mode picker
  |  - WebSocket for streaming tokens + audio
  v
FastAPI backend (Python, localhost)
  |-- /stt        Whisper (faster-whisper)
  |-- /chat (WS)  Dialogue Manager -> Retriever -> Prompt -> LLM (stream) -> Citation check
  |-- /tts        Piper (or browser TTS for quick start)
  |-- /texts      chapter list / verse lookup
  v
Local model + data layer
  - Ollama or llama.cpp  (LLM)
  - sentence-transformers (embeddings)
  - FAISS or ChromaDB    (vector index)
  - data/  (chunked, verse-tagged corpus + cross-ref index)
```

The web version keeps the **exact same logical modules** as the device (STT → Dialogue Manager → RAG → LLM → Citation Verifier → TTS). You are building the real system; only the packaging differs.

---

## 3. Recommended stack (laptop)

| Layer | Tool | Why |
|-------|------|-----|
| Backend | **FastAPI** + Uvicorn | async, WebSocket streaming, simple |
| Front-end | **React + Vite** (or plain HTML/JS to start) | quick UI, mic + audio APIs |
| STT | **faster-whisper** (`base` or `small`) | fast, accurate, local |
| VAD | webrtcvad / silero-vad | endpointing + barge-in |
| LLM | **Ollama** running `qwen3:4b` / `phi4-mini` (or llama.cpp) | trivial local setup, streaming API |
| Embeddings | **sentence-transformers** `bge-small-en-v1.5` | strong small embedder |
| Vector DB | **FAISS** (or ChromaDB for convenience) | fast local retrieval |
| TTS | **Piper** (device-grade) or browser `speechSynthesis` (quick start) | offline speech out |

**Quick-start shortcut:** for the very first end-to-end loop, use the browser's built-in `speechSynthesis` (TTS) and `SpeechRecognition` where available — get the pipeline working, then swap in faster-whisper + Piper to match the device.

---

## 4. Suggested project structure

```
shastra-samvad/
├── backend/
│   ├── main.py                # FastAPI app, routes, WebSocket
│   ├── stt.py                 # faster-whisper wrapper
│   ├── tts.py                 # Piper wrapper (+ browser fallback)
│   ├── dialogue_manager.py    # intent classification + state machine
│   ├── retriever.py           # FAISS search, mode-specific strategies
│   ├── crossref.py            # Gita <-> Mahabharata linking
│   ├── prompt_builder.py      # mode templates + persona
│   ├── llm.py                 # Ollama/llama.cpp streaming client
│   ├── citation_verifier.py   # "no verse, no answer" check
│   └── config.py
├── data/
│   ├── raw/                   # downloaded Gita + Mahabharata text
│   ├── chunks/                # verse-tagged JSONL chunks
│   ├── index/                 # FAISS index + id map
│   └── crossref/              # precomputed cross-reference map
├── scripts/
│   ├── 01_clean_texts.py      # normalize raw text
│   ├── 02_chunk_and_tag.py    # verse-anchored chunking
│   ├── 03_build_embeddings.py # embed + build FAISS
│   └── 04_build_crossref.py   # build cross-reference index
├── frontend/
│   ├── src/App.jsx            # chat UI, mic, playback, pickers
│   └── ...
└── README.md
```

---

## 5. Data preparation pipeline (do this first)

The quality of everything downstream depends on clean, verse-tagged data.

**Sources (public-domain):**
- Mahabharata — Ganguli translation, all 18 Parvas — sacred-texts.com/hin/maha/
- Bhagavad Gita — Telang translation — sacred-texts.com/hin/sbe08/

**Steps:**
1. **Clean** (`01_clean_texts.py`): strip HTML/footnotes, normalize whitespace, keep chapter/section markers.
2. **Chunk + tag** (`02_chunk_and_tag.py`): split into retrievable chunks, each with a **verse anchor** in metadata:
   ```json
   {"id": "gita_2_47", "text": "You have a right to action alone...", "source": "Gita", "chapter": 2, "verse": 47}
   {"id": "mbh_udyoga_33", "text": "...", "source": "Mahabharata", "parva": "Udyoga", "section": 33}
   ```
   - Gita: chunk per verse (or small group of verses).
   - Mahabharata (huge, ~1.8M words): chunk per section/passage; aggressive chunking is fine — retrieval stays fast.
3. **Embed + index** (`03_build_embeddings.py`): embed each chunk with `bge-small-en-v1.5`, build a FAISS index, save an `id → metadata` map.
4. **Cross-reference** (`04_build_crossref.py`): for each Gita verse, find the most semantically similar Mahabharata passages (cosine over embeddings); keep the top links, optionally hand-curate the strongest ones. Save as `crossref/gita_to_mbh.json`.

---

## 6. The four modes

The Dialogue Manager classifies each turn and picks a retrieval strategy + prompt template.

| Mode | Trigger | Retrieval | Prompt behavior |
|------|---------|-----------|-----------------|
| **TEACH** | user starts/continues a chapter | sequential verses in the chapter | explain the current verse(s) simply, then invite questions |
| **DOUBT** | question during a lesson | focused search around doubt + lesson context | answer the specific doubt, tie back to the verse |
| **DEBATE** | user challenges/argues | verses for AND against the position | argue the scriptural position, engage counter-points, stay grounded |
| **COUNSEL** | general life/ethical question | broad semantic search for principles | map situation to principles, ethical + wise, refuse harmful/out-of-scope |

**Intent classification (simple to start):** keyword/heuristic rules + embedding similarity to a few labeled examples per mode. Upgrade to a small classifier later. The current chapter/lesson state biases the decision (e.g., a question during an active lesson leans DOUBT).

---

## 7. Prompt template skeleton

```
SYSTEM:
You are a wise, patient Guru teaching from the {source}. You speak clearly and kindly.
You may ONLY assert what is supported by the provided verses. If the verses do not
support an answer, say you cannot find scriptural basis and offer what the texts do say.
Cite verses by their anchors (e.g., Gita 2.47).

MODE: {mode}
{mode_specific_instruction}

RETRIEVED VERSES:
{for each verse}: [{anchor}] {text}

CONVERSATION SO FAR:
{history}

USER: {query}

GURU:
```

`{mode_specific_instruction}` swaps per mode (teach / clarify / debate / counsel). Keep the reasoning scratchpad short to protect latency.

---

## 8. Backend endpoints

| Endpoint | Type | Purpose |
|----------|------|---------|
| `POST /stt` | HTTP | audio blob → transcript (or do STT client-side to start) |
| `WS /chat` | WebSocket | streams: user turn in → mode + retrieved verses + token stream + cited verse ids out |
| `POST /tts` | HTTP | sentence → audio (Piper); or use browser TTS |
| `GET /texts` | HTTP | list texts/chapters |
| `GET /verse/{id}` | HTTP | fetch a verse by anchor (for citation display) |

**Streaming contract on `WS /chat`:** emit events in order — `{type: "mode", mode}`, `{type: "verses", items:[...]}`, then repeated `{type: "token", text}` / `{type: "sentence", text, audio_url}`, then `{type: "citations", ids:[...]}`, then `{type: "done"}`. The front-end plays each sentence's audio as it arrives (streaming TTS) and renders citations.

---

## 9. Front-end responsibilities

- Chapter/text/mode pickers.
- Mic capture (push-to-talk button to start) → send audio or transcript.
- Chat transcript with the Guru's answer streaming in.
- **Verse citation panel** showing the anchors behind each answer (this visibly demonstrates grounding — great for demos and patent evidence).
- Audio playback of streamed sentences; a "stop/interrupt" control (prototype of barge-in).

---

## 10. Build order (fastest path to a working loop)

1. **Data pipeline** (§5) — get chunks + FAISS index built. Nothing works without this.
2. **Retriever** — query → top-k verses in the terminal. Verify retrieval quality by hand.
3. **LLM + prompt** — pipe retrieved verses into Ollama, get a grounded text answer. Text-only, no speech yet.
4. **Web loop** — FastAPI `WS /chat` + a minimal React page: type a question → streamed grounded answer + citations.
5. **Add speech** — faster-whisper for input, Piper (or browser TTS) for output; stream sentence-by-sentence.
6. **Modes** — add the Dialogue Manager and the four mode templates.
7. **Citation Verifier** — enforce "no verse, no answer"; show citations.
8. **Cross-reference + counsel guardrail + barge-in** — the differentiators (and the patent points).

Get steps 1–4 solid first; that is the whole system in text form. Everything after is layering.

---

## 11. Laptop → device parity checklist

When you move to the Jetson Nano 4GB, only these change:
- Ollama → `llama.cpp` with the same 4B model quantized to Q4.
- faster-whisper → `whisper.cpp` `base.en`.
- Browser TTS → Piper (if you used the browser shortcut).
- React web page → the same front-end served locally to the touchscreen (or a lightweight kiosk UI).

The Dialogue Manager, Retriever, cross-reference index, prompt templates, and citation verifier are **identical** — that's the point of building it this way. Build once on the laptop; port the shells.

---

## 12. Models to pull on the laptop

```
# LLM (pick one to start)
ollama pull qwen3:4b
ollama pull phi4-mini

# Embeddings (via sentence-transformers, auto-downloads)
bge-small-en-v1.5

# STT
faster-whisper  (base or small)

# TTS
piper  (an English voice, e.g. en_US-*medium)
```

Start with `qwen3:4b` for its reasoning mode (good for Debate). If your laptop is tight on RAM, `phi4-mini` is lighter and still a strong reasoner.
