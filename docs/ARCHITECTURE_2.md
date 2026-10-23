# Shastra Samvad — Architecture 2 (proposed four-layer design)

> **This is a proposed design, not what is built today.** The repository
> currently implements the seven-stage pipeline in the root
> [ARCHITECTURE.md](../ARCHITECTURE.md): FAISS and bge-small retrieval, a
> cosine plus NLI citation verifier, and the Gita and Mahabharata only. The
> Vyakarana engine, ChromaDB, the 0.72 gate, the Ramayana and Panchatantra,
> the Unity avatar, Rhubarb lip sync and MediaPipe gestures below do not exist
> in the code yet.

Four layers, named for the parts of a person: **Deha** (body), **Shravana**
(listening), **Smriti** (memory) and **Vāṇī** (speech).

## Data flow

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ LAYER 1 · DEHA (देह) — hardware                                        │
 │  Orin Nano 8GB · 7" touchscreen · USB codec · amp + speaker · fan      │
 └──────────────────────────────────────────────────────────────────────┘
        │ mic audio                                        ▲ speaker audio
        ▼                                                  │ + avatar frames
 ┌─────────────────────────────┐                  ┌──────────────────────┐
 │ LAYER 2 · SHRAVANA (श्रवण)    │                  │ LAYER 4 · VĀṆĪ (वाणी)  │
 │ mic → Silero VAD →          │                  │ LLM (Ollama, quant.) │
 │ Whisper.cpp (CUDA) → text   │                  │  → Piper/Coqui TTS   │
 └──────────────┬──────────────┘                  │  → Rhubarb visemes   │
                │ text query                      │  → Unity 3D avatar   │
                ▼                                 │  ← MediaPipe gesture │
 ┌───────────────────────────────────────────┐    └──────────▲───────────┘
 │ LAYER 3 · SMRITI (स्मृति) — memory / RAG      │               │ grounded
 │ Vyakarana pre-process (Sandhi split, Dhatu)│               │ context
 │   → embed (all-MiniLM-L6-v2, 384-d)        │               │
 │   → ChromaDB retrieval                     │               │
 │   → cosine gate ≥ 0.72 ─── pass ───────────┼───────────────┘
 │            └── fail → "I cannot find a     │
 │                grounded answer" (no LLM)   │
 └───────────────────────────────────────────┘
```

A turn is: **speak → Layer 2 → Layer 3 → (gate) → Layer 4 → hear and see the
answer.** A query that fails the gate skips the language model entirely and
returns the fixed refusal through the same speech and avatar path.

## Layer 1 — Deha (देह): body / hardware

| Part | Choice |
|---|---|
| Compute | NVIDIA Jetson Orin Nano 8 GB — 1024-core Ampere GPU, 40 TOPS, Arm Cortex-A78AE CPU |
| Display | Waveshare 7" HDMI LCD, 1024×600, capacitive, USB touch digitizer, driven through a DisplayPort-to-HDMI adapter |
| Audio in/out | USB audio codec (Waveshare USB Sound Card HAT): microphone in, line out |
| Amplifier + speaker | PAM8403 amplifier board + DFRobot FIT0502 passive speaker (3 W, 8 Ω) |
| Power | The kit's own 19 V supply through the DC barrel jack |
| Cooling | Heatsink and fan that ship with the kit |
| Storage | 64 GB microSD (JetPack OS, models, corpus) |
| Cabling | DisplayPort-to-HDMI adapter, 30 cm HDMI, USB-A to micro-USB (touch digitizer), Ethernet |

> Use the supply that came with the kit. Do not connect a 5V/4A adapter or a
> USB-C PD bank, and check the datasheet before powering the board any other
> way.

## Layer 2 — Shravana (श्रवण): listening / STT

- **Silero VAD** detects speech start and end, and detects the user speaking
  over the avatar so playback can stop.
- **Whisper.cpp** (tiny or base model, CUDA-accelerated on the Orin Nano)
  transcribes the segmented audio.
- **Input:** USB audio codec microphone, 16 kHz mono.
- **Pipeline:** mic → VAD → Whisper STT → text query.

## Layer 3 — Smriti (स्मृति): memory / RAG

- **Corpus:** SQLite staging database → ChromaDB vector store.
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions, sentence-transformers).
- **Guardrail Protocol:** a hard cosine-similarity threshold of **0.72**.
  If the best retrieved passage scores below it, the system answers "I cannot
  find a grounded answer" and the language model is never called.
- **Vyakarana Grammar Engine:** Sanskrit Sandhi-splitting and Dhatu-mapping,
  applied as a pre-processor to both corpus text and query before embedding.
- **Cross-textual linking:** Dhatu-based concept mapping across the Gita,
  Mahabharata, Ramayana and Panchatantra.
- **Query pipeline:** text query → Vyakarana pre-processing → embedding →
  ChromaDB retrieval → cosine threshold gate → context assembly.

The 0.72 value is fixed by the claim, but the score distribution depends on
the embedding model, so measure real in-corpus and out-of-corpus queries
against it (see the development guide) before treating it as validated.

## Layer 4 — Vāṇī (वाणी): speech / avatar output

- **LLM:** Ollama with a quantized small model (likely Phi-3-mini or similar)
  that fits the Orin Nano's 8 GB shared memory alongside every other
  component. It receives only the gated, retrieved context.
- **TTS:** Piper or Coqui TTS, offline and low-latency.
- **3D avatar:** a Unity-based animated guru character on the touchscreen.
- **Lip sync:** Rhubarb Lip Sync maps phonemes to visemes for the avatar's
  mouth.
- **Gesture tracking:** MediaPipe hand and body gesture recognition for
  interactive control.
- **Output:** the avatar speaks the grounded answer with synchronized lip
  movement on the 7" display.

## Memory budget (8 GB shared CPU/GPU)

Everything below must be resident together, so the budget is a real
constraint, not a footnote. Fill in measured numbers during bring-up.

| Component | Estimate | Measured |
|---|---|---|
| LLM, 4-bit quantized | ~2–3 GB | TBD |
| Unity avatar runtime | TBD | TBD |
| Whisper tiny/base | ~0.2–0.5 GB | TBD |
| MiniLM + ChromaDB | ~0.3 GB | TBD |
| TTS voice | ~0.1 GB | TBD |
| OS + JetPack | ~1.5 GB | TBD |
