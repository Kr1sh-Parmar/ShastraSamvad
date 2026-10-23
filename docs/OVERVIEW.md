# Shastra Samvad (शास्त्र संवाद) — What We Are Building

*"Dialogue with the Scriptures."*

## In one paragraph

Shastra Samvad is a fully offline, portable device that recreates the
traditional Guru–Shishya (teacher–student) experience through conversation.
A student picks a text and chapter on a touchscreen, then speaks or types.
The device finds the most relevant verses of the **Bhagavad Gita** and the
**Mahabharata**, hands them to a small on-device language model, and speaks
the answer back as "the Guru". Every sentence is checked against the verses
it claims to cite before it is spoken, so the Guru does not invent scripture.
Nothing leaves the device: speech recognition, retrieval, generation,
verification and speech synthesis all run locally with no internet.

## What the student can do

Four interaction modes, chosen automatically from what the student says (or
forced from the screen):

| Mode | When | What the Guru does |
|---|---|---|
| **TEACH** | "teach me chapter 2", "continue" | Walks the chapter or parva verse by verse, two verses per turn, with the Gita commentary for each |
| **DOUBT** | a question during a lesson | Answers using the verse currently being taught plus focused search |
| **DEBATE** | "I disagree", "prove it" | Retrieves verses on both sides of the claim and argues from the text |
| **COUNSEL** | "I feel lost", "what should I do" | Maps a life situation onto principles found across both texts |

Mode is picked by rule phrases first, then by embedding similarity to example
sentences per mode. An active lesson nudges a question toward DOUBT.

## How a turn works

```
 speech/text → STT → dialogue routing → verse retrieval → prompt
             → LLM → per-sentence citation check → TTS → speaker + screen
```

1. **STT.** faster-whisper on a laptop, whisper.cpp on the device.
2. **Routing.** Classifies the mode, tracks lesson position and debate
   stance, and screens harmful requests before any generation.
3. **Retrieval.** A FAISS index with a different strategy per mode
   (sequential for TEACH, anchored for DOUBT, two-sided for DEBATE, broad for
   COUNSEL). A cross-reference map adds Mahabharata passages that illustrate a
   Gita verse.
4. **Prompt.** Persona, rules, mode instruction and the retrieved verses with
   their anchors.
5. **LLM.** phi4-mini, through Ollama on a laptop or llama-cpp-python on the
   device, streamed.
6. **Citation check, per sentence.** A sentence is withheld if it cites a
   verse that was not retrieved, drifts away from the verse it cites,
   asserts something no retrieved verse supports, or contradicts the verse it
   cites (checked by an NLI model). This is the "no verse, no answer" rule.
7. **TTS.** Piper, sentence by sentence, so the first sentence can be spoken
   while the next is still generating. The browser's own voice is the fallback.

The screen shows the transcript, which sentences were spoken or withheld
(with the reason), and the verses consulted.

The student can interrupt the Guru mid-answer. A browser build detects this
by microphone loudness; a kiosk build is designed to use Silero VAD on the
server.

## Source texts

| Text | Content in the index |
|---|---|
| Bhagavad Gita | 657 verse translations and 1,322 commentary chunks |
| Mahabharata (Ganguli translation, Project Gutenberg) | 14,572 passages across 18 parvas, 2,108 sections |
| **Total indexed** | **16,551 chunks** |

Plus 1,576 Gita-to-Mahabharata links kept at cosine similarity 0.80 or above.

Each chunk's id is its citation anchor, for example `gita_2_47` or
`mbh_udyoga_33`. Scope is **English only**. The Ganguli translation is public
domain, but the Gita commentary is not (see the License section of the
README), so check its terms before any redistribution.

## Target hardware

**NVIDIA Jetson Nano 4GB**, college-provided and not counted against the
budget. Around it: a 5" or 7" HDMI/USB capacitive touchscreen, a USB audio
codec for microphone and audio out, a small powered speaker, a 5V/4A power
bank into the barrel jack, and a heatsink with a PWM fan. The itemized parts
list, estimated prices and power notes are in the README. Planned spend is
about ₹14,600 against a ₹30,000 cap (prices are AI-approximated).

All models run resident at once, about 2.8 GB of the Nano's 4 GB (see the
memory table in the README).

## Goals

1. Fully offline conversational device for the Gita and Mahabharata.
2. Four modes: Teach, Doubt, Debate, Counsel.
3. Grounded answers: every claim traceable to a cited verse.
4. A natural 5–10 second response, with first spoken words in roughly 4–6
   seconds. This is a design target; it has not been measured on the Nano.
5. A working, portable product within budget.
6. Defensible intellectual property around the novel methods.

## Novelty and patent angle

The team's patent plan rests on five method claims plus one apparatus claim.
Each maps to code that exists in this repository:

| Claim | Where it lives |
|---|---|
| Verse-anchored, citation-enforced generation ("no verse, no answer") | `backend/citation_verifier.py`, `backend/nli.py` |
| Multimode Guru–Shishya dialogue controller | `backend/dialogue_manager.py`, `backend/retriever.py`, `backend/prompt_builder.py` |
| Low-latency sentence-streaming speech with barge-in | `backend/main.py`, `backend/tts.py`, `backend/vad.py` |
| Dual-corpus cross-referential retrieval | `backend/crossref.py`, `scripts/04_build_crossref.py` |
| On-device ethical guardrail with refusal and crisis reply | `dialogue_manager.screen` |
| Apparatus: the portable, network-independent device | hardware list in the README |

In India, Section 3(k) excludes a computer programme "per se", so each claim
is framed as a method with a technical effect on a specific device. File a
provisional application through a registered patent agent **before any public
demonstration**, run a prior-art search first, and keep a dated, signed
invention log. This is not legal advice.

## Where the project stands

**Working on a laptop with Ollama:** the full seven-stage pipeline, all four
modes, the two-stage citation verifier with a pytest suite, harm screening,
sentence-by-sentence streaming with Piper, push-to-talk, browser barge-in, and
a standalone `kiosk.html` frontend.

**Written but never run on real hardware:** the llama-cpp-python, whisper.cpp
and Silero VAD paths, and the `/mic` endpoint.

**Known gaps:**

- The `/mic` endpoint passes raw PCM to a decoder that expects an encoded
  container, so server-side barge-in will fail until that is fixed.
- No latency or memory measurement on the Jetson Nano yet.
- The FAISS index is not in the repo and must be built with `scripts/00` to
  `04`.
- No formal citation-accuracy benchmark beyond the unit tests.
- No LICENSE file.

## Out of scope

- Other languages (Sanskrit, Hindi) and other texts such as the Ramayana and
  Panchatantra.
- Cloud features and multi-user accounts.
- A 3D avatar, lip sync and gesture control. The interface today is the
  touchscreen web page with audio.

## Related documents

- [README.md](../README.md) — setup, configuration, hardware list
- [software_architecture.md](../software_architecture.md) — module contracts
- [ARCHITECTURE.md](../ARCHITECTURE.md) — pipeline diagram
