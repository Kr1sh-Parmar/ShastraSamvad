# Testing and Measurement

How to check the system works, and how to record numbers you can show a
professor or use as patent evidence. Run these on the college GPU computer
once [GPU_COMPUTER_SETUP.md](GPU_COMPUTER_SETUP.md) is done.

## 1. Unit tests

```powershell
python -m pytest backend/test_pipeline.py -q
```

The suite covers the citation verifier (invented anchors, drift, unsupported
claims, negation), dialogue routing, retrieval, prompt formatting, corpus
chunking and the harm screen.

Build the index first. Several tests read the real index and nothing skips
them, so they fail on a machine that has not run `scripts/00` to `04`. The
first run also downloads the embedder and NLI model.

## 2. Smoke test through the live server

With the backend running (`python -m uvicorn backend.main:app --port 8000`):

```powershell
python scripts/smoke_chat.py          # guardrail only, no LLM
python scripts/smoke_chat.py --llm    # four real turns through the model
```

The first form checks routing, retrieval and the refusal path without calling
the model. The second runs four real turns and prints the mode, the retrieved
verse ids and each sentence.

## 3. Measure whole turns

```powershell
python scripts/06_measure_turns.py              # 12 turns, about 3 minutes
python scripts/06_measure_turns.py run1.json    # also save the detail
```

It sends two turns per mode, plus a continued lesson, across both texts, and
reports per turn and in total:

| Measure | Meaning |
|---|---|
| spoken | sentences that passed the verifier |
| withheld | sentences blocked, grouped by reason |
| echoed | sentences copying a retrieved passage word for word |
| empty turns | turns that fell through to "no scriptural basis" |

### Read the variance before believing a difference

The script's own notes say twelve turns with **no code change** give a spoken
rate anywhere from 48% to 62% and an echo rate from 0% to 20%. So a single
before and after pair says nothing about a prompt change. Trust categorical
changes instead, such as a withheld reason appearing or disappearing (for
example `invented anchor` going from 3 to 0 and staying at 0 over three runs).

## 4. What to record, and what it can claim

Make a dated entry for every run. Copy this table into your logbook:

| Date | Machine | GPU and driver | LLM | Test | Result | Saved file |
|---|---|---|---|---|---|---|
| | | | phi4-mini via Ollama | pytest | passed / failed | |
| | | | phi4-mini via Ollama | smoke --llm | | |
| | | | phi4-mini via Ollama | 06_measure_turns | spoken % / withheld % / echo % | run1.json |

Also record these by hand, since no script measures them yet:

- time from pressing send to the first spoken sentence, for 5 or more turns
- peak GPU memory during a turn (`nvidia-smi`)

What these numbers can and cannot claim:

- **Can show:** the verifier withholds unsupported sentences, the four modes
  route and retrieve differently, and sentence-by-sentence streaming reaches
  the speaker before the answer is finished.
- **Cannot show:** the Jetson Nano's latency, memory fit, or the 5 to 10
  second target. The Nano paths (llama-cpp-python, whisper.cpp, Silero VAD)
  are not run on this machine. Label every figure with the hardware that
  produced it.

## 5. Missing measurements

These do not exist yet and need adding if they are to be claimed:

- a latency script that logs time to first sentence for every turn
- a labeled question set scoring citation precision and recall, beyond the
  unit tests
- anything run on the Jetson Nano
