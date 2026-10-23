# Smoke Run: What Connects to What

Setup for the first test: the **college GPU computer** is the brain, the
**Jetson Orin Nano** is the kiosk (screen, touch, microphone, speaker), and
the **7" Waveshare LCD** is the Jetson's display.

> The target board is the **Jetson Orin Nano Developer Kit** and nothing
> else. Port details below come from that kit's documentation; check them
> against your board's labels before wiring.

## The picture

```
                    ┌────────────────────────────────────────────┐
                    │  COLLEGE GPU COMPUTER   (the brain)         │
                    │                                            │
                    │   Ollama + phi4-mini  ──(GPU)              │
                    │   FastAPI backend  :8000                   │
                    │     · dialogue routing   · retrieval       │
                    │     · citation verifier  · embedder + NLI  │
                    │     · speech-to-text     · Piper voice     │
                    │   data/  (FAISS index + corpus)            │
                    └──────────────────┬─────────────────────────┘
                                       │
                       Ethernet cable (or same LAN)
                       HTTP + WebSocket on port 8000
                                       │
 ┌─────────────────────────────────────┴───────────────────────────────────┐
 │  JETSON ORIN NANO   (the kiosk, runs no AI model in this test)           │
 │                                                                         │
 │   Chromium --kiosk  →  frontend/kiosk.html                              │
 │   draws the screen, captures the mic, plays the Guru's voice            │
 │                                                                         │
 │   [DisplayPort]──► DP-to-HDMI adapter ──► HDMI cable ──┐                │
 │   [USB-A]──► USB-A to micro-USB cable (touch) ─────────┤                │
 │   [USB-A]──► USB audio codec ──┬─ mic in ◄── microphone │                │
 │                                └─ 3.5mm out ─► PAM8403 amp ─► speaker    │
 │   [DC jack]◄── 19V power adapter     [fan header]◄── 40mm PWM fan        │
 │   [microSD]  JetPack OS + kiosk.html                                    │
 └─────────────────────────────────────────────────────┬───────────────────┘
                                                       ▼
                                        ┌────────────────────────────┐
                                        │ 7" WAVESHARE LCD 1024x600  │
                                        │ HDMI in + USB touch        │
                                        └────────────────────────────┘
```

## One question, across both machines

1. The student holds the talk button on the touchscreen and speaks.
2. The Jetson's browser sends the audio to the computer: `POST /stt`.
3. The computer transcribes it and returns the text.
4. The browser sends the question over the WebSocket: `WS /chat`.
5. The computer picks the mode, retrieves verses, builds the prompt and asks
   Ollama (on the GPU) to answer.
6. Each sentence is checked against the retrieved verses. Passing sentences
   go back to the Jetson, each with an audio address.
7. The Jetson fetches the audio (`GET /tts/ID`), plays it on the speaker and
   shows the sentence and citations on the LCD.

## Connection table

| From | To | Cable or link | Carries |
|---|---|---|---|
| Jetson DisplayPort | LCD HDMI input | DP-to-HDMI adapter plus HDMI cable | video |
| Jetson USB-A | LCD micro-USB | USB-A to micro-USB | touch (some models also take power here) |
| Jetson USB-A | USB audio codec | USB | microphone in, audio out |
| Audio codec 3.5 mm out | PAM8403 amp | 3.5 mm to wire | line-level audio |
| PAM8403 amp | 3 W 8 Ω speaker | speaker wire | sound |
| Jetson DC jack | wall adapter | the dev kit's own supply | power |
| Jetson fan header | 40 mm PWM fan | fan connector | cooling |
| Jetson Ethernet | college computer or switch | Ethernet | all backend traffic |

Things that catch people out:

- **The Orin Nano Developer Kit outputs video on DisplayPort, not HDMI.** The
  Waveshare LCD takes HDMI, so you need a DisplayPort-to-HDMI adapter.
  Confirm against your board's port labels.
- **Power:** use the supply that came with the kit. Do not connect a 5V/4A
  adapter to an Orin kit without checking its datasheet.
- **Do not power the amp from the Jetson's pins** unless you have checked the
  voltage and current; use the amp's own supply.

## Which parts run where

| Part | Runs on | Why |
|---|---|---|
| Ollama and phi4-mini | college computer (GPU) | the Jetson is not running any model in this test |
| Backend, retrieval, verifier, speech-to-text, Piper | college computer | one place, one config |
| `kiosk.html` in Chromium | Jetson | it is the screen, mic and speaker |
| FAISS index and corpus | college computer | built with `scripts/00` to `04` |

## Bring-up order

Test each link before adding the next, so a failure points at one cable.

1. **Computer alone.** Run the backend and the Next.js page on the college
   computer ([GPU_COMPUTER_SETUP.md](GPU_COMPUTER_SETUP.md)). Nothing else is
   needed until this works.
2. **Jetson and LCD alone.** Boot the Jetson with the LCD attached. Confirm
   the picture appears at the right size and that touch moves the pointer.
3. **Audio on the Jetson.** Record and play a clip through the codec and
   speaker. Fix volume and which device is selected now.
4. **Network.** Connect the Jetson and the computer. If the college network
   isolates devices from each other, use a direct Ethernet cable with fixed
   addresses, for example `192.168.50.1` (computer) and `192.168.50.2`
   (Jetson). From the Jetson, ping the computer.
5. **Open the backend to the network.** On the computer:
   - start it with `--host 0.0.0.0`, because the default only accepts
     requests from the same machine:

     ```powershell
     python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
     ```
   - allow inbound TCP port 8000 in the computer's firewall.
6. **Point the kiosk at the computer.** On the Jetson, edit line 135 of
   `frontend/kiosk.html`:

   ```js
   const BACKEND_HTTP = "http://192.168.50.1:8000";   // the computer's address
   ```

   The backend only accepts browser requests from `localhost:3000` and
   `localhost:3001` (`CORS_ORIGINS` in `backend/config.py`). Add the origin
   the kiosk page is loaded from, or its requests will be blocked. If you
   open the file directly its origin is `file://`, which browsers report as
   `null`; serving the file from a small static server on the Jetson and
   adding that address is cleaner.
7. **Check from the Jetson.** Open `http://<computer-ip>:8000/health` in
   Chromium. Every field should be true except `vad`.
8. **Run the kiosk:**

   ```bash
   chromium-browser --kiosk /path/to/frontend/kiosk.html
   ```

   Ask a typed question first, then a spoken one.

If the microphone is blocked, remember browsers only allow it on `localhost`
or HTTPS. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Port summary

| Port | On | Used by |
|---|---|---|
| 8000 | college computer | backend: `/chat` (WebSocket), `/stt`, `/tts`, `/texts`, `/health` |
| 11434 | college computer | Ollama, used only by the backend on the same machine |
| 3001 | college computer | Next.js dev page, only for testing on the computer itself |

## What this test proves and what it does not

**Proves:** the screen, touch, microphone, speaker, network and the whole
spoken question-and-answer loop work together.

**Does not prove:** anything about the Jetson running the AI. In this layout
the Jetson loads no model, so the llama-cpp-python, whisper.cpp and Silero VAD
paths stay untested, and the Jetson's own latency and memory remain unknown.
That is the next stage, with `EMBEDDED_MODE` on and everything on the Jetson,
and it should be measured separately and labeled as such.

## Three stages, in order

| Stage | What runs where | Wiring |
|---|---|---|
| 1. Computer only | everything on the college computer, its own screen | none |
| 2. This document | brain on the computer, kiosk on the Jetson | the picture above |
| 3. Standalone | everything on the Jetson (`EMBEDDED_MODE = True`) | the Jetson plus LCD, audio and power only; no computer |
