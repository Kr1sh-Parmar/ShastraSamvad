# Troubleshooting

Symptom, cause, fix. Every entry comes from how the code actually behaves.

## Start-up and health

**`RuntimeError: No index at ...corpus.faiss. Run scripts/00..03 first.`**
The index has not been built. `data/` is not in git. Follow step 5 of
[GPU_COMPUTER_SETUP.md](GPU_COMPUTER_SETUP.md). The backend loads the index at
start-up, so it will not come up without it.

**`/health` shows `"index": false`.** Same cause as above.

**`/health` shows `"ollama": false`.** Ollama is not running, or the model is
missing. Start Ollama, run `ollama pull phi4-mini`, and check
`ollama list`.

**`/health` shows `"whisper": false` or `"piper": false`.**
- `whisper` reports "loaded and ready", so it is false until start-up has
  finished loading it. Wait for "Application startup complete" and refresh.
- `piper` is false until `scripts/05_fetch_piper.py` has run. This is fine:
  the browser's voice is used instead.

**`/health` shows `"crossref": false`.** `scripts/04_build_crossref.py` has
not run, or there was no Mahabharata text to link to. Answers still work but
no illustrating passages are added.

**Start-up is slow, or the page disconnects while the backend loads.** Every
model loads before the first question (the embedder alone takes about 30
seconds cold). Wait for "Application startup complete" before opening the
page. `kiosk.html` reconnects with backoff if it opened too early.

**Port already in use.** Run uvicorn on another port, for example
`--port 8001`, and update the address the frontend uses. Next.js is on 3001;
the backend only allows `localhost:3000` and `localhost:3001`.

## Installing

**Import error when running `import faiss`, or a numpy/ABI warning.**
`faiss-cpu` 1.9 is built against numpy 1.x. Reinstall in a clean environment:

```powershell
pip install "numpy<2" "faiss-cpu>=1.9"
```

Also check that a `faiss-cpu` wheel exists for your Python version.

**`torch.cuda.is_available()` is `False` on Windows.** You have the CPU-only
PyTorch. Install the CUDA build from pytorch.org. See step 3 of the setup
guide.

**`ollama ps` shows CPU, not GPU.** Update the NVIDIA driver and restart
Ollama. The model will still run on CPU, just slowly.

## Building the corpus

**The Gita is missing from the results, only Mahabharata verses appear.** The
Bhagavad-gita As It Is PDF was not in `data/raw/`. `00_fetch_mahabharata.py`
only downloads the Mahabharata; the Gita PDF has to be copied in by hand.
Re-run `01`, `02`, `03` and `04` after adding it.

**Chunk counts are far below 16,551.** One text did not load. Check
`data/raw/` and the output of `02_chunk_and_tag.py`.

**Weird or cut-off errors after stopping a build part-way.** The index and
`id_map.json` are written one after the other, so an interrupted
`03_build_embeddings.py` can leave a mismatched pair, and the backend will
then fail with a raw error rather than a clear message. Delete `data/index/`
and re-run `03` and `04`.

**`00_fetch_mahabharata.py` fails.** It needs internet and downloads from
Project Gutenberg. Check the lab firewall.

## Using it

**"I cannot find scriptural basis for that in the verses before me."** This
is the guardrail working. It appears when no sentence the model wrote could be
verified against a retrieved verse. For an out-of-scope question it is the
correct answer. For an in-scope question, run
`python scripts/06_measure_turns.py` and read the withheld reasons (see
[TESTING_AND_MEASUREMENT.md](TESTING_AND_MEASUREMENT.md)).

**Sentences appear greyed out.** Those were withheld, and the reason is shown.
`invented anchor` on a verse that *was* retrieved is a bug worth reporting.

**The microphone does nothing.** Browsers allow the microphone only on
`localhost` or HTTPS. Open the page on the same computer as `localhost`, not
by IP address. Also check the browser's site permission for the microphone.

**The Guru answers in a robotic voice.** That is the browser fallback. Run
`scripts/05_fetch_piper.py` for Piper.

**Typed questions work, spoken ones do not.** `POST /stt` needs
`python-multipart` and faster-whisper, and the model downloads on first use.
Check the backend terminal for the error.

## Not for this machine

**`/mic` closes immediately with code 4000.** Expected. That endpoint only runs
when `VAD_USE_SILERO` is on, which is for the kiosk build. It also has a known
unfixed bug: it passes raw audio to a decoder that expects an encoded
container, so it will fail even when on. Use the browser's push-to-talk.

**Piper script fails on Linux.** `05_fetch_piper.py` downloads the Windows
build. Install Piper manually and set `PIPER_BIN` in `backend/config.py`.
