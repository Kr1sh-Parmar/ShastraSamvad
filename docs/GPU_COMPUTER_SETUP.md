# Running Shastra Samvad on the College GPU Computer

First-run guide for the laptop/dev path (Ollama, faster-whisper, Piper). This
is **not** the Jetson Orin Nano build: leave `EMBEDDED_MODE = False` in
`backend/config.py`.

Commands are for Windows PowerShell. Linux differences are noted where they
exist.

## 0. Before you go to the lab

The pipeline is offline at run time, but three things download once. If the
lab network is restricted, do these at home and carry them over:

- the Python packages (`pip install -r requirements.txt`)
- the embedder `BAAI/bge-small-en-v1.5` and the NLI model
  `cross-encoder/nli-deberta-v3-small` (Hugging Face, fetched on first use)
- the LLM: `ollama pull phi4-mini` (about 2.5 GB)

Also bring the **Bhagavad-gita As It Is PDF**. It is not in the repo and no
script downloads it (see step 5).

## 1. Check the GPU

```powershell
nvidia-smi
```

You need a working NVIDIA driver. phi4-mini at 4-bit needs roughly 2.5 GB of
GPU memory, so any card with 4 GB or more is enough. Note the GPU model and
driver version; you will record them with every measurement.

## 2. Install the tools

- Python 3.10 or newer
- git
- [Ollama](https://ollama.com/download)
- Node.js, only if you want the Next.js frontend (`kiosk.html` needs nothing)

## 3. Get the code and Python packages

```powershell
git clone https://github.com/Kr1sh-Parmar/ShastraSamvad.git
cd ShastraSamvad
git checkout fix/phase-2-integration
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Make sure PyTorch can see the GPU

On Windows, `pip` installs a **CPU-only** PyTorch by default. Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, install the CUDA build from pytorch.org, using the
install command it gives for your CUDA version, then re-run the check. On
Linux the default wheel is usually the CUDA one.

## 4. Pull the language model

```powershell
ollama pull phi4-mini
ollama run phi4-mini "say hello"
ollama ps
```

`ollama ps` should show the model on **GPU**. If it says CPU, update the
NVIDIA driver and restart Ollama.

## 5. Build the corpus and index

`data/` is git-ignored, so it does not come with the clone.

1. Create `data/raw/` and copy the **Bhagavad-gita As It Is PDF** into it.
   `scripts/01_clean_texts.py` is written for that book's layout. Its
   commentary is not public domain, so keep the file out of git and check its
   terms before sharing the built index.
2. Run the pipeline in order:

```powershell
python scripts/00_fetch_mahabharata.py
python scripts/01_clean_texts.py
python scripts/02_chunk_and_tag.py
python scripts/03_build_embeddings.py
python scripts/04_build_crossref.py
```

`00` downloads the Mahabharata from Project Gutenberg (about 12 MB), so it
needs internet. `03` embeds about 16.5 thousand chunks and shows a progress
bar.

**Check:** the counts should be close to 657 Gita verses, 1,322 commentary
chunks and 14,572 Mahabharata passages (16,551 total). A much smaller total
means one text did not load.

## 6. Get the voice (optional)

```powershell
python scripts/05_fetch_piper.py
```

This downloads the **Windows** build of Piper into `piper/`. Without it the
browser's own voice speaks instead, which is fine for a first run. On Linux
the script does not work as written: download the Linux Piper build and a
voice by hand, then set `PIPER_BIN` in `backend/config.py` (it points at
`piper.exe`).

## 7. Start the backend

```powershell
python -m uvicorn backend.main:app --port 8000
```

The first start takes a while because every model loads up front (about 30
seconds cold for the embedder alone). Wait for "Application startup
complete", then open `http://localhost:8000/health`. Every field should be
true except `vad`, which stays false unless `VAD_USE_SILERO` is on.

## 8. Start a frontend

Use the Next.js app first. Its address is the one the backend allows:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3001` **on the same computer**. Browsers only allow
microphone access on `localhost` or HTTPS, and the backend only accepts
requests from `localhost:3000` and `localhost:3001` (`CORS_ORIGINS`). Opening
the page from another machine over the network will break the mic.

Once that works, try `frontend/kiosk.html`.

## What runs on the GPU

| Part | Runs on | Notes |
|---|---|---|
| LLM (phi4-mini) | GPU | Ollama does this automatically |
| Embedder (bge-small) | GPU if PyTorch sees CUDA | sentence-transformers picks it automatically |
| NLI verifier | GPU if PyTorch sees CUDA | same |
| Speech to text | **CPU** | `backend/stt.py:25` hard-codes `device="cpu"`; the `base` model on CPU is fine for short questions |
| FAISS search | CPU | 16.5 thousand vectors; milliseconds |
| Piper voice | CPU | |

Moving speech to text onto the GPU means editing `stt.py` and installing the
CUDA and cuDNN libraries faster-whisper needs. Skip that unless speech to
text is clearly the slow step.

## What this machine can and cannot tell you

Numbers measured here describe **this computer**. They do not predict the
Jetson Orin Nano's latency or memory use, and its paths (llama-cpp-python,
whisper.cpp, Silero VAD) are not exercised at all. Label every measurement
with the machine and GPU it came from. See
[TESTING_AND_MEASUREMENT.md](TESTING_AND_MEASUREMENT.md).

If something fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
