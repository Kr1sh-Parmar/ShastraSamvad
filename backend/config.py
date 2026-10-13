"""Every tunable in one place. Retune here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CHUNKS = DATA / "chunks"
INDEX = DATA / "index"
CROSSREF = DATA / "crossref"

CORPUS_JSONL = CHUNKS / "corpus.jsonl"
FAISS_INDEX = INDEX / "corpus.faiss"
ID_MAP = INDEX / "id_map.json"
CROSSREF_MAP = CROSSREF / "gita_to_mbh.json"

# --- models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# NOT qwen3:4b (arch doc 12's first choice). It is a reasoning model: on these
# rule-heavy prompts it spent 1800 tokens deliberating and emitted no answer at
# all, and with think=false it wrote the deliberation into the reply instead.
# phi4-mini answers in 110-220 tokens, ~5s, citing anchors correctly.
LLM_MODEL = "phi4-mini"
OLLAMA_URL = "http://localhost:11434"
# Ollama unloads an idle model after 5 minutes by default, so a pause between
# demo questions costs a full cold load on the next one — the single largest
# latency item in the pipeline (retrieval is 13ms, the verifier 0.2s/turn).
LLM_KEEP_ALIVE = "30m"
# Ollama needs a daemon; embedded boards run llm.py against llama-cpp-python
# directly instead. True keeps the dev/demo path (Ollama) unchanged; False
# switches to the in-process GGUF path below. Both are kept side by side so a
# board can be brought up before the Ollama path is retired.
LLM_USE_OLLAMA = True
LLM_MODEL_PATH = ROOT / "models" / "phi4-mini-q4.gguf"
# 0 = pure CPU. Jetson Nano 4GB can offload some layers to its CUDA cores by
# raising this — measure on-device before committing to a layer count, since GPU VRAM is shared with system RAM on the Nano.
LLM_GPU_LAYERS = 0
LLM_CONTEXT_SIZE = 2048
WHISPER_MODEL = "base"          # base.en on device
WHISPER_COMPUTE = "int8"        # CPU
# faster-whisper is the dev/demo path (pip install, no separate build step).
# whisper.cpp base.en on CPU is the architecture doc's actual STT engine for
# the device — a plain C++ binary with no Python ML stack to cross-compile for
# the Jetson Nano. STT_USE_WHISPER_CPP switches stt.py to it; False
# keeps faster-whisper unchanged for laptop development.
STT_USE_WHISPER_CPP = False
WHISPER_CPP_MODEL_PATH = ROOT / "models" / "ggml-base.en.bin"
# Arch doc 4.1's VAD for endpointing and barge-in. The browser build does this
# itself (page.jsx: RMS loudness over the mic stream) since it already owns
# the mic and a JS-side threshold is free. A kiosk build with no browser in
# front of the mic has nothing to do that job, so the backend does it with
# Silero VAD instead — same purpose, run where the audio actually arrives.
VAD_USE_SILERO = False
VAD_THRESHOLD = 0.5             # Silero speech-probability cutoff per chunk
VAD_MIN_SPEECH_MS = 250         # below this, a blip isn't a barge-in
VAD_MIN_SILENCE_MS = 700        # silence this long after speech = end of turn
# Entailment stage behind the cosine gate — see the citation verifier block
# below for why this exists and what it is allowed to reject. ~70ms per pair.
#
# NOT the xsmall sibling, despite it being the obvious pick for the device. It
# reads "let the results go" as abandoning the duty, so it called "Do your duty
# and let the results go" — the plainest true reading of 2.47 there is — a
# contradiction at 0.970, against a weakest true contradiction of 0.997. A
# 0.027 margin is not a gate. Measured separation on the same 16 sentences:
#   nli-deberta-v3-small      +0.661   <- this one
#   nli-MiniLM2-L6-H768       +0.087
#   nli-distilroberta-base    +0.082
#   nli-deberta-v3-xsmall     +0.027
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
# Keeps NLI_MODEL's ~400MB resident between turns by default — the same
# tradeoff LLM_KEEP_ALIVE makes for the LLM, just smaller stakes here since
# 400MB matters far more on a 4GB board than a laptop. True unloads it after
# each check() pass and reloads on the next verified sentence: ~200ms paid
# per NLI call instead of per idle gap, in exchange for 400MB freed the rest
# of the time. See EMBEDDED_MODE below for where this is meant to flip on.
NLI_MODEL_LAZY = False

# --- retrieval ---
TOP_K = 6
# One verse per turn, not three. Handed three, the model attributes a sentence
# to the wrong one of them — and a cited sentence is scored only against the
# verse it named, so a faithful paraphrase of 2.4 labelled [gita_2_5] is
# withheld as drift and the lesson goes silent. Measured: faithful third-person
# paraphrases of narrative verses score 0.61-0.71 against the RIGHT verse, which
# clears CITED_MIN_SIM but not the stricter uncited floor. So the sentence has
# to carry an anchor, and the anchor is only reliable when there is one to pick.
TEACH_WINDOW = 2                # verses per teaching turn
# Debate gets more than TOP_K, not less. It used to get TOP_K // 2 per side and,
# after dedup, argued from 3 verses where COUNSEL had 6 — so the model reached
# into memory for the rest and the verifier withheld the entire turn. This is
# the one mode that has to hold two positions at once; starving it is what made
# it fabricate.
DEBATE_TOP_K = 8
CROSSREF_TOP_K = 3
# Measured against the real Mahabharata, which 0.55 predated. Both texts are the
# same register — Sanskrit epic in English prose about dharma, kings and battle
# — so cosine runs high across the board and the whole distribution lives
# between 0.58 and 0.89. At 0.55 every one of the 1,979 Gita chunks kept a link,
# which is not a curated map (arch doc 4.4 asks for "the strongest links"), it is
# no filter at all. Sampled by hand:
#   0.70  noise. "There is no truth superior to Me" linked to Bhima boasting he
#         is the superior brother for carrying Arjuna — matched on one word.
#   0.78  mixed. One real parallel, one block of translator footnotes.
#   0.80+ the thing this index exists for. Arjuna seeing his relatives arrayed
#         (1.26) links to Drona Parva 28, where he kills those same uncles.
# 0.80 keeps 1,576 links over 711 of the 1,979 chunks. Verses with no strong
# illustration now correctly get none, instead of getting the nearest noise.
CROSSREF_MIN_SCORE = 0.80

# --- generation ---
MAX_TOKENS = 400                # answer-length cap (arch doc 4.6); real turns use 110-220
TEMPERATURE = 0.3

# --- citation verifier ("no verse, no answer") ---
# Thresholds measured, not guessed (bge-small floors around 0.45 even for
# unrelated English, so a single bar cannot separate the two cases):
#   true paraphrases of a verse        0.69 - 0.86
#   sentence drifting off its citation 0.52
#   fabrications sharing topic         0.45 - 0.68
# A sentence citing a real anchor is checked against THAT verse: 0.60 sits
# between the drift case and the loosest honest paraphrase. A sentence asserting
# something with no anchor has nothing vouching for it, so it must land
# near-verbatim on some retrieved verse. Withholding is the safe failure.
#
CITED_MIN_SIM = 0.60            # sentence vs the verse it cited
CITATION_MIN_SIM = 0.65         # uncited assertion vs any retrieved verse

# Cosine cannot catch a sentence that NEGATES the verse it cites: measured 0.777
# for the negation against 0.840 for the faithful version, because cosine scores
# topic, not entailment. So a cross-encoder runs behind it (NLI_MODEL above),
# premise = the verse, hypothesis = the sentence.
#
# It may only REJECT ON CONTRADICTION. Requiring entailment would silence honest
# teaching — measured against the verse each paraphrases:
#   faithful paraphrase                      entailment    0.97
#   faithful teaching with life-framing      NEUTRAL       0.81
#   faithful third-person paraphrase         NEUTRAL       0.99
# The last two are correct answers that carry no entailment label at all.
#
# A confidence floor, not argmax: the turn-taking line "Say continue when you
# are ready" — no scriptural claim in it — scored contradiction 0.66. Sentences
# like that are already short-circuited by _is_conversational before they reach
# here; this threshold is the second line of defence, not the first.
#
# 0.90 measured over 10 faithful and 6 contradicting sentences on gita_2_47,
# 2_13 and 1_1: worst faithful 0.338, weakest contradiction 0.999. The gap is
# wide, so this is a floor with room either side, not a tuned edge.
NLI_CONTRADICTION_MAX = 0.90
NO_BASIS_MESSAGE = (
    "I cannot find scriptural basis for that in the verses before me. "
    "Let me tell you instead what the texts do say."
)

# --- speech ---
# A push-to-talk turn is seconds of Opus. The cap is a trust boundary, not a
# tuning knob: /stt reads from the network into memory before Whisper sees it.
MAX_AUDIO_BYTES = 8 * 1024 * 1024
USE_BROWSER_TTS = False         # True falls back to the browser's speechSynthesis
PIPER_BIN = ROOT / "piper" / "piper.exe"
PIPER_VOICE = ROOT / "piper" / "en_US-lessac-medium.onnx"

# --- server ---
PORT = 8000
CORS_ORIGINS = ["http://localhost:3001", "http://127.0.0.1:3001",
                "http://localhost:3000", "http://127.0.0.1:3000"]

# --- embedded deployment ---
# Target board: NVIDIA Jetson Nano 4GB (B01). Every model
# in this file's --- models --- section runs at once, for the whole life of the
# process — there is no request-scoped loading on a kiosk with one user.
#
# Quantization: LLM_MODEL_PATH expects a Q4_K_M GGUF (~2.0GB for phi4-mini's
# ~3.8B params) — the standard "fits in 4GB with room for everything else"
# quant, not the smaller Q4_0/Q3 variants, which measurably degrade citation
# accuracy on a model this size already picked for being small. Pull it with
# llama.cpp's own convert+quantize tooling, or a pre-quantized GGUF from
# Hugging Face (search "phi-4-mini gguf Q4_K_M") — same file either way,
# config.py only cares that LLM_MODEL_PATH points at it.
#
# Resident memory, everything loaded simultaneously:
#   LLM (phi4-mini, Q4_K_M GGUF)         ~2.0 GB
#   Embedder (bge-small)                 ~130 MB
#   NLI (deberta-v3-small)               ~400 MB   (0 once idle, if NLI_MODEL_LAZY)
#   Whisper (base.en)                    ~150 MB
#   Piper TTS                            ~60 MB
#   FAISS index + corpus                 ~50 MB
#   --------------------------------------------
#   Total                                ~2.8 GB   -> ~1.2 GB left for OS/buffers on 4GB
#
# Jetson Nano: LLM_GPU_LAYERS > 0 offloads that many transformer layers to the
# Nano's 128-core Maxwell GPU via CUDA, trading GPU VRAM (shared with system
# RAM on this board, so it still counts against the 4GB above) for tokens/sec
# — 0 is the safe CPU-only default until that tradeoff is measured on-device.
#
# EMBEDDED_MODE is the one flag to set when bringing up the Nano: it forces
# every individual backend flag above to its embedded value, so a deployment
# script doesn't have to know or repeat that LLM_USE_OLLAMA, STT_USE_WHISPER_CPP
# and VAD_USE_SILERO all need to flip together. Leave it False for laptop
# dev — that leaves every flag it would touch at the default already set
# above, so nothing changes.
EMBEDDED_MODE = False

if EMBEDDED_MODE:
    LLM_USE_OLLAMA = False
    STT_USE_WHISPER_CPP = True
    VAD_USE_SILERO = True
