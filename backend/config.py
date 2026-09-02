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
WHISPER_MODEL = "base"          # base.en on device
WHISPER_COMPUTE = "int8"        # CPU
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

# --- retrieval ---
TOP_K = 6
# One verse per turn, not three. Handed three, the model attributes a sentence
# to the wrong one of them — and a cited sentence is scored only against the
# verse it named, so a faithful paraphrase of 2.4 labelled [gita_2_5] is
# withheld as drift and the lesson goes silent. Measured: faithful third-person
# paraphrases of narrative verses score 0.61-0.71 against the RIGHT verse, which
# clears CITED_MIN_SIM but not the stricter uncited floor. So the sentence has
# to carry an anchor, and the anchor is only reliable when there is one to pick.
TEACH_WINDOW = 1                # verses per teaching turn
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
CITATION_MIN_SIM = 0.70         # uncited assertion vs any retrieved verse

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
