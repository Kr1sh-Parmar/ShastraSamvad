"""Which kind of turn is this? (arch doc 4.2)

Rules first, embedding similarity as the tiebreak. Simple on purpose — spec 6
says start here and upgrade to a real classifier only if this proves too blunt.
"""
import numpy as np

from . import config, embed

MODES = ("TEACH", "DOUBT", "DEBATE", "COUNSEL")

# Strong signals. If one of these fires, we don't bother with embeddings.
RULES = {
    "TEACH": ("teach me", "start the lesson", "explain chapter", "begin chapter",
              "continue", "go on", "next verse", "read the next"),
    "DEBATE": ("i disagree", "that's wrong", "thats wrong", "but surely", "prove it",
               "how can you say", "i don't accept", "i dont accept", "contradict",
               "isn't that", "isnt that"),
    "DOUBT": ("what does that mean", "i don't understand", "i dont understand",
              "can you clarify", "what do you mean", "confused", "say that again"),
    "COUNSEL": ("i feel", "i am struggling", "im struggling", "should i",
                "my father", "my mother", "my job", "my friend", "i am afraid",
                "what should i do"),
}

EXAMPLES = {
    "TEACH": ["teach me the second chapter", "explain this chapter from the start",
              "what does the Gita say in chapter two", "continue the lesson",
              "walk me through these verses"],
    "DOUBT": ["what does detachment mean here", "I don't follow that last part",
              "why does he say that in this verse", "can you explain that word",
              "how does that fit with what you just said"],
    "DEBATE": ["I think that reasoning is flawed", "surely violence is never justified",
               "that contradicts what you said earlier", "prove that from the text",
               "I disagree with the idea of duty above all"],
    "COUNSEL": ["I am anxious about my future", "my family wants me to take a job I hate",
                "I cannot forgive someone", "I feel lost and without purpose",
                "how should I deal with anger at work"],
}

_centroids: dict[str, np.ndarray] = {}


def _centroid(mode: str) -> np.ndarray:
    if mode not in _centroids:
        v = embed.encode(EXAMPLES[mode]).mean(axis=0)
        _centroids[mode] = v / np.linalg.norm(v)
    return _centroids[mode]


def classify(text: str, state: dict) -> str:
    low = text.lower()
    for mode, phrases in RULES.items():
        if any(p in low for p in phrases):
            return mode

    vec = embed.encode(text)[0]
    scores = {m: float(vec @ _centroid(m)) for m in MODES}

    # An active lesson biases a question toward DOUBT rather than a fresh COUNSEL turn.
    if state.get("lesson_active"):
        scores["DOUBT"] += 0.08
    return max(scores, key=scores.get)


# Ethical guardrail (arch doc 4.8). A trust boundary — checked before the model
# is ever asked, so a harmful request costs nothing and gets no generated text.
HARMFUL = (
    "kill myself", "kill my self", "suicide", "end my life", "take my life",
    "hurt myself", "harm myself", "kill him", "kill her", "kill them",
    "murder", "poison someone", "make a bomb", "build a bomb", "how to hurt",
    "take revenge on", "get away with",
)
CRISIS_REPLY = (
    "What you are carrying is too heavy to carry alone, and it is beyond what I "
    "can answer from the texts. Please speak to someone you trust or a doctor "
    "today. The scriptures say the self is never destroyed; your life has worth "
    "that this moment cannot measure."
)
REFUSAL = (
    "I cannot help with that, and the texts I teach from would not counsel it. "
    "Ask me instead what they say about anger, duty, or the harm we do ourselves "
    "by acting from it."
)


def screen(text: str, mode: str) -> str | None:
    """-> a reply to send INSTEAD of generating, or None to proceed."""
    low = text.lower()
    if not any(p in low for p in HARMFUL):
        return None
    self_harm = any(p in low for p in
                    ("myself", "my self", "suicide", "my life"))
    return CRISIS_REPLY if self_harm else REFUSAL


def new_state() -> dict:
    # `teaching` is the (source, chapter) the lesson position belongs to; it
    # starts as the null pair so a fresh state is already "on" that lesson.
    return {"source": None, "chapter": None, "position": 0, "lesson_active": False,
            "teaching": (None, None), "current_verse_text": "", "stance": None,
            "history": []}


def route(text: str, state: dict | None = None) -> dict:
    """-> {mode, query, state}  (contract: arch doc 8)"""
    state = state or new_state()
    mode = (state.get("forced_mode") or classify(text, state)).upper()

    if mode == "TEACH":
        state["lesson_active"] = True
        # A different text or chapter is a different lesson, so the position
        # cannot carry over. It did: three "continue"s in chapter 2 then a
        # switch to chapter 12 opened at verse 5 of 12, and _span announced it
        # as such. Every turn passes through here, so this is the one place to
        # catch it — main.py sets state["chapter"] without knowing about
        # position, and it should stay that way.
        lesson = (state.get("source"), state.get("chapter"))
        if lesson != state.get("teaching"):
            state["teaching"] = lesson
            state["position"] = 0
        elif any(w in text.lower() for w in ("continue", "go on", "next")):
            # By a whole window, not by one. The retriever slices
            # verses[pos : pos + TEACH_WINDOW], so stepping by 1 would re-teach
            # two of the three verses the student just heard.
            state["position"] = int(state.get("position", 0)) + config.TEACH_WINDOW
    elif mode == "DEBATE":
        state["stance"] = text

    return {"mode": mode, "query": text, "state": state}
