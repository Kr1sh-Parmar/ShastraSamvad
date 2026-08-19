"""Does this sentence contradict the verse it cites?

The second stage of the citation verifier. Cosine similarity scores topic, so
it passes a sentence that says the exact opposite of the verse it names — the
one failure the "no verse, no answer" guardrail could not see. A cross-encoder
reads the pair instead of comparing two independent vectors, which is what it
takes to tell a verse from its own negation.

Loaded lazily and once, like embed.py. ~45ms per pair on CPU.
"""
from . import config

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.NLI_MODEL)
    return _model


def contradicts(premises: str | list[str], hypothesis: str) -> float:
    """-> the highest contradiction probability over (verse, sentence) pairs.

    `premises` are the verse texts the sentence is answerable to, `hypothesis`
    the Guru's sentence. Highest wins: contradicting any one of the verses it
    cited is enough to withhold it.
    """
    if isinstance(premises, str):
        premises = [premises]
    if not premises:
        return 0.0
    m = model()
    # Label order is the model's, not an assumption: nli-deberta-v3 happens to
    # put contradiction at 0, but reading it from the config survives a swap.
    idx = next(i for i, name in m.config.id2label.items()
               if name.lower().startswith("contradict"))
    scores = m.predict([(p, hypothesis) for p in premises], apply_softmax=True)
    return max(float(row[idx]) for row in scores)
