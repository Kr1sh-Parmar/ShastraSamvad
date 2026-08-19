"""No verse, no answer. (arch doc 4.7 — the anti-hallucination guardrail.)

Runs BEFORE a sentence reaches the speaker, so an unsupported claim is never
spoken. A sentence passes only if it is anchored to a retrieved verse, or is
plainly conversational rather than a claim about scripture.

Two stages: cosine similarity for invention and drift, then a cross-encoder
(nli.py) for the case cosine is blind to — a sentence that negates the very
verse it cites.
"""
import re

import numpy as np

from . import config, embed, nli

_ID = r"[a-z]{3,4}_[a-z0-9_]+"
# Models write both [gita_2_47] and [gita_12_11, gita_4_14]. Miss the second form
# and the sentence reads as uncited, so the panel highlights whatever verse the
# fallback match picked — a citation pointing at the wrong verse.
# Parentheses are accepted alongside brackets: the persona asks for [gita_2_1] but
# small models drift to (gita_2_1), and a real citation read as no citation puts a
# faithful sentence under the stricter uncited floor and silences the whole answer.
# The id must still exist in the retrieved set either way — this loosens parsing,
# not the guardrail.
ANCHOR_RE = re.compile(rf"[\[(]({_ID}(?:\s*,\s*{_ID})*)[\])]", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# Turn-taking phrases carry no scriptural claim, so they need no citation.
CONVERSATIONAL = (
    "let me", "shall we", "do you", "would you like", "tell me", "listen",
    "come closer", "sit", "very well", "good question", "as you wish",
    "i will explain", "ask me", "my child", "dear student",
    "continue", "say continue", "when you are ready", "when you're ready",
)


def strip_anchors(text: str) -> str:
    """Remove anchors for speech, leaving no punctuation debris behind.

    "- [gita_4_14]: No work affects Me." would otherwise be spoken as
    "- : No work affects Me."
    """
    out = ANCHOR_RE.sub("", text)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)             # emptied brackets
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)            # space before punctuation
    out = re.sub(r"(?:\s*,)+", ",", out)                  # "as in , and ," -> one comma
    out = re.sub(r"(?:\s*[,;:])+(\s*[.!?])", r"\1", out)  # "provided:,,." -> "provided."
    out = re.sub(r"^[\s\-*•:,;]+", "", out)               # bullet / orphaned lead-in colon
    return re.sub(r"\s{2,}", " ", out).strip()


def _say_one(anchor: str) -> str:
    """One anchor as the student should hear it.

    gita_12_11    -> "Gita 12.11"
    gita_2_5_p1   -> "Gita 2.5"          (the purport tag is not spoken)
    mbh_udyoga_33 -> "the Mahabharata, Udyoga Parva, section 33"

    The Mahabharata case needs its own branch: its anchor carries a parva NAME
    where the Gita carries a chapter number, and the digits-only rule below
    would drop the name and read the prefix out literally, as "Mbh 33".
    """
    src, *rest = anchor.strip().split("_")
    nums = [n for n in rest if n.isdigit()]          # drops the _p1 purport tag
    if src.lower() == "mbh" and rest and not rest[0].isdigit():
        section = f", section {nums[0]}" if nums else ""
        return f"the Mahabharata, {rest[0].title()} Parva{section}"
    return f"{src.title()} {'.'.join(nums)}".strip()


def speakable(text: str) -> str:
    """Anchors become spoken references: [gita_12_11] -> "Gita 12.11".

    Deleting them instead leaves the grammar dangling — "as per [a] and [b]"
    becomes "as per and". The student should hear the citation anyway; that is
    the whole point of anchoring answers to verses.
    """
    def say(m: re.Match) -> str:
        return "(" + ", ".join(_say_one(a) for a in m.group(1).split(",")) + ")"
    out = re.sub(r"^[\s\-*•]+", "", ANCHOR_RE.sub(say, text))
    return re.sub(r"\s{2,}", " ", out).strip()


# A model that cannot recall the anchor writes the reference in prose instead —
# "as Gita 2.37 says". That is a citation the student hears as one, and since
# speakable() renders real anchors the same way, the two are indistinguishable.
# Unchecked, it is the anchor guardrail with a door left open beside it.
PROSE_REF = re.compile(r"\b(?:bhagavad[\s-]*)?gita\s*(\d+)\s*[.:]\s*(\d+)", re.I)
# The same door on the Mahabharata side, which speakable() now also renders in
# prose — "the Udyoga Parva, section 33". Matches with or without the leading
# "Mahabharata", since the model usually names the parva alone once it is in
# the middle of a turn.
# A comma with no section word is prose, not a citation ("the Udyoga Parva, 5
# kings gathered"), so the number must either follow a section word or sit
# directly against the parva name.
PROSE_REF_MBH = re.compile(
    r"\b(?:mahabharata[,\s]*)?([a-z]{3,20})[\s-]*parva\b"
    r"(?:[,\s]*(?:section|chapter|§)\s*|\s+)(\d+)", re.I)


def anchors(text: str) -> list[str]:
    """Every verse id cited, flattening [a, b] into two. Prose refs count too.

    Lowercased, because ANCHOR_RE matches case-insensitively but the ids it is
    checked against are lowercase by construction. Measured: the model writes
    [MBH_SANTI_176_5] often enough to matter, and the mismatch made check()
    call a verse it had actually been handed an INVENTED anchor — rejecting the
    one thing an anchor is supposed to prove. Same failure the bracket/paren
    leniency above exists to prevent, arriving by a different route.
    """
    bracketed = [a.strip().lower() for g in ANCHOR_RE.findall(text)
                 for a in g.split(",")]
    return (bracketed
            + [f"gita_{c}_{v}" for c, v in PROSE_REF.findall(text)]
            + [f"mbh_{p.lower()}_{s}" for p, s in PROSE_REF_MBH.findall(text)])


def _is_conversational(sentence: str) -> bool:
    low = sentence.lower().strip()
    if low.endswith("?") and len(low) < 90:
        return True
    return len(low) < 90 and any(low.startswith(p) for p in CONVERSATIONAL)


def check(sentence: str, verses: list[dict]) -> dict:
    """-> {ok, cited_ids, reason}  (contract: arch doc 8)"""
    known = {v["verse_id"]: v["text"] for v in verses}
    # "Gita 18.8" is satisfied by the purport chunk gita_18_8_p1.
    resolved = {a: (a if a in known
                    else next((k for k in known if k.startswith(a + "_")), None))
                for a in anchors(sentence)}
    valid = [r for r in resolved.values() if r]
    invented = [a for a, r in resolved.items() if not r]

    # A fabricated anchor is the worst failure mode — reject outright.
    if invented:
        return {"ok": False, "cited_ids": [], "reason": f"invented anchor {invented}"}

    bare = strip_anchors(sentence)
    if not bare:
        return {"ok": True, "cited_ids": valid, "reason": "empty"}

    if _is_conversational(bare):
        return {"ok": True, "cited_ids": valid, "reason": "conversational"}

    if not known:
        return {"ok": False, "cited_ids": [], "reason": "no verses retrieved"}

    # Stage 1 — cosine. Catches invention and drift cheaply, but scores topic
    # rather than entailment, so it cannot tell a verse from its own negation.
    targets = valid or list(known)
    sim = embed.encode(bare) @ embed.encode([known[t] for t in targets]).T
    best = int(np.argmax(sim[0]))
    score = float(sim[0][best])

    floor = config.CITED_MIN_SIM if valid else config.CITATION_MIN_SIM
    if score < floor:
        why = "drifts from cited verse" if valid else "unsupported"
        return {"ok": False, "cited_ids": [], "reason": f"{why} ({score:.2f})"}

    # Stage 2 — entailment, on what stage 1 let through. Rejects contradiction
    # ONLY: faithful teaching often reads as `neutral` to an NLI model (measured
    # 0.81 and 0.99 for two correct paraphrases), so demanding entailment would
    # silence the honest answers along with the false ones. See config.
    cited_ids = valid or [targets[best]]
    against = [known[t] for t in cited_ids]
    contra = nli.contradicts(against, bare)
    if contra > config.NLI_CONTRADICTION_MAX:
        return {"ok": False, "cited_ids": [],
                "reason": f"contradicts the cited verse ({contra:.2f})"}
    return {"ok": True, "cited_ids": cited_ids, "reason": f"{score:.2f}"}


def split_stream(buffer: str) -> tuple[list[str], str]:
    """Cut completed sentences off the front of a growing buffer.

    -> (finished sentences, remaining partial). Used to start speaking sentence N
    while the model is still writing sentence N+1.
    """
    parts = SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer
    return [p for p in parts[:-1] if p.strip()], parts[-1]
