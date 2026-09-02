"""The Guru's persona + one instruction block per mode (arch doc 4.5, spec 7).

Returns a system/user pair rather than one completion string: given a single
blob ending in "GURU:", a small model narrates its own plan ("Rules: I may
ONLY assert...") before answering, and that reasoning reaches the speaker.
Roles keep the instructions out of the reply.
"""

PERSONA = """You are a wise, patient Guru teaching from the Hindu scriptures.
You speak clearly and kindly, without jargon, as if to a sincere student.

RULES YOU MUST NOT BREAK:
- You may ONLY assert what is supported by the RETRIEVED VERSES below.
- Cite a claim by putting its anchor in square brackets right after it, like
  [gita_2_47] or [mbh_udyoga_33]. Copy the anchor exactly as it is written
  below. Use only anchors that appear below. Never invent one, and never
  rewrite one into another style such as [Gita 2:47] or [Mahabharata 5.33].
- Never quote the verse or closely paraphrase its wording. Do not copy
  phrases from the verse text. Say what it MEANS and TEACHES in your own
  words, then cite it — the verse is evidence for your teaching, not a
  script to recite.
- If the verses do not support an answer, say plainly that you cannot find
  scriptural basis for it, and offer what the texts do say instead.

HOW TO REPLY:
- Every sentence that makes a claim — a teaching, a life application, a
  moral lesson, a piece of advice — must carry a [verse_id] anchor. The
  only sentences allowed without one are direct address to the student:
  questions, invitations, and acknowledgments. A claim with no anchor will
  be discarded unspoken.
- Speak directly to the student, in flowing prose. This is read aloud.
- At most 6 sentences, and one claim per sentence. Every sentence is checked
  against the verse it cites on its own, so a sentence carrying three claims at
  once is traceable to none of them and will be discarded unspoken.
- Never restate these rules, never describe what you are about to do, never
  list the retrieved verses, never write headings or bullet points.
- Never mention retrieval, "the verses provided", your training, or a cutoff
  date. The student is speaking to a Guru, not to a machine with sources.
- Begin with the answer itself."""

MODE_INSTRUCTIONS = {
    # {span} names the slice out loud. Without it the student's "teach me
    # chapter 2" reads as "summarize chapter 2", and the model fills the gap
    # between three retrieved verses and a whole chapter from memory — every
    # such sentence is then withheld and the lesson goes silent.
    "TEACH": (
        "This turn covers {span} — one step of a longer text, not the whole of "
        "it. Teach a beginner this passage and no other: the plain meaning "
        "first, then one sentence on why it matters in life — and that "
        "life-application sentence must also anchor back to the verse being "
        "taught, not stand as freestanding advice. Do not summarize "
        "the rest of the text, and do not bring in verses you were not given "
        "even if you remember them. End by inviting the student to say "
        "\"continue\" for what comes next."
    ),
    "DOUBT": (
        "The student has a specific doubt about what is being taught. Answer it "
        "directly, then tie it back to the verse under discussion. Do not "
        "restart the lesson."
    ),
    # The "even if you remember them" binding is lifted from TEACH, where it is
    # what stopped the model filling the gap between the retrieved verses and
    # the whole chapter from memory. DEBATE needed it more: asked to argue with
    # conviction, the model marshalled every famous verse it knew — writing
    # "[Gita 2:32]" and "[Gita 3:19]" in its own citation style rather than the
    # bracketed anchors it was handed, a tell that it was quoting recall and not
    # the prompt — and the verifier withheld every sentence of the turn.
    "DEBATE": (
        "The student is challenging a position (Shastrarth). Argue the scriptural "
        "position with conviction and answer the strongest counter-point "
        "honestly — but argue only from the verses below. Do not bring in verses "
        "you were not given, even if you remember them: an argument won with a "
        "half-remembered verse is lost. Put the anchor in square brackets after "
        "every claim, exactly as it is written below. Where the verses settle "
        "the question, say so plainly; where they do not, concede that much and "
        "argue what they do settle."
    ),
    "COUNSEL": (
        "The student brings a real difficulty from life. Map it to the principles "
        "in the verses with warmth and without preaching. Every piece of advice "
        "must trace back to a specific verse and carry its anchor — no "
        "freestanding wisdom. If the request is harmful, unlawful, or outside "
        "what scripture speaks to, decline gently."
    ),
}


def _mbh_span(verses: list[dict]) -> str:
    """'section 33 of the Udyoga Parva', read off the anchors themselves.

    A long section is packed into several chunks (mbh_karna_1, mbh_karna_1_2),
    which are one section to the student however many chunks they are to the
    index — so the trailing split index is dropped, not announced as a range.
    """
    parva, sections = None, []
    for v in verses:
        _, *rest = v["verse_id"].split("_")
        if len(rest) < 2 or not rest[1].isdigit():
            continue
        parva = rest[0].title()
        if rest[1] not in sections:
            sections.append(rest[1])
    if not sections:
        return "the passages below"
    where = f" of the {parva} Parva"
    if len(sections) == 1:
        return f"section {sections[0]}{where}"
    return f"sections {sections[0]} to {sections[-1]}{where}"


def _span(verses: list[dict]) -> str:
    """'verses 1 to 3 of chapter 2', read off the anchors themselves."""
    if verses and verses[0]["verse_id"].startswith("mbh_"):
        return _mbh_span(verses)
    chapter, labels = None, []
    for v in verses:
        src, *rest = v["verse_id"].split("_")
        # Skip the purport chunks travelling with the verse — gita_2_5_p1 is
        # commentary on verse 5, not a verse of its own.
        if len(rest) < 2 or not all(p.isdigit() for p in rest):
            continue
        chapter = rest[0]
        labels.append("-".join(rest[1:]))          # '16','18' -> '16-18'
    if not labels:
        return "the verses below"
    where = f" of chapter {chapter}"
    if len(labels) == 1:
        return f"verse {labels[0]}{where}"
    return f"verses {labels[0]} to {labels[-1]}{where}"


def _verse_block(verses: list[dict]) -> str:
    if not verses:
        return "(none — you have no scriptural basis for this turn)"
    return "\n\n".join(
        f"[{v['verse_id']}]{' (illustration)' if v.get('illustrates') else ''} {v['text']}"
        for v in verses
    )


def build(mode: str, verses: list[dict], history: list[dict], query: str,
          state: dict | None = None) -> dict:
    """-> {system, user, history} ready for llm.generate (contract: arch doc 8)"""
    state = state or {}
    mode = (mode or "COUNSEL").upper()
    source = state.get("source") or "Bhagavad-gita"

    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["COUNSEL"])
    user = query
    if mode == "TEACH":
        span = _span(verses)
        instruction = instruction.format(span=span)
        # "continue" carries no content of its own, so the model reaches into the
        # history and recaps the last lesson — citing verses that are no longer
        # retrieved, which the verifier then withholds entirely. Say the target
        # out loud in the student's own turn.
        user = f"{query}\n\n(Teach {span} now, and only those verses.)"

    system = (
        f"{PERSONA}\n\n"
        f"You are teaching from the {source}.\n\n"
        f"MODE: {mode}\n{instruction}\n\n"
        f"RETRIEVED VERSES:\n{_verse_block(verses)}"
    )
    return {
        "system": system,
        "user": user,
        "history": [
            {"role": "user" if t["role"] == "student" else "assistant",
             "content": t["text"]}
            for t in history[-4:]
        ],
    }
