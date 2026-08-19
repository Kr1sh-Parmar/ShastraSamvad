"""Verse-anchored chunking. THE most important script in the project.

Every chunk's `id` is a citation anchor the Guru will speak and the UI will show.
Wrong anchors here poison every answer downstream.

Reads  data/chunks/clean_*.txt
Writes data/chunks/corpus.jsonl
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config

MAX_CHARS = 1200          # long purports / Mahabharata sections split here
MIN_CHARS = 40            # drop headers and stubs

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

GITA_BLOCK = re.compile(
    r"^### GITA (\d+)\.([\d_]+)\s*\n@TRANSLATION\s*\n(.*?)(?:\n@PURPORT\s*\n(.*?))?"
    r"(?=\n### GITA |\Z)",
    re.S | re.M,
)
# Every volume delimits its parvas with `BOOK n`, uniformly, across all 18 —
# which is what makes the two parva-name styles below stop mattering. The names
# themselves are inconsistent (`ADI PARVA` in volumes 1-2, `Karna-parva` in 3-4,
# and volume 3 uses both), and they repeat a few hundred lines later as a
# running head, so they are read as a label and never used to find a boundary.
BOOK_RE = re.compile(r"^BOOK\s+(\d+)\s*$", re.M)
PARVA_NAME_RE = re.compile(r"^\W*(.+?)[\s-]*parva\W*$", re.I)
SECTION_RE = re.compile(r"^\s*SECTION\s+([IVXLCDM]+|\d+)\b.*$", re.I | re.M)
# Ganguli's footnotes are collected as numbered paragraphs at the end of a
# section — "109. The Bengal reading of the second line is vicious." They are
# scholarly apparatus, not the epic, and they carry the section's anchor, so
# left in they are retrieved and cited as though Vyasa had written them.
# 5.65% of the Mahabharata text; sampled across all four volumes, nothing but
# footnotes matches this. Mahabharata only — the Gita's purports are commentary
# we index on purpose, under their own `kind`.
FOOTNOTE_PARA = re.compile(r"^\d{1,4}\.\s")
# The second section style: a bare integer alone on its own line. Ganguli's prose
# is hard-wrapped, so a line holding nothing but a number is a header; the years
# that could be confused with one ("[1883-1896]") are bracketed.
BARE_SECTION_RE = re.compile(r"^(\d{1,4})$", re.M)


def to_int(token: str) -> int:
    if token.isdigit():
        return int(token)
    total = prev = 0
    for ch in reversed(token.upper()):
        val = ROMAN.get(ch, 0)
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _fit(para: str, limit: int) -> list[str]:
    """Cut one over-long paragraph at sentence bounds, then hard if it must.

    The Gita's purports are ordinary paragraphs, so nothing here fires for them.
    Ganguli runs a whole section of dialogue as a single paragraph — the longest
    measured 60,770 characters, and a paragraph is never cut by pack() below.
    That silently breaks retrieval twice over: bge-small truncates at 512 tokens
    (~2,000 chars), so such a chunk is indexed on its opening 3% and the rest can
    never be found, and if it IS retrieved the whole 60k goes into the prompt.
    """
    if len(para) <= limit:
        return [para]
    out, buf = [], ""
    for s in re.split(r"(?<=[.!?])\s+", para):
        while len(s) > limit:                  # a "sentence" with no end in sight
            out.append(s[:limit])
            s = s[limit:]
        if buf and len(buf) + len(s) + 1 > limit:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        out.append(buf)
    return out


def pack(paras: list[str], limit: int = MAX_CHARS) -> list[str]:
    """Greedily fill chunks up to `limit`, cutting mid-paragraph only if forced."""
    chunks, buf = [], ""
    for p in (part for para in paras for part in _fit(para, limit)):
        if buf and len(buf) + len(p) + 2 > limit:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def chunk_gita(text: str) -> list[dict]:
    """One chunk per verse translation, plus the purport split into chunks.

    The translation is scripture and is what the Guru cites; the purport is
    commentary that explains it. Both are retrievable, and `kind` keeps them
    distinguishable at prompt time.
    """
    out = []
    for m in GITA_BLOCK.finditer(text):
        chapter, verse, translation, purport = m.group(1), m.group(2), m.group(3), m.group(4)
        translation = re.sub(r"\s+", " ", translation).strip()
        if len(translation) < MIN_CHARS:
            continue

        out.append({
            "id": f"gita_{chapter}_{verse}",
            "text": translation,
            "source": "Gita",
            "kind": "verse",
            "chapter": int(chapter),
            "verse": verse,
            "ref": f"Bhagavad-gita {chapter}.{verse.replace('_', '-')}",
        })

        for i, chunk in enumerate(pack(paragraphs(purport or "")), start=1):
            if len(chunk) >= MIN_CHARS:
                out.append({
                    "id": f"gita_{chapter}_{verse}_p{i}",
                    "text": re.sub(r"[ \t]+", " ", chunk).strip(),
                    "source": "Gita",
                    "kind": "purport",
                    "chapter": int(chapter),
                    "verse": verse,
                    "ref": f"Purport to Bhagavad-gita {chapter}.{verse.replace('_', '-')}",
                })
    return out


def split_on(pattern: re.Pattern, text: str) -> list[tuple[int, str]]:
    """-> [(number, body)] for each header; body runs to the next header."""
    matches = list(pattern.finditer(text))
    return [
        (to_int(m.group(1)),
         text[m.end():(matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip())
        for i, m in enumerate(matches)
    ]


def parva_name(body: str, book: int) -> str:
    """The parva label on the first non-blank line after its `BOOK n` header.

    'ADI PARVA' and 'Karna-parva' both -> 'Karna'-shaped: one capitalized word,
    which is what the anchor and the spoken reference are built from.
    """
    for line in body.split("\n"):
        if line.strip():
            m = PARVA_NAME_RE.match(line.strip())
            return m.group(1).strip().capitalize() if m else f"book{book}"
    return f"book{book}"


def chunk_mahabharata(text: str) -> list[dict]:
    """One chunk per section, split at paragraph bounds when a section is long.

    Split by book first, then by section within it. Section numbers restart at 1
    in every parva, so a file-wide section split would collide those anchors —
    and a single parva name read off the top of a volume would misattribute the
    other three to five parvas underneath it.
    """
    out = []
    for book, body in split_on(BOOK_RE, text):
        parva = parva_name(body, book)
        # Whichever style this book uses. split_on returns nothing before the
        # first header, which is how the translator's preface and the
        # sacred-texts scanning credits are dropped rather than made citable.
        sections = split_on(SECTION_RE, body) or split_on(BARE_SECTION_RE, body)
        for section, passage in sections:
            body = [p for p in paragraphs(passage) if not FOOTNOTE_PARA.match(p)]
            for i, chunk in enumerate(pack(body), start=1):
                chunk = re.sub(r"[ \t]+", " ", chunk).strip()
                if len(chunk) < MIN_CHARS:
                    continue
                suffix = "" if i == 1 else f"_{i}"
                out.append({
                    "id": f"mbh_{parva.lower()}_{section}{suffix}",
                    "text": chunk,
                    "source": "Mahabharata",
                    "kind": "passage",
                    "parva": parva,
                    "book": book,
                    "section": section,
                    "ref": f"Mahabharata, {parva} Parva, Section {section}",
                })
    return out


def main() -> None:
    files = sorted(config.CHUNKS.glob("clean_*.txt"))
    if not files:
        raise SystemExit("No clean_*.txt — run 01_clean_texts.py first.")

    records, seen = [], set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        recs = chunk_gita(text) if GITA_BLOCK.search(text) else chunk_mahabharata(text)

        for r in recs:                      # ids are anchors — they must be unique
            if r["id"] in seen:
                raise SystemExit(f"duplicate anchor {r['id']} — fix the parser")
            seen.add(r["id"])
        records.extend(recs)
        kinds = {k: sum(1 for r in recs if r["kind"] == k) for k in {r["kind"] for r in recs}}
        print(f"{path.name:45s} -> {len(recs):>6,} chunks  {kinds}")

    config.CORPUS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with config.CORPUS_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(records):,} chunks -> {config.CORPUS_JSONL}")

    print("\n--- spot-check these anchors against the source by hand ---")
    verses = [r for r in records if r["kind"] == "verse"]
    for r in verses[:: max(1, len(verses) // 5)][:5]:
        print(f"[{r['id']}] {r['ref']}\n    {r['text'][:110]}...")


if __name__ == "__main__":
    main()
