"""Raw source -> clean, marked-up text. Keeps chapter/verse structure explicit.

Reads  data/raw/*.pdf|*.txt|*.htm|*.html
Writes data/chunks/clean_<name>.txt

PDF path is built for the Bhagavad-gita As It Is layout: each verse is a
`TEXT n` block containing TRANSLATION and (usually) PURPORT. The PDF's own
table of contents supplies chapter.verse labels; the two line up 1:1, which is
checked at runtime rather than assumed.
"""
import html
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config

# --- plain text / html -------------------------------------------------------
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
FOOTNOTE_REF = re.compile(r"\[\s*(p\.\s*)?\d+\s*\]|\{p\.\s*\d+\}")
FOOTNOTE_LINE = re.compile(r"^\s*\d+:\s.*$", re.M)
PAGE_MARKER = re.compile(r"^\s*p\.\s*\d+\s*$", re.M)
BLANKS = re.compile(r"\n{3,}")
SPACES = re.compile(r"[ \t]+")

# Project Gutenberg wraps each text in a licence. Left in, it is embedded,
# indexed and retrieved exactly like scripture — the Guru would cite the terms
# of use as though they were the Mahabharata. The markers below are what the
# body is cut out from between.
PG_START = re.compile(r"^\*\*\* ?START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.M)
PG_END = re.compile(r"^\*\*\* ?END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.M)

# --- pdf ---------------------------------------------------------------------
COPYRIGHT = re.compile(r"^Copyright.*Reserved\.\s*$", re.M)
TEXT_HEADER = re.compile(r"^TEXTS?\s+(\d+)(?:\s*[-–]\s*(\d+))?\s*$", re.M)
TOC_VERSE = re.compile(r"^Texts?\s+(\d+)\.(\d+)(?:\s*[-–]\s*(\d+))?$")

# The BBT diacritic font parks Sanskrit letters in Latin-1 slots. Folding them
# to plain ASCII gives the spellings people actually say and read
# (Krsna -> Krishna, ksatriya -> kshatriya), which is what the embedder indexes
# and the TTS has to pronounce.
# ponytail: fold to ASCII, don't preserve diacritics. This is a *spoken* Guru;
# scholarly transliteration would only have to be undone before synthesis.
BBT = {
    0xE0: "m", 0xE4: "a", 0xE5: "ri", 0xE7: "sh", 0xE9: "i", 0xEB: "n",
    0xEC: "n", 0xEF: "n", 0xF1: "sh", 0xF2: "d", 0xF6: "t", 0xF9: "h",
    0xFC: "u", 0xC4: "A", 0xC5: "Ri", 0xC7: "Sh", 0xC9: "I", 0xCB: "N",
    0xCC: "N", 0xCF: "N", 0xD1: "Sh", 0xD6: "T", 0xDC: "U",
}
PUNCT = {"—": "-", "–": "-", "’": "'", "‘": "'",
         "“": '"', "”": '"', "©": "(c)", " ": " "}
FOLD = str.maketrans({chr(k): v for k, v in BBT.items()} | PUNCT)
PUNCT_ONLY = str.maketrans(PUNCT)


def fold(text: str) -> str:
    """Decode the diacritic font, then drop anything still non-ASCII."""
    return text.translate(FOLD).encode("ascii", "ignore").decode()


def strip_gutenberg(text: str, name: str) -> str:
    """Keep only what lies between the START and END markers.

    Refuses rather than guesses: a Gutenberg file whose markers have moved is a
    file we cannot safely cut the licence out of, and shipping it uncut would
    make the licence citable scripture. Same posture as clean_pdf's TOC check.
    """
    if "PROJECT GUTENBERG" not in text.upper():
        return text
    start, end = PG_START.search(text), PG_END.search(text)
    if not (start and end and start.end() < end.start()):
        raise SystemExit(
            f"{name}: Project Gutenberg text with no usable START/END markers. "
            "Refusing to emit the licence as scripture."
        )
    return text[start.end():end.start()]


def to_ascii(text: str) -> str:
    """Decompose accents and drop what will not fold.

    PUNCT first, or the drop takes the punctuation with it: volume 1 alone holds
    13,539 curly quotes and apostrophes, and Ganguli's prose is nested direct
    speech throughout. Deleting them turns "Drona's son" into "Dronas son" and
    strips every quotation mark out of the dialogue.

    NOT fold(): that table decodes the BBT diacritic font, which parks Sanskrit
    letters in Latin-1 slots. Applied to a file with genuine Latin-1 accents it
    would read 0xE9 as a Sanskrit vowel and turn 'e' into 'i'.
    """
    text = text.translate(PUNCT_ONLY)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def clean_markup(text: str) -> str:
    if "<" in text and ">" in text:
        text = SCRIPT_STYLE.sub(" ", text)
        text = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>", "\n", text, flags=re.I)
        text = TAG.sub("", text)
    text = html.unescape(text)
    text = FOOTNOTE_REF.sub("", text)
    text = FOOTNOTE_LINE.sub("", text)
    text = PAGE_MARKER.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANKS.sub("\n\n", text).strip()


def clean_pdf(path: Path) -> str:
    import fitz

    doc = fitz.open(path)
    body = "".join(COPYRIGHT.sub("", doc[p].get_text()) for p in range(doc.page_count))

    blocks = list(TEXT_HEADER.finditer(body))
    labels = [TOC_VERSE.match(t.strip()).groups()
              for _, t, _ in doc.get_toc() if TOC_VERSE.match(t.strip())]

    if len(blocks) != len(labels):
        raise SystemExit(
            f"{path.name}: {len(blocks)} TEXT blocks vs {len(labels)} TOC verses. "
            "The layout assumption does not hold — inspect before trusting anchors."
        )
    # Anchors are only trustworthy if the TOC verse number matches the block header.
    for i, (blk, lab) in enumerate(zip(blocks, labels)):
        if blk.group(1) != lab[1]:
            raise SystemExit(
                f"{path.name}: block {i} is 'TEXT {blk.group(1)}' but the TOC says "
                f"{lab[0]}.{lab[1]}. Refusing to emit misnumbered anchors."
            )

    out = []
    for i, (blk, (chapter, v1, v2)) in enumerate(zip(blocks, labels)):
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(body)
        chunk = body[blk.end():end]

        ti = chunk.find("TRANSLATION")
        if ti < 0:
            continue
        pi = chunk.find("PURPORT", ti)
        translation = chunk[ti + len("TRANSLATION"):pi if pi > 0 else len(chunk)]
        purport = chunk[pi + len("PURPORT"):] if pi > 0 else ""

        verse = f"{v1}_{v2}" if v2 else v1
        out.append(f"### GITA {chapter}.{verse}")
        out.append("@TRANSLATION")
        out.append(re.sub(r"\s+", " ", fold(translation)).strip())
        if purport.strip():
            out.append("@PURPORT")
            out.append(BLANKS.sub("\n\n", SPACES.sub(" ", fold(purport))).strip())
        out.append("")
    return "\n".join(out)


def main() -> None:
    files = sorted(p for p in config.RAW.iterdir()
                   if p.suffix.lower() in {".pdf", ".txt", ".htm", ".html"})
    if not files:
        raise SystemExit(f"No source files in {config.RAW}. Put the corpus there first.")

    config.CHUNKS.mkdir(parents=True, exist_ok=True)
    for src in files:
        if src.suffix.lower() == ".pdf":
            cleaned = clean_pdf(src)
        else:
            raw = src.read_text(encoding="utf-8", errors="ignore")
            cleaned = to_ascii(clean_markup(strip_gutenberg(raw, src.name)))
        out = config.CHUNKS / f"clean_{src.stem}.txt"
        out.write_text(cleaned, encoding="utf-8")
        print(f"{src.name:40s} -> {out.name:40s} {len(cleaned):>9,} chars")


if __name__ == "__main__":
    main()
