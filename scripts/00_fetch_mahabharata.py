"""Download the Ganguli Mahabharata into data/raw/ (spec 5, arch doc 4.3).

    python scripts/00_fetch_mahabharata.py

Until this runs the corpus is Gita-only, so 04_build_crossref.py has nothing to
link against and /health reports crossref: false.

sacred-texts.com — the source the spec names — is behind Cloudflare now (403).
Project Gutenberg carries the same Ganguli translation, complete, in four
volumes covering all 18 parvas.

Reads  nothing
Writes data/raw/mahabharata_vol{1..4}.txt   (~12 MB total)
"""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config

# Volume -> Gutenberg ebook id. Between them: BOOK 1 through BOOK 18.
VOLUMES = {1: 15474, 2: 15475, 3: 15476, 4: 15477}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# These files carry the Project Gutenberg licence around the text. Stripping it
# is 01_clean_texts.py's job — but if that ever silently stopped happening, the
# licence would be embedded and retrieved as scripture, so the marker it keys on
# is checked here too, at the point the bytes arrive.
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def fetch(url: str) -> bytes:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        buf = bytearray()
        for chunk in r.iter_bytes(1 << 18):
            buf += chunk
            if total:
                print(f"\r  {len(buf) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
        print()
        return bytes(buf)


def main() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)

    for vol, ebook in VOLUMES.items():
        out = config.RAW / f"mahabharata_vol{vol}.txt"
        if out.exists():
            print(f"{out.name} already present, skipping")
            continue

        print(f"{out.name}  (Gutenberg #{ebook}) ...")
        body = fetch(URL.format(id=ebook))
        text = body.decode("utf-8", errors="replace")
        if START_MARKER not in text or END_MARKER not in text:
            raise SystemExit(
                f"#{ebook}: no Project Gutenberg START/END markers. The licence "
                "cannot be stripped from a file shaped like this — inspect it "
                "before letting it into the corpus."
            )
        # Bytes, not write_text: these files are CRLF, and a text-mode write on
        # Windows turns every \r\n into \r\r\n. Read back under universal
        # newlines that is two line breaks, so the prose arrives with a blank
        # line between every wrapped line and each one chunks as its own
        # paragraph. Raw stays byte-for-byte what the source served.
        out.write_bytes(body)

    got = sorted(config.RAW.glob("mahabharata_vol*.txt"))
    print(f"\n{len(got)}/{len(VOLUMES)} volumes -> {config.RAW}")
    if len(got) != len(VOLUMES):
        raise SystemExit("Some volumes are missing — see above.")
    print("Now re-run 01_clean_texts.py, 02, 03 and 04 to rebuild the index.")


if __name__ == "__main__":
    main()
