"""Verse-anchored retrieval. Strategy varies by mode (arch doc 4.3)."""
import json
from itertools import zip_longest

import numpy as np

from . import config, embed

_index = None
_records: list[dict] = []
_by_id: dict[str, dict] = {}
_share: dict[str, float] = {}


def _load():
    global _index, _records, _by_id, _share
    if _index is not None:
        return
    import faiss
    if not config.FAISS_INDEX.exists():
        raise RuntimeError(
            f"No index at {config.FAISS_INDEX}. Run scripts/00..03 first."
        )
    _index = faiss.read_index(str(config.FAISS_INDEX))
    _records = json.loads(config.ID_MAP.read_text(encoding="utf-8"))
    _by_id = {r["id"]: r for r in _records}
    counts: dict[str, int] = {}
    for r in _records:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    _share = {s: n / len(_records) for s, n in counts.items()}


def get(verse_id: str) -> dict | None:
    _load()
    return _by_id.get(verse_id)


def all_records() -> list[dict]:
    _load()
    return _records


def _hit(i: int, score: float) -> dict:
    r = _records[i]
    return {"verse_id": r["id"], "text": r["text"], "score": round(float(score), 4),
            "source": r["source"]}


def _semantic(query: str, k: int, source: str | None = None) -> list[dict]:
    _load()
    # Over-fetch when filtering by source, so the filter can't starve the result
    # — scaled by how much of the corpus that source actually is, not a flat
    # multiple. The flat 8x was set when the index held nothing but the Gita, so
    # the filter discarded nothing. The Gita is now 11% of 17k chunks, and 8x
    # returned 6 verses of the 8 asked for on the DEBATE counter-query: the same
    # starvation that once had the model arguing from memory with every sentence
    # withheld. Searching 280 of 17,380 in a flat index costs single-digit ms.
    fetch = int(k / _share.get(source, 1.0) * 4) if source else k
    scores, ids = _index.search(embed.encode(query), min(fetch, _index.ntotal))
    hits = []
    for i, s in zip(ids[0], scores[0]):
        if i < 0:
            continue
        if source and _records[i]["source"] != source:
            continue
        hits.append(_hit(i, s))
        if len(hits) == k:
            break
    return hits


def _lesson_order(rec: dict) -> tuple[int, int]:
    """Where a chunk sits in the lesson: (unit number, split index).

    Gita:         verse number. '47' -> 47, '16_18' -> 16. Verse ids are
                  strings, so sorting them lexically would teach chapter 2 as
                  1, 10, 11, ... 2, 20.
    Mahabharata:  section number, then which chunk of that section. A long
                  section is packed into several — mbh_karna_1, mbh_karna_1_2 —
                  and they must be taught in that order, not interleaved.
    """
    if rec["source"] == "Gita":
        return int(str(rec.get("verse", 0)).split("_")[0]), 0
    section = int(rec.get("section", 0))
    # The trailing number is the section itself on the first chunk, and the
    # split index on the rest: mbh_karna_1 vs mbh_karna_1_2.
    if rec["id"].endswith(f"_{section}"):
        return section, 1
    tail = rec["id"].rsplit("_", 1)[-1]
    return section, int(tail) if tail.isdigit() else 1


def _purports(verse_id: str) -> list[dict]:
    """The commentary chunks belonging to one verse: gita_2_5 -> gita_2_5_p1.

    A lesson needs them. The verse itself is first-person speech ("how can I
    counterattack Bhishma"); the Guru teaches it in third-person prose, and a
    faithful paraphrase measured only 0.560 against the verse but 0.715 against
    its purport. Without the purport in the retrieved set, correct teaching
    scores as drift and the whole turn is withheld.
    """
    _load()
    ps = [r for r in _records if r["id"].startswith(verse_id + "_p")]
    # ponytail: first chunk only. Verse 2.2 carries four, and handing the model
    # all of them turned the lesson into a chapter-wide recap citing verses it
    # had not been given. Take more per verse only if a lesson reads thin.
    return ps[:1]


def _lesson_chunks(source: str, unit: int | str) -> list[dict]:
    """The teachable units of one chapter (Gita) or one parva (Mahabharata).

    A lesson walks the scripture itself; the Gita's purports are commentary and
    would otherwise consume the teaching window, so they are excluded here and
    added back per verse by _purports. The Mahabharata has no purports — its
    passages are already narrative prose, which is what a lesson reads out.
    """
    _load()
    if source == "Gita":
        units = [r for r in _records
                 if r["source"] == source and r.get("chapter") == unit
                 and r.get("kind") == "verse"]
    else:
        # Case-insensitive: the parva name is a label chosen by the client.
        want = str(unit).lower()
        units = [r for r in _records
                 if r["source"] == source
                 and str(r.get("parva", "")).lower() == want]
    return sorted(units, key=_lesson_order)


def search(query: str, mode: str = "COUNSEL", state: dict | None = None) -> list[dict]:
    """-> [{verse_id, text, score, source}]  (contract: arch doc 8)"""
    state = state or {}
    source = state.get("source")
    mode = (mode or "COUNSEL").upper()

    if mode == "TEACH":
        # Sequential walk through the chapter or parva — no embedding search.
        unit = state.get("chapter")
        units = _lesson_chunks(source or "Gita", unit) if unit else []
        if units:
            pos = int(state.get("position", 0))
            window = units[pos:pos + config.TEACH_WINDOW]
            if window:
                return [{"verse_id": r["id"], "text": r["text"], "score": 1.0,
                         "source": r["source"]}
                        for v in window for r in [v, *_purports(v["id"])]]
        return _semantic(query, config.TOP_K, source)   # unit unknown -> fall back

    if mode == "DOUBT":
        # Anchor the doubt to the verse currently being taught.
        lesson = state.get("current_verse_text", "")
        return _semantic(f"{query}\n{lesson}".strip(), config.TOP_K, source)

    if mode == "DEBATE":
        # Verses supporting AND countering the position, so the Guru can argue
        # both — and more of them than any other mode gets. Debate has the most
        # to say; handed three verses the model argued from memory instead, and
        # every sentence of the turn was withheld as unsupported.
        #
        # Two measured changes. Each side now fetches DEBATE_TOP_K rather than
        # TOP_K // 2 — after dedup the old split returned 3 verses where COUNSEL
        # got 6. And the counter-query names the rebuttal: "arguments against:
        # X" proved to be a reordering of X (1-2 of 6 verses were new), because
        # the embedder scores the topic and ignores the prefix. Asking for the
        # refutation turns up 2-5 verses the plain query never sees.
        sides = [_semantic(q, config.DEBATE_TOP_K, source) for q in
                 (query, f"the scriptural answer refuting the claim that {query}")]
        merged, seen = [], set()
        # Round-robin, so the cap cannot leave one side unrepresented.
        for rank in zip_longest(*sides):
            for h in rank:
                if h and h["verse_id"] not in seen:
                    seen.add(h["verse_id"])
                    merged.append(h)
        return merged[:config.DEBATE_TOP_K]

    # COUNSEL — broad search across both texts, mapping situation to principles.
    return _semantic(query, config.TOP_K)


if __name__ == "__main__":
    for q in ["what is my duty when I do not want to fight",
              "how do I act without craving the result",
              "my mind will not stay still"]:
        print(f"\n=== {q}")
        for h in search(q):
            print(f"  {h['score']:.3f}  [{h['verse_id']}]  {h['text'][:90]}...")
