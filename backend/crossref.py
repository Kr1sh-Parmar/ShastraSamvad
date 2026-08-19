"""Gita <-> Mahabharata links (arch doc 4.4). A dict lookup, nothing more."""
import json

from . import config, retriever

_map: dict[str, list[dict]] | None = None


def _load() -> dict:
    global _map
    if _map is None:
        _map = (json.loads(config.CROSSREF_MAP.read_text(encoding="utf-8"))
                if config.CROSSREF_MAP.exists() else {})
    return _map


def expand(verse_ids: list[str], limit: int = 2) -> list[dict]:
    """-> [{verse_id, text}] illustrating passages for the given verses."""
    links = _load()
    out, seen = [], set(verse_ids)
    for vid in verse_ids:
        for link in links.get(vid, []):
            if link["verse_id"] in seen:
                continue
            rec = retriever.get(link["verse_id"])
            if rec:
                seen.add(rec["id"])
                out.append({"verse_id": rec["id"], "text": rec["text"],
                            "score": link["score"], "source": rec["source"],
                            "illustrates": vid})
            if len(out) >= limit:
                return out
    return out
