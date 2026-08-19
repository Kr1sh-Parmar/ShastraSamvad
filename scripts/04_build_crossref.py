"""Gita verse -> the Mahabharata passages that illustrate it.

Lets the Guru explain a principle and cite the narrative that shows it lived out.

Reads  data/index/*
Writes data/crossref/gita_to_mbh.json
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config


def main() -> None:
    if not config.FAISS_INDEX.exists():
        raise SystemExit("No index — run 03_build_embeddings.py first.")

    records = json.loads(config.ID_MAP.read_text(encoding="utf-8"))
    index = faiss.read_index(str(config.FAISS_INDEX))
    vecs = index.reconstruct_n(0, index.ntotal).astype(np.float32)

    gita = [i for i, r in enumerate(records) if r["source"] == "Gita"]
    mbh = [i for i, r in enumerate(records) if r["source"] == "Mahabharata"]
    if not gita or not mbh:
        raise SystemExit("Need both Gita and Mahabharata chunks to cross-reference.")

    # Search only the Mahabharata half, so a Gita verse can't link to itself.
    sub = faiss.IndexFlatIP(vecs.shape[1])
    sub.add(vecs[mbh])
    scores, hits = sub.search(vecs[gita], config.CROSSREF_TOP_K)

    out, kept = {}, 0
    for row, gi in enumerate(gita):
        links = [
            {"verse_id": records[mbh[h]]["id"], "score": round(float(s), 4)}
            for h, s in zip(hits[row], scores[row])
            if s >= config.CROSSREF_MIN_SCORE
        ]
        if links:
            out[records[gi]["id"]] = links
            kept += len(links)

    config.CROSSREF.mkdir(parents=True, exist_ok=True)
    config.CROSSREF_MAP.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out):,}/{len(gita):,} Gita verses linked, {kept:,} links "
          f"(min score {config.CROSSREF_MIN_SCORE}) -> {config.CROSSREF_MAP}")


if __name__ == "__main__":
    main()
