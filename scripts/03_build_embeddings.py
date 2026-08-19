"""Embed every chunk, build the FAISS index.

Reads  data/chunks/corpus.jsonl
Writes data/index/corpus.faiss, data/index/id_map.json
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config


def main() -> None:
    if not config.CORPUS_JSONL.exists():
        raise SystemExit("No corpus.jsonl — run 02_chunk_and_tag.py first.")

    records = [json.loads(line) for line in config.CORPUS_JSONL.open(encoding="utf-8")]
    print(f"embedding {len(records):,} chunks with {config.EMBED_MODEL} ...")

    model = SentenceTransformer(config.EMBED_MODEL)
    vecs = model.encode(
        [r["text"] for r in records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,       # normalized + inner product == cosine
        convert_to_numpy=True,
    ).astype(np.float32)

    # ponytail: flat index — exact, no training, fast enough at this scale.
    # Switch to IVF only if search time is measured as a problem.
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    config.INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX))
    config.ID_MAP.write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{index.ntotal:,} vectors (dim {vecs.shape[1]}) -> {config.FAISS_INDEX}")


if __name__ == "__main__":
    main()
