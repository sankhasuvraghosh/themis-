"""Hybrid index: BM25 + dense embeddings fused with reciprocal-rank fusion (RRF).

At this corpus size (about 1,000 sections) a plain NumPy matrix product is fast enough;
swap in FAISS if you scale up.
"""
import json
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from src.config import EMBED_MODEL, INDEX_DIR, RRF_K, TOP_K_RETRIEVE

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


class SectionIndex:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.emb = embeddings
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])
        self.acts = np.array([c["act"] for c in chunks])
        self.by_section: dict[tuple, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_section[(c["act"].upper(), c["section"].upper())].append(i)

    @classmethod
    def build(cls, chunks: list[dict]) -> "SectionIndex":
        emb = get_embedder().encode([c["embed_text"] for c in chunks], batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=True)
        return cls(chunks, np.asarray(emb, dtype="float32"))

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(INDEX_DIR / "embeddings.npy", self.emb)

    @classmethod
    def load(cls) -> "SectionIndex":
        try:
            with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
                chunks = [json.loads(l) for l in f]
            emb = np.load(INDEX_DIR / "embeddings.npy")
        except FileNotFoundError:
            raise SystemExit("Index not found. Run: python -m src.ingest.build_index")
        return cls(chunks, emb)

    def _mask(self, scores: np.ndarray, acts) -> np.ndarray:
        if acts:
            return np.where(np.isin(self.acts, list(acts)), scores, -np.inf)
        return scores

    def search(self, query: str, k: int = TOP_K_RETRIEVE, acts=None, pool: int = 50) -> list[int]:
        """Return chunk indices ranked by fused BM25 + dense score. `acts` filters by act (metadata filter)."""
        bm = self._mask(np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float64"), acts)
        q = BGE_QUERY_PREFIX + query if "bge" in EMBED_MODEL.lower() else query
        qv = get_embedder().encode([q], normalize_embeddings=True)[0]
        de = self._mask(self.emb @ qv, acts)

        fused: dict[int, float] = defaultdict(float)
        for scores, floor in ((bm, 0.0), (de, -np.inf)):
            for rank, idx in enumerate(np.argsort(-scores)[:pool]):
                if scores[idx] > floor:
                    fused[int(idx)] += 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)[:k]
