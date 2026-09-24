"""Cross-encoder reranker for the top hybrid-search candidates."""
from src.config import RERANK_MODEL, TOP_K_CONTEXT


class Reranker:
    def __init__(self, name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = TOP_K_CONTEXT) -> list[dict]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c["embed_text"]) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
        return [dict(c, rerank_score=float(s)) for c, s in ranked[:top_k]]
