"""End-to-end RAG pipeline: mapping lookup + hybrid retrieval + rerank + grounded generation."""
from src.config import TOP_K_CONTEXT, TOP_K_RETRIEVE
from src.generation.llm import LLM
from src.generation.prompts import SYSTEM_PROMPT, build_user_message
from src.ingest.mapping import find_old_refs, lookup_old, new_section_bases
from src.retrieval.index import SectionIndex
from src.retrieval.reranker import Reranker


class Themis:
    def __init__(self, llm: LLM | None = None):
        self.index = SectionIndex.load()
        self.reranker = Reranker()
        self.llm = llm or LLM()

    def _pinned(self, query: str) -> tuple[list[dict], list[dict]]:
        """If the query names an old IPC/CrPC/Evidence section, pin its new-code section into the context."""
        rows, pinned = [], []
        for act, sec in find_old_refs(query):
            for row in lookup_old(act, sec):
                rows.append(row)
                for base in new_section_bases(row):
                    for i in self.index.by_section.get((row["new_act"].upper(), base), [])[:2]:
                        pinned.append(self.index.chunks[i])
        return pinned, rows

    def retrieve(self, query: str) -> tuple[list[dict], list[dict]]:
        pinned, mappings = self._pinned(query)
        candidates = [self.index.chunks[i] for i in self.index.search(query, TOP_K_RETRIEVE)]
        top = self.reranker.rerank(query, candidates, TOP_K_CONTEXT)
        seen = {c["id"] for c in pinned}
        context = pinned + [c for c in top if c["id"] not in seen]
        return context[: TOP_K_CONTEXT + len(pinned)], mappings

    def answer(self, query: str) -> dict:
        context, mappings = self.retrieve(query)
        text = self.llm.chat(SYSTEM_PROMPT, build_user_message(query, context, mappings))
        return {"answer": text, "context": context, "mappings": mappings}
