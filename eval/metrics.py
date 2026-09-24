"""Metrics for the Themis test set (no extra dependencies).

Test-set format (eval/testset.jsonl), one JSON object per line:
  {"id": "q001", "question": "...", "gold_sections": ["BNS 103"], "answerable": true, "category": "mapping"}
Section numbers are compared by act + base number, so "BNS 318(4)" counts as "BNS 318".
"""
import json
import re
import statistics
from pathlib import Path

CITE_1 = re.compile(r"\b(BNSS|BNS|BSA)\b[\s,]*(?:§|section|sec\.?|s\.)?\s*(\d{1,3})(?!\d)", re.I)
CITE_2 = re.compile(r"(?:§|section|sec\.?)\s*(\d{1,3})(?:\s*\(\w+\))*\s*of\s*(?:the\s*)?(BNSS|BNS|BSA)\b", re.I)
REFUSAL = re.compile(r"not_found|not found|cannot answer|can't answer|out of scope|outside the scope|no information", re.I)


def extract_citations(text: str) -> set[str]:
    cites = {f"{m.group(1).upper()} {m.group(2)}" for m in CITE_1.finditer(text)}
    cites |= {f"{m.group(2).upper()} {m.group(1)}" for m in CITE_2.finditer(text)}
    return cites


BASE_SEC = re.compile(r"\d+[A-Za-z]*")


def norm_gold(g: str) -> str:
    act, sec = g.split(maxsplit=1)
    return act.upper() + " " + BASE_SEC.match(sec).group(0).upper()


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score(rec: dict, answer: str, context_labels: list[str] | None = None) -> dict:
    """context_labels: e.g. ['BNS 103', 'BNS 104'] (RAG only; enables recall@k and grounding)."""
    out: dict = {}
    cited = extract_citations(answer)
    if rec["answerable"]:
        gold = {norm_gold(g) for g in rec["gold_sections"]}
        out["cite_hit"] = bool(gold & cited)
        out["cite_exact"] = bool(gold) and gold <= cited
        if context_labels is not None:
            ctx = set(context_labels)
            out["recall_at_k"] = gold <= ctx
            out["cites_only_context"] = cited <= ctx
    else:
        out["refused"] = bool(REFUSAL.search(answer))
    return out


def summarize(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r["scores"]})
    return {k: round(statistics.mean(float(r["scores"][k]) for r in rows if k in r["scores"]), 3) for k in keys} | {"n": len(rows)}


def report(name: str, rows: list[dict]) -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    summary = summarize(rows)
    (out_dir / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n== {name} ==")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
