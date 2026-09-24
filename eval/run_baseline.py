"""Baseline: the same LLM with NO retrieval.   Run: python eval/run_baseline.py [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from metrics import load_testset, report, score
from src.generation.llm import LLM
from src.generation.prompts import BASELINE_SYSTEM

TESTSET = Path(__file__).resolve().parent / "testset.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    records = load_testset(TESTSET)[: args.limit]
    llm = LLM()
    rows = []
    for rec in tqdm(records):
        answer = llm.chat(BASELINE_SYSTEM, rec["question"])
        rows.append({**rec, "answer": answer, "scores": score(rec, answer)})
    report("baseline", rows)


if __name__ == "__main__":
    main()
