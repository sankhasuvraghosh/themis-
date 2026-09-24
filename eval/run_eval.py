"""Full Themis pipeline evaluation.   Run: python eval/run_eval.py [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from metrics import load_testset, report, score
from src.generation.pipeline import Themis

TESTSET = Path(__file__).resolve().parent / "testset.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    records = load_testset(TESTSET)[: args.limit]
    themis = Themis()
    rows = []
    for rec in tqdm(records):
        out = themis.answer(rec["question"])
        labels = [f'{c["act"]} {c["section"]}' for c in out["context"]]
        rows.append({**rec, "answer": out["answer"], "context": labels, "scores": score(rec, out["answer"], labels)})
    report("themis", rows)


if __name__ == "__main__":
    main()
