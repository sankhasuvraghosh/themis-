"""Create supervised fine-tuning data that teaches the answer FORMAT (citations and NOT_FOUND).
Answers are built from the section text itself, so they are correct by construction.
Sections are split into train/val BEFORE examples are generated, to avoid leakage.

Run: python -m src.finetune.make_sft_data
"""
import json
import random
import zlib

from src.config import PROC_DIR, SECTIONS_PATH
from src.generation.prompts import SYSTEM_PROMPT, build_user_message
from src.ingest.chunking import chunk_section
from src.ingest.mapping import load_mapping, new_section_bases
from src.ingest.parse_pdf import load_sections

random.seed(42)
ASK = ["What does {act} section {n} provide?", "Explain {act} section {n}.", "What is {act} section {n} about?"]


def snippet(text: str, n: int = 350) -> str:
    if len(text) <= n:
        return text
    return text[: text.rfind(" ", 0, n)] + " ..."


def example(question, chunks, mappings, answer):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks, mappings)},
        {"role": "assistant", "content": answer},
    ]}


def make_examples(secs: list[dict]) -> list[dict]:
    out = []
    for s in secs:
        c = chunk_section(s)[0]
        q = random.choice(ASK).format(act=s["act"], n=s["section"])
        out.append(example(q, [c], [], f'{s["title"]}: {snippet(c["text"])} [{c["label"]}]'))
        # Distractor context that does NOT contain the asked section -> must answer NOT_FOUND
        others = [chunk_section(o)[0] for o in random.sample(secs, min(3, len(secs))) if o is not s]
        if others:
            q = random.choice(ASK).format(act=s["act"], n=s["section"])
            out.append(example(q, others, [], f'NOT_FOUND: The provided context does not contain {s["act"]} section {s["section"]}.'))
    return out


def make_mapping_examples(all_secs: list[dict], val: bool) -> list[dict]:
    by_key = {(s["act"].upper(), s["section"].upper()): s for s in all_secs}
    out = []
    for r in load_mapping():
        if (zlib.crc32((r["new_act"] + r["new_section"]).encode()) % 10 == 0) != val:
            continue
        chunks = [chunk_section(by_key[(r["new_act"].upper(), b)])[0]
                  for b in new_section_bases(r) if (r["new_act"].upper(), b) in by_key]
        refs = " ".join(f'[{c["label"]}]' for c in chunks)
        q = f'Which {r["new_act"]} section corresponds to {r["old_act"]} section {r["old_section"]}?'
        ans = f'{r["old_act"]} section {r["old_section"]} corresponds to {r["new_act"]} section {r["new_section"]}. {r.get("change_summary", "")} {refs}'.strip()
        out.append(example(q, chunks, [r], ans))
    return out


def main() -> None:
    secs = load_sections()
    random.shuffle(secs)
    cut = int(0.9 * len(secs))
    train = make_examples(secs[:cut]) + make_mapping_examples(secs, val=False)
    val = make_examples(secs[cut:]) + make_mapping_examples(secs, val=True)
    random.shuffle(train)
    for name, data in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        with open(PROC_DIR / name, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} examples")


if __name__ == "__main__":
    main()
