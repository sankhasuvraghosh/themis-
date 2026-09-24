"""Parse bare-act PDFs into one record per section.

Bare acts are formatted like:  "103. Punishment for murder.—(1) Whoever commits murder ..."
Parsing is heuristic: ALWAYS inspect data/processed/sections.jsonl after running it.
"""
import json
import re
from collections import Counter
from pathlib import Path

from src.config import ACT_FILES, EXPECTED_SECTIONS, PROC_DIR, RAW_DIR, SECTIONS_PATH

SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+(\S.*)$")
CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)\b", re.I)
HEADING_RE = re.compile(r"^(.+?)\.\s*[—–\-]+\s*(.*)$")


def load_pdf_lines(path: Path) -> list[str]:
    """Read a PDF and return clean text lines (page numbers and running headers removed)."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages = [[l.strip() for l in p.get_text().splitlines() if l.strip()] for p in doc]
    n = len(pages)
    counts = Counter(l for pg in pages for l in set(pg))
    noisy = {l for l, c in counts.items() if n >= 5 and c > 0.4 * n}  # repeated headers/footers
    return [l for pg in pages for l in pg if l not in noisy and not re.fullmatch(r"\d+", l)]


def _is_next(num: str, last: int) -> bool:
    """Accept only section numbers that plausibly follow the previous one."""
    m = re.match(r"(\d+)([A-Z]*)$", num)
    base, suffix = int(m.group(1)), m.group(2)
    if base == 1 and last >= 10:  # table of contents ended, body restarts at 1
        return True
    if base == last and suffix:   # e.g. 12A after 12
        return True
    return last < base <= last + 10


def split_heading(head: str) -> tuple[str, str]:
    m = HEADING_RE.match(head)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return head.rstrip(". ").strip(), ""


def parse_sections(lines: list[str], act: str) -> list[dict]:
    sections, cur = [], None
    last, chapter, chapter_title, want_title = 0, "", "", False
    for line in lines:
        cm = CHAPTER_RE.match(line)
        if cm:
            chapter, chapter_title, want_title = cm.group(1).upper(), "", True
            continue
        if want_title:
            want_title = False
            if line.isupper():
                chapter_title = line.title()
                continue
        sm = SECTION_RE.match(line)
        if sm and _is_next(sm.group(1), last):
            if cur:
                sections.append(cur)
            num = sm.group(1)
            last = int(re.match(r"\d+", num).group())
            title, body = split_heading(sm.group(2))
            cur = {"act": act, "chapter": chapter, "chapter_title": chapter_title,
                   "section": num, "title": title, "text": body}
        elif cur:
            cur["text"] = (cur["text"] + " " + line).strip()
    if cur:
        sections.append(cur)

    # The table of contents and the body both yield the same numbers: keep the longest text.
    best: dict[tuple, dict] = {}
    for s in sections:
        key = (s["act"], s["section"])
        if key not in best or len(s["text"]) > len(best[key]["text"]):
            best[key] = s
    return list(best.values())


def parse_all() -> list[dict]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for act, fname in ACT_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        secs = parse_sections(load_pdf_lines(path), act)
        exp = EXPECTED_SECTIONS.get(act)
        flag = "" if exp and abs(len(secs) - exp) <= 0.05 * exp else f"  <-- expected about {exp}, check the parser"
        print(f"{act}: {len(secs)} sections{flag}")
        records.extend(secs)
    with open(SECTIONS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_sections() -> list[dict]:
    with open(SECTIONS_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
