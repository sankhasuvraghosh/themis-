"""Old-to-new section mapping (IPC -> BNS, CrPC -> BNSS, Evidence Act -> BSA).

data/mapping/mapping.csv columns: old_act,old_section,new_act,new_section,change_summary
  old_act in {IPC, CrPC, IEA}; new_act in {BNS, BNSS, BSA}.
Fill it from the official comparison tables, and spot-check it by hand.
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

from src.config import MAPPING_CSV

_ACT = r"(ipc|i\.p\.c|cr\.?\s?p\.?\s?c|iea|(?:indian\s+)?evidence\s+act)(?![a-z])"
_SEC = r"(\d{1,3}[a-z]{0,2})(?!\d)"
_PAT_A = re.compile(rf"\b{_ACT}\.?\s*[,:\-]?\s*(?:sections?|sec\.?|s\.)?\s*{_SEC}", re.I)
_PAT_B = re.compile(rf"\b(?:sections?|sec\.?|s\.)\s*{_SEC}\s*(?:of\s+)?(?:the\s+)?(?:old\s+)?{_ACT}", re.I)


def _canon(act: str) -> str:
    a = re.sub(r"[\s.]", "", act.lower())
    return {"ipc": "IPC", "crpc": "CRPC"}.get(a, "IEA")


def find_old_refs(query: str) -> list[tuple[str, str]]:
    """Find mentions like 'IPC 420', 'section 154 of CrPC' -> [('IPC','420'), ...]."""
    refs = [(_canon(m.group(1)), m.group(2).upper()) for m in _PAT_A.finditer(query)]
    refs += [(_canon(m.group(2)), m.group(1).upper()) for m in _PAT_B.finditer(query)]
    return list(dict.fromkeys(refs))


@lru_cache(maxsize=1)
def load_mapping(path: Path = MAPPING_CSV) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def lookup_old(act: str, section: str) -> list[dict]:
    return [r for r in load_mapping()
            if r["old_act"].upper() == act.upper() and r["old_section"].upper() == section.upper()]


def new_section_bases(row: dict) -> list[str]:
    """'318(4)' -> ['318'];  '103, 105' -> ['103', '105']."""
    s = re.sub(r"\([^)]*\)", "", row["new_section"].upper())
    return re.findall(r"\d+[A-Z]{0,2}", s)
