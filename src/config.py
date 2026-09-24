"""Central configuration. Most values can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
MAPPING_CSV = ROOT / "data" / "mapping" / "mapping.csv"
SECTIONS_PATH = PROC_DIR / "sections.jsonl"

# Put the official PDFs in data/raw/ with these names.
ACT_FILES = {"BNS": "bns.pdf", "BNSS": "bnss.pdf", "BSA": "bsa.pdf"}
# Used only as a sanity check after parsing.
EXPECTED_SECTIONS = {"BNS": 358, "BNSS": 531, "BSA": 170}

# Retrieval models (small, free, run on CPU)
EMBED_MODEL = os.getenv("THEMIS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("THEMIS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM: "local" runs an open-weights model with transformers (no token needed).
#      "api" uses the Hugging Face Inference API (needs HF_TOKEN; free tier is rate-limited).
LLM_BACKEND = os.getenv("THEMIS_LLM_BACKEND", "local")
_DEFAULT_LLM = {"local": "Qwen/Qwen2.5-1.5B-Instruct", "api": "Qwen/Qwen2.5-7B-Instruct"}
LLM_MODEL = os.getenv("THEMIS_LLM_MODEL", _DEFAULT_LLM.get(LLM_BACKEND, "Qwen/Qwen2.5-1.5B-Instruct"))
ADAPTER_PATH = os.getenv("THEMIS_ADAPTER")  # optional LoRA adapter from src/finetune
HF_TOKEN = os.getenv("HF_TOKEN")

TOP_K_RETRIEVE = 20   # candidates from hybrid search
TOP_K_CONTEXT = 5     # chunks passed to the LLM after reranking
MAX_NEW_TOKENS = 400
RRF_K = 60            # reciprocal-rank-fusion constant
