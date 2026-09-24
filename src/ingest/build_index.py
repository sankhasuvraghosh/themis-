"""Parse the PDFs, chunk them and build the hybrid index.   Run: python -m src.ingest.build_index"""
import sys

from src.ingest.chunking import chunk_section
from src.ingest.parse_pdf import parse_all
from src.retrieval.index import SectionIndex


def main() -> None:
    records = parse_all()
    if not records:
        sys.exit("No sections parsed. Put bns.pdf, bnss.pdf, bsa.pdf in data/raw/ (see README).")
    chunks = [c for s in records for c in chunk_section(s)]
    print(f"{len(records)} sections -> {len(chunks)} chunks. Embedding...")
    SectionIndex.build(chunks).save()
    print("Index saved to data/index/")


if __name__ == "__main__":
    main()
