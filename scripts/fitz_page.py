"""Extract text from a PDF page using PyMuPDF (fitz) — usually handles
Hebrew embedded fonts better than pypdf.

Usage inside the container:
    docker exec electricity-rag python -m scripts.fitz_page 75
"""

from __future__ import annotations

import sys

import fitz

from backend.config import UPLOADS_DIR


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: fitz_page.py PAGE_NUMBER")
        return 1
    page_num = int(argv[0])
    pdfs = [f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
    if not pdfs:
        print("No PDF in uploads/")
        return 1
    pdf = pdfs[0]
    doc = fitz.open(str(pdf))
    try:
        if page_num < 1 or page_num > doc.page_count:
            print(f"Page {page_num} out of range (total: {doc.page_count})")
            return 1
        page = doc.load_page(page_num - 1)

        print(f"== {pdf.name} page {page_num} via PyMuPDF (get_text) ==")
        raw = page.get_text("text")
        print(raw)
        hb = sum(1 for c in raw if "\u0590" <= c <= "\u05FF")
        print(f"\n-- Hebrew chars: {hb}, total chars: {len(raw)} --")
    finally:
        doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
