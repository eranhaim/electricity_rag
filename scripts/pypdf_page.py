"""Extract raw text from a specific PDF page using pypdf, for comparison
against the vision-extracted markdown.

Usage inside the container:
    docker exec electricity-rag python -m scripts.pypdf_page 75
"""

from __future__ import annotations

import sys

from langchain_community.document_loaders import PyPDFLoader

from backend.config import UPLOADS_DIR


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: pypdf_page.py PAGE_NUMBER")
        return 1

    page_num = int(argv[0])
    pdfs = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    if not pdfs:
        print("No PDF in uploads/")
        return 1

    pdf = pdfs[0]
    print(f"== {pdf.name} page {page_num} via pypdf ==")
    loader = PyPDFLoader(str(pdf))
    docs = loader.load()
    if page_num < 1 or page_num > len(docs):
        print(f"Page {page_num} out of range (total: {len(docs)})")
        return 1
    print(docs[page_num - 1].page_content)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
