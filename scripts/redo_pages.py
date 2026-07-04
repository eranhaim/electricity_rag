"""Delete specific cached page markdown files and re-extract them via the
vision pipeline. Useful when a small number of pages are known-corrupted and
we don't want to reprocess the entire PDF.

Usage inside container:
    docker exec electricity-rag python -m scripts.redo_pages 49 50
"""

from __future__ import annotations

import asyncio
import sys

from backend.config import UPLOADS_DIR, PROCESSED_DIR
from backend.pdf_llm_ocr import _extract_page
from backend.pdf_llm_ocr import CONCURRENCY  # type: ignore


async def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: redo_pages.py PAGE [PAGE ...]")
        return 1

    page_nums = [int(a) for a in argv]

    pdfs = [f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
    if not pdfs:
        print("No PDF")
        return 1
    pdf = pdfs[0]

    cache_dir = PROCESSED_DIR / "pages" / pdf.stem
    for p in page_nums:
        f = cache_dir / f"page_{p:04d}.md"
        if f.exists():
            f.unlink()
            print(f"Deleted cached {f.name}")

    sem = asyncio.Semaphore(min(CONCURRENCY, len(page_nums)))
    tasks = [_extract_page(pdf, p, sem) for p in page_nums]
    results = await asyncio.gather(*tasks)
    for r in results:
        marker = "OK" if r.markdown.strip() else "EMPTY"
        print(f"page {r.page_num}: {marker} ({len(r.markdown)} chars, cached={r.from_cache})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
