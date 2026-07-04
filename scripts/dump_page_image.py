"""Render a page to PNG and save it locally so we can visually inspect what
gpt-4o is actually seeing.

Usage inside the container:
    docker exec electricity-rag python -m scripts.dump_page_image 75 300
    docker cp electricity-rag:/tmp/page_0075.png ./page_0075.png
"""

from __future__ import annotations

import sys

import fitz

from backend.config import UPLOADS_DIR


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: dump_page_image.py PAGE_NUMBER [DPI]")
        return 1
    page_num = int(argv[0])
    dpi = int(argv[1]) if len(argv) > 1 else 200

    pdfs = [f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
    if not pdfs:
        print("No PDF")
        return 1

    doc = fitz.open(str(pdfs[0]))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi)
    out = f"/tmp/page_{page_num:04d}.png"
    pix.save(out)
    doc.close()
    print(f"Saved {out} ({pix.width}x{pix.height} @ {dpi} DPI)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
