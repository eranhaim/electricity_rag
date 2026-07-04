"""Print the extracted markdown for a range of pages.

Usage inside container:
    docker exec electricity-rag python -m scripts.show_pages 51 57
"""

from __future__ import annotations

import sys

from backend.config import PROCESSED_DIR


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("Usage: show_pages.py START [END]  (inclusive)")
        return 1
    start = int(argv[0])
    end = int(argv[1]) if len(argv) > 1 else start

    root = PROCESSED_DIR / "pages"
    stems = sorted(d for d in root.iterdir() if d.is_dir())
    if not stems:
        print("No cached pages.")
        return 1
    pdf_dir = stems[0]

    for p in range(start, end + 1):
        f = pdf_dir / f"page_{p:04d}.md"
        if not f.exists():
            print(f"\n=== page {p} (missing) ===\n")
            continue
        text = f.read_text(encoding="utf-8")
        print(f"\n=== page {p} ({len(text)} chars) ===\n{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
