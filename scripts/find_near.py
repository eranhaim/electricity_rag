"""Search extracted pages for lines where a term appears near a specific
measurement pattern (numbers with units).

Usage inside container:
    docker exec electricity-rag python -m scripts.find_near לוח
    docker exec electricity-rag python -m scripts.find_near "פס השוואת פוטנציאלים"
"""

from __future__ import annotations

import re
import sys

from backend.config import PROCESSED_DIR

UNIT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:מטר|מ\"מ|מ״מ|ס\"מ|ס״מ|ממ\"ר|מגאום|אמפר|וולט|kV|mA|Hz)")


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: find_near.py TERM [TERM ...]")
        return 1
    terms = argv

    root = PROCESSED_DIR / "pages"
    hits = 0
    for md_file in sorted(root.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            if all(t in line for t in terms) and UNIT_RE.search(line):
                hits += 1
                page = md_file.stem
                print(f"{page}: {line.strip()[:280]}")
    print(f"\n== {hits} lines contained ALL {terms} AND a measurement ==")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
