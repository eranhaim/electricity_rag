"""Search the extracted per-page markdown files for lines mentioning any of
several Hebrew terms. Useful for QA'ing whether specific content survived the
PDF → Markdown pipeline.

Usage inside the container:
    docker exec electricity-rag python -m scripts.find_topic "לוח חשמל" גובה מטר
"""

from __future__ import annotations

import sys

from backend.config import PROCESSED_DIR


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: find_topic.py TERM [TERM ...] (line must contain ALL terms)")
        return 1

    root = PROCESSED_DIR / "pages"
    if not root.exists():
        print(f"No cache dir at {root}")
        return 1

    hits = 0
    for md_file in sorted(root.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if all(term in line for term in argv):
                hits += 1
                page = md_file.stem  # page_0017
                print(f"{page}:L{i}: {line.strip()[:250]}")
    print(f"\n== {hits} lines matched (ALL terms: {argv}) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
