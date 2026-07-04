"""Scan cached page markdown files, detect pathological hallucinations, and
delete them so a subsequent reindex re-fetches them from the vision model.

Also runs the current cleanup pass on every remaining page (idempotent).

Usage inside container:
    docker exec electricity-rag python -m scripts.heal_pages           # scan only
    docker exec electricity-rag python -m scripts.heal_pages --delete  # actually delete bad pages
"""

from __future__ import annotations

import sys

from backend.config import PROCESSED_DIR
from backend.pdf_llm_ocr import _clean_model_output, _looks_pathological


def main(argv: list[str]) -> int:
    delete = "--delete" in argv
    root = PROCESSED_DIR / "pages"
    if not root.exists():
        print("No cache dir.")
        return 0

    total = 0
    cleaned = 0
    bad: list[str] = []
    for md_file in sorted(root.rglob("*.md")):
        total += 1
        original = md_file.read_text(encoding="utf-8")
        after = _clean_model_output(original)
        if after != original:
            md_file.write_text(after, encoding="utf-8")
            cleaned += 1
        if _looks_pathological(after):
            bad.append(str(md_file.relative_to(root)))
            if delete:
                md_file.unlink(missing_ok=True)

    print(f"Cleaned {cleaned}/{total} pages.")
    print(f"Pathological pages: {len(bad)}")
    for b in bad:
        print(f"  {b}")
    if bad and not delete:
        print("\n(rerun with --delete to remove them so reindex re-fetches from vision)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
