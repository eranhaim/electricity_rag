"""One-shot cleanup of cached page markdown files.

Apply ``pdf_llm_ocr._clean_model_output`` to every cached per-page ``.md`` so
we strip stray ```markdown fences and collapse runaway repetitions without
re-invoking the LLM.

Usage inside the container:
    python -m scripts.clean_pages
"""

from __future__ import annotations

import sys

from backend.config import PROCESSED_DIR
from backend.pdf_llm_ocr import _clean_model_output


def main() -> int:
    root = PROCESSED_DIR / "pages"
    if not root.exists():
        print(f"No cache dir at {root}, nothing to do.")
        return 0

    total = 0
    changed = 0
    for md_file in root.rglob("*.md"):
        total += 1
        original = md_file.read_text(encoding="utf-8")
        cleaned = _clean_model_output(original)
        if cleaned != original:
            md_file.write_text(cleaned, encoding="utf-8")
            changed += 1
    print(f"Cleaned {changed}/{total} page files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
