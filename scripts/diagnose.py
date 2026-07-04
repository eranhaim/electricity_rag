"""Diagnose whether the processed PDF text contains actual Hebrew characters
or garbage. Print counts and a small sample.
"""

from __future__ import annotations

import sys

from backend.config import PROCESSED_DIR


def _count_ranges(text: str) -> dict[str, int]:
    counts = {
        "total": len(text),
        "hebrew": 0,
        "ascii_letter": 0,
        "digit": 0,
        "latin1_supp": 0,
        "cid_markers": text.count("(cid:"),
    }
    for ch in text:
        cp = ord(ch)
        if 0x0590 <= cp <= 0x05FF:
            counts["hebrew"] += 1
        elif 0x0041 <= cp <= 0x007A:
            counts["ascii_letter"] += 1
        elif 0x0030 <= cp <= 0x0039:
            counts["digit"] += 1
        elif 0x0080 <= cp <= 0x00FF:
            counts["latin1_supp"] += 1
    return counts


def main() -> int:
    files = sorted(PROCESSED_DIR.glob("*.txt")) + sorted(PROCESSED_DIR.glob("*.md"))
    if not files:
        print("No processed files.")
        return 1
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        counts = _count_ranges(text)
        print(f"\n=== {f.name} ===")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        hebrew_ratio = counts["hebrew"] / max(counts["total"], 1)
        print(f"  hebrew ratio: {hebrew_ratio:.1%}")
        idx = text.find("תקנה")
        if idx == -1:
            idx = 0
        print(f"  sample @ {idx}:\n    {text[idx: idx + 300]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
