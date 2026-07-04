"""End-to-end smoke test that hits the live API on EC2 and prints answers
for a handful of Hebrew and English questions. Run locally:

    python -m scripts.smoke_test
"""

from __future__ import annotations

import json
import sys
import urllib.request

API = "http://54.173.144.0"

QUESTIONS = [
    ("he", "מהו הגובה המינימלי להתקנת קופסת חיבור מהרצפה לפי תקנות החשמל?"),
    ("he", "מהם המרחקים המינימליים בין מוליכים מבודדים במתח נמוך? הצג בטבלה"),
    ("he", "מה קובעות התקנות לגבי הארקת יסוד?"),
    ("he", "מהי רמת התנגדות הבידוד המינימלית בבדיקה תקופתית?"),
    ("en", "What are the requirements for grounding systems?"),
    ("en", "What is the minimum cross-section for a copper conductor in a high voltage installation?"),
]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    for i, (lang, q) in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 80}\n[{i}] ({lang}) {q}\n{'-' * 80}")
        session = post("/api/sessions", {})
        sid = session["id"]
        try:
            resp = post(f"/api/sessions/{sid}/chat", {"message": q})
        except Exception as e:  # noqa: BLE001
            print(f"  !! request failed: {e}")
            continue
        print("ANSWER:")
        print(resp.get("answer", "").strip())
        print("\nSOURCES:")
        for s in resp.get("sources", []):
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
