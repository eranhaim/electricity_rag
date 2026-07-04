"""End-to-end smoke test hitting the live API. Uses questions from the
partner's POC roadmap plus additional common ones.

Outputs a JSONL file with question, answer, and sources for each so we can
review offline. Run locally:

    python -m scripts.smoke_test
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://54.173.144.0"

# Questions from the partner's roadmap (RAG_Electricians_POC_Workplan_HE.pdf)
# plus additional common ones we expect to be in חוק החשמל.
QUESTIONS = [
    ("he", "מה הגובה המותר להתקנת לוח חשמל בדירה?"),
    ("he", "מה הדרישות לגבי הארקת יסוד?"),
    ("he", "מתי נדרש מפסק מגן?"),
    ("he", "מה הדרישות לגבי לוח חשמל פלסטי?"),
    ("he", "מה אומר החוק לגבי התנגדות הארקה?"),
    ("he", "מתי יש להשתמש במוליך הארקה בחתך מסוים?"),
    ("he", "מה הדרישות לגבי התקנת שקע בחדר רחצה?"),
    ("he", "מהי הרגישות הנדרשת של מפסק מגן במעגל סופי?"),
    ("he", "מהו העומק המינימלי להנחת כבל באדמה?"),
    ("he", "מהם צבעי הבידוד הנדרשים למוליך אפס והארקה?"),
    ("he", "מהו חתך המוליך המינימלי במעגל סופי?"),
    ("he", "מהי רמת התנגדות הבידוד המינימלית בבדיקה תקופתית?"),
    ("he", "מה קובעות התקנות לגבי הארקת מובל מתכתי בחדר רחצה?"),
    ("he", "מהו המרחק המזערי בין קווי חשמל למבנים?"),
    ("en", "What are the requirements for foundation grounding?"),
    ("en", "When is a residual current device required?"),
]


def post(path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(__file__).parent.parent / f"smoke_results_{ts}.jsonl"
    print(f"Writing results to: {out_path}\n")

    with out_path.open("w", encoding="utf-8") as f:
        for i, (lang, q) in enumerate(QUESTIONS, start=1):
            print(f"[{i}/{len(QUESTIONS)}] ({lang}) {q}")
            session = post("/api/sessions", {})
            sid = session["id"]
            try:
                resp = post(f"/api/sessions/{sid}/chat", {"message": q})
            except Exception as e:  # noqa: BLE001
                resp = {"answer": f"[ERROR: {e}]", "sources": []}

            answer = resp.get("answer", "").strip()
            refused = (
                "המידע לא נמצא במאגר הידע" in answer
                or "not in the knowledge base" in answer.lower()
            )
            record = {
                "lang": lang,
                "question": q,
                "answer": answer,
                "sources": resp.get("sources", []),
                "refused": refused,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            preview = answer.replace("\n", " ")[:180]
            marker = "REFUSED" if refused else "OK"
            print(f"       [{marker}] {preview}")
            print(f"       sources: {len(resp.get('sources', []))}\n")

    print(f"\nDone. Results in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
