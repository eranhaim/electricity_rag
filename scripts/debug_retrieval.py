"""Print exactly what the retriever pulls back for a given question, so we
can debug why the LLM says "not in the knowledge base"."""

from __future__ import annotations

import asyncio
import sys

from backend.rag_pipeline import get_vectorstore, _retrieve


QUESTIONS = [
    "מהו הגובה המינימלי להתקנת קופסת חיבור מהרצפה?",
    "מה קובעות התקנות לגבי הארקת יסוד?",
    "מהי רמת התנגדות הבידוד המינימלית בבדיקה תקופתית?",
]


async def main() -> int:
    vs = get_vectorstore()
    if vs is None:
        print("No vector store loaded.")
        return 1

    for q in QUESTIONS:
        print(f"\n{'=' * 80}\nQ: {q}\n{'-' * 80}")

        hits = vs.similarity_search_with_score(q, k=8)
        for i, (doc, score) in enumerate(hits, start=1):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "")
            snippet = (doc.page_content or "").replace("\n", " ")[:200]
            print(
                f"  [{i}] score={score:.3f} p.{page} sec='{section}'\n"
                f"      {snippet}"
            )

        mmr_docs = _retrieve(vs, q, k=6, fetch_k=20)
        print(f"\n  MMR selected pages: {[d.metadata.get('page') for d in mmr_docs]}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
