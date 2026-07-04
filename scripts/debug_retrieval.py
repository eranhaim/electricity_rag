"""Show exactly what the current hybrid retriever pulls back for a set of
questions. Prints both pure-vector scores and the fused (BM25 + FAISS/MMR)
ranking. Run inside the container:

    docker exec electricity-rag python -m scripts.debug_retrieval
"""

from __future__ import annotations

import sys

from backend.rag_pipeline import get_vectorstore, _retrieve, _bm25


QUESTIONS = [
    "מה הגובה המותר של פס השוואות פוטנציאלים מהרצפה?",
    "מהו הגובה המינימלי להתקנת קופסת חיבור?",
    "מה קובעות התקנות לגבי הארקת יסוד?",
    "מתי נדרש מפסק מגן?",
    "מה הדרישות לגבי לוח חשמל פלסטי?",
]


def main() -> int:
    vs = get_vectorstore()
    if vs is None:
        print("No vector store loaded.")
        return 1

    for q in QUESTIONS:
        print(f"\n{'=' * 80}\nQ: {q}\n{'-' * 80}")

        print("── FAISS similarity (top-8) ──")
        hits = vs.similarity_search_with_score(q, k=8)
        for i, (doc, score) in enumerate(hits, start=1):
            page = doc.metadata.get("page", "?")
            section = (doc.metadata.get("section") or "")[:70]
            snippet = (doc.page_content or "").replace("\n", " ")[:150]
            print(f"  {i:>2}. score={score:.3f} p.{page} sec='{section}'")
            print(f"      {snippet}")

        print("\n── Hybrid retriever (BM25 + FAISS/MMR ensemble, k=8) ──")
        docs = _retrieve(vs, q, k=8, fetch_k=24)
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "?")
            section = (doc.metadata.get("section") or "")[:70]
            snippet = (doc.page_content or "").replace("\n", " ")[:150]
            print(f"  {i:>2}. p.{page} sec='{section}'")
            print(f"      {snippet}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
