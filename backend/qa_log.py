"""Append-only Q&A audit log.

Every user question, the chunks that were retrieved for it, and the final
answer are appended as a single JSONL record. Used by the admin panel to
review answer quality — a POC roadmap requirement (section 15: "prepare a
log of question, retrieved sources, and answer").

Format: one JSON object per line, encoded UTF-8. Never mutated in place.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langchain.schema import Document

from backend.config import DATA_DIR

LOG_FILE = DATA_DIR / "qa_log.jsonl"


def _summarize_chunk(doc: Document) -> dict:
    meta = doc.metadata or {}
    content = doc.page_content or ""
    return {
        "source": meta.get("source"),
        "section": meta.get("section") or None,
        "page": meta.get("page"),
        "chunk_id": meta.get("chunk_id"),
        "snippet": content[:400],
        "full_length": len(content),
    }


def log_qa(
    *,
    session_id: str | None,
    question: str,
    answer: str,
    retrieved: Iterable[Document],
    sources: list[str],
    refused: bool,
    extra: dict | None = None,
) -> None:
    """Append a Q&A entry to the audit log. Never raises — logging failures
    must not break the chat flow.
    """
    try:
        record = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "refused": refused,
            "sources_cited": sources,
            "retrieved_chunks": [_summarize_chunk(d) for d in retrieved],
        }
        if extra:
            record["extra"] = extra

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[qa_log] failed to log entry: {e}")


def read_recent(limit: int = 50) -> list[dict]:
    """Return the most recent ``limit`` log entries, newest first."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:]
    tail.reverse()
    out: list[dict] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def count_entries() -> int:
    if not LOG_FILE.exists():
        return 0
    return sum(1 for _ in LOG_FILE.open("r", encoding="utf-8"))
