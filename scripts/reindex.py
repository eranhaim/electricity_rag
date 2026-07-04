"""Re-process all files currently in data/uploads/ and rebuild the FAISS index.

Run this inside the container after changing the file processor or splitter:
    docker exec electricity-rag python -m scripts.reindex
"""

from __future__ import annotations

import asyncio
import sys

from backend.config import UPLOADS_DIR
from backend.file_processor import process_file
from backend.rag_pipeline import rebuild_vectorstore


async def main() -> int:
    files = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
    if not files:
        print("No files in uploads/, nothing to do.")
        return 0

    print(f"Reprocessing {len(files)} file(s):")
    for f in files:
        print(f"  - {f.name}")
        try:
            out = await process_file(f)
            print(f"      -> wrote {out.name}")
        except Exception as e:  # noqa: BLE001
            print(f"      !! failed: {e}")

    chunks = await rebuild_vectorstore()
    print(f"\nRebuilt vector store: {chunks} chunks indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
