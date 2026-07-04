"""
File processor for the Electricity RAG.

Strategy per format:
  * PDFs → LLM-vision (gpt-4o) via ``pdf_llm_ocr``. Each page image goes
    through a strict Hebrew-preserving prompt that emits structured Markdown
    (# פרק, ### תקנה N., tables, lists) with page markers. This dramatically
    outperforms text-only PDF extraction on Hebrew legal documents. Falls
    back to PyPDFLoader if vision fails (network/API errors).
  * Everything else (DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPub, images,
    audio, ...) → Microsoft MarkItDown, which produces clean Markdown.

The extraction NEVER translates, summarizes, or paraphrases source content.
Preserving the original Hebrew and every numeric value is essential.

Debug artifacts:
    ``data/processed/<name>.md`` for MarkItDown and vision outputs
    ``data/processed/pages/<name>/page_NNNN.md`` for per-page vision caches

The vector store is (re)built from the returned Documents directly by
``rag_pipeline.rebuild_vectorstore()``. This module returns one Document per
uploaded file whose ``page_content`` is the full Markdown; header-aware
chunking happens later.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain.schema import Document

from backend.config import PROCESSED_DIR
from backend.pdf_llm_ocr import convert_pdf_to_markdown


_markitdown = None


def _get_markitdown():
    """Lazily construct a single MarkItDown instance."""
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown

        _markitdown = MarkItDown(enable_plugins=False)
    return _markitdown


def _pdf_fallback_pypdf(file_path: Path) -> str:
    """Emergency fallback: PyPDFLoader per page → concatenated Markdown with
    ``<!-- page: N -->`` markers so downstream chunking still gets page
    metadata."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()

    parts: list[str] = []
    for i, d in enumerate(raw_docs, start=1):
        parts.append(f"<!-- page: {i} -->")
        parts.append((d.page_content or "").strip())
    return "\n\n".join(parts) + "\n"


async def _load_pdf_via_vision(file_path: Path) -> str:
    """Primary PDF path: LLM-vision → structured Hebrew Markdown."""

    def _progress(done: int, total: int, result) -> None:
        marker = "cached" if result.from_cache else "processed"
        print(f"[file_processor] {file_path.name}: {done}/{total} pages ({marker})")

    return await convert_pdf_to_markdown(file_path, on_progress=_progress)


async def _load_pdf(file_path: Path) -> list[Document]:
    try:
        markdown = await _load_pdf_via_vision(file_path)
        if markdown.strip():
            return [
                Document(
                    page_content=markdown,
                    metadata={"source": file_path.name, "extractor": "gpt-4o-vision"},
                )
            ]
        raise RuntimeError("vision returned empty markdown")
    except Exception as e:  # noqa: BLE001
        print(f"[file_processor] vision extraction failed for {file_path.name}: {e}")
        markdown = _pdf_fallback_pypdf(file_path)
        if not markdown.strip():
            return []
        return [
            Document(
                page_content=markdown,
                metadata={"source": file_path.name, "extractor": "pypdf-fallback"},
            )
        ]


def _load_via_markitdown(file_path: Path) -> list[Document]:
    md = _get_markitdown()
    text = ""
    try:
        result = md.convert(str(file_path))
        text = (result.text_content or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"[file_processor] MarkItDown failed for {file_path.name}: {e}")

    if not text:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:  # noqa: BLE001
            text = ""

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": file_path.name, "extractor": "markitdown"},
        )
    ]


async def load_file_documents_async(file_path: Path) -> list[Document]:
    """Async loader (needed for PDF vision path)."""
    if file_path.suffix.lower() == ".pdf":
        return await _load_pdf(file_path)
    return _load_via_markitdown(file_path)


def load_file_documents(file_path: Path) -> list[Document]:
    """Sync loader (non-PDF only).

    PDFs require the async vision path — call ``load_file_documents_async``
    from an async context instead. Using this sync function for a PDF raises
    a clear error rather than deadlocking on nested event loops.
    """
    if file_path.suffix.lower() == ".pdf":
        raise RuntimeError(
            "PDFs must be loaded via load_file_documents_async (async vision "
            "extraction); calling the sync path would deadlock."
        )
    return _load_via_markitdown(file_path)


def _write_debug_artifact(file_path: Path, docs: list[Document]) -> Path:
    """Write a human-readable ``.md`` artifact of the extraction, per file.
    NOT used for retrieval — the vector store is built from ``docs`` directly.
    """
    output_path = PROCESSED_DIR / (file_path.stem + ".md")
    output_path.write_text(
        "\n\n---\n\n".join(d.page_content for d in docs),
        encoding="utf-8",
    )
    return output_path


async def process_file(file_path: Path) -> Path:
    """Extract text from a file into Documents (no LLM rewriting of content),
    write a debug ``.md`` artifact, and return the artifact path.

    The retrieval index is (re)built from the raw Documents by
    ``rag_pipeline.rebuild_vectorstore()``, which the upload handler calls
    immediately after this function.
    """
    docs = await load_file_documents_async(file_path)
    if not docs:
        raise ValueError(f"No extractable text in {file_path.name}")
    return _write_debug_artifact(file_path, docs)
