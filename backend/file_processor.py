"""
File processor for the Electricity RAG.

Uses Microsoft's MarkItDown library to convert any supported file
(PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPub, ZIP, images, audio, ...)
into structured Markdown. Markdown preserves:
    - Headings (chapters, sections, regulation numbers)
    - Tables (regulation values, distances, thresholds)
    - Lists (requirements, exceptions)
    - Text order and Hebrew content

The resulting Markdown is what feeds the vector index — NO LLM rewriting is
applied. Preserving the original Hebrew content is essential.

Each processed file yields a single LangChain ``Document`` whose ``page_content``
is the full Markdown of the source file, and whose metadata includes
``source`` (original filename). Downstream chunking (in ``rag_pipeline``) uses
``MarkdownHeaderTextSplitter`` to attach section/regulation headers as
per-chunk metadata for accurate citations.

A ``.md`` artifact is also written to ``data/processed/`` for human debugging.
"""

from __future__ import annotations

from pathlib import Path

from langchain.schema import Document

from backend.config import PROCESSED_DIR


_markitdown = None


def _get_markitdown():
    """Lazily construct a single MarkItDown instance (cheap but reusable)."""
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown

        _markitdown = MarkItDown(enable_plugins=False)
    return _markitdown


def _convert_to_markdown(file_path: Path) -> str:
    """Convert any supported file to Markdown via MarkItDown.

    Falls back to a plain-text read for extensions MarkItDown cannot handle
    (e.g. .txt), so we never lose content.
    """
    md = _get_markitdown()
    try:
        result = md.convert(str(file_path))
        text = (result.text_content or "").strip()
        if text:
            return text
    except Exception as e:  # noqa: BLE001
        print(f"[file_processor] MarkItDown failed for {file_path.name}: {e}")

    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return ""


def load_file_documents(file_path: Path) -> list[Document]:
    """Convert a file to Markdown and return it as a single Document.

    Chunking with markdown-header awareness happens in ``rag_pipeline``, so
    here we only produce ONE Document per file that carries the entire
    Markdown payload plus ``source`` metadata.
    """
    markdown = _convert_to_markdown(file_path)
    if not markdown:
        return []

    return [
        Document(
            page_content=markdown,
            metadata={"source": file_path.name},
        )
    ]


def _write_debug_markdown(file_path: Path, docs: list[Document]) -> Path:
    """Write the converted Markdown to ``data/processed/<name>.md`` so a human
    can inspect exactly what the retriever will see. This file is NOT used
    for retrieval — the vector store is built from ``docs`` directly."""
    output_path = PROCESSED_DIR / (file_path.stem + ".md")
    output_path.write_text(
        "\n\n---\n\n".join(d.page_content for d in docs),
        encoding="utf-8",
    )
    return output_path


async def process_file(file_path: Path) -> Path:
    """Convert an uploaded file to Markdown Documents, write a debug ``.md``
    artifact, and return the artifact path.

    The retrieval index is (re)built from the raw Documents by
    ``rag_pipeline.rebuild_vectorstore()``, which the upload handler calls
    immediately after this function.
    """
    docs = load_file_documents(file_path)
    if not docs:
        raise ValueError(f"No extractable text in {file_path.name}")
    return _write_debug_markdown(file_path, docs)
