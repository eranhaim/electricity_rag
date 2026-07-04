"""
File processor for the Electricity RAG.

Hybrid strategy:
  * PDFs → PyPDFLoader (per-page). PyPDFLoader (via pypdf) handles Hebrew
    text correctly. MarkItDown/pdfminer returns garbled CID codes for many
    Hebrew PDFs with embedded fonts, so it's NOT used here.
  * Everything else (DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPub, images,
    audio, ...) → MarkItDown. Markdown preserves headings, tables, and
    lists — the structural cues that ``MarkdownHeaderTextSplitter`` uses
    downstream for regulation-level chunking.

No LLM is applied to the extracted text — preserving the original Hebrew
content is essential (the earlier "LLM optimization" step corrupted values).

A debug artifact (``.md`` for MarkItDown formats, ``.txt`` for PDFs) is written
to ``data/processed/`` so a human can inspect exactly what the retriever
sees. The vector store is (re)built from the returned Documents directly.
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


def _load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF into one Document per page using PyPDFLoader.

    Kept as its own path because MarkItDown/pdfminer returns garbage
    ``(cid:NNN)`` codes for many Hebrew PDFs; pypdf's extraction preserves
    the original Hebrew characters.
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()

    docs: list[Document] = []
    for i, d in enumerate(raw_docs):
        text = (d.page_content or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "page": i + 1,
                    "total_pages": len(raw_docs),
                },
            )
        )
    return docs


def _load_via_markitdown(file_path: Path) -> list[Document]:
    """Convert a file to Markdown via MarkItDown, returned as a single
    Document. Header-aware chunking (in ``rag_pipeline``) will split it into
    section-tagged chunks."""
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
        Document(page_content=text, metadata={"source": file_path.name})
    ]


def load_file_documents(file_path: Path) -> list[Document]:
    """Dispatch to the right loader based on file extension.

    Returns Documents with faithfully preserved content and ``source``
    metadata (plus ``page`` for PDFs). NEVER translates or paraphrases.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)

    return _load_via_markitdown(file_path)


def _write_debug_artifact(file_path: Path, docs: list[Document]) -> Path:
    """Write a human-readable artifact for transparency/debugging.

    * ``.md`` for MarkItDown-processed files (Markdown output)
    * ``.txt`` for PDFs, with per-page separators (plain text output)

    This artifact is NOT used for retrieval — the vector store is built
    from the Document objects directly.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        output_path = PROCESSED_DIR / (file_path.stem + ".txt")
        parts: list[str] = []
        for d in docs:
            page = d.metadata.get("page", "?")
            parts.append(f"--- {file_path.name} | page {page} ---\n{d.page_content}")
        output_path.write_text("\n\n".join(parts), encoding="utf-8")
    else:
        output_path = PROCESSED_DIR / (file_path.stem + ".md")
        output_path.write_text(
            "\n\n---\n\n".join(d.page_content for d in docs),
            encoding="utf-8",
        )
    return output_path


async def process_file(file_path: Path) -> Path:
    """Extract text from a file into Documents (no LLM rewriting), write a
    debug artifact, and return the artifact path.

    The retrieval index is (re)built from the raw Documents by
    ``rag_pipeline.rebuild_vectorstore()``, which the upload handler calls
    immediately after this function.
    """
    docs = load_file_documents(file_path)
    if not docs:
        raise ValueError(f"No extractable text in {file_path.name}")
    return _write_debug_artifact(file_path, docs)
