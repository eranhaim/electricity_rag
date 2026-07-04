"""
RAG pipeline for Hebrew Israeli electricity law.

Design notes:
  * Files are converted to Markdown by ``file_processor`` (MarkItDown) which
    preserves the original Hebrew content plus structural cues: headings,
    tables, lists.
  * Chunking is two-stage:
      1. ``MarkdownHeaderTextSplitter`` splits by Markdown header hierarchy
         (# / ## / ###). Each resulting chunk carries the header path as
         metadata (e.g. {"h1": "פרק ב", "h2": "התקנת מוליכים",
         "h3": "תקנה 17"}). This lets the LLM cite the actual regulation
         name/number instead of guessing a page.
      2. ``RecursiveCharacterTextSplitter`` further splits any header
         section that is still too large for a single retrieval unit.
    This is markdown-aware chunking, which massively improves recall on
    structured legal text vs. blind character splitting.
  * Retrieval: MMR (Maximal Marginal Relevance) to fetch diverse chunks and
    avoid returning six near-duplicates of the same paragraph.
  * Cross-lingual retrieval: when the question is in English, we ALSO run
    retrieval with a Hebrew translation of the question, because the corpus
    is Hebrew. Results are merged and deduplicated.
  * Answer generation: LCEL-style flow (retrieve -> format context ->
    prompt -> LLM). The system prompt enforces language matching, Markdown
    formatting, and citation of regulation numbers taken from chunk headers.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import AIMessage, Document, HumanMessage, SystemMessage

from backend.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    UPLOADS_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from backend.file_processor import load_file_documents

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None
# Keep the raw chunks in memory alongside the FAISS index so we can build a
# BM25 keyword retriever without re-parsing the PDF on every query. Hebrew
# legal questions frequently rely on exact terms (e.g. "תיבה" vs "קופסה")
# that vector similarity alone under-recalls.
_chunks: list[Document] = []
_bm25: BM25Retriever | None = None

SYSTEM_PROMPT = """You are a senior expert assistant specializing in Israeli electricity regulations, standards, and electrical safety codes ("תקנות החשמל"). You answer questions based ONLY on the provided context documents from the knowledge base.

LANGUAGE RULE (HIGHEST PRIORITY — MUST FOLLOW):
- Detect the language of the user's CURRENT question.
- If the question is in Hebrew → your ENTIRE answer MUST be in Hebrew.
- If the question is in English → your ENTIRE answer MUST be in English (translate content from Hebrew context into English).
- NEVER mix languages in a single answer.

ACCURACY & CITATION RULES:
- Use ONLY information explicitly present in the provided context. NEVER invent regulation numbers, values, or facts.
- Read the ENTIRE context carefully before concluding "not found". The context may be fragmented (e.g. RTL Hebrew PDF extraction can produce jumbled word order); if any chunk contains information that plausibly answers the question, USE IT and cite that chunk. Do not require an exact keyword match.
- ONLY refuse ("המידע לא נמצא במאגר הידע" / "This information is not in the knowledge base") when the context truly contains no relevant material. When you refuse, briefly say which related topics DID appear in the context so the user can rephrase.
- ALWAYS cite the specific regulation / section / clause when it appears in the context (e.g., "לפי תקנה 17", "according to regulation 17"). Regulation numbers usually appear at the start of a paragraph in the Hebrew text.
- Take citations from any [section:] header shown at the top of the chunk, and from the source filename shown in [source:] plus the [page:] number when present.
- If a fact spans multiple regulations, cite each one.
- Preserve exact numeric values, units, and thresholds from the source. Do not round or paraphrase numbers.

FORMATTING RULES — ALWAYS use rich Markdown:
- Use **bold** for key terms, regulation names, and threshold values.
- Use bullet or numbered lists for multiple items, steps, or requirements.
- Use ## and ### headers to organize longer answers into scannable sections.
- When presenting comparative data, distances, measurements, intervals, thresholds, resistances, currents, or any structured data — ALWAYS present it as a Markdown table with clear column headers.
- Use > blockquotes for direct quotations from regulations.
- If the user asks for a table, you MUST answer with a table.
- Keep answers thorough but scannable — avoid long unbroken paragraphs.

ANSWER STRUCTURE for complex questions:
1. Short direct answer / summary (2-3 lines).
2. Detailed explanation with citations to regulation numbers / sections.
3. Table or list of specific values/requirements when applicable.
4. Notes, exceptions, or related regulations.
"""


# ── Embeddings & store ───────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            chunk_size=100,
        )
    return _embeddings


def get_vectorstore(force_reload: bool = False) -> FAISS | None:
    global _vectorstore, _chunks, _bm25
    if force_reload:
        _vectorstore = None
        _chunks = []
        _bm25 = None
    if _vectorstore is not None:
        return _vectorstore
    index_path = VECTORSTORE_DIR / "index.faiss"
    if index_path.exists():
        _vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        # Rehydrate the in-memory chunk list from the FAISS docstore so BM25
        # can index the same corpus on next query without a full reprocess.
        try:
            _chunks = list(_vectorstore.docstore._dict.values())  # type: ignore[attr-defined]
            _bm25 = _build_bm25(_chunks) if _chunks else None
        except Exception:  # noqa: BLE001
            _chunks = []
            _bm25 = None
    return _vectorstore


def rebuild_vectorstore() -> int:
    """Rebuild the FAISS index (and in-memory BM25 corpus) from every file
    currently in ``data/uploads``.

    Loads raw Documents (with source + page metadata) directly via
    ``load_file_documents`` — the ``data/processed/*.md``/``.txt`` artifacts
    are only for human debugging and are NOT used for retrieval.

    Returns the number of chunks indexed.
    """
    global _vectorstore, _chunks, _bm25

    upload_files = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]

    if not upload_files:
        _vectorstore = None
        _chunks = []
        _bm25 = None
        for stale in VECTORSTORE_DIR.glob("index.*"):
            stale.unlink(missing_ok=True)
        return 0

    documents: list[Document] = []
    for f in upload_files:
        try:
            documents.extend(load_file_documents(f))
        except Exception as e:  # noqa: BLE001
            print(f"[rag_pipeline] failed to load {f.name}: {e}")

    if not documents:
        _vectorstore = None
        _chunks = []
        _bm25 = None
        for stale in VECTORSTORE_DIR.glob("index.*"):
            stale.unlink(missing_ok=True)
        return 0

    chunks = _split_markdown_documents(documents)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    _vectorstore = FAISS.from_documents(chunks, get_embeddings())
    _vectorstore.save_local(str(VECTORSTORE_DIR))

    _chunks = chunks
    _bm25 = _build_bm25(chunks)
    return len(chunks)


def _build_bm25(chunks: list[Document]) -> BM25Retriever:
    """Build a BM25 keyword retriever over the same chunks that live in the
    FAISS index. BM25Retriever accepts an optional tokenizer; the default
    whitespace tokenizer works fine for Hebrew because Hebrew words are
    space-separated. k is set high because it's cheap and we merge with
    vector results via EnsembleRetriever."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 10
    return retriever


# ── Markdown-aware chunking ──────────────────────────────────────────────────

# Header hierarchy we care about. Matches MarkItDown output which uses
# ATX headings (# ## ###). Values become metadata keys on each chunk.
_HEADER_LEVELS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _split_markdown_documents(documents: list[Document]) -> list[Document]:
    """Two-stage split:
        1. Split by Markdown headers → each chunk carries its header path as
           metadata (h1/h2/h3/h4), giving the LLM a real "section" to cite.
        2. Any header section that is still larger than ``CHUNK_SIZE`` is
           further split by ``RecursiveCharacterTextSplitter`` while keeping
           the parent header metadata intact.
    Documents that contain no Markdown headers (e.g. PDF pages extracted by
    PyPDFLoader) fall through to plain character splitting while preserving
    the original metadata (including ``page`` for PDFs).
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_LEVELS,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        parent_meta = dict(doc.metadata)

        header_chunks = header_splitter.split_text(doc.page_content)

        if not header_chunks:
            for c in char_splitter.split_documents([doc]):
                merged = {**parent_meta, **c.metadata}
                merged.setdefault("section", "")
                c.metadata = merged
                all_chunks.append(c)
            continue

        for hc in header_chunks:
            section_parts = [
                hc.metadata[k] for k in ("h1", "h2", "h3", "h4") if hc.metadata.get(k)
            ]
            section = " > ".join(section_parts)

            base_meta = {**parent_meta, "section": section, **hc.metadata}

            if len(hc.page_content) <= CHUNK_SIZE:
                all_chunks.append(
                    Document(page_content=hc.page_content, metadata=base_meta)
                )
                continue

            sub_docs = char_splitter.split_documents(
                [Document(page_content=hc.page_content, metadata=base_meta)]
            )
            for sd in sub_docs:
                merged = {**base_meta, **sd.metadata}
                sd.metadata = merged
                all_chunks.append(sd)

    return all_chunks


# ── Retrieval helpers ────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Return 'he' if the text contains any Hebrew codepoint, else 'en'."""
    for ch in text:
        if "\u0590" <= ch <= "\u05FF":
            return "he"
    return "en"


async def _translate_to_hebrew(question: str) -> str:
    """Translate an English question into short, keyword-rich Hebrew to boost
    retrieval recall against a Hebrew corpus. Best-effort — falls back to the
    original text on failure."""
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
        resp = await llm.ainvoke([
            SystemMessage(
                content=(
                    "You translate short user questions from English into "
                    "concise Hebrew suitable for keyword search in Israeli "
                    "electricity regulations. Reply with the Hebrew "
                    "translation ONLY, no quotes, no explanation."
                )
            ),
            HumanMessage(content=question),
        ])
        translated = (resp.content or "").strip()
        return translated or question
    except Exception:  # noqa: BLE001
        return question


def _retrieve(vs: FAISS, query: str, k: int = 8, fetch_k: int = 24) -> list[Document]:
    """Hybrid retrieval: BM25 (exact keyword match) + FAISS/MMR (semantic).

    BM25 is critical for Hebrew legal text because the corpus uses very
    specific vocabulary (e.g. "תיבה" for junction box, "מבדד" for
    insulator). Vector similarity alone often misses these when the user
    phrases the question with a different synonym.

    We combine the two with LangChain's ``EnsembleRetriever`` at 50/50
    weights (Reciprocal Rank Fusion under the hood).
    """
    vector_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )

    if _bm25 is None:
        return vector_retriever.invoke(query)

    # Tune BM25's own k for this query so both retrievers contribute a
    # similar-sized candidate pool before fusion.
    _bm25.k = k
    ensemble = EnsembleRetriever(
        retrievers=[_bm25, vector_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble.invoke(query)


def _dedupe(docs: list[Document]) -> list[Document]:
    """Deduplicate by (source, section, first 80 chars) so multiple retrieval
    passes don't return the same chunk twice."""
    seen: set[tuple] = set()
    unique: list[Document] = []
    for d in docs:
        key = (
            d.metadata.get("source", ""),
            d.metadata.get("section", ""),
            (d.page_content or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


# ── Prompt construction ──────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved chunks into a numbered, cite-friendly context block.

    Each chunk header exposes:
      - source: original filename (so the LLM can cite it)
      - section: the Markdown header path (e.g. "פרק ב > תקנה 17") when
        available — this enables regulation-number citations.
      - page: 1-based page number for PDF chunks.
    """
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        section = d.metadata.get("section") or ""
        page = d.metadata.get("page")

        header_parts = [f"chunk {i}", f"source: {src}"]
        if section:
            header_parts.append(f"section: {section}")
        if page is not None:
            header_parts.append(f"page: {page}")

        lines.append("[" + " | ".join(header_parts) + "]\n" + d.page_content)
    return "\n\n---\n\n".join(lines)


def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict] | None,
) -> list:
    lang = _detect_language(question)
    lang_reminder = (
        "IMPORTANT: The user's question is in English. Answer ENTIRELY in "
        "English. Translate any Hebrew content from the context into English."
        if lang == "en"
        else "חשוב: שאלת המשתמש בעברית — יש לענות באופן מלא בעברית."
    )

    system_content = (
        SYSTEM_PROMPT
        + "\n\nContext from the knowledge base (each chunk is annotated with "
        "its source file and page number for citation):\n\n"
        + context
    )

    messages: list = [SystemMessage(content=system_content)]

    if chat_history:
        for msg in chat_history[-8:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=f"{lang_reminder}\n\nQuestion: {question}"))
    return messages


# ── Public API ───────────────────────────────────────────────────────────────

async def ask(question: str, session_messages: list[dict] | None = None) -> dict:
    """Answer a question grounded in the vector store.

    Returns ``{"answer": str, "sources": list[str]}`` — the same shape the
    frontend already consumes.
    """
    vs = get_vectorstore()
    if vs is None:
        return {
            "answer": (
                "מאגר הידע עדיין לא נטען. יש לפנות למנהל המערכת להעלאת "
                "מסמכים.\n\nNo knowledge base loaded. Please ask an admin to "
                "upload documents first."
            ),
            "sources": [],
        }

    lang = _detect_language(question)
    docs = _retrieve(vs, question, k=8, fetch_k=24)

    if lang == "en":
        hebrew_query = await _translate_to_hebrew(question)
        if hebrew_query and hebrew_query.strip() and hebrew_query != question:
            docs = _dedupe(docs + _retrieve(vs, hebrew_query, k=6, fetch_k=20))

    context = _format_context(docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    messages = _build_messages(question, context, session_messages)
    response = await llm.ainvoke(messages)

    sources: list[str] = []
    seen_pairs: set[tuple] = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        section = d.metadata.get("section") or ""
        page = d.metadata.get("page")
        key = (src, section, page)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        label = src
        if section:
            label += f" — {section}"
        elif page is not None:
            label += f" (p.{page})"
        sources.append(label)

    return {"answer": response.content, "sources": sources}


async def debug_retrieve(question: str, k: int = 8) -> list[dict]:
    """Helper for offline debugging — returns raw scored hits so we can see
    what the retriever is actually pulling back for a given question."""
    vs = get_vectorstore()
    if vs is None:
        return []
    hits = vs.similarity_search_with_score(question, k=k)
    return [
        {
            "score": float(score),
            "source": doc.metadata.get("source"),
            "section": doc.metadata.get("section"),
            "snippet": (doc.page_content or "")[:400],
        }
        for doc, score in hits
    ]
