"""
RAG pipeline for Hebrew Israeli electricity law.

Design notes:
  * The vector store is built from raw Documents produced by
    ``file_processor.load_file_documents`` — Hebrew is preserved as-is.
  * Chunking: RecursiveCharacterTextSplitter, chunk_size=800, overlap=150.
    Dense Hebrew legal text benefits from smaller, more focused chunks so a
    single regulation lands inside one or two chunks (previous 1500/300 was
    diluting relevance).
  * Retrieval: MMR (Maximal Marginal Relevance) to fetch diverse chunks and
    avoid returning six near-duplicates of the same paragraph.
  * Cross-lingual retrieval: when the question is in English, we ALSO run
    retrieval with a Hebrew translation of the question, because the corpus
    is Hebrew. Results are merged and deduplicated. This gives better recall
    without requiring a full agent loop.
  * Answer generation: LCEL-style flow (retrieve -> format context ->
    prompt -> LLM). The system prompt enforces language matching, markdown
    formatting, and citation of regulation numbers + page numbers from
    document metadata.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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

SYSTEM_PROMPT = """You are a senior expert assistant specializing in Israeli electricity regulations, standards, and electrical safety codes ("תקנות החשמל"). You answer questions based ONLY on the provided context documents from the knowledge base.

LANGUAGE RULE (HIGHEST PRIORITY — MUST FOLLOW):
- Detect the language of the user's CURRENT question.
- If the question is in Hebrew → your ENTIRE answer MUST be in Hebrew.
- If the question is in English → your ENTIRE answer MUST be in English (translate content from Hebrew context into English).
- NEVER mix languages in a single answer.

ACCURACY & CITATION RULES:
- Use ONLY information explicitly present in the provided context. If the answer is not in the context, say so clearly ("המידע לא נמצא במאגר הידע" / "This information is not in the knowledge base"). NEVER invent regulation numbers, values, or facts.
- ALWAYS cite the specific regulation, section, article, or clause number when it appears in the context (e.g., "לפי תקנה 17", "according to regulation 17").
- ALWAYS cite the page number of the source document when relevant, in the format "(עמ' N)" for Hebrew or "(p. N)" for English. Take page numbers from the [page: N] markers in the context.
- If a fact spans multiple regulations or pages, cite each one.
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
2. Detailed explanation with citations to regulation numbers and page numbers.
3. Table or list of specific values/requirements when applicable.
4. Notes, exceptions, or related regulations.
"""


# ── Embeddings & store ───────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore(force_reload: bool = False) -> FAISS | None:
    global _vectorstore
    if force_reload:
        _vectorstore = None
    if _vectorstore is not None:
        return _vectorstore
    index_path = VECTORSTORE_DIR / "index.faiss"
    if index_path.exists():
        _vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def rebuild_vectorstore() -> int:
    """Rebuild the FAISS index from every file currently in ``data/uploads``.

    Loads raw Documents (with source + page metadata) directly via
    ``load_file_documents`` — the ``data/processed/*.txt`` artifacts are only
    for human debugging and are NOT used for retrieval. This is the fix that
    prevents the previous "LLM-optimized" corrupted text from ever entering
    the index again.

    Returns the number of chunks indexed.
    """
    global _vectorstore

    upload_files = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]

    if not upload_files:
        _vectorstore = None
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
        for stale in VECTORSTORE_DIR.glob("index.*"):
            stale.unlink(missing_ok=True)
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    _vectorstore = FAISS.from_documents(chunks, get_embeddings())
    _vectorstore.save_local(str(VECTORSTORE_DIR))
    return len(chunks)


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


def _retrieve(vs: FAISS, query: str, k: int = 6, fetch_k: int = 20) -> list[Document]:
    """MMR retrieval — balances relevance and diversity so the LLM sees
    multiple distinct regulations rather than near-duplicate passages."""
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )
    return retriever.invoke(query)


def _dedupe(docs: list[Document]) -> list[Document]:
    """Deduplicate by (source, page, first 80 chars) to merge results from
    multiple retrieval passes without losing distinct chunks."""
    seen: set[tuple] = set()
    unique: list[Document] = []
    for d in docs:
        key = (
            d.metadata.get("source", ""),
            d.metadata.get("page", d.metadata.get("chunk_id", "")),
            (d.page_content or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


# ── Prompt construction ──────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved chunks into a numbered, cite-friendly context block."""
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", d.metadata.get("sheet", "?"))
        lines.append(f"[chunk {i} | source: {src} | page: {page}]\n{d.page_content}")
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
    docs = _retrieve(vs, question, k=6, fetch_k=20)

    if lang == "en":
        hebrew_query = await _translate_to_hebrew(question)
        if hebrew_query and hebrew_query.strip() and hebrew_query != question:
            docs = _dedupe(docs + _retrieve(vs, hebrew_query, k=4, fetch_k=16))

    context = _format_context(docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    messages = _build_messages(question, context, session_messages)
    response = await llm.ainvoke(messages)

    sources: list[str] = []
    seen_pairs: set[tuple] = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        sources.append(f"{src} (p.{page})" if page is not None else src)

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
            "page": doc.metadata.get("page"),
            "snippet": (doc.page_content or "")[:400],
        }
        for doc, score in hits
    ]
