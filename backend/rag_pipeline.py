"""
RAG pipeline for Hebrew Israeli electricity law.

Design notes:
  * Files → Markdown by ``file_processor`` (LLM-vision for PDFs, MarkItDown
    for other formats), preserving Hebrew content + headings + tables.
  * Chunking is markdown-aware:
      0. Per-page split via ``<!-- page: N -->`` markers, so each chunk
         carries a ``page`` metadata field for citation.
      1. ``MarkdownHeaderTextSplitter`` splits by ``# / ## / ### / ####``.
         Each chunk carries the header path as ``section`` metadata (e.g.
         "פרק ב' > תקנה 17"), so the LLM can cite the actual regulation.
      2. ``RecursiveCharacterTextSplitter`` further splits any header
         section that is still too large.
  * Retrieval — several layers, applied in order:
      a. **Query expansion**: an LLM produces 2-3 alternate Hebrew
         phrasings of the user's question (synonyms, related terms). This
         dramatically improves recall on Hebrew legal text where the same
         concept has multiple names (e.g. "קופסה" vs "תיבה").
      b. **Hybrid retrieval per query**: BM25 (keyword) + FAISS MMR
         (semantic) fused via ``EnsembleRetriever`` (RRF).
      c. **Regulation-reference chasing**: if any first-pass chunk mentions
         "תקנה N" or "תקנה N(א)", we do a targeted BM25 lookup for that
         specific regulation number to pull in its actual text.
      d. **Cross-lingual**: English questions are additionally re-issued
         in Hebrew for extra recall against the Hebrew corpus.
  * Answering: LCEL-style. System prompt enforces language matching, rich
    Markdown formatting, "פרשנות מקצועית" markers when combining regs, and
    citation of regulation number + page from chunk metadata.
  * Every Q&A pair is appended to an audit log (backend/qa_log.py) so the
    professional reviewer can spot-check answers — a POC roadmap deliverable.
"""

from __future__ import annotations

import os
import re
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
from backend.file_processor import load_file_documents_async
from backend.pdf_llm_ocr import split_markdown_by_page_markers
from backend.qa_log import log_qa

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None
# Keep the raw chunks in memory alongside the FAISS index so we can build a
# BM25 keyword retriever without re-parsing the PDF on every query. Hebrew
# legal questions frequently rely on exact terms (e.g. "תיבה" vs "קופסה")
# that vector similarity alone under-recalls.
_chunks: list[Document] = []
_bm25: BM25Retriever | None = None

SYSTEM_PROMPT = """You are a senior expert assistant for Israeli electricity regulations ("תקנות החשמל"), advising licensed electricians and electrical engineers. You answer questions using ONLY the provided context from the knowledge base.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE (HIGHEST PRIORITY — NEVER BREAK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Detect the language of the user's CURRENT question.
- Hebrew question → entire answer in Hebrew.
- English question → entire answer in English (translate context as needed).
- NEVER mix languages in one answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCURACY, GROUNDING, AND CITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use ONLY information explicitly present in the provided context. NEVER invent regulation numbers, values, or facts.
- Be thorough: read the ENTIRE context before concluding "not found". Hebrew PDFs sometimes have jumbled word order or OCR noise; if the meaning is clearly present in any chunk, USE IT.
- Prefer specific facts (numbers, thresholds, regulation numbers) from the context over general summaries.
- ALWAYS cite the specific regulation and, when available, the page number. Format:
  * Hebrew: "לפי תקנה 17 (עמ' 33)" or "כאמור בתקנה 49(ג)"
  * English: "under regulation 17 (p. 33)" or "per regulation 49(c)"
- Take citations from the ``[section: ...]`` header path or the leading "### תקנה N." line inside the chunk, and from the ``[page: N]`` marker.
- If information spans multiple regulations, cite each one.
- Preserve EXACT numeric values, units, and thresholds. Never round, convert, or paraphrase numbers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECT LAW vs PROFESSIONAL INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distinguish clearly between what the law directly says and what you're inferring:
- **📖 מהחוק ישירות / Direct from the law**: quote or paraphrase from a single explicit regulation. Prefer > blockquotes for verbatim text.
- **🧠 פרשנות מקצועית / Professional interpretation**: when you combine multiple regulations, apply them to a specific scenario, or explain implications. Prefix with a small header: `**🧠 פרשנות מקצועית:**` (Hebrew) or `**🧠 Professional interpretation:**` (English).
- Never blur the two.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN INFORMATION IS PARTIAL OR MISSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If the context has PARTIAL info (some but not all aspects of the question): answer the parts that ARE covered with citations, then explicitly state what wasn't covered ("היבטים נוספים כמו X לא מופיעים במאגר").
- If the context has NO relevant info at all: reply "המידע לא נמצא במאגר הידע" (Hebrew) or "This information is not in the knowledge base" (English), then list the related topics that DID appear so the user can rephrase.
- Do not refuse if any chunk plausibly answers the question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING — always use rich Markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **Bold** for key terms, regulation names, and threshold values.
- Bullet / numbered lists for multiple items, steps, requirements.
- ## and ### headers to organize longer answers.
- Markdown tables for any comparative or tabular data (distances, cross-sections, currents, intervals, resistances, thresholds).
- > blockquotes for verbatim regulation text.
- Keep answers thorough but scannable — avoid long unbroken paragraphs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD ANSWER STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For non-trivial questions, structure the answer as:
1. **תשובה קצרה / Short answer** — 2-3 lines with the direct requirement + main citation.
2. **פירוט / Detail** — the relevant regulation text, with tables or bullets.
3. **חריגים / הערות / Exceptions & notes** — related regs, exceptions, edge cases.
4. **פרשנות מקצועית** — if you combined multiple regs, label the inference here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY DISCLAIMER (always end with this line, in the same language as the answer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hebrew: "_מערכת זו היא כלי עזר ואינה מחליפה שיקול דעת של מהנדס חשמל מוסמך._"
English: "_This system is an aid and does not replace the judgement of a licensed electrical engineer._"
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


async def rebuild_vectorstore() -> int:
    """Rebuild the FAISS index (and in-memory BM25 corpus) from every file
    currently in ``data/uploads``.

    Uses the async loader because PDFs go through LLM-vision extraction.
    All page-level extraction is cached to disk, so re-runs are cheap.

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
            documents.extend(await load_file_documents_async(f))
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


def _pages_from_document(doc: Document) -> list[tuple[int | None, str]]:
    """If the document's Markdown contains ``<!-- page: N -->`` markers,
    split it into (page, markdown) pairs. Otherwise return a single
    (None, whole_content) pair."""
    content = doc.page_content or ""
    if "<!-- page:" not in content:
        return [(None, content)]
    pairs = split_markdown_by_page_markers(content)
    return [(p, md) for p, md in pairs if md.strip()]


def _split_markdown_documents(documents: list[Document]) -> list[Document]:
    """Three-stage split for a Markdown corpus:
        0. Peel off per-page sections using ``<!-- page: N -->`` markers so
           every downstream chunk carries a ``page`` metadata field for
           citation (PDF vision output uses these markers).
        1. Split by Markdown headers (# / ## / ### / ####) → each chunk
           carries its header path as ``section`` metadata (e.g.
           "פרק ב' > תקנה 17. גובה התקנת תיבה"). This gives the LLM a real
           regulation to cite instead of guessing.
        2. Any header section still larger than ``CHUNK_SIZE`` is further
           split by ``RecursiveCharacterTextSplitter`` while keeping parent
           header + page metadata intact.
    Documents without headers fall through to plain character splitting with
    metadata preserved.
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
        parent_meta = dict(doc.metadata)
        source = parent_meta.get("source", "unknown")

        for page_num, page_md in _pages_from_document(doc):
            page_meta = dict(parent_meta)
            if page_num is not None:
                page_meta["page"] = page_num

            header_chunks = header_splitter.split_text(page_md)

            if not header_chunks:
                page_doc = Document(page_content=page_md, metadata=page_meta)
                for c in char_splitter.split_documents([page_doc]):
                    merged = {**page_meta, **c.metadata}
                    merged.setdefault("section", "")
                    c.metadata = merged
                    all_chunks.append(c)
                continue

            for hc in header_chunks:
                section_parts = [
                    hc.metadata[k]
                    for k in ("h1", "h2", "h3", "h4")
                    if hc.metadata.get(k)
                ]
                section = " > ".join(section_parts)
                base_meta = {**page_meta, "section": section, **hc.metadata}

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


async def _expand_query(question: str, lang: str) -> list[str]:
    """Ask the LLM for 2-3 alternate phrasings of the question using
    domain-specific synonyms. Returns the alternate queries (without the
    original — the caller merges that in). Best-effort.

    For Hebrew legal text this is a big deal because the same object has
    many names (קופסה / תיבה / קופסת חיבור / קופסת הסתעפות for junction box).
    Vector similarity partly bridges this, BM25 does not, so alt phrasings
    let both retrievers hit the right chunks.
    """
    try:
        prompt_lang = "Hebrew" if lang == "he" else "English"
        system = (
            "You are a query rewriter for a retrieval system over Israeli "
            "electricity regulations (חוק החשמל).\n"
            f"Given the user's question in {prompt_lang}, output 2 to 3 "
            f"alternate short {prompt_lang} phrasings that use different but "
            "equivalent electrical/legal terminology (synonyms, hyponyms, "
            "regulation-style vocabulary). Focus on nouns and technical terms.\n"
            "Rules:\n"
            "- Output ONE alternate query per line. No numbering, no quotes, "
            "no commentary.\n"
            "- Keep each alt query short (≤ 12 words).\n"
            "- Never invent regulation numbers.\n"
            "- If the question is in Hebrew, output alt queries in Hebrew. "
            "If in English, output in English."
        )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        resp = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=question),
        ])
        raw = (resp.content or "").strip()
        alts: list[str] = []
        for line in raw.splitlines():
            line = line.strip("- •\t ").strip()
            if not line or line == question:
                continue
            alts.append(line)
        return alts[:3]
    except Exception:  # noqa: BLE001
        return []


# Matches Hebrew regulation references like "תקנה 17", "תקנה 99(א)",
# "תקנה 7(ג)(2)". Group 1 = the raw number+clause suffix.
_REG_REF_RE = re.compile(r"תקנה\s+(\d+(?:\([א-ת0-9]+\))*)")


def _extract_reg_references(docs: list[Document], max_refs: int = 5) -> list[str]:
    """Scan retrieved chunks for regulation references ('תקנה N', 'תקנה N(א)')
    and return the top-N most-frequent ones. Used to trigger a targeted BM25
    lookup that pulls in the actual text of referenced regulations.
    """
    if not docs:
        return []

    counts: dict[str, int] = {}
    for d in docs:
        for m in _REG_REF_RE.finditer(d.page_content or ""):
            key = m.group(1)
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [key for key, _ in ranked[:max_refs]]


def _bm25_regulation_lookup(reg_ids: list[str], k_per_ref: int = 2) -> list[Document]:
    """For each referenced regulation id (e.g. "17" or "7(ג)"), pull the top
    BM25 hits whose text or section explicitly names that regulation. This
    fetches the actual regulation text when other chunks only reference it.
    """
    if not reg_ids or _bm25 is None:
        return []
    _bm25.k = k_per_ref
    results: list[Document] = []
    for reg_id in reg_ids:
        query = f"תקנה {reg_id}"
        try:
            results.extend(_bm25.invoke(query))
        except Exception:  # noqa: BLE001
            continue
    return results


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

async def ask(
    question: str,
    session_messages: list[dict] | None = None,
    session_id: str | None = None,
) -> dict:
    """Answer a question grounded in the vector store.

    Multi-stage retrieval:
      1. Original query → hybrid retrieval (BM25 + FAISS MMR)
      2. LLM query expansion → 2-3 alt Hebrew/English phrasings → same
         hybrid retrieval per alt query
      3. Cross-lingual: if question is English, also retrieve on a Hebrew
         translation
      4. Regulation-reference chasing: if first-pass hits mention "תקנה N",
         BM25-fetch the actual text of those regulations

    All retrieved chunks are deduplicated and passed to the LLM.

    Returns ``{"answer": str, "sources": list[str]}`` — the same shape the
    frontend consumes.
    """
    vs = get_vectorstore()
    if vs is None:
        answer = (
            "מאגר הידע עדיין לא נטען. יש לפנות למנהל המערכת להעלאת מסמכים."
            "\n\nNo knowledge base loaded. Please ask an admin to upload "
            "documents first."
        )
        return {"answer": answer, "sources": []}

    lang = _detect_language(question)

    # --- Stage 1: primary hybrid retrieval on the original query ---
    all_docs: list[Document] = _retrieve(vs, question, k=8, fetch_k=24)

    # --- Stage 2: query expansion (LLM-generated alt phrasings) ---
    alt_queries = await _expand_query(question, lang)
    for alt in alt_queries:
        all_docs.extend(_retrieve(vs, alt, k=5, fetch_k=16))

    # --- Stage 3: cross-lingual boost for English questions ---
    hebrew_query: str | None = None
    if lang == "en":
        hebrew_query = await _translate_to_hebrew(question)
        if hebrew_query and hebrew_query.strip() and hebrew_query != question:
            all_docs.extend(_retrieve(vs, hebrew_query, k=6, fetch_k=20))

    docs = _dedupe(all_docs)

    # --- Stage 4: regulation-reference chasing ---
    referenced = _extract_reg_references(docs, max_refs=5)
    if referenced:
        chased = _bm25_regulation_lookup(referenced, k_per_ref=2)
        if chased:
            docs = _dedupe(docs + chased)

    # Cap total context to a reasonable size (best hits first, deduped).
    docs = docs[:20]

    context = _format_context(docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    messages = _build_messages(question, context, session_messages)
    response = await llm.ainvoke(messages)
    answer = response.content or ""

    sources = _build_source_list(docs)
    refused = _detect_refusal(answer)

    log_qa(
        session_id=session_id,
        question=question,
        answer=answer,
        retrieved=docs,
        sources=sources,
        refused=refused,
        extra={
            "lang": lang,
            "alt_queries": alt_queries,
            "hebrew_query": hebrew_query,
            "referenced_regs": referenced,
        },
    )

    return {"answer": answer, "sources": sources}


def _build_source_list(docs: list[Document]) -> list[str]:
    """Produce a compact, unique source list for the frontend footer."""
    sources: list[str] = []
    seen: set[tuple] = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        section = d.metadata.get("section") or ""
        page = d.metadata.get("page")
        key = (src, section, page)
        if key in seen:
            continue
        seen.add(key)

        label = src
        if section:
            label += f" — {section}"
        if page is not None:
            label += f" (p.{page})"
        sources.append(label)
    return sources


def _detect_refusal(answer: str) -> bool:
    """Return True if the answer contains the "not in knowledge base" refusal
    phrase (used for the Q&A audit log)."""
    if not answer:
        return False
    lowered = answer.lower()
    return (
        "המידע לא נמצא במאגר הידע" in answer
        or "not in the knowledge base" in lowered
    )


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
