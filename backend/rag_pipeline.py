import os
import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, SystemMessage

from backend.config import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    PROCESSED_DIR, VECTORSTORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None

SYSTEM_PROMPT = """You are a senior expert assistant specializing in Israeli electricity regulations, standards, and electrical safety codes. You answer questions based ONLY on the provided context documents from the knowledge base.

LANGUAGE RULE (HIGHEST PRIORITY — MUST FOLLOW):
- Detect the language of the user's CURRENT question below.
- If the question is in Hebrew → your ENTIRE answer MUST be in Hebrew.
- If the question is in English → your ENTIRE answer MUST be in English.
- NEVER mix languages. Even though the context documents are in Hebrew, if the question is in English you MUST translate all content into English.

ACCURACY & CITATION RULES:
- Only use information explicitly found in the provided context. If the answer is NOT in the context, clearly state that. NEVER fabricate or guess information.
- Always cite the specific regulation, standard, section, or clause number when available (e.g., "לפי תקנה 17", "According to regulation 17").
- When multiple regulations are relevant, cite each one separately.

FORMATTING RULES — ALWAYS use rich Markdown:
- Use **bold** for key terms, regulation names, important values, and thresholds.
- Use bullet points or numbered lists for multiple items, steps, or requirements.
- Use headers (## and ###) to organize long answers into clear sections.
- When presenting comparative data, specifications, distances, measurements, intervals, thresholds, or any structured data — ALWAYS use a Markdown table with proper column headers.
- Use > blockquotes for direct quotes from regulations.
- Keep answers thorough, well-structured, and easy to scan — avoid long unbroken paragraphs.
- If the user explicitly asks for a table, you MUST present the answer as a table.

ANSWER STRUCTURE for complex questions:
1. Brief direct answer or summary
2. Detailed explanation with citations
3. Table or list of specific requirements (if applicable)
4. Important notes or exceptions (if any)"""


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore(force_reload: bool = False) -> FAISS | None:
    global _vectorstore
    index_path = VECTORSTORE_DIR / "index.faiss"
    if force_reload:
        _vectorstore = None
    if _vectorstore is not None:
        return _vectorstore
    if index_path.exists():
        _vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR), get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def rebuild_vectorstore() -> int:
    """Rebuild the FAISS index from all processed .txt files."""
    global _vectorstore

    processed_files = list(PROCESSED_DIR.glob("*.txt"))
    if not processed_files:
        _vectorstore = None
        vs_path = VECTORSTORE_DIR / "index.faiss"
        if vs_path.exists():
            shutil.rmtree(VECTORSTORE_DIR)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        return 0

    documents: list[Document] = []
    for f in processed_files:
        text = f.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": f.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    _vectorstore = FAISS.from_documents(chunks, get_embeddings())
    _vectorstore.save_local(str(VECTORSTORE_DIR))
    return len(chunks)


def _detect_language(text: str) -> str:
    """Simple heuristic: if text contains Hebrew characters, it's Hebrew."""
    for ch in text:
        if "\u0590" <= ch <= "\u05FF":
            return "he"
    return "en"


def _build_messages(
    question: str,
    context_docs: list[Document],
    chat_history: list[dict] | None,
) -> list:
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    lang = _detect_language(question)
    lang_reminder = (
        "IMPORTANT: The user's question is in ENGLISH. You MUST answer ENTIRELY in English. "
        "Translate all Hebrew content from the context into English."
        if lang == "en"
        else ""
    )

    system_content = SYSTEM_PROMPT + "\n\nContext from knowledge base:\n" + context_text

    messages = [SystemMessage(content=system_content)]

    if chat_history:
        for msg in chat_history[-8:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain.schema import AIMessage
                messages.append(AIMessage(content=msg["content"]))

    user_content = question
    if lang_reminder:
        user_content = f"{lang_reminder}\n\nQuestion: {question}"

    messages.append(HumanMessage(content=user_content))
    return messages


async def ask(question: str, session_messages: list[dict] | None = None) -> dict:
    vs = get_vectorstore()
    if vs is None:
        return {
            "answer": "No knowledge base loaded yet. Please ask an admin to upload documents first.",
            "sources": [],
        }

    retriever = vs.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(question)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    messages = _build_messages(question, docs, session_messages)
    response = await llm.ainvoke(messages)

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    return {
        "answer": response.content,
        "sources": sources,
    }
