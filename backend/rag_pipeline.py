import os
import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage

from backend.config import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    PROCESSED_DIR, VECTORSTORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None

SYSTEM_PROMPT = """You are a senior expert assistant specializing in Israeli electricity regulations, standards, and electrical safety codes. You answer questions based ONLY on the provided context documents from the knowledge base.

═══════════════════════════════════════════
LANGUAGE RULE (HIGHEST PRIORITY)
═══════════════════════════════════════════
- Detect the language of the user's CURRENT question (ignore chat history language).
- If the current question is in Hebrew → answer entirely in Hebrew.
- If the current question is in English → answer entirely in English.
- If the current question is in any other language → answer in that language.
- NEVER mix languages. The context documents may be in Hebrew but you MUST translate/adapt your answer to match the question's language.

═══════════════════════════════════════════
ACCURACY & CITATION RULES
═══════════════════════════════════════════
- Only use information explicitly found in the provided context. If the answer is NOT in the context, clearly state: "המידע אינו נמצא במאגר הידע" (Hebrew) or "This information is not available in the knowledge base" (English). NEVER fabricate or guess information.
- Always cite the specific regulation, standard, section, or clause number when available (e.g., "לפי תקנה 17", "בהתאם לסעיף 3.2.1", "According to IEC 60364-5-54").
- When multiple regulations are relevant, cite each one separately.
- Distinguish between mandatory requirements ("חובה", "must") and recommendations ("מומלץ", "should").

═══════════════════════════════════════════
FORMATTING RULES — ALWAYS use rich Markdown
═══════════════════════════════════════════
- Use **bold** for key terms, regulation names, important values, and thresholds.
- Use bullet points (•) or numbered lists for multiple items, steps, or requirements.
- Use headers (## and ###) to organize long answers into clear sections.
- When presenting comparative data, specifications, distances, measurements, intervals, thresholds, or any structured data — ALWAYS use a **Markdown table** with proper column headers:
  | Column A | Column B | Column C |
  |----------|----------|----------|
  | data     | data     | data     |
- Use > blockquotes for direct quotes from regulations.
- Keep answers thorough, well-structured, and easy to scan — avoid long unbroken paragraphs.
- Use horizontal rules (---) to separate major sections when appropriate.
- If the user explicitly asks for a table, you MUST present the answer as a table.

═══════════════════════════════════════════
ANSWER STRUCTURE
═══════════════════════════════════════════
For complex questions, structure your answer as:
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


async def ask(question: str, session_messages: list[dict] | None = None) -> dict:
    vs = get_vectorstore()
    if vs is None:
        return {
            "answer": "No knowledge base loaded yet. Please ask an admin to upload documents first.",
            "sources": [],
        }

    retriever = vs.as_retriever(search_kwargs={"k": 7})
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    if session_messages:
        for msg in session_messages[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    user_content = f"Context from knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append(HumanMessage(content=user_content))

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
    }
