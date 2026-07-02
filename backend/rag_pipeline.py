import os
import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

from backend.config import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    PROCESSED_DIR, VECTORSTORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None


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


def get_chain(session_messages: list[dict] | None = None):
    vs = get_vectorstore()
    if vs is None:
        return None

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, streaming=True)
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    if session_messages:
        for msg in session_messages[-10:]:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


async def ask(question: str, session_messages: list[dict] | None = None) -> dict:
    chain = get_chain(session_messages)
    if chain is None:
        return {
            "answer": "No knowledge base loaded yet. Please ask an admin to upload documents first.",
            "sources": [],
        }

    result = chain.invoke({"question": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})
    return {
        "answer": result["answer"],
        "sources": sources,
    }
