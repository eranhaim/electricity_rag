"""
Extracts text from uploaded files and sends it to a high-end LLM
to produce a RAG-optimized text file.
"""
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config import OPENAI_API_KEY, UPLOADS_DIR, PROCESSED_DIR

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

RAG_OPTIMIZATION_PROMPT = """You are a document processing expert. Your task is to take raw text extracted from documents about electricity and power systems and produce a clean, well-structured, RAG-optimized version.

Instructions:
1. Remove all formatting artifacts, headers/footers, page numbers, and noise.
2. Organize the content into clear, self-contained sections with descriptive headings.
3. Each section should be a coherent chunk that can stand alone when retrieved.
4. Preserve all factual information, numbers, formulas, standards, and technical details exactly.
5. Expand abbreviations on first use.
6. Add brief context at the start of each section so a reader understands the topic without needing surrounding text.
7. Use clear, concise language. Remove redundancy but keep completeness.
8. If there are tables, convert them into readable text or structured lists.
9. Output plain text only (no markdown formatting).

Process the following document text:"""


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        lines = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            lines.append(f"Sheet: {sheet}")
            for row in ws.iter_rows(values_only=True):
                lines.append("\t".join(str(c) if c is not None else "" for c in row))
        return "\n".join(lines)

    if suffix == ".csv":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    return file_path.read_text(encoding="utf-8", errors="ignore")


async def process_file(file_path: Path) -> Path:
    """Extract text from a file, optimize it via LLM, save the result, and return the output path."""
    raw_text = _extract_text(file_path)

    if not raw_text.strip():
        raise ValueError(f"No text could be extracted from {file_path.name}")

    MAX_CHARS = 100_000
    if len(raw_text) > MAX_CHARS:
        chunks = [raw_text[i:i + MAX_CHARS] for i in range(0, len(raw_text), MAX_CHARS)]
    else:
        chunks = [raw_text]

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=16000)

    optimized_parts = []
    for i, chunk in enumerate(chunks):
        messages = [
            SystemMessage(content=RAG_OPTIMIZATION_PROMPT),
            HumanMessage(content=chunk),
        ]
        response = await llm.ainvoke(messages)
        optimized_parts.append(response.content)

    optimized_text = "\n\n".join(optimized_parts)
    output_name = file_path.stem + "_optimized.txt"
    output_path = PROCESSED_DIR / output_name
    output_path.write_text(optimized_text, encoding="utf-8")

    return output_path
