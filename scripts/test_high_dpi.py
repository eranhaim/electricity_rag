"""Send a single PDF page to gpt-4o vision at high DPI with an improved
prompt that emphasizes exact preservation of numbers. Compares against the
existing cached (150 DPI) extraction.

Usage inside the container:
    docker exec electricity-rag python -m scripts.test_high_dpi 75
"""

from __future__ import annotations

import asyncio
import base64
import sys

import fitz
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config import UPLOADS_DIR, PROCESSED_DIR
from backend.pdf_llm_ocr import _clean_model_output


HIGH_DPI_SYSTEM_PROMPT = """You are an expert legal document parser specializing in Israeli electricity regulations (תקנות החשמל). You receive an image of a single Hebrew legal document page. Extract its content into clean, structured Markdown.

CRITICAL RULES — NEVER BREAK:
1. Preserve ALL Hebrew text EXACTLY. NEVER translate. NEVER paraphrase. NEVER summarize.
2. **NUMBERS ARE SACRED**: read every digit, decimal point, comma, unit, and symbol character-by-character. Numbers like "2.1 מטר", "10 מ\"מ", "0.03 אמפר", "1,000 וולט" MUST be reproduced exactly as printed. If you cannot clearly read a digit, output "[?]" rather than guess.
3. Preserve ALL regulation numbers and clause identifiers EXACTLY (e.g., "תקנה 17", "תקנה 99(א)(2)", "פרק ב'").
4. Preserve tables as valid GitHub-Flavored Markdown tables. Every cell in every row.
5. If a page has multiple columns, read them right-to-left (natural Hebrew RTL order).
6. If any content is genuinely unreadable, output "[...]" for that spot — never fabricate.

STRUCTURE RULES:
1. "# פרק X' — [title]" for chapter headings.
2. "## [group title]" for numbered subsections.
3. "### תקנה N. [title]" for individual regulations. ALWAYS include the regulation number.
4. Sub-clauses "(א)", "(ב)", "(ג)", numeric "(1)", "(2)" appear as nested list items.
5. Preserve blank lines between logical sections.
6. If the page is purely a header, TOC, footer, or blank, output only what appears.

OUTPUT: only the Markdown extracted from the page. No preface, no explanation, no code fences.
"""


async def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: test_high_dpi.py PAGE_NUMBER [DPI]")
        return 1
    page_num = int(argv[0])
    dpi = int(argv[1]) if len(argv) > 1 else 300

    pdfs = [f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
    if not pdfs:
        print("No PDF")
        return 1
    pdf = pdfs[0]

    doc = fitz.open(str(pdf))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi)
    png = pix.tobytes("png")
    doc.close()
    print(f"Rendered page {page_num} at {dpi} DPI: {len(png):,} bytes")

    b64 = base64.b64encode(png).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=4096)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"Extract the Hebrew Markdown from page {page_num}."},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]
    )
    resp = await llm.ainvoke([SystemMessage(content=HIGH_DPI_SYSTEM_PROMPT), msg])
    md = _clean_model_output(resp.content or "")

    print(f"\n== NEW EXTRACTION ({dpi} DPI, high-detail) ==\n")
    print(md)

    # Compare against the cached 150-DPI extraction
    cache = PROCESSED_DIR / "pages" / pdf.stem / f"page_{page_num:04d}.md"
    if cache.exists():
        print("\n\n== EXISTING CACHED EXTRACTION (150 DPI) ==\n")
        print(cache.read_text(encoding="utf-8"))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
