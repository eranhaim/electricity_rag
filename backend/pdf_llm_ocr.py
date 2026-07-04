"""
LLM-vision PDF → structured Hebrew Markdown processor.

Why not use a plain PDF text extractor?
    Hebrew PDFs with embedded fonts (like Israeli legal documents) frequently
    produce jumbled/mangled text with plain extractors — RTL word order gets
    reversed, ligatures fail, columns interleave. This makes retrieval brittle.

Approach:
    1. Render each PDF page to a high-DPI PNG image using PyMuPDF (fitz).
    2. Send each page image to GPT-4o (vision) with a strict prompt that
       preserves Hebrew, numbers, regulation IDs, and tables — while adding
       Markdown structure (# פרק, ### תקנה N).
    3. Cache each page's Markdown in ``data/processed/pages/<pdf-stem>/page_NNNN.md``
       so re-runs skip pages we've already processed.
    4. Concatenate all pages into a single Markdown document with
       ``<!-- page: N -->`` markers, so downstream chunkers can attach page
       numbers to each chunk for citation.

Cost/perf notes:
    ~285 pages of Hebrew law → ~$5-6 on gpt-4o (one-time), then cached.
    Uses ``asyncio.Semaphore`` to bound concurrent page requests.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config import OPENAI_API_KEY, PROCESSED_DIR


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Vision-capable model. Overridable via env for future upgrades.
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

# Render DPI. 150 is a good balance for Hebrew legibility vs image tokens.
RENDER_DPI = 150

# Concurrency: keep small to respect rate limits (5 requests/sec-ish is safe
# for tier 1/2 accounts; adjust upward if you have higher limits).
CONCURRENCY = 4


SYSTEM_PROMPT = """You are an expert legal document parser specializing in Israeli electricity regulations (תקנות החשמל). You receive an image of a single page from a Hebrew legal document. Extract its content into clean, structured Markdown.

CRITICAL RULES — NEVER BREAK:
1. Preserve ALL Hebrew text EXACTLY as it appears. Do NOT translate to English. Do NOT paraphrase. Do NOT summarize.
2. Preserve ALL numeric values EXACTLY (e.g., "2.1 מטר", "10 מ\"מ", "1.29 מגאום", "0.9 ממ\"ר"). Do NOT round, convert units, or reformat.
3. Preserve ALL regulation numbers and clause identifiers EXACTLY (e.g., "תקנה 17", "תקנה 99(א)(2)", "פרק ב'", "סעיף 5").
4. Preserve tables as valid GitHub-Flavored Markdown tables.
5. If a page has multiple columns, read them in the natural Hebrew RTL reading order (right column first, then left).
6. If content is unclear, cut off, or unreadable at a page edge, output "[...]" — do NOT guess.
7. Do NOT add commentary, headers, footers, page numbers, or any content that isn't on the page. Do NOT wrap the output in a code fence.

STRUCTURE RULES:
1. Use "# פרק X' — [title]" for chapter/part headings (e.g., "# פרק ב' — התקנת מוליכים").
2. Use "## [group title]" for numbered subsections / topic groups.
3. Use "### תקנה N. [title]" for individual regulations (e.g., "### תקנה 17. גובה התקנת תיבה"). Always include the regulation number in the heading.
4. Sub-clauses "(א)", "(ב)", "(ג)" appear as nested list items under their parent.
5. Preserve blank lines between logical sections so downstream Markdown splitters can chunk correctly.
6. If the page is purely a header page, table of contents, or blank, output only what appears — a short Markdown line or an empty response is fine.

OUTPUT: only the Markdown extracted from the page. No preface, no explanation, no code fences.
"""


@dataclass
class PageResult:
    page_num: int  # 1-based
    markdown: str
    from_cache: bool


def _cache_dir_for(pdf_path: Path) -> Path:
    d = PROCESSED_DIR / "pages" / pdf_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_file(pdf_path: Path, page_num: int) -> Path:
    return _cache_dir_for(pdf_path) / f"page_{page_num:04d}.md"


def _render_page_png(pdf_path: Path, page_num: int, dpi: int = RENDER_DPI) -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _count_pages(pdf_path: Path) -> int:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


_CODE_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _collapse_repeated_lines(text: str) -> str:
    """Collapse runs of 3+ consecutive identical (non-blank) lines to a
    single occurrence. Catches "1. Xxxx / 2. Xxxx / 3. Xxxx / ..." style
    hallucinations where each numbered line has the same body."""
    out: list[str] = []
    prev_stripped: str | None = None
    run = 0
    for line in text.splitlines():
        s = line.strip()
        if s and s == prev_stripped:
            run += 1
            if run < 3:
                out.append(line)
        else:
            out.append(line)
            prev_stripped = s
            run = 1
    return "\n".join(out)


def _clean_model_output(markdown: str) -> str:
    """Post-process model output:
      * Strip stray leading/trailing ```markdown code fences.
      * Collapse absurd repetitions (both short phrases and full lines).
      * Collapse "N. body" patterns where the body is identical across
        many numbered items — a strong signal of hallucination.
    """
    md = markdown.strip()
    md = _CODE_FENCE_RE.sub("", md).strip()

    md = _collapse_repeated_lines(md)

    # Substring repetition — extend up to 200 chars so we catch sentence-
    # length repeated phrases (a known gpt-4o failure mode on Hebrew pages).
    for phrase_len in range(3, 201):
        pattern = re.compile(r"(.{" + str(phrase_len) + r"})\1{2,}", re.DOTALL)
        md = pattern.sub(r"\1", md)

    # Collapse "N. same_body" runs of 5+ enumerated lines whose bodies match.
    lines = md.split("\n")
    enum_re = re.compile(r"^\s*\d+[.)]\s+(.*)$")
    kept: list[str] = []
    i = 0
    while i < len(lines):
        m = enum_re.match(lines[i])
        if not m:
            kept.append(lines[i])
            i += 1
            continue
        body = m.group(1).strip()
        end = i
        while end + 1 < len(lines):
            nm = enum_re.match(lines[end + 1])
            if nm and nm.group(1).strip() == body:
                end += 1
            else:
                break
        if end - i >= 4:
            kept.append(lines[i])
            i = end + 1
        else:
            kept.append(lines[i])
            i += 1
    return "\n".join(kept)


def _looks_pathological(markdown: str) -> bool:
    """True if the output is very likely a runaway hallucination:
    the same non-trivial line appears 5+ times, OR > 60 % of length is a
    single repeated substring. Used to trigger a retry with different
    sampling temperature."""
    if not markdown:
        return False
    lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
    if not lines:
        return False
    counts: dict[str, int] = {}
    for ln in lines:
        if len(ln) >= 25:
            counts[ln] = counts.get(ln, 0) + 1
    if any(c >= 5 for c in counts.values()):
        return True
    total = len(markdown)
    for phrase_len in (60, 120):
        m = re.search(
            r"(.{" + str(phrase_len) + r"})\1{4,}",
            markdown,
            flags=re.DOTALL,
        )
        if m and (len(m.group(0))) / total > 0.6:
            return True
    return False


def _looks_valid(markdown: str) -> bool:
    """Sanity check: non-empty, no refusal, not a pathological hallucination.
    Blank/short pages are OK (some pages really are near-empty)."""
    if markdown is None:
        return False
    md = markdown.strip()
    lower = md.lower()
    if any(x in lower for x in ("i can't", "i cannot", "sorry, i", "as an ai")):
        return False
    if _looks_pathological(md):
        return False
    return True


async def _extract_page(
    pdf_path: Path,
    page_num: int,
    sem: asyncio.Semaphore,
) -> PageResult:
    cache_path = _cache_file(pdf_path, page_num)
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8")
        return PageResult(page_num, cached, True)

    png_bytes = await asyncio.to_thread(_render_page_png, pdf_path, page_num)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    user_msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Extract the Hebrew Markdown from page {page_num} of "
                    "the attached PDF page image, following ALL rules above."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]
    )

    async with sem:
        last_err: Exception | None = None
        # On each retry raise the temperature so a stuck repetition loop
        # gets a chance to sample different tokens next time.
        for attempt, temperature in enumerate([0.0, 0.3, 0.7]):
            try:
                llm = ChatOpenAI(
                    model=VISION_MODEL,
                    temperature=temperature,
                    max_tokens=4096,
                )
                resp = await llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    user_msg,
                ])
                md = _clean_model_output(resp.content or "")
                if _looks_valid(md):
                    cache_path.write_text(md, encoding="utf-8")
                    return PageResult(page_num, md, False)
                last_err = RuntimeError(
                    f"invalid response at attempt {attempt + 1} (temp={temperature})"
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))

    print(f"[pdf_llm_ocr] page {page_num} failed after retries: {last_err}")
    # Persist an empty placeholder so we don't retry indefinitely on next run.
    cache_path.write_text("", encoding="utf-8")
    return PageResult(page_num, "", False)


PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


async def convert_pdf_to_markdown(
    pdf_path: Path,
    on_progress: callable | None = None,
) -> str:
    """Convert a PDF into one big Markdown string with per-page HTML markers
    so downstream chunkers can attach a ``page`` metadata field to each
    chunk. Cached per-page — subsequent calls are fast.
    """
    total = _count_pages(pdf_path)
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        asyncio.create_task(_extract_page(pdf_path, p, sem))
        for p in range(1, total + 1)
    ]

    done = 0
    results: list[PageResult] = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done += 1
        if on_progress:
            on_progress(done, total, r)

    results.sort(key=lambda r: r.page_num)

    parts: list[str] = []
    for r in results:
        parts.append(f"<!-- page: {r.page_num} -->")
        if r.markdown.strip():
            parts.append(r.markdown.strip())
    return "\n\n".join(parts) + "\n"


def split_markdown_by_page_markers(markdown: str) -> list[tuple[int, str]]:
    """Split a full-document Markdown into (page_num, page_markdown) tuples
    using the ``<!-- page: N -->`` markers this module emits."""
    pages: list[tuple[int, str]] = []
    current_page: int | None = None
    buf: list[str] = []
    for line in markdown.splitlines():
        m = PAGE_MARKER_RE.search(line)
        if m:
            if current_page is not None:
                pages.append((current_page, "\n".join(buf).strip()))
            current_page = int(m.group(1))
            buf = []
        else:
            buf.append(line)
    if current_page is not None:
        pages.append((current_page, "\n".join(buf).strip()))
    return pages
