import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import ADMIN_PASSWORD, UPLOADS_DIR
from backend.sessions import (
    create_session, list_sessions, get_session,
    add_message, update_title, delete_session,
)
from backend.rag_pipeline import ask, rebuild_vectorstore, get_vectorstore
from backend.file_processor import process_file
from backend.qa_log import read_recent as read_qa_log, count_entries as count_qa_log

app = FastAPI(title="Electricity RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helper ──────────────────────────────────────────────────────────────

def verify_admin(x_admin_password: str = Header(...)):
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")


# ── Chat endpoints ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class TitleUpdate(BaseModel):
    title: str


@app.get("/api/sessions")
def api_list_sessions():
    return list_sessions()


@app.post("/api/sessions")
def api_create_session():
    return create_session()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.patch("/api/sessions/{session_id}")
def api_update_session(session_id: str, body: TitleUpdate):
    s = update_title(session_id, body.title)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.post("/api/sessions/{session_id}/chat")
async def api_chat(session_id: str, body: ChatRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    add_message(session_id, "user", body.message)

    result = await ask(
        body.message,
        session.get("messages"),
        session_id=session_id,
    )
    add_message(session_id, "assistant", result["answer"])

    if len(session["messages"]) == 0:
        short_title = body.message[:50] + ("..." if len(body.message) > 50 else "")
        update_title(session_id, short_title)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }


# ── Admin endpoints ──────────────────────────────────────────────────────────

@app.post("/api/admin/login")
def api_admin_login(x_admin_password: str = Header(...)):
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid password")
    return {"ok": True}


@app.get("/api/admin/files", dependencies=[Depends(verify_admin)])
def api_list_files():
    uploads = [
        {"name": f.name, "size": f.stat().st_size}
        for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
    return uploads


@app.post("/api/admin/upload", dependencies=[Depends(verify_admin)])
async def api_upload_file(file: UploadFile = File(...)):
    dest = UPLOADS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        await process_file(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"Processing failed: {e}")

    chunk_count = await rebuild_vectorstore()
    return {
        "ok": True,
        "filename": file.filename,
        "chunks_indexed": chunk_count,
    }


@app.delete("/api/admin/files/{filename}", dependencies=[Depends(verify_admin)])
async def api_delete_file(filename: str):
    upload_path = UPLOADS_DIR / filename
    if not upload_path.exists():
        raise HTTPException(404, "File not found")
    upload_path.unlink()

    processed_dir = UPLOADS_DIR.parent / "processed"
    stem = Path(filename).stem
    for candidate in [
        processed_dir / f"{stem}.md",
        processed_dir / f"{stem}.txt",
        processed_dir / f"{stem}_optimized.txt",
    ]:
        candidate.unlink(missing_ok=True)

    chunk_count = await rebuild_vectorstore()
    return {"ok": True, "chunks_indexed": chunk_count}


@app.post("/api/admin/reindex", dependencies=[Depends(verify_admin)])
async def api_reindex():
    """Re-process every file in ``data/uploads`` and rebuild the FAISS index
    from the raw content. Use this after upgrading the pipeline or if the
    processed/vectorstore artifacts get out of sync with the uploads."""
    upload_files = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
    errors: list[str] = []
    for f in upload_files:
        try:
            await process_file(f)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}: {e}")

    chunk_count = await rebuild_vectorstore()
    return {
        "ok": True,
        "files_processed": len(upload_files) - len(errors),
        "chunks_indexed": chunk_count,
        "errors": errors,
    }


@app.get("/api/admin/status", dependencies=[Depends(verify_admin)])
def api_admin_status():
    vs = get_vectorstore()
    return {
        "vectorstore_loaded": vs is not None,
        "upload_count": len([f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]),
        "qa_log_count": count_qa_log(),
    }


@app.get("/api/admin/qa_log", dependencies=[Depends(verify_admin)])
def api_qa_log(limit: int = 50):
    """Return the most recent Q&A audit entries so a reviewer (Ori) can
    spot-check answer quality. Newest first."""
    limit = max(1, min(limit, 500))
    return {
        "total": count_qa_log(),
        "entries": read_qa_log(limit=limit),
    }


# ── Serve frontend (production) ─────────────────────────────────────────────

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
