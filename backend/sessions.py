import json
import uuid
from datetime import datetime
from pathlib import Path
from backend.config import SESSIONS_FILE


def _load_all() -> dict:
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_all(data: dict):
    SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_session(title: str | None = None) -> dict:
    sid = str(uuid.uuid4())
    session = {
        "id": sid,
        "title": title or "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    data = _load_all()
    data[sid] = session
    _save_all(data)
    return session


def list_sessions() -> list[dict]:
    data = _load_all()
    sessions = sorted(data.values(), key=lambda s: s["created_at"], reverse=True)
    return [{"id": s["id"], "title": s["title"], "created_at": s["created_at"]} for s in sessions]


def get_session(sid: str) -> dict | None:
    data = _load_all()
    return data.get(sid)


def add_message(sid: str, role: str, content: str) -> dict | None:
    data = _load_all()
    session = data.get(sid)
    if not session:
        return None
    session["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    })
    _save_all(data)
    return session


def update_title(sid: str, title: str) -> dict | None:
    data = _load_all()
    session = data.get(sid)
    if not session:
        return None
    session["title"] = title
    _save_all(data)
    return session


def delete_session(sid: str) -> bool:
    data = _load_all()
    if sid in data:
        del data[sid]
        _save_all(data)
        return True
    return False
