import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SESSIONS_FILE = DATA_DIR / "sessions.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

for d in [UPLOADS_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
