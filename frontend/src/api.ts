const API = "/api";

export interface Session {
  id: string;
  title: string;
  created_at: string;
  messages?: Message[];
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
}

export interface UploadedFile {
  name: string;
  size: number;
}

export interface QALogChunk {
  source: string | null;
  section: string | null;
  page: number | null;
  chunk_id: number | null;
  snippet: string;
  full_length: number;
}

export interface QALogEntry {
  ts: string;
  session_id: string | null;
  question: string;
  answer: string;
  refused: boolean;
  sources_cited: string[];
  retrieved_chunks: QALogChunk[];
  extra?: {
    lang?: string;
    alt_queries?: string[];
    hebrew_query?: string | null;
    referenced_regs?: string[];
  };
}

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch(`${API}/sessions`);
  return res.json();
}

export async function createSession(): Promise<Session> {
  const res = await fetch(`${API}/sessions`, { method: "POST" });
  return res.json();
}

export async function fetchSession(id: string): Promise<Session> {
  const res = await fetch(`${API}/sessions/${id}`);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API}/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(id: string, title: string): Promise<void> {
  await fetch(`${API}/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
  const res = await fetch(`${API}/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}

export async function adminLogin(password: string): Promise<boolean> {
  const res = await fetch(`${API}/admin/login`, {
    method: "POST",
    headers: { "x-admin-password": password },
  });
  return res.ok;
}

export async function adminListFiles(password: string): Promise<UploadedFile[]> {
  const res = await fetch(`${API}/admin/files`, {
    headers: { "x-admin-password": password },
  });
  return res.json();
}

export async function adminUploadFile(password: string, file: File): Promise<{ ok: boolean; filename: string; chunks_indexed: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/admin/upload`, {
    method: "POST",
    headers: { "x-admin-password": password },
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function adminDeleteFile(password: string, filename: string): Promise<void> {
  await fetch(`${API}/admin/files/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: { "x-admin-password": password },
  });
}

export async function adminStatus(password: string): Promise<{ vectorstore_loaded: boolean; upload_count: number; qa_log_count?: number }> {
  const res = await fetch(`${API}/admin/status`, {
    headers: { "x-admin-password": password },
  });
  return res.json();
}

export async function adminQaLog(
  password: string,
  limit: number = 50,
): Promise<{ total: number; entries: QALogEntry[] }> {
  const res = await fetch(`${API}/admin/qa_log?limit=${limit}`, {
    headers: { "x-admin-password": password },
  });
  return res.json();
}
