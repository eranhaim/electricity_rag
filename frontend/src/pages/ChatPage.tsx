import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  MessageSquarePlus, Trash2, Send, Zap, Settings, Pencil, Check, X, Menu, ChevronLeft,
} from "lucide-react";
import {
  fetchSessions, createSession, fetchSession, deleteSession,
  renameSession, sendMessage, Session, Message,
} from "../api";
import styles from "./ChatPage.module.css";

export default function ChatPage() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadSessions() {
    const s = await fetchSessions();
    setSessions(s);
  }

  async function handleNewChat() {
    const s = await createSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setMessages([]);
  }

  async function handleSelectSession(id: string) {
    setActiveId(id);
    const s = await fetchSession(id);
    setMessages(s.messages || []);
  }

  async function handleDelete(id: string) {
    await deleteSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeId === id) {
      setActiveId(null);
      setMessages([]);
    }
  }

  async function handleRename(id: string) {
    if (!editTitle.trim()) return;
    await renameSession(id, editTitle.trim());
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title: editTitle.trim() } : s)));
    setEditingId(null);
  }

  async function handleSend() {
    if (!input.trim() || loading) return;

    let sessionId = activeId;
    if (!sessionId) {
      const s = await createSession();
      setSessions((prev) => [s, ...prev]);
      sessionId = s.id;
      setActiveId(s.id);
    }

    const userMsg: Message = { role: "user", content: input.trim(), timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await sendMessage(sessionId, userMsg.content);
      const assistantMsg: Message = { role: "assistant", content: res.answer, timestamp: new Date().toISOString() };
      setMessages((prev) => [...prev, assistantMsg]);

      if (messages.length === 0) {
        const shortTitle = userMsg.content.slice(0, 50) + (userMsg.content.length > 50 ? "..." : "");
        setSessions((prev) =>
          prev.map((s) => (s.id === sessionId ? { ...s, title: shortTitle } : s))
        );
      }
    } catch {
      const errMsg: Message = { role: "assistant", content: "Sorry, something went wrong. Please try again.", timestamp: new Date().toISOString() };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className={styles.layout}>
      {/* Sidebar */}
      <aside className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`}>
        <div className={styles.sidebarHeader}>
          <div className={styles.logo}>
            <Zap size={20} />
            <span>Electricity RAG</span>
          </div>
          <button className={styles.iconBtn} onClick={() => setSidebarOpen(false)} title="Close sidebar">
            <ChevronLeft size={18} />
          </button>
        </div>

        <button className={styles.newChatBtn} onClick={handleNewChat}>
          <MessageSquarePlus size={18} />
          <span>New Chat</span>
        </button>

        <div className={styles.sessionList}>
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`${styles.sessionItem} ${s.id === activeId ? styles.active : ""}`}
              onClick={() => handleSelectSession(s.id)}
            >
              {editingId === s.id ? (
                <div className={styles.editRow} onClick={(e) => e.stopPropagation()}>
                  <input
                    className={styles.editInput}
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleRename(s.id)}
                    autoFocus
                  />
                  <button className={styles.iconBtnSm} onClick={() => handleRename(s.id)}><Check size={14} /></button>
                  <button className={styles.iconBtnSm} onClick={() => setEditingId(null)}><X size={14} /></button>
                </div>
              ) : (
                <>
                  <span className={styles.sessionTitle}>{s.title}</span>
                  <div className={styles.sessionActions}>
                    <button
                      className={styles.iconBtnSm}
                      onClick={(e) => { e.stopPropagation(); setEditingId(s.id); setEditTitle(s.title); }}
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      className={styles.iconBtnSm}
                      onClick={(e) => { e.stopPropagation(); handleDelete(s.id); }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        <button className={styles.adminLink} onClick={() => navigate("/admin")}>
          <Settings size={16} />
          <span>Admin Panel</span>
        </button>
      </aside>

      {/* Main */}
      <main className={styles.main}>
        {!sidebarOpen && (
          <button className={styles.menuBtn} onClick={() => setSidebarOpen(true)}>
            <Menu size={20} />
          </button>
        )}

        <div className={styles.chatArea}>
          {messages.length === 0 && !loading ? (
            <div className={styles.emptyState}>
              <Zap size={48} className={styles.emptyIcon} />
              <h2>Electricity Knowledge Assistant</h2>
              <p>Ask me anything about electricity, power systems, regulations, and more.</p>
            </div>
          ) : (
            <div className={styles.messages}>
              {messages.map((m, i) => (
                <div key={i} className={`${styles.messageRow} ${styles[m.role]}`}>
                  <div className={styles.avatar}>
                    {m.role === "user" ? "U" : <Zap size={16} />}
                  </div>
                  <div className={styles.messageBubble}>
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                </div>
              ))}
              {loading && (
                <div className={`${styles.messageRow} ${styles.assistant}`}>
                  <div className={styles.avatar}><Zap size={16} /></div>
                  <div className={styles.messageBubble}>
                    <div className={styles.typing}>
                      <span /><span /><span />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <textarea
              ref={inputRef}
              className={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about electricity..."
              rows={1}
              disabled={loading}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!input.trim() || loading}
            >
              <Send size={18} />
            </button>
          </div>
          <p className={styles.disclaimer}>
            Answers are based on uploaded electricity documents. Always verify critical information.
          </p>
        </div>
      </main>
    </div>
  );
}
