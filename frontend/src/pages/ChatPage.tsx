import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  MessageSquarePlus, Trash2, Send, Zap, Settings, Pencil, Check, X, Menu, ChevronLeft, BookOpen,
} from "lucide-react";
import {
  fetchSessions, createSession, fetchSession, deleteSession,
  renameSession, sendMessage, Session, Message,
} from "../api";
import { useI18n } from "../i18n";
import styles from "./ChatPage.module.css";

export default function ChatPage() {
  const navigate = useNavigate();
  const { t, toggleLocale } = useI18n();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  // Sources are only returned by the /chat endpoint (not persisted per-message
  // in the session store), so we keep them locally keyed by message index in
  // the current view. Reset whenever we switch sessions.
  const [messageSources, setMessageSources] = useState<Record<number, string[]>>({});
  const [openSources, setOpenSources] = useState<Record<number, boolean>>({});
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
    setMessageSources({});
    setOpenSources({});
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
      setMessages((prev) => {
        const next = [...prev, assistantMsg];
        // Store the sources against the (index of the assistant message we
        // just appended). The user message is at length-1 before append; the
        // assistant is at length after append. Use next.length-1.
        setMessageSources((sp) => ({ ...sp, [next.length - 1]: res.sources || [] }));
        return next;
      });

      if (messages.length === 0) {
        const shortTitle = userMsg.content.slice(0, 50) + (userMsg.content.length > 50 ? "..." : "");
        setSessions((prev) =>
          prev.map((s) => (s.id === sessionId ? { ...s, title: shortTitle } : s))
        );
      }
    } catch {
      const errMsg: Message = { role: "assistant", content: t.errorMessage, timestamp: new Date().toISOString() };
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
            <span>{t.appName}</span>
          </div>
          <div className={styles.headerActions}>
            <button className={styles.langBtn} onClick={toggleLocale} title="Switch language">
              {t.langToggle}
            </button>
            <button className={styles.iconBtn} onClick={() => setSidebarOpen(false)} title={t.closeSidebar}>
              <ChevronLeft size={18} />
            </button>
          </div>
        </div>

        <button className={styles.newChatBtn} onClick={handleNewChat}>
          <MessageSquarePlus size={18} />
          <span>{t.newChat}</span>
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
          <span>{t.adminPanel}</span>
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
              <h2>{t.emptyTitle}</h2>
              <p>{t.emptyDescription}</p>
            </div>
          ) : (
            <div className={styles.messages}>
              {messages.map((m, i) => {
                const sources = messageSources[i];
                const sourcesOpen = openSources[i];
                return (
                  <div key={i} className={`${styles.messageRow} ${styles[m.role]}`}>
                    <div className={styles.avatar}>
                      {m.role === "user" ? "U" : <Zap size={16} />}
                    </div>
                    <div className={styles.messageBubble}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      {m.role === "assistant" && sources && sources.length > 0 && (
                        <div className={styles.sourcesBlock}>
                          <button
                            className={styles.sourcesToggle}
                            onClick={() =>
                              setOpenSources((sp) => ({ ...sp, [i]: !sp[i] }))
                            }
                          >
                            <BookOpen size={13} />
                            <span>
                              {t.sourcesLabel} ({sources.length})
                            </span>
                          </button>
                          {sourcesOpen && (
                            <ul className={styles.sourcesList}>
                              {sources.map((s, si) => (
                                <li key={si}>{s}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
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
              placeholder={t.inputPlaceholder}
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
            {t.disclaimer}
          </p>
        </div>
      </main>
    </div>
  );
}
