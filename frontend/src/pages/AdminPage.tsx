import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, Upload, Trash2, FileText, Lock, Loader2, CheckCircle2, AlertCircle, Zap, Database,
} from "lucide-react";
import {
  adminLogin, adminListFiles, adminUploadFile, adminDeleteFile, adminStatus, UploadedFile,
} from "../api";
import styles from "./AdminPage.module.css";

export default function AdminPage() {
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(false);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [status, setStatus] = useState<{ vectorstore_loaded: boolean; upload_count: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = sessionStorage.getItem("admin_pw");
    if (saved) {
      setPassword(saved);
      setAuthenticated(true);
    }
  }, []);

  useEffect(() => {
    if (authenticated) {
      loadFiles();
      loadStatus();
    }
  }, [authenticated]);

  async function handleLogin() {
    const ok = await adminLogin(password);
    if (ok) {
      setAuthenticated(true);
      setAuthError(false);
      sessionStorage.setItem("admin_pw", password);
    } else {
      setAuthError(true);
    }
  }

  async function loadFiles() {
    const f = await adminListFiles(password);
    setFiles(f);
  }

  async function loadStatus() {
    const s = await adminStatus(password);
    setStatus(s);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);
    try {
      const result = await adminUploadFile(password, file);
      setUploadStatus({
        type: "success",
        message: `"${result.filename}" processed successfully. ${result.chunks_indexed} chunks indexed.`,
      });
      loadFiles();
      loadStatus();
    } catch (err: any) {
      setUploadStatus({
        type: "error",
        message: err.message || "Upload failed",
      });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(filename: string) {
    if (!confirm(`Delete "${filename}" and rebuild the knowledge base?`)) return;
    await adminDeleteFile(password, filename);
    loadFiles();
    loadStatus();
  }

  function formatSize(bytes: number) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  if (!authenticated) {
    return (
      <div className={styles.loginPage}>
        <div className={styles.loginCard}>
          <div className={styles.loginIcon}>
            <Lock size={32} />
          </div>
          <h2>Admin Access</h2>
          <p>Enter the admin password to manage documents.</p>
          <div className={styles.loginForm}>
            <input
              type="password"
              className={styles.passwordInput}
              placeholder="Password"
              value={password}
              onChange={(e) => { setPassword(e.target.value); setAuthError(false); }}
              onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              autoFocus
            />
            <button className={styles.loginBtn} onClick={handleLogin}>
              Sign In
            </button>
          </div>
          {authError && (
            <div className={styles.authError}>
              <AlertCircle size={14} /> Incorrect password
            </div>
          )}
          <button className={styles.backLink} onClick={() => navigate("/")}>
            <ArrowLeft size={14} /> Back to chat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.adminPage}>
      <header className={styles.header}>
        <button className={styles.backBtn} onClick={() => navigate("/")}>
          <ArrowLeft size={18} />
          <span>Back to Chat</span>
        </button>
        <h1>
          <Zap size={22} />
          Admin Panel
        </h1>
      </header>

      <div className={styles.content}>
        {/* Status cards */}
        <div className={styles.statusGrid}>
          <div className={styles.statusCard}>
            <Database size={20} />
            <div>
              <div className={styles.statusLabel}>Vector Store</div>
              <div className={styles.statusValue}>
                {status?.vectorstore_loaded ? (
                  <span className={styles.statusActive}><CheckCircle2 size={14} /> Active</span>
                ) : (
                  <span className={styles.statusInactive}><AlertCircle size={14} /> Empty</span>
                )}
              </div>
            </div>
          </div>
          <div className={styles.statusCard}>
            <FileText size={20} />
            <div>
              <div className={styles.statusLabel}>Documents</div>
              <div className={styles.statusValue}>{status?.upload_count ?? 0} files</div>
            </div>
          </div>
        </div>

        {/* Upload section */}
        <div className={styles.section}>
          <h2>Upload Documents</h2>
          <p className={styles.sectionDesc}>
            Upload PDF, DOCX, TXT, XLSX, or CSV files. Each file will be processed by an LLM
            to create an optimized knowledge base document, then indexed for retrieval.
          </p>

          <div
            className={`${styles.dropZone} ${uploading ? styles.dropZoneDisabled : ""}`}
            onClick={() => !uploading && fileInputRef.current?.click()}
          >
            {uploading ? (
              <>
                <Loader2 size={32} className={styles.spinner} />
                <span>Processing file with LLM... This may take a minute.</span>
              </>
            ) : (
              <>
                <Upload size={32} />
                <span>Click to select a file</span>
                <span className={styles.dropHint}>PDF, DOCX, TXT, XLSX, CSV</span>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.xlsx,.xls,.csv"
              onChange={handleUpload}
              hidden
            />
          </div>

          {uploadStatus && (
            <div className={`${styles.alert} ${styles[uploadStatus.type]}`}>
              {uploadStatus.type === "success" ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
              {uploadStatus.message}
            </div>
          )}
        </div>

        {/* File list */}
        <div className={styles.section}>
          <h2>Uploaded Files</h2>
          {files.length === 0 ? (
            <p className={styles.emptyText}>No files uploaded yet.</p>
          ) : (
            <div className={styles.fileList}>
              {files.map((f) => (
                <div key={f.name} className={styles.fileItem}>
                  <FileText size={18} className={styles.fileIcon} />
                  <div className={styles.fileInfo}>
                    <span className={styles.fileName}>{f.name}</span>
                    <span className={styles.fileSize}>{formatSize(f.size)}</span>
                  </div>
                  <button className={styles.deleteBtn} onClick={() => handleDelete(f.name)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
