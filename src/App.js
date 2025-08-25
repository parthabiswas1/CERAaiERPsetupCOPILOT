import { useEffect, useState } from "react";
import "./styles.css";

/** ===== CONFIG ===== */
const BASE_URL = "https://ceraai-erp-setup-copilot.onrender.com"; // API URL
const USER = "demo";
const PASS = "demo";

/** ===== HELPERS ===== */
function useRunId() {
  const [runId, setRunId] = useState(() => {
    return localStorage.getItem("ceraai_run_id") || `ui-${Date.now()}`;
  });
  useEffect(() => localStorage.setItem("ceraai_run_id", runId), [runId]);
  return [runId, setRunId];
}

export default function App() {
  const [runId] = useRunId();
  const AUTH = "Basic " + btoa(`${USER}:${PASS}`);

  const api = async (path, { method = "GET", body, headers = {} } = {}) => {
    const res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: {
        Authorization: AUTH,
        "Content-Type": "application/json",
        "X-Run-ID": runId,
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} :: ${txt}`);
    }
    return res.json();
  };

  /** ===== UI STATE ===== */
  const [messages, setMessages] = useState([]);
  const [field, setField] = useState("");      // e.g., country / ein / ca_edd_id
  const [value, setValue] = useState("");      // e.g., US / 12-3456789 / 123-4567
  const [state, setState] = useState(null);
  const [mapped, setMapped] = useState(null);
  const [execRes, setExecRes] = useState(null);
  const [audit, setAudit] = useState([]);
  const [error, setError] = useState("");

  /** ===== ACTIONS ===== */
  const nextQuestion = async () => {
    setError("");
    try {
      const data = await api("/interview/next");
      setMessages((m) => [...m, { sender: "ceraai", text: data.question }]);
      await refreshState();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const submitAnswer = async () => {
    if (!field || !value) return;
    setError("");
    try {
      setMessages((m) => [...m, { sender: "user", text: `${field}: ${value}` }]);
      await api("/interview/answer", { method: "POST", body: { field, value } });
      setField(""); setValue("");
      await refreshState();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const refreshState = async () => {
    setError("");
    try {
      const s = await api("/state");
      setState(s);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const runMap = async () => {
    setError("");
    try {
      const res = await api("/map", { method: "POST", body: {} });
      setMapped(res.mapped);
      await refreshState();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const runExecute = async () => {
    setError("");
    try {
      const res = await api("/execute", {
        method: "POST",
        body: {}, // uses mapped payload saved in state
        headers: { "Idempotency-Key": "ui-exec-1" },
      });
      setExecRes(res);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const loadAudit = async () => {
    setError("");
    try {
      const res = await api(`/audit/logs?run_id=${encodeURIComponent(runId)}`);
      setAudit(res.logs || []);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  /** Auto-load first question */
  useEffect(() => { nextQuestion().catch(() => {}); /* eslint-disable-next-line */ }, []);

  return (
    <div className="app">
      {/* Header */}
      <nav className="header">
        <div className="brand">CERAai – ERP-SETUP-COPILOT</div>
        <div className="runid">Run ID: <code>{runId}</code></div>
      </nav>

      <div className="body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="step">🗨️ Interview</div>
          <div className="step">✅ Validator</div>
          <div className="step">🔄 Mapper</div>
          <div className="step">🚀 Executor</div>
          <div className="step">📜 Auditor</div>
        </aside>

        {/* Main */}
        <main className="main">
          {/* Errors */}
          {error ? <div className="error">{error}</div> : null}

          {/* Chat */}
          <section className="card chat">
            <div className="card-head">
              <h2>Interview</h2>
              <div className="actions">
                <button className="btn" onClick={nextQuestion}>Next</button>
              </div>
            </div>
            <div className="chat-body">
              {messages.map((m, i) => (
                <div key={i} className={`bubble ${m.sender === "user" ? "user" : "bot"}`}>
                  {m.text}
                </div>
              ))}
            </div>
            <div className="chat-input">
              <input
                placeholder="field (e.g., country, ein, ca_edd_id)"
                value={field}
                onChange={(e) => setField(e.target.value)}
              />
              <input
                placeholder="value"
                value={value}
                onChange={(e) => setValue(e.target.value)}
              />
              <button className="btn primary" onClick={submitAnswer}>Submit</button>
            </div>
          </section>

          {/* Validator / State */}
          <section className="card">
            <div className="card-head">
              <h2>Validator / State</h2>
              <div className="actions">
                <button className="btn" onClick={refreshState}>Refresh</button>
              </div>
            </div>
            <pre className="code">{JSON.stringify(state || {}, null, 2)}</pre>
          </section>

          {/* Mapper */}
          <section className="card">
            <div className="card-head">
              <h2>Mapper</h2>
              <div className="actions">
                <button className="btn primary" onClick={runMap}>Run Mapper</button>
              </div>
            </div>
            <pre className="code">{JSON.stringify(mapped || {}, null, 2)}</pre>
          </section>

          {/* Executor */}
          <section className="card">
            <div className="card-head">
              <h2>Executor</h2>
              <div className="actions">
                <button className="btn success" onClick={runExecute}>Execute (Idempotent)</button>
              </div>
            </div>
            <pre className="code">{JSON.stringify(execRes || {}, null, 2)}</pre>
          </section>

          {/* Auditor */}
          <section className="card">
            <div className="card-head">
              <h2>Audit Trail</h2>
              <div className="actions">
                <button className="btn" onClick={loadAudit}>Refresh Audit</button>
              </div>
            </div>
            <ul className="audit">
              {(audit || []).map((e, i) => (
                <li key={i}>
                  <span className="dim">{e.timestamp}</span> — {e.action || e.entity || "event"}
                  {e.run_id ? <> (<code>{e.run_id}</code>)</> : null}
                </li>
              ))}
            </ul>
          </section>
        </main>
      </div>
    </div>
  );
}
