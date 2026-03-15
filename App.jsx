import { useState, useEffect, useRef, useCallback } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

// ─── Utility hooks ────────────────────────────────────────────────────────────
function useInterval(fn, delay) {
  const saved = useRef(fn);
  useEffect(() => { saved.current = fn; }, [fn]);
  useEffect(() => {
    if (delay === null) return;
    const id = setInterval(() => saved.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// ─── Mini Components ──────────────────────────────────────────────────────────

function GlowDot({ active, color = "#00ffc8" }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: active ? color : "#333",
      boxShadow: active ? `0 0 8px ${color}, 0 0 20px ${color}55` : "none",
      transition: "all 0.4s ease",
    }} />
  );
}

function ScopeTag({ label, color }) {
  return (
    <span style={{
      fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
      letterSpacing: "0.12em", padding: "2px 8px", borderRadius: 3,
      background: `${color}18`, border: `1px solid ${color}55`, color,
      textTransform: "uppercase", userSelect: "none",
    }}>
      {label}
    </span>
  );
}

function EnergyMeter({ value }) {
  // energy typically ranges from ~-2 to 2; normalise to 0–1
  const norm   = Math.max(0, Math.min(1, (value + 2) / 4));
  const hue    = Math.round((1 - norm) * 140);           // green → red
  const color  = `hsl(${hue}, 90%, 55%)`;
  const pct    = Math.round(norm * 100);
  const arc    = 283 - norm * 283;                       // SVG circle path

  return (
    <div style={{ position: "relative", width: 140, height: 140, margin: "0 auto" }}>
      <svg width="140" height="140" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="none" stroke="#1a1a2e" strokeWidth="8" />
        <circle cx="50" cy="50" r="45" fill="none"
          stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray="283" strokeDashoffset={arc}
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1), stroke 0.8s" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0, display: "flex",
        flexDirection: "column", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 22, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace",
          textShadow: `0 0 12px ${color}` }}>
          {value.toFixed(3)}
        </span>
        <span style={{ fontSize: 9, color: "#888", letterSpacing: "0.1em", marginTop: 2 }}>ENERGY</span>
      </div>
    </div>
  );
}

function SoundnessBadge({ sound, reason }) {
  const color = sound ? "#00ffc8" : "#ff4d6d";
  const label = sound ? "✓  LOGICALLY SOUND" : "✗  LOGIC FLAW";
  return (
    <div style={{
      marginTop: 16, padding: "14px 18px", borderRadius: 8,
      border: `1px solid ${color}44`,
      background: `${color}0e`,
      display: "flex", flexDirection: "column", gap: 6,
    }}>
      <span style={{
        fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
        color, fontWeight: 700, letterSpacing: "0.06em",
        textShadow: `0 0 10px ${color}88`,
      }}>
        {label}
      </span>
      <span style={{ fontSize: 11, color: "#aaa", fontFamily: "'JetBrains Mono', monospace" }}>
        {reason}
      </span>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 18, height: 18, border: "2px solid #ffffff22",
      borderTop: "2px solid #00ffc8", borderRadius: "50%",
      animation: "spin 0.7s linear infinite", display: "inline-block",
    }} />
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [problem,   setProblem]   = useState("");
  const [solution,  setSolution]  = useState("");
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");

  const [health,    setHealth]    = useState(null);
  const [connected, setConnected] = useState(false);

  const [pdfStatus, setPdfStatus] = useState(null);     // { filename, char_count, figures }
  const [dragging,  setDragging]  = useState(false);
  const dropRef = useRef(null);

  // ── Health polling ─────────────────────────────────────────────────────────
  const fetchHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/health`);
      if (r.ok) {
        const d = await r.json();
        setHealth(d);
        setConnected(true);
      } else { setConnected(false); }
    } catch { setConnected(false); }
  }, []);

  useEffect(() => { fetchHealth(); }, [fetchHealth]);
  useInterval(fetchHealth, 5000);

  // ── Query submission ───────────────────────────────────────────────────────
  async function handleQuery() {
    if (!problem.trim() || !solution.trim()) {
      setError("Both fields are required."); return;
    }
    setError(""); setLoading(true); setResult(null);
    try {
      const r = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem, solution }),
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      setResult(await r.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  // ── PDF drag-and-drop ──────────────────────────────────────────────────────
  function onDragOver(e) { e.preventDefault(); setDragging(true); }
  function onDragLeave()  { setDragging(false); }

  async function handleDrop(e) {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer?.files[0] ?? e.target.files?.[0];
    if (!file || !file.name.endsWith(".pdf")) {
      setError("Only PDF files accepted."); return;
    }
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(`${API}/api/upload-pdf`, { method: "POST", body: fd });
      if (!r.ok) throw new Error("Upload failed");
      const d = await r.json();
      setPdfStatus(d);
      fetchHealth();   // refresh figure count
    } catch (e) { setError(e.message); }
  }

  const fileRef = useRef(null);

  // ─── CSS-in-JS global tokens ───────────────────────────────────────────────
  const panel = {
    background: "rgba(10,11,24,0.78)",
    backdropFilter: "blur(14px)",
    border: "1px solid rgba(0,255,200,0.12)",
    borderRadius: 12,
    padding: "24px 28px",
  };

  const inputStyle = {
    width: "100%", boxSizing: "border-box",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(0,255,200,0.2)",
    borderRadius: 8, color: "#e8e8ff",
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 13, padding: "12px 14px", resize: "vertical",
    outline: "none", transition: "border-color 0.2s",
  };

  const labelStyle = {
    fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase",
    color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
    marginBottom: 6, display: "flex", alignItems: "center", gap: 8,
  };

  const btnPrimary = {
    background: "linear-gradient(135deg, #00ffc8 0%, #00b4ff 100%)",
    color: "#050810", border: "none", borderRadius: 8,
    padding: "13px 32px", fontFamily: "'JetBrains Mono', monospace",
    fontWeight: 700, fontSize: 13, letterSpacing: "0.08em",
    cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
    boxShadow: "0 4px 24px #00ffc844",
    transition: "opacity 0.2s, transform 0.15s",
  };

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #050810 0%, #080c1e 40%, #0a0714 100%)",
      color: "#dde0ff",
      fontFamily: "'IBM Plex Sans', 'JetBrains Mono', sans-serif",
      padding: "0 0 40px",
    }}>

      {/* Global keyframes injected once */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
        * { margin:0; padding:0; box-sizing:border-box; }
        ::selection { background: #00ffc844; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        textarea:focus { border-color: rgba(0,255,200,0.55) !important; }
        textarea::placeholder { color: #3a3a5a; }
        ::-webkit-scrollbar { width:5px; }
        ::-webkit-scrollbar-track { background:#0a0a18; }
        ::-webkit-scrollbar-thumb { background:#1e1e3e; border-radius:4px; }
      `}</style>

      {/* ── Top Bar ─────────────────────────────────────────────────────────── */}
      <header style={{
        borderBottom: "1px solid rgba(0,255,200,0.08)",
        padding: "18px 36px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "rgba(5,8,16,0.9)", backdropFilter: "blur(10px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg, #00ffc8, #00b4ff)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 700, color: "#050810",
          }}>⚛</div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 600, letterSpacing: "0.04em", color: "#e8e8ff" }}>
              SciOracle
            </div>
            <div style={{ fontSize: 10, color: "#6a6a9a", letterSpacing: "0.12em", fontFamily: "'JetBrains Mono', monospace" }}>
              MATH-EBM DISCOVERY SHELL
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#6a6a9a",
            display: "flex", alignItems: "center", gap: 8 }}>
            <GlowDot active={connected} />
            {connected ? "BACKEND CONNECTED" : "DISCONNECTED"}
          </div>
          {health && (
            <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#00b4ff",
              display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ opacity: 0.5 }}>◈</span>
              {health.device.toUpperCase()}
            </div>
          )}
        </div>
      </header>

      {/* ── Body Grid ───────────────────────────────────────────────────────── */}
      <div style={{
        maxWidth: 1320, margin: "0 auto", padding: "32px 24px",
        display: "grid", gridTemplateColumns: "280px 1fr 340px", gap: 20,
      }}>

        {/* ══════════ LEFT: Live Status Sidebar ══════════ */}
        <aside>
          <div style={{ ...panel, animation: "fadeIn 0.4s ease" }}>
            <div style={{ fontSize: 10, letterSpacing: "0.16em", color: "#6a6a9a",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 20 }}>
              ◈ SYSTEM STATUS
            </div>

            {/* Connectivity */}
            <StatusRow label="Backend API" active={connected}
              value={connected ? "LIVE" : "OFFLINE"} />
            <StatusRow label="Checkpoint"
              active={health?.checkpoint_exists}
              value={health?.checkpoint_exists ? "LOADED" : "MISSING"} />

            {/* Device */}
            <div style={{ margin: "20px 0 8px", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16 }}>
              <div style={{ fontSize: 10, letterSpacing: "0.14em", color: "#6a6a9a",
                fontFamily: "'JetBrains Mono', monospace", marginBottom: 12 }}>
                HARDWARE
              </div>
              <div style={{
                padding: "10px 12px", borderRadius: 8,
                background: "rgba(0,180,255,0.07)", border: "1px solid rgba(0,180,255,0.15)",
                fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: "#00b4ff",
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <span style={{ fontSize: 18 }}>{health?.device?.includes("cuda") ? "⚡" : "🖥"}</span>
                {health?.device?.toUpperCase() || "—"}
              </div>
            </div>

            {/* Knowledge Base */}
            <div style={{ margin: "20px 0 8px", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16 }}>
              <div style={{ fontSize: 10, letterSpacing: "0.14em", color: "#6a6a9a",
                fontFamily: "'JetBrains Mono', monospace", marginBottom: 12 }}>
                KNOWLEDGE BASE
              </div>
              <KBStat icon="◼" label="Figures indexed" value={health?.figures_in_kb ?? 0} color="#b060ff" />
              {pdfStatus && <KBStat icon="◻" label="Last PDF chars" value={pdfStatus.char_count.toLocaleString()} color="#00ffc8" />}
            </div>

            {/* Dual-Encoder scope legend */}
            <div style={{ margin: "20px 0 0", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16 }}>
              <div style={{ fontSize: 10, letterSpacing: "0.14em", color: "#6a6a9a",
                fontFamily: "'JetBrains Mono', monospace", marginBottom: 12 }}>
                ENCODER SCOPE
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <ScopeTag label="LLM Sequence Scope" color="#00b4ff" />
                <ScopeTag label="AST Graph Scope"    color="#b060ff" />
              </div>
            </div>

            {/* VRAM guard notice */}
            <div style={{
              marginTop: 20, padding: "10px 12px", borderRadius: 8,
              background: "rgba(255,200,0,0.05)", border: "1px solid rgba(255,200,0,0.15)",
              fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: "#ffcc0099",
              letterSpacing: "0.06em",
            }}>
              ⚠ VRAM GUARD: 6 GB · RTX 2060
            </div>

            {/* Privacy badge */}
            <div style={{
              marginTop: 10, padding: "10px 12px", borderRadius: 8,
              background: "rgba(0,255,200,0.05)", border: "1px solid rgba(0,255,200,0.12)",
              fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: "#00ffc899",
              letterSpacing: "0.06em", display: "flex", alignItems: "center", gap: 6,
            }}>
              🔒 AIR-GAPPED · LOCAL SQLITE
            </div>
          </div>
        </aside>

        {/* ══════════ CENTRE: Discovery Shell ══════════ */}
        <main style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Query Panel */}
          <div style={{ ...panel, animation: "fadeIn 0.5s ease 0.05s both" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#e8e8ff", letterSpacing: "0.02em" }}>
                  Discovery Shell
                </div>
                <div style={{ fontSize: 11, color: "#6a6a9a", marginTop: 3, fontFamily: "'JetBrains Mono', monospace" }}>
                  Evaluate energy of a (problem, solution) pair
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <ScopeTag label="Dual Encoder" color="#00ffc8" />
              </div>
            </div>

            {/* Problem field */}
            <div style={{ marginBottom: 16 }}>
              <div style={labelStyle}>
                <ScopeTag label="LLM Sequence Scope" color="#00b4ff" />
                Natural Language Problem
              </div>
              <textarea
                rows={4} value={problem}
                onChange={e => setProblem(e.target.value)}
                placeholder="e.g. Prove that sin²(x) + cos²(x) = 1 for all real x …"
                style={inputStyle}
              />
            </div>

            {/* Solution field */}
            <div style={{ marginBottom: 20 }}>
              <div style={labelStyle}>
                <ScopeTag label="AST Graph Scope" color="#b060ff" />
                Mathematical Proposed Solution
              </div>
              <textarea
                rows={4} value={solution}
                onChange={e => setSolution(e.target.value)}
                placeholder="e.g. Eq(sin(x)**2 + cos(x)**2, 1)  — SymPy notation preferred"
                style={inputStyle}
              />
            </div>

            {/* Submit */}
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <button
                onClick={handleQuery}
                disabled={loading || !connected}
                style={{ ...btnPrimary, opacity: (loading || !connected) ? 0.5 : 1 }}
                onMouseEnter={e => { if (!loading) e.target.style.transform = "scale(1.02)"; }}
                onMouseLeave={e => { e.target.style.transform = "scale(1)"; }}
              >
                {loading ? <Spinner /> : "⚡"}
                {loading ? "Evaluating …" : "Evaluate Energy"}
              </button>
              {error && (
                <span style={{ fontSize: 12, color: "#ff4d6d", fontFamily: "'JetBrains Mono', monospace" }}>
                  ✗ {error}
                </span>
              )}
            </div>
          </div>

          {/* PDF Ingestion Zone */}
          <div style={{ ...panel, animation: "fadeIn 0.5s ease 0.1s both" }}>
            <div style={{ marginBottom: 16, fontSize: 12, letterSpacing: "0.12em",
              color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace" }}>
              ◈ PDF INGESTION ZONE
            </div>

            <div
              ref={dropRef}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? "#00ffc8" : "rgba(0,255,200,0.18)"}`,
                borderRadius: 10, padding: "32px 20px",
                textAlign: "center", cursor: "pointer",
                background: dragging ? "rgba(0,255,200,0.04)" : "transparent",
                transition: "all 0.25s",
              }}
            >
              <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={handleDrop} />
              <div style={{ fontSize: 28, marginBottom: 10 }}>📄</div>
              <div style={{ fontSize: 13, color: "#8888aa", fontFamily: "'JetBrains Mono', monospace" }}>
                Drop a scientific PDF here, or click to browse
              </div>
              <div style={{ fontSize: 10, color: "#555", marginTop: 6, fontFamily: "'JetBrains Mono', monospace" }}>
                Text + figures extracted · Figures saved to /figures/
              </div>
            </div>

            {pdfStatus && (
              <div style={{ marginTop: 16, animation: "fadeIn 0.35s ease" }}>
                <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#00ffc8",
                  marginBottom: 10 }}>
                  ✓ {pdfStatus.filename}
                </div>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                  <StatPill label="Characters" value={pdfStatus.char_count.toLocaleString()} />
                  <StatPill label="Figures found" value={pdfStatus.figures?.length ?? 0} color="#b060ff" />
                </div>
                {pdfStatus.figures?.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 10, color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
                      marginBottom: 6 }}>EXTRACTED FIGURES</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {pdfStatus.figures.map(f => (
                        <a key={f.id}
                          href={`${API}/figures/${f.filename}`}
                          target="_blank" rel="noreferrer"
                          style={{
                            fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                            padding: "3px 9px", borderRadius: 4,
                            background: "rgba(176,96,255,0.12)",
                            border: "1px solid rgba(176,96,255,0.25)",
                            color: "#b060ff", textDecoration: "none",
                          }}>
                          p{f.page} fig
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>

        {/* ══════════ RIGHT: Energy Render Panel ══════════ */}
        <aside>
          <div style={{ ...panel, animation: "fadeIn 0.5s ease 0.12s both" }}>
            <div style={{ fontSize: 10, letterSpacing: "0.16em", color: "#6a6a9a",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 20 }}>
              ◈ ENERGY RENDER
            </div>

            {!result && !loading && (
              <div style={{ textAlign: "center", padding: "40px 0" }}>
                <div style={{ fontSize: 36, opacity: 0.2, marginBottom: 12 }}>⚛</div>
                <div style={{ fontSize: 12, color: "#4a4a6a", fontFamily: "'JetBrains Mono', monospace" }}>
                  Submit a query to<br />visualise the energy
                </div>
              </div>
            )}

            {loading && (
              <div style={{ textAlign: "center", padding: "48px 0", display: "flex",
                flexDirection: "column", alignItems: "center", gap: 16 }}>
                <Spinner />
                <div style={{ fontSize: 11, color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
                  animation: "pulse 1.5s ease infinite" }}>
                  Running inference …
                </div>
              </div>
            )}

            {result && !loading && (
              <div style={{ animation: "fadeIn 0.4s ease" }}>
                <EnergyMeter value={result.energy} />

                <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 10 }}>
                  <ResultRow label="Query ID"  value={result.id.slice(0, 8) + "…"} mono />
                  <ResultRow label="Device"    value={result.device.toUpperCase()} mono />
                  <ResultRow label="Raw Score" value={result.energy.toFixed(6)} mono color="#00b4ff" />
                </div>

                <SoundnessBadge sound={result.sound} reason={result.sound_reason} />

                {/* Scope readout */}
                <div style={{
                  marginTop: 16, padding: "12px 14px", borderRadius: 8,
                  background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                  display: "flex", flexDirection: "column", gap: 8,
                }}>
                  <div style={{ fontSize: 10, color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
                    letterSpacing: "0.1em" }}>ENCODER TRACES</div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <ScopeTag label="LLM Sequence" color="#00b4ff" />
                    <span style={{ fontSize: 10, color: "#00b4ff88", fontFamily: "'JetBrains Mono', monospace" }}>
                      {problem.length} tokens
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <ScopeTag label="AST Graph" color="#b060ff" />
                    <span style={{ fontSize: 10, color: "#b060ff88", fontFamily: "'JetBrains Mono', monospace" }}>
                      {solution.length} chars
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => setResult(null)}
                  style={{
                    marginTop: 16, width: "100%", padding: "10px",
                    background: "transparent", border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 8, color: "#6a6a9a", cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
                    letterSpacing: "0.08em", transition: "border-color 0.2s",
                  }}
                  onMouseEnter={e => e.target.style.borderColor = "rgba(0,255,200,0.3)"}
                  onMouseLeave={e => e.target.style.borderColor = "rgba(255,255,255,0.1)"}
                >
                  CLEAR RENDER
                </button>
              </div>
            )}
          </div>
        </aside>

      </div>
    </div>
  );
}

// ─── Small helper components ───────────────────────────────────────────────────

function StatusRow({ label, active, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
      marginBottom: 10, paddingBottom: 10, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <span style={{ fontSize: 11, color: "#7a7a9a", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <GlowDot active={active} color={active ? "#00ffc8" : "#ff4d6d"} />
        <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
          color: active ? "#00ffc8" : "#ff4d6d", letterSpacing: "0.08em" }}>
          {value}
        </span>
      </div>
    </div>
  );
}

function KBStat({ icon, label, value, color = "#00ffc8" }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
      marginBottom: 8 }}>
      <span style={{ fontSize: 11, color: "#7a7a9a", fontFamily: "'JetBrains Mono', monospace",
        display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color }}>{icon}</span> {label}
      </span>
      <span style={{ fontSize: 13, fontWeight: 600, color,
        fontFamily: "'JetBrains Mono', monospace" }}>{value}</span>
    </div>
  );
}

function StatPill({ label, value, color = "#00ffc8" }) {
  return (
    <div style={{
      padding: "6px 12px", borderRadius: 6,
      background: `${color}10`, border: `1px solid ${color}25`,
      display: "flex", flexDirection: "column", gap: 2,
    }}>
      <span style={{ fontSize: 9, color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.1em" }}>{label.toUpperCase()}</span>
      <span style={{ fontSize: 16, fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace" }}>
        {value}
      </span>
    </div>
  );
}

function ResultRow({ label, value, mono, color = "#b0b0cc" }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 10, color: "#6a6a9a", fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.1em" }}>{label}</span>
      <span style={{
        fontSize: 12, color, fontFamily: mono ? "'JetBrains Mono', monospace" : "inherit",
      }}>{value}</span>
    </div>
  );
}
