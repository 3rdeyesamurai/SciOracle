import { useState, useEffect, useRef } from 'react';
import './index.css';

const API_BASE = "http://localhost:8000/api";

function App() {
  const [status, setStatus] = useState({ online: false, model_loaded: false, device: 'N/A', figures_count: 0 });

  // Query State
  const [problemText, setProblemText] = useState('');
  const [solutionText, setSolutionText] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // PDF Analysis State
  const [isDragging, setIsDragging] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Poll Backend Status
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus({ online: true, ...data });
        }
      } catch (err) {
        setStatus(s => ({ ...s, online: false }));
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!problemText || !solutionText) return;

    setIsQuerying(true);
    setQueryResult(null);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem: problemText, solution: solutionText })
      });
      const data = await res.json();
      setQueryResult(data);
    } catch (err) {
      console.error(err);
      setQueryResult({ error: "Failed to connect to SciOracle Backend." });
    } finally {
      setIsQuerying(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file || file.type !== 'application/pdf') {
      alert("Please upload a valid PDF document.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setAnalysisResult(data);
      // Let's seed the query panel with extracted context
      if (data.context_preview) {
        setProblemText(data.context_preview);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to analyze document.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Drag Drop Handlers
  const onDragOver = e => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = e => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar / Dashboard */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <div className="brand-icon">Ω</div>
          <h1>SciOracle EBM</h1>
        </div>

        <div className="status-card glass-panel" style={{ background: 'rgba(0,0,0,0.3)' }}>
          <div className="status-header">
            System Core
            <div className="status-indicator">
              {status.online ? 'Online' : 'Offline'}
              <div className={`dot ${!status.online ? 'offline' : ''}`} />
            </div>
          </div>
          <div className="stat-row">
            <span>Hardware Target</span>
            <span className="stat-value" style={{ textTransform: 'uppercase' }}>{status.device}</span>
          </div>
          <div className="stat-row">
            <span>Model State</span>
            <span className="stat-value" style={{ color: status.model_loaded ? '#10b981' : '#f59e0b' }}>
              {status.model_loaded ? 'Hot-Swapped' : 'Training...'}
            </span>
          </div>
          <div className="stat-row">
            <span>Extracted Figures</span>
            <span className="stat-value">{status.figures_count} Items</span>
          </div>
        </div>

        <div style={{ marginTop: 'auto', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <p>Dual-Encoder GNN Engine enabled.</p>
          <p>Continuous Self-Improvement active.</p>
        </div>
      </aside>

      {/* Main Execution View */}
      <main className="main-content">

        {/* PDF Ingestion Zone */}
        <section
          className={`upload-zone glass-panel ${isDragging ? 'drag-active' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileUpload(e.target.files[0])}
            accept=".pdf"
            style={{ display: 'none' }}
          />
          <div className="upload-icon">📄</div>
          {isAnalyzing ? (
            <>
              <div className="loader-spinner" />
              <p>PyMuPDF Extracting Knowledge...</p>
            </>
          ) : (
            <>
              <h3>Upload PDF for Structural Analysis</h3>
              <p>Drag and drop scientific documents here to extract Graphical Figures and NLP representations.</p>
            </>
          )}
        </section>

        {analysisResult && (
          <div className="result-card glass-panel" style={{ borderLeft: '4px solid var(--accent)' }}>
            <h3>PDF Breakdown: {analysisResult.filename}</h3>
            <div className="result-grid">
              <div>
                <p style={{ color: 'var(--text-muted)' }}>Characters Evaluated</p>
                <div className="energy-score" style={{ fontSize: '1.8rem', background: 'linear-gradient(to right, #60a5fa, #3b82f6)' }}>
                  {analysisResult.characters_extracted}
                </div>
              </div>
              <div>
                <p style={{ color: 'var(--text-muted)' }}>Figures Emitted</p>
                <div className="energy-score" style={{ fontSize: '1.8rem', background: 'linear-gradient(to right, #60a5fa, #3b82f6)' }}>
                  {analysisResult.figures_extracted}
                </div>
              </div>
            </div>
            {analysisResult.figures_extracted > 0 && (
              <div style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Extracted and linked: {analysisResult.figure_names.join(', ')}
              </div>
            )}
          </div>
        )}

        {/* Math Reasoning EBM Platform */}
        <section className="query-card glass-panel">
          <h2>Interactive EBM Discovery Shell</h2>
          <form onSubmit={handleQuerySubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="input-group">
              <label htmlFor="problemText">Vibe Coding Intent (Natural Language Vibe)</label>
              <input
                id="problemText"
                type="text"
                value={problemText}
                onChange={e => setProblemText(e.target.value)}
                placeholder="e.g., I want an equation describing energy in mass."
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="solutionText">Mathematical Proposed State (AST Graph Scope)</label>
              <input
                id="solutionText"
                type="text"
                value={solutionText}
                onChange={e => setSolutionText(e.target.value)}
                placeholder="e.g., E = m*c**2"
                required
              />
            </div>

            <button type="submit" className="btn" disabled={isQuerying || !status.model_loaded}>
              {isQuerying ? <div className="loader-spinner" /> : 'Evaluate Structural Energy'}
            </button>
          </form>
        </section>

        {/* EBM Energy Render */}
        {queryResult && (
          <div className="result-card glass-panel">
            {queryResult.error ? (
              <div style={{ color: 'var(--error)' }}>
                <h4>Evaluation Failed</h4>
                <p>{queryResult.error}</p>
              </div>
            ) : (
              <>
                <h3 style={{ borderBottom: '1px solid var(--panel-border)', paddingBottom: '1rem' }}>EBM Verdict & Dual-Encoder Mapping</h3>

                <div className="result-grid">
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Predicted Landscape Energy</span>
                    <div>
                      <span className="energy-score">{queryResult.energy.toFixed(4)}</span>
                    </div>

                    <div className={`soundness-badge ${queryResult.is_sound ? 'true' : 'false'}`}>
                      {queryResult.is_sound ? '✓ Mathematically Validated' : '✗ Logic Flaw Detected'}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>NLP Extracted Math Vector</span>
                      <div className="code-block">{queryResult.problem_math}</div>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>AST Parsed Math Vector</span>
                      <div className="code-block">{queryResult.solution_math}</div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
