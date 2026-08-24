import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api.js'

const PIPELINE = [
  ['01', 'Forensic ingestion', 'SHA-256 fingerprint, container structure, incremental-update chain, invisible text layers, timestamp inversions.'],
  ['02', 'Default-deny triage', 'Every byte lands at tier L3. Promotion to the evidence tier requires a clean injection scan; nothing is ever promoted to the command tier.'],
  ['03', 'System 1 extraction', 'Parties, defined terms, obligations, dates and attributes — each bound to the exact character span it came from.'],
  ['04', 'Discrepancy detection', 'Ambiguity, omission and direct contradiction, within an instrument and across the corpus, plus an impossible-chronology check.'],
  ['05', 'Equation archiving', 'Formulae lifted into Strict Content MathML, OpenMath and OMDoc, canonicalised by equality saturation over an e-graph.'],
  ['06', 'System 2 adjudication', 'SymPy decides, Wolfram corroborates, Lean 4 certifies. An unavailable prover is reported as unavailable — never as a pass.'],
]

export default function Landing() {
  const [plans, setPlans] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.plans().then((r) => setPlans(r.plans)).catch(() => {})
    api.health().then(setHealth).catch(() => {})
  }, [])

  return (
    <div className="landing">
      <div className="topbar">
        <strong>Lexegis</strong>
        <div className="row">
          <a href="#how">How it works</a>
          <a href="#pricing">Pricing</a>
          <Link to="/login">Sign in</Link>
          <Link to="/signup"><button className="primary">Start free</button></Link>
        </div>
      </div>

      <section className="hero">
        <h1>Evidence that argues back gets checked, not believed.</h1>
        <p className="lede">
          Lexegis is a dual-process engine for international disputes. A fast heuristic layer reads the
          paperwork; a deterministic symbolic layer decides what is actually true — contradictions across
          instruments, impossible chronologies, and mathematical claims that are archived, canonicalised
          and formally verified rather than paraphrased.
        </p>
        <div className="row" style={{ marginTop: '1.4rem' }}>
          <Link to="/signup"><button className="primary">Create an account</button></Link>
          <Link to="/login"><button>Sign in and load the demo matter</button></Link>
        </div>
        {health && (
          <div className="row small muted" style={{ marginTop: '1.2rem' }}>
            <span className="badge ok">engine online</span>
            <span>SymPy {health.system2_backends.sympy.version}</span>
            <span>Lean 4 {health.system2_backends.lean4.available ? 'attached' : 'not attached'}</span>
            <span>Wolfram {health.system2_backends.wolfram.available ? 'attached' : 'not attached'}</span>
          </div>
        )}
      </section>

      <section id="how">
        <h2>The pipeline</h2>
        <p className="sub">Ordering is a security property, not a convenience: nothing is read for meaning
          before it has been fingerprinted, and nothing reaches a reasoning context before triage has ruled on it.</p>
        <div className="pipeline">
          {PIPELINE.map(([n, title, body]) => (
            <div className="step" key={n}>
              <div className="n">{n}</div>
              <h3>{title}</h3>
              <p className="small muted" style={{ margin: 0 }}>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section style={{ marginTop: '3rem' }}>
        <div className="grid two">
          <div className="card">
            <h3>Algebraic obfuscation does not create novelty</h3>
            <p className="small muted">
              A claim rewritten to look new is matched against the archive by equality saturation over an
              e-graph — every rewrite rule applied to every equivalent form at once, so congruence is decided
              rather than searched for.
            </p>
            <pre>{`R = 0.07·N + 0.03·N   ≡   R = 0.1·N
(a + b)²              ≡   a² + 2ab + b²
canonical key ······· (+ (* num:-1/10 var:N) var:R)`}</pre>
          </div>
          <div className="card">
            <h3>Every conclusion is a hash-chained ledger entry</h3>
            <p className="small muted">
              Forensic verdicts, triage decisions, canonicalisations and proof attempts are appended to a
              chained log. Re-verification recomputes the chain and names the first broken link, so a tampered
              audit trail cannot pass as an intact one.
            </p>
            <pre>{`GET /api/v1/ledger/verify
{ "valid": true, "entries": 41, "head": "9f3c…" }`}</pre>
          </div>
        </div>
      </section>

      <section id="pricing" style={{ marginTop: '3rem' }}>
        <h2>Pricing</h2>
        <p className="sub">Plan limits gate how much analysis you may run. They never change an analytical result.</p>
        <div className="grid three">
          {plans && Object.entries(plans).map(([key, plan]) => (
            <div className={`price ${key === 'pro' ? 'featured' : ''}`} key={key}>
              <div className="badge">{key}</div>
              <h3 style={{ marginTop: '.5rem' }}>{plan.name}</h3>
              <div className="amount">${plan.price_usd_month.toLocaleString()}<span className="muted small">/mo</span></div>
              <ul>{plan.features.map((f) => <li key={f}>{f}</li>)}</ul>
              <div className="small muted" style={{ marginTop: '.8rem' }}>
                {plan.limits.documents_per_month.toLocaleString()} documents · {plan.limits.matters} matters · {plan.limits.seats} seats
              </div>
              <Link to="/signup"><button className={key === 'pro' ? 'primary' : ''} style={{ marginTop: '.9rem', width: '100%' }}>
                {plan.price_usd_month === 0 ? 'Start free' : `Choose ${plan.name}`}
              </button></Link>
            </div>
          ))}
        </div>
      </section>

      <footer className="muted small" style={{ marginTop: '3rem', borderTop: '1px solid var(--line-soft)', paddingTop: '1rem' }}>
        Lexegis produces analytical findings for review by qualified counsel. It does not provide legal advice.
      </footer>
    </div>
  )
}
