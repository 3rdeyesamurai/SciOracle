import { useState } from 'react'
import { api } from '../lib/api.js'
import { Badge, ErrorNote, Spinner } from '../components/Bits.jsx'

const TONE = { refuted: 'critical', conditional: 'high', verified: 'low', certified: 'low', definitional: 'info' }

export default function MathLab() {
  const [tab, setTab] = useState('Adjudicate')
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Equation lab</h1>
          <p className="sub">
            Formalise a claim, canonicalise it, and decide it. Congruence is settled by equality saturation
            over an e-graph; truth is settled by SymPy, corroborated by Wolfram and certified by Lean 4 where attached.
          </p>
        </div>
      </div>
      <div className="tabs">
        {['Adjudicate', 'Compare', 'Extract from text'].map((name) => (
          <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name}</button>
        ))}
      </div>
      {tab === 'Adjudicate' && <Adjudicate />}
      {tab === 'Compare' && <Compare />}
      {tab === 'Extract from text' && <Extract />}
    </>
  )
}

function Adjudicate() {
  const [expression, setExpression] = useState('(a + b)^2 = a^2 + b^2')
  const [useLean, setUseLean] = useState(false)
  const [archive, setArchive] = useState(true)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function run() {
    setBusy(true); setError(null); setResult(null)
    try {
      setResult(await api.analyseEquation({ expression, use_lean: useLean, use_wolfram: false, archive }))
    } catch (err) { setError(err) } finally { setBusy(false) }
  }

  return (
    <div className="grid two">
      <div className="card">
        <div className="field">
          <label htmlFor="expr">Claim</label>
          <textarea id="expr" rows={3} value={expression} onChange={(e) => setExpression(e.target.value)} />
        </div>
        <div className="row">
          <label className="row small muted" style={{ margin: 0 }}>
            <input type="checkbox" checked={useLean} onChange={(e) => setUseLean(e.target.checked)} style={{ width: 'auto' }} />
            Attempt Lean 4 certification
          </label>
          <label className="row small muted" style={{ margin: 0 }}>
            <input type="checkbox" checked={archive} onChange={(e) => setArchive(e.target.checked)} style={{ width: 'auto' }} />
            Archive
          </label>
          <button className="primary" onClick={run} disabled={busy}>{busy ? 'Deciding…' : 'Adjudicate'}</button>
        </div>
        <ErrorNote error={error} />
        <p className="small muted" style={{ marginTop: '.8rem' }}>
          Try <code>R = 0.07*N + 0.03*N</code>, then <code>R = 0.1*N</code> — the second is flagged as congruent
          prior art. Or <code>(a+b)^2 = a^2 + b^2</code> for a refutation with an explicit counterexample.
        </p>
      </div>

      <div className="card">
        {busy && <Spinner label="Running System 2" />}
        {!result && !busy && <span className="muted small">Results appear here.</span>}
        {result && (
          <>
            <div className="row">
              <Badge tone={TONE[result.verification.verdict] || 'info'}>{result.verification.verdict}</Badge>
              <span className="muted small">authority: {result.verification.authority}</span>
            </div>
            <h4 style={{ marginTop: '.6rem' }}>{result.pretty}</h4>
            <div className="small muted mono">canonical key: {result.canonical_key}</div>
            <div className="small muted">
              saturation: {result.saturation.iterations} rounds, {result.saturation.enodes} e-nodes,
              {result.saturation.saturated ? ' reached fixpoint' : ' hit budget'}
            </div>

            {result.verification.backends.map((backend) => (
              <div className="span-quote" key={backend.backend}>
                <strong>{backend.backend}</strong> — {backend.status}: {backend.detail}
                {backend.counterexample && `\ncounterexample ${JSON.stringify(backend.counterexample.assignment)} → ${backend.counterexample.lhs_value} ≠ ${backend.counterexample.rhs_value}`}
              </div>
            ))}

            {result.prior_art?.length > 0 && (
              <>
                <h4 style={{ marginTop: '.8rem' }}>Congruent prior art</h4>
                {result.prior_art.map((match, i) => (
                  <div className="span-quote" key={i}>{match.source_text} — {match.detail}</div>
                ))}
              </>
            )}

            <details style={{ marginTop: '.8rem' }}>
              <summary className="small muted">Strict Content MathML</summary>
              <pre>{result.content_mathml}</pre>
            </details>
            <details>
              <summary className="small muted">OpenMath</summary>
              <pre>{result.openmath}</pre>
            </details>
            <details>
              <summary className="small muted">OMDoc</summary>
              <pre>{result.omdoc}</pre>
            </details>
            {result.verification.backends.find((b) => b.backend === 'lean4')?.source && (
              <details>
                <summary className="small muted">Lean 4 obligation</summary>
                <pre>{result.verification.backends.find((b) => b.backend === 'lean4').source}</pre>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function Compare() {
  const [left, setLeft] = useState('G = (a + b)^2')
  const [right, setRight] = useState('G = a^2 + 2*a*b + b^2')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function run() {
    setBusy(true); setError(null); setResult(null)
    try { setResult(await api.compareEquations(left, right)) } catch (err) { setError(err) } finally { setBusy(false) }
  }

  return (
    <div className="card">
      <div className="grid two">
        <div className="field">
          <label>Claim A</label>
          <textarea rows={2} value={left} onChange={(e) => setLeft(e.target.value)} />
        </div>
        <div className="field">
          <label>Claim B</label>
          <textarea rows={2} value={right} onChange={(e) => setRight(e.target.value)} />
        </div>
      </div>
      <button className="primary" onClick={run} disabled={busy}>{busy ? 'Saturating…' : 'Decide congruence'}</button>
      <ErrorNote error={error} />
      {result && (
        <div style={{ marginTop: '1rem' }}>
          <div className="row">
            <Badge tone={result.result.equivalent ? 'critical' : 'ok'}>
              {result.result.equivalent ? 'congruent' : 'no congruence derived'}
            </Badge>
            <span className="muted small">
              {result.result.iterations} rounds · {result.result.enodes} e-nodes ·
              {result.result.saturated ? ' saturated' : ' budget reached'}
            </span>
          </div>
          <p className="small" style={{ marginTop: '.6rem' }}>{result.interpretation}</p>
          <div className="grid two">
            <div className="span-quote">A canonical: {result.left.canonical_key}</div>
            <div className="span-quote">B canonical: {result.right.canonical_key}</div>
          </div>
          <details style={{ marginTop: '.6rem' }}>
            <summary className="small muted">Rules applied during saturation</summary>
            <pre>{JSON.stringify(result.result.rules_applied, null, 1)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

function Extract() {
  const [text, setText] = useState(
    'The applicant states the estimator gain as G = (a + b)^2 and asserts at paragraph 44 that G = a^2 + b^2.\n'
    + 'The royalty payable is R = 0.07*N + 0.03*N where N denotes Net Revenue.\n'
    + 'The Effective Date = 1 April 2023 is not a formula.')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function run() {
    setBusy(true); setError(null)
    try { setResult(await api.extractEquations(text)) } catch (err) { setError(err) } finally { setBusy(false) }
  }

  return (
    <div className="card">
      <div className="field">
        <label>Paste an excerpt — LaTeX, plain text or a PDF text layer</label>
        <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} />
      </div>
      <button className="primary" onClick={run} disabled={busy}>{busy ? 'Scanning…' : 'Extract equations'}</button>
      <ErrorNote error={error} />
      {result && (
        <table style={{ marginTop: '1rem' }}>
          <thead><tr><th>Source</th><th>Notation</th><th>Formalised</th><th>Offset</th></tr></thead>
          <tbody>
            {result.candidates.map((candidate, i) => (
              <tr key={i}>
                <td className="mono">{candidate.source_text}</td>
                <td className="muted small">{candidate.notation}</td>
                <td>{candidate.parsed
                  ? <span className="mono ok">{candidate.pretty}</span>
                  : <span className="muted small">{candidate.error}</span>}</td>
                <td className="muted small mono">{candidate.span.start}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
