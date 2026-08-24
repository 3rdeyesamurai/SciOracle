import { useState } from 'react'
import { Badge, Empty } from './Bits.jsx'

const TONE = { refuted: 'critical', conditional: 'high', verified: 'low', certified: 'low', definitional: 'info' }

export default function Equations({ equations, documents, unparsed = [] }) {
  const [open, setOpen] = useState(null)
  const titleFor = (id) => documents?.find((d) => d.id === id)?.filename || id

  if (!equations.length) return <Empty>No equations were recovered from this corpus.</Empty>

  return (
    <>
      <table>
        <thead>
          <tr><th>Claim</th><th>Verdict</th><th>Authority</th><th>Prior art</th><th>Source</th><th /></tr>
        </thead>
        <tbody>
          {equations.map((eq) => (
            <tr key={eq.equation_id || eq.id}>
              <td className="mono">{eq.source_text}</td>
              <td><Badge tone={TONE[eq.verdict] || 'info'}>{eq.verdict}</Badge></td>
              <td className="muted small">{eq.authority || (eq.verification?.authority ?? '—')}</td>
              <td>{(eq.prior_art || []).length > 0
                ? <Badge tone="high">{eq.prior_art.length} congruent</Badge>
                : <span className="muted small">none</span>}</td>
              <td className="muted small">{titleFor(eq.document_id)}</td>
              <td><button className="ghost small" onClick={() => setOpen(open === eq ? null : eq)}>
                {open === eq ? 'Hide' : 'Detail'}</button></td>
            </tr>
          ))}
        </tbody>
      </table>

      {open && (
        <div className="card" style={{ marginTop: '.8rem' }}>
          <h4>{open.source_text}</h4>
          <div className="small muted mono">canonical key: {open.canonical_key}</div>
          {open.latex && <pre>{open.latex}</pre>}
          {(open.prior_art || []).map((match, i) => (
            <div className="span-quote" key={i}>
              congruent to “{match.source_text}” — {match.detail} (via {match.match})
            </div>
          ))}
          <div className="small muted" style={{ marginTop: '.5rem' }}>
            backends consulted: {(open.backends || []).join(', ') || 'sympy'}
          </div>
        </div>
      )}

      {unparsed.length > 0 && (
        <div className="card tight" style={{ marginTop: '1rem' }}>
          <h4>Equation-shaped text that could not be formalised</h4>
          <p className="small muted">Archived unadjudicated rather than guessed at. A human should confirm the notation.</p>
          {unparsed.slice(0, 12).map((item, i) => (
            <div className="span-quote" key={i}>{item.source_text} — {item.reason}</div>
          ))}
        </div>
      )}
    </>
  )
}
