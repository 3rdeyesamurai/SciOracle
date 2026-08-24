import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { Badge, Spinner } from './Bits.jsx'

export default function DocumentPanel({ documentId, onClose }) {
  const [dossier, setDossier] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setDossier(null)
    api.document(documentId).then(setDossier).catch(setError)
  }, [documentId])

  if (error) return <div className="card err">{String(error.detail || error.message)}</div>
  if (!dossier) return <div className="card"><Spinner label="Opening dossier" /></div>

  const forensics = dossier.forensics || {}
  const injection = forensics.injection || {}
  const triage = dossier.triage || {}

  return (
    <div className="card">
      <div className="row between">
        <h3 style={{ margin: 0 }}>{dossier.filename}</h3>
        <button className="ghost" onClick={onClose}>Close</button>
      </div>
      <div className="row" style={{ margin: '.5rem 0' }}>
        <Badge tone={dossier.quarantined ? 'critical' : 'ok'}>tier {dossier.tier}</Badge>
        {dossier.quarantined && <Badge tone="critical">quarantined</Badge>}
        <Badge tone={injection.verdict === 'clean' ? 'ok' : injection.verdict === 'hostile' ? 'critical' : 'high'}>
          injection: {injection.verdict}
        </Badge>
        <Badge tone={forensics.max_severity === 'none' ? 'ok' : forensics.max_severity}>
          forensics: {forensics.max_severity}
        </Badge>
        <span className="muted small mono">{dossier.byte_size?.toLocaleString()} bytes</span>
      </div>

      <div className="grid two">
        <div>
          <h4>Forensic report</h4>
          <div className="small mono muted" style={{ wordBreak: 'break-all' }}>sha256 {dossier.sha256}</div>
          <div className="small muted">integrity score {forensics.integrity_score}</div>
          {(forensics.findings || []).length === 0
            ? <div className="small muted">No forensic anomalies.</div>
            : (forensics.findings || []).map((f, i) => (
              <div className="span-quote" key={i}><strong>{f.code}</strong> [{f.severity}] — {f.detail}</div>
            ))}
          {forensics.container && Object.keys(forensics.container).length > 0 && (
            <pre>{JSON.stringify(forensics.container, null, 1)}</pre>
          )}
        </div>
        <div>
          <h4>Triage decision</h4>
          <ul className="small muted">{(triage.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
          {(injection.signals || []).length > 0 && (
            <>
              <h4>Injection signals</h4>
              {injection.signals.slice(0, 5).map((s, i) => (
                <div className="span-quote" key={i}>
                  {s.signal} · weight {s.weight} · salience ×{s.salience_ratio} · offset {s.offset}
                  {'\n'}{(s.match || '').slice(0, 200)}
                </div>
              ))}
            </>
          )}
          {triage.gated_summary && (
            <>
              <h4>Gated summary (extractive)</h4>
              <div className="span-quote">{triage.gated_summary}</div>
            </>
          )}
        </div>
      </div>

      <h4 style={{ marginTop: '1rem' }}>Ledger entries for this document</h4>
      <table>
        <thead><tr><th>seq</th><th>action</th><th>hash</th><th>time</th></tr></thead>
        <tbody>
          {dossier.ledger.map((entry) => (
            <tr key={entry.seq}>
              <td className="mono">{entry.seq}</td>
              <td>{entry.action}</td>
              <td className="mono muted">{entry.entry_hash.slice(0, 16)}…</td>
              <td className="muted small mono">{entry.ts.slice(0, 19).replace('T', ' ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
