import { useState } from 'react'
import { api } from '../lib/api.js'
import { Badge, Empty } from './Bits.jsx'

const CATEGORIES = ['all', 'in_text', 'cross_document', 'mathematical', 'epistemic']

export default function Findings({ findings, documents }) {
  const [category, setCategory] = useState('all')
  const [severity, setSeverity] = useState('all')
  const [statuses, setStatuses] = useState({})

  const titleFor = (id) => documents?.find((d) => d.id === id)?.filename || id
  const shown = findings.filter((f) =>
    (category === 'all' || f.category === category) && (severity === 'all' || f.severity === severity))

  async function setStatus(finding, value) {
    setStatuses((prev) => ({ ...prev, [finding.id]: value }))
    if (finding.id) { try { await api.updateFinding(finding.id, value) } catch { /* optimistic */ } }
  }

  return (
    <>
      <div className="row between" style={{ marginBottom: '.8rem' }}>
        <div className="row">
          {CATEGORIES.map((c) => (
            <button key={c} className={category === c ? 'primary' : 'ghost'} onClick={() => setCategory(c)}>
              {c.replace('_', ' ')}
            </button>
          ))}
        </div>
        <select value={severity} onChange={(e) => setSeverity(e.target.value)} style={{ width: 'auto' }}>
          <option value="all">all severities</option>
          {['critical', 'high', 'medium', 'low'].map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {shown.length === 0 ? <Empty>No findings in this view.</Empty> : shown.map((finding, i) => {
        const status = statuses[finding.id] || finding.status || 'open'
        return (
          <div className={`finding ${finding.severity}`} key={finding.id || i}>
            <div className="row between">
              <div className="row">
                <Badge tone={finding.severity}>{finding.severity}</Badge>
                <Badge>{finding.subtype?.replace(/_/g, ' ')}</Badge>
                <span className="muted small">{finding.category?.replace('_', ' ')} · confidence {finding.confidence}</span>
              </div>
              <div className="row">
                {status !== 'open' && <Badge tone={status === 'dismissed' ? 'low' : 'high'}>{status}</Badge>}
                <button className="ghost small" onClick={() => setStatus(finding, 'accepted')}>Accept</button>
                <button className="ghost small" onClick={() => setStatus(finding, 'dismissed')}>Dismiss</button>
              </div>
            </div>
            <div className="title">{finding.title}</div>
            <div className="detail">{finding.detail}</div>
            {(finding.spans || []).slice(0, 4).map((span, j) => (
              <div className="span-quote" key={j}>
                {span.document || titleFor(span.doc_id)}
                {span.clause ? ` · clause ${span.clause}` : ''}
                {` · chars ${span.start}–${span.end}`}
                {span.text ? `\n${span.text}` : ''}
              </div>
            ))}
            {(finding.provenance || []).length > 0 && (
              <div className="small muted mono" style={{ marginTop: '.4rem' }}>
                provenance: {finding.provenance.map((p) => Object.entries(p).map(([k, v]) => `${k}=${v}`).join(' ')).join(' | ')}
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}
