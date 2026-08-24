import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../lib/api.js'
import { Badge, Empty, ErrorNote, Spinner, Stat } from '../components/Bits.jsx'
import Findings from '../components/Findings.jsx'
import Equations from '../components/Equations.jsx'
import GraphView from '../components/GraphView.jsx'
import DocumentPanel from '../components/DocumentPanel.jsx'

const TABS = ['Findings', 'Documents', 'Chronology', 'Equations', 'Knowledge graph']
const DOC_TYPES = [['ip_licence', 'IP licence / patent'], ['commercial', 'Commercial contract'], ['treaty', 'Treaty / instrument']]

export default function Matter() {
  const { matterId } = useParams()
  const [data, setData] = useState(null)
  const [freshAnalysis, setAnalysis] = useState(null)
  const [tab, setTab] = useState('Findings')
  const [docType, setDocType] = useState('ip_licence')
  const [useLean, setUseLean] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [openDoc, setOpenDoc] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)

  const load = useCallback(async () => {
    try { setData(await api.matter(matterId)) } catch (err) { setError(err) }
  }, [matterId])
  useEffect(() => { load() }, [load])

  async function upload(files) {
    setError(null); setBusy(true)
    try {
      for (const file of files) await api.uploadDocument(matterId, file)
      await load()
    } catch (err) { setError(err) } finally { setBusy(false) }
  }

  async function runAnalysis() {
    setError(null); setBusy(true)
    try {
      setAnalysis(await api.analyse(matterId, { doc_type: docType, use_lean: useLean, use_wolfram: false }))
      await load()
    } catch (err) { setError(err) } finally { setBusy(false) }
  }

  if (error && !data) return <div className="card err">{String(error.detail || error.message)}</div>
  if (!data) return <Spinner label="Loading matter" />
  const analysis = freshAnalysis || data.analysis

  const findings = analysis?.findings || data.findings || []
  const counts = findings.reduce((acc, f) => ({ ...acc, [f.severity]: (acc[f.severity] || 0) + 1 }), {})
  const equations = analysis?.equations || (data.equations || []).map((e) => ({
    ...e, equation_id: e.id, verdict: e.verification?.verdict, authority: e.verification?.authority,
  }))

  return (
    <>
      <div className="page-head">
        <div>
          <Link to="/matters" className="muted small">← Matters</Link>
          <h1>{data.matter.name}</h1>
          <p className="sub">
            {data.matter.forum || 'no forum recorded'} · {data.documents.length} documents
            {data.matter.jurisdictions?.length ? ` · ${data.matter.jurisdictions.join(', ')}` : ''}
          </p>
        </div>
        <div className="row">
          <select value={docType} onChange={(e) => setDocType(e.target.value)} style={{ width: 'auto' }}>
            {DOC_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
          <label className="row small muted" style={{ margin: 0 }}>
            <input type="checkbox" checked={useLean} onChange={(e) => setUseLean(e.target.checked)}
                   style={{ width: 'auto' }} /> Lean 4
          </label>
          <button className="primary" onClick={runAnalysis} disabled={busy || data.documents.length === 0}>
            {busy ? 'Analysing…' : 'Run analysis'}
          </button>
        </div>
      </div>

      <ErrorNote error={error} />

      <div className="grid three" style={{ marginBottom: '1rem' }}>
        <Stat label="Critical" value={counts.critical || 0} tone="critical" />
        <Stat label="High" value={counts.high || 0} tone="high" />
        <Stat label="Medium / low" value={(counts.medium || 0) + (counts.low || 0)} />
        <Stat label="Equations archived" value={equations.length} />
        <Stat label="Quarantined documents" value={data.documents.filter((d) => d.quarantined).length}
              tone={data.documents.some((d) => d.quarantined) ? 'critical' : undefined} />
      </div>

      <div className="tabs">
        {TABS.map((name) => (
          <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name}</button>
        ))}
      </div>

      {tab === 'Findings' && (findings.length
        ? <Findings findings={findings} documents={data.documents} />
        : <Empty>No findings yet — upload documents and run the analysis.</Empty>)}

      {tab === 'Documents' && (
        <>
          <div className={`dropzone ${dragging ? 'drag' : ''}`}
               onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
               onDragLeave={() => setDragging(false)}
               onDrop={(e) => { e.preventDefault(); setDragging(false); upload(Array.from(e.dataTransfer.files)) }}
               onClick={() => fileInput.current?.click()}>
            <input ref={fileInput} type="file" multiple hidden
                   onChange={(e) => upload(Array.from(e.target.files))} />
            {busy ? <Spinner label="Ingesting" /> : 'Drop PDFs or text files here, or click to select. Every file is fingerprinted and triaged before it is read.'}
          </div>

          <div className="card tight" style={{ marginTop: '1rem' }}>
            <table>
              <thead><tr><th>File</th><th>Tier</th><th>SHA-256</th><th>Size</th><th /></tr></thead>
              <tbody>
                {data.documents.map((doc) => (
                  <tr key={doc.id}>
                    <td>{doc.filename}</td>
                    <td><Badge tone={doc.quarantined ? 'critical' : 'ok'}>
                      {doc.tier}{doc.quarantined ? ' · quarantined' : ''}</Badge></td>
                    <td className="mono muted small">{doc.sha256.slice(0, 20)}…</td>
                    <td className="muted small">{doc.byte_size.toLocaleString()} B</td>
                    <td><button className="ghost small" onClick={() => setOpenDoc(doc.id)}>Dossier</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {openDoc && <div style={{ marginTop: '1rem' }}>
            <DocumentPanel documentId={openDoc} onClose={() => setOpenDoc(null)} />
          </div>}
        </>
      )}

      {tab === 'Chronology' && (
        analysis?.chronology?.length ? (
          <div className="card">
            <div className="timeline">
              {analysis.chronology.map((event, i) => (
                <div className="event" key={i}>
                  <div className="date">{event.date}</div>
                  <div className="small">{event.assertion}</div>
                  <div className="muted small">{event.document}{event.clause ? ` · clause ${event.clause}` : ''}</div>
                </div>
              ))}
            </div>
          </div>
        ) : <Empty>Run the analysis to build the source-linked chronology.</Empty>
      )}

      {tab === 'Equations' && (
        <Equations equations={equations} documents={data.documents}
                   unparsed={analysis?.unparsed_equations || []} />
      )}

      {tab === 'Knowledge graph' && (
        analysis?.graph?.nodes?.length ? (
          <div className="card">
            <div className="row between">
              <h3 style={{ margin: 0 }}>Matter knowledge graph</h3>
              <span className="muted small">
                {analysis.graph.stats.node_count} nodes · {analysis.graph.stats.edge_count} edges
              </span>
            </div>
            <GraphView graph={analysis.graph} />
          </div>
        ) : <Empty>Run the analysis to build the knowledge graph.</Empty>
      )}
    </>
  )
}
