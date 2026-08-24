import { Fragment, useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { Badge, Empty, Spinner } from '../components/Bits.jsx'

export default function Ledger() {
  const [entries, setEntries] = useState(null)
  const [head, setHead] = useState('')
  const [verification, setVerification] = useState(null)
  const [open, setOpen] = useState(null)

  useEffect(() => {
    api.ledger().then((r) => { setEntries(r.entries); setHead(r.head) }).catch(() => setEntries([]))
  }, [])

  async function verify() {
    setVerification(null)
    setVerification(await api.verifyLedger())
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Audit ledger</h1>
          <p className="sub">
            Append-only and hash-chained. Each entry commits to its payload and to the entry before it, so a
            retroactive edit anywhere in the trail is detectable — and named.
          </p>
        </div>
        <button className="primary" onClick={verify}>Verify chain</button>
      </div>

      {verification && (
        <div className="card">
          {verification.valid
            ? <><Badge tone="ok">chain intact</Badge> <span className="small muted">
                {verification.entries} entries · head {verification.head?.slice(0, 24)}…</span></>
            : <><Badge tone="critical">chain broken</Badge> <span className="small">
                first broken link at sequence {verification.broken_at}: {verification.reason}</span></>}
        </div>
      )}

      {entries === null ? <Spinner label="Loading ledger" />
        : entries.length === 0 ? <Empty>The ledger is empty.</Empty> : (
          <div className="card tight">
            <div className="small muted mono" style={{ marginBottom: '.5rem' }}>head: {head}</div>
            <table>
              <thead><tr><th>seq</th><th>action</th><th>subject</th><th>actor</th><th>entry hash</th><th>time</th><th /></tr></thead>
              <tbody>
                {entries.map((entry) => (
                  <Fragment key={entry.seq}>
                    <tr>
                      <td className="mono">{entry.seq}</td>
                      <td>{entry.action}</td>
                      <td className="mono muted small">{entry.subject || '—'}</td>
                      <td className="muted small">{entry.actor}</td>
                      <td className="mono muted small">{entry.entry_hash.slice(0, 14)}…</td>
                      <td className="muted small mono">{entry.ts.slice(0, 19).replace('T', ' ')}</td>
                      <td><button className="ghost small"
                                  onClick={() => setOpen(open === entry.seq ? null : entry.seq)}>payload</button></td>
                    </tr>
                    {open === entry.seq && (
                      <tr>
                        <td colSpan={7}><pre>{JSON.stringify(entry.payload, null, 2)}</pre></td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
    </>
  )
}
