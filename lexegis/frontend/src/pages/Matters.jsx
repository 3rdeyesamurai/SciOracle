import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import { useAuth } from '../lib/auth.jsx'
import { Badge, Empty, ErrorNote, Spinner } from '../components/Bits.jsx'

export default function Matters() {
  const [matters, setMatters] = useState(null)
  const [name, setName] = useState('')
  const [forum, setForum] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const { refresh } = useAuth()
  const navigate = useNavigate()

  const load = useCallback(async () => {
    try { setMatters((await api.matters()).matters) } catch (err) { setError(err) }
  }, [])
  useEffect(() => { load() }, [load])

  async function create(event) {
    event.preventDefault()
    setError(null); setBusy(true)
    try {
      const matter = await api.createMatter({ name, forum: forum || null, jurisdictions: [] })
      setName(''); setForum('')
      await refresh()
      navigate(`/matters/${matter.id}`)
    } catch (err) { setError(err) } finally { setBusy(false) }
  }

  async function seedDemo() {
    setError(null); setBusy(true)
    try {
      const result = await api.seedDemo()
      await refresh()
      navigate(`/matters/${result.matter_id}`)
    } catch (err) { setError(err) } finally { setBusy(false) }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Matters</h1>
          <p className="sub">A matter is one evidentiary corpus: the instruments, exhibits and expert reports analysed together.</p>
        </div>
        <button onClick={seedDemo} disabled={busy}>Load demonstration matter</button>
      </div>

      <div className="card">
        <form className="row" onSubmit={create}>
          <div style={{ flex: '2 1 260px' }}>
            <label htmlFor="name">Matter name</label>
            <input id="name" value={name} onChange={(e) => setName(e.target.value)} required minLength={2}
                   placeholder="Helios / Meridian — licence dispute" />
          </div>
          <div style={{ flex: '1 1 180px' }}>
            <label htmlFor="forum">Forum (optional)</label>
            <input id="forum" value={forum} onChange={(e) => setForum(e.target.value)}
                   placeholder="UNCITRAL (Geneva)" />
          </div>
          <button className="primary" disabled={busy} style={{ alignSelf: 'flex-end' }}>Create matter</button>
        </form>
        <ErrorNote error={error} />
      </div>

      {matters === null ? <Spinner label="Loading matters" />
        : matters.length === 0 ? <Empty>No matters yet. Create one above, or load the demonstration corpus.</Empty>
        : (
          <div className="card tight">
            <table>
              <thead>
                <tr><th>Matter</th><th>Forum</th><th>Documents</th><th>Findings</th><th>Created</th></tr>
              </thead>
              <tbody>
                {matters.map((matter) => (
                  <tr key={matter.id}>
                    <td><Link to={`/matters/${matter.id}`}><strong>{matter.name}</strong></Link></td>
                    <td className="muted">{matter.forum || '—'}</td>
                    <td>{matter.document_count}</td>
                    <td>
                      <div className="row">
                        {Object.entries(matter.finding_counts || {}).length === 0
                          ? <span className="muted small">not analysed</span>
                          : Object.entries(matter.finding_counts).map(([severity, n]) => (
                            <Badge key={severity} tone={severity}>{severity} {n}</Badge>
                          ))}
                      </div>
                    </td>
                    <td className="muted small mono">{matter.created_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
    </>
  )
}
