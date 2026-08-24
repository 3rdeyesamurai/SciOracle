import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { useAuth } from '../lib/auth.jsx'
import { Badge, ErrorNote, Spinner } from '../components/Bits.jsx'

export default function Settings() {
  const { profile, refresh } = useAuth()
  const [keys, setKeys] = useState(null)
  const [plans, setPlans] = useState(null)
  const [created, setCreated] = useState(null)
  const [name, setName] = useState('')
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      setKeys((await api.apiKeys()).keys)
      setPlans((await api.plans()).plans)
    } catch (err) { setError(err) }
  }, [])
  useEffect(() => { load() }, [load])

  const entitlements = profile?.entitlements

  async function createKey(event) {
    event.preventDefault()
    setError(null)
    try {
      setCreated(await api.createApiKey(name))
      setName('')
      await load()
    } catch (err) { setError(err) }
  }

  async function changePlan(plan) {
    setError(null)
    try { await api.changePlan(plan); await refresh() } catch (err) { setError(err) }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p className="sub">{profile?.org?.name} · {profile?.user?.email}</p>
        </div>
      </div>
      <ErrorNote error={error} />

      <div className="card">
        <h3>Plan and usage</h3>
        {!entitlements ? <Spinner label="Loading" /> : (
          <>
            <div className="row">
              <Badge>{entitlements.plan_name}</Badge>
              <span className="muted small">billing period {entitlements.period}</span>
            </div>
            <table style={{ marginTop: '.6rem' }}>
              <thead><tr><th>Metric</th><th>Used</th><th>Limit</th></tr></thead>
              <tbody>
                {Object.entries(entitlements.limits).map(([metric, limit]) => (
                  <tr key={metric}>
                    <td>{metric.replace(/_/g, ' ')}</td>
                    <td className={entitlements.usage[metric] >= limit ? 'err' : ''}>{entitlements.usage[metric]}</td>
                    <td className="muted">{limit.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {plans && (
          <div className="grid three" style={{ marginTop: '1rem' }}>
            {Object.entries(plans).map(([key, plan]) => (
              <div className={`price ${profile?.org?.plan === key ? 'featured' : ''}`} key={key}>
                <h4>{plan.name}</h4>
                <div className="amount">${plan.price_usd_month.toLocaleString()}<span className="muted small">/mo</span></div>
                <button style={{ width: '100%', marginTop: '.6rem' }}
                        className={profile?.org?.plan === key ? '' : 'primary'}
                        disabled={profile?.org?.plan === key}
                        onClick={() => changePlan(key)}>
                  {profile?.org?.plan === key ? 'Current plan' : `Switch to ${plan.name}`}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h3>API keys</h3>
        <p className="sub">Service credentials for the REST API. Keys are stored as digests — the plaintext is shown once.</p>
        <form className="row" onSubmit={createKey}>
          <div style={{ flex: '1 1 240px' }}>
            <label htmlFor="keyname">Key name</label>
            <input id="keyname" value={name} onChange={(e) => setName(e.target.value)} required
                   placeholder="ci-pipeline" />
          </div>
          <button className="primary" style={{ alignSelf: 'flex-end' }}>Create key</button>
        </form>

        {created && (
          <div className="span-quote" style={{ marginTop: '.8rem' }}>
            {created.api_key}
            {'\n'}{created.warning}
          </div>
        )}

        {keys === null ? <Spinner label="Loading keys" /> : (
          <table style={{ marginTop: '.8rem' }}>
            <thead><tr><th>Name</th><th>Prefix</th><th>Created</th><th>Status</th><th /></tr></thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id}>
                  <td>{key.name}</td>
                  <td className="mono muted small">{key.prefix}…</td>
                  <td className="muted small mono">{key.created_at.slice(0, 10)}</td>
                  <td>{key.revoked_at ? <Badge>revoked</Badge> : <Badge tone="ok">active</Badge>}</td>
                  <td>{!key.revoked_at && (
                    <button className="danger small"
                            onClick={async () => { await api.revokeApiKey(key.id); load() }}>Revoke</button>
                  )}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>Using the API</h3>
        <pre>{`curl -X POST https://<your-host>/api/v1/math/analyse \\
  -H "X-API-Key: lxg_..." -H "Content-Type: application/json" \\
  -d '{"expression": "(a+b)^2 = a^2 + b^2", "archive": true}'`}</pre>
        <p className="small muted">Full OpenAPI schema at <code>/docs</code> on the engine host.</p>
      </div>
    </>
  )
}
