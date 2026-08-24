import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth.jsx'
import { ErrorNote } from '../components/Bits.jsx'

export default function Login({ mode = 'login' }) {
  const isSignup = mode === 'signup'
  const { login, signup } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [organisation, setOrganisation] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (isSignup) await signup(email, password, organisation)
      else await login(email, password)
      navigate('/matters')
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-wrap">
      <form className="card auth-card" onSubmit={submit}>
        <Link to="/" className="muted small">← Lexegis</Link>
        <h2 style={{ marginTop: '.6rem' }}>{isSignup ? 'Create an account' : 'Sign in'}</h2>
        <p className="sub">{isSignup
          ? 'Organisations are isolated tenants. Your evidence never leaves your organisation.'
          : 'Access your matters, equation archive and audit ledger.'}</p>

        {isSignup && (
          <div className="field">
            <label htmlFor="org">Organisation</label>
            <input id="org" value={organisation} onChange={(e) => setOrganisation(e.target.value)}
                   placeholder="Chambers or firm name" required minLength={2} />
          </div>
        )}
        <div className="field">
          <label htmlFor="email">Email</label>
          <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="password">Password{isSignup && <span className="muted"> — at least 10 characters</span>}</label>
          <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                 required minLength={isSignup ? 10 : 1} />
        </div>
        <ErrorNote error={error} />
        <button className="primary" style={{ width: '100%', marginTop: '.6rem' }} disabled={busy}>
          {busy ? 'Working…' : isSignup ? 'Create account' : 'Sign in'}
        </button>
        <div className="small muted" style={{ marginTop: '.8rem' }}>
          {isSignup ? <>Already have an account? <Link to="/login">Sign in</Link></>
                    : <>No account yet? <Link to="/signup">Start free</Link></>}
        </div>
      </form>
    </div>
  )
}
