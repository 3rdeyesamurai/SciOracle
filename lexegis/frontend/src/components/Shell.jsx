import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth.jsx'

const LINKS = [
  ['/matters', 'Matters'],
  ['/math', 'Equation lab'],
  ['/ledger', 'Audit ledger'],
  ['/settings', 'Settings'],
]

export default function Shell({ children }) {
  const { profile, logout } = useAuth()
  const navigate = useNavigate()
  const entitlements = profile?.entitlements

  return (
    <div className="shell">
      <aside className="sidebar">
        <NavLink to="/matters" className="brand">
          Lexegis
          <small>Neuro-symbolic legal engine</small>
        </NavLink>
        <nav>
          {LINKS.map(([to, label]) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? 'active' : '')}>{label}</NavLink>
          ))}
        </nav>
        <div className="foot">
          <div>{profile?.org?.name}</div>
          <div className="muted">{profile?.user?.email}</div>
          <div className="row" style={{ marginTop: '.5rem' }}>
            <span className="badge">{entitlements?.plan_name || profile?.org?.plan}</span>
            <button className="ghost small" onClick={() => { logout(); navigate('/') }}>Sign out</button>
          </div>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  )
}
