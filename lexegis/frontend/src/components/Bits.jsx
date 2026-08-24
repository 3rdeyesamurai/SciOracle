export function Badge({ children, tone }) {
  return <span className={`badge ${tone || ''}`}>{children}</span>
}

export function Stat({ label, value, tone }) {
  return (
    <div className="stat">
      <div className="value" style={tone ? { color: `var(--${tone})` } : undefined}>{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}

export function Empty({ children }) {
  return <div className="card muted small">{children}</div>
}

export function ErrorNote({ error }) {
  if (!error) return null
  const detail = error.detail ?? error.message
  return <div className="err">{typeof detail === 'string' ? detail : JSON.stringify(detail)}</div>
}

export function Spinner({ label }) {
  return <span className="row small muted"><span className="spinner" /> {label}</span>
}
