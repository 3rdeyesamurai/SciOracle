import { Navigate, Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing.jsx'
import Login from './pages/Login.jsx'
import Matters from './pages/Matters.jsx'
import Matter from './pages/Matter.jsx'
import MathLab from './pages/MathLab.jsx'
import Ledger from './pages/Ledger.jsx'
import Settings from './pages/Settings.jsx'
import Shell from './components/Shell.jsx'
import { useAuth } from './lib/auth.jsx'

function Protected({ children }) {
  const { profile, loading } = useAuth()
  if (loading) return <div className="auth-wrap"><span className="spinner" /></div>
  if (!profile) return <Navigate to="/login" replace />
  return <Shell>{children}</Shell>
}

export default function App() {
  const { profile, loading } = useAuth()
  return (
    <Routes>
      <Route path="/" element={loading ? null : profile ? <Navigate to="/matters" replace /> : <Landing />} />
      <Route path="/login" element={profile ? <Navigate to="/matters" replace /> : <Login />} />
      <Route path="/signup" element={profile ? <Navigate to="/matters" replace /> : <Login mode="signup" />} />
      <Route path="/matters" element={<Protected><Matters /></Protected>} />
      <Route path="/matters/:matterId" element={<Protected><Matter /></Protected>} />
      <Route path="/math" element={<Protected><MathLab /></Protected>} />
      <Route path="/ledger" element={<Protected><Ledger /></Protected>} />
      <Route path="/settings" element={<Protected><Settings /></Protected>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
