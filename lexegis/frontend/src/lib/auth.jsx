import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api } from './api.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    if (!localStorage.getItem('lexegis.token')) {
      setProfile(null)
      setLoading(false)
      return null
    }
    try {
      const me = await api.me()
      setProfile(me)
      return me
    } catch {
      localStorage.removeItem('lexegis.token')
      setProfile(null)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const value = useMemo(() => ({
    profile,
    loading,
    refresh,
    async login(email, password) {
      const result = await api.login({ email, password })
      localStorage.setItem('lexegis.token', result.access_token)
      return refresh()
    },
    async signup(email, password, organisation) {
      const result = await api.signup({ email, password, organisation })
      localStorage.setItem('lexegis.token', result.access_token)
      return refresh()
    },
    logout() {
      localStorage.removeItem('lexegis.token')
      setProfile(null)
    },
  }), [profile, loading, refresh])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)
