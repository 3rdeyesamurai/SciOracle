const BASE = import.meta.env.VITE_API_BASE || ''

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail))
    this.status = status
    this.detail = detail
  }
}

function authHeaders() {
  const token = localStorage.getItem('lexegis.token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, { method = 'GET', body, raw } = {}) {
  const headers = { ...authHeaders() }
  let payload = body
  if (body && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  const response = await fetch(`${BASE}${path}`, { method, headers, body: payload })
  if (response.status === 204) return null
  const text = await response.text()
  let data = text
  try { data = JSON.parse(text) } catch { /* non-JSON response */ }
  if (!response.ok) throw new ApiError(response.status, data?.detail ?? data)
  return raw ? text : data
}

export const api = {
  health: () => request('/api/v1/health'),
  signup: (body) => request('/api/v1/auth/signup', { method: 'POST', body }),
  login: (body) => request('/api/v1/auth/login', { method: 'POST', body }),
  me: () => request('/api/v1/auth/me'),
  apiKeys: () => request('/api/v1/auth/api-keys'),
  createApiKey: (name) => request('/api/v1/auth/api-keys', { method: 'POST', body: { name } }),
  revokeApiKey: (id) => request(`/api/v1/auth/api-keys/${id}`, { method: 'DELETE' }),

  matters: () => request('/api/v1/matters'),
  createMatter: (body) => request('/api/v1/matters', { method: 'POST', body }),
  matter: (id) => request(`/api/v1/matters/${id}`),
  deleteMatter: (id) => request(`/api/v1/matters/${id}`, { method: 'DELETE' }),
  uploadDocument: (id, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/v1/matters/${id}/documents`, { method: 'POST', body: form })
  },
  ingestText: (id, filename, text) =>
    request(`/api/v1/matters/${id}/documents/text`, { method: 'POST', body: { filename, text } }),
  analyse: (id, body) => request(`/api/v1/matters/${id}/analyse`, { method: 'POST', body }),
  document: (id) => request(`/api/v1/documents/${id}`),
  updateFinding: (id, status) => request(`/api/v1/findings/${id}?status=${status}`, { method: 'PATCH' }),

  ledger: (limit = 200) => request(`/api/v1/ledger?limit=${limit}`),
  verifyLedger: () => request('/api/v1/ledger/verify'),

  analyseEquation: (body) => request('/api/v1/math/analyse', { method: 'POST', body }),
  compareEquations: (left, right) => request('/api/v1/math/compare', { method: 'POST', body: { left, right } }),
  extractEquations: (text) => request('/api/v1/math/extract', { method: 'POST', body: { text } }),
  archive: () => request('/api/v1/math/archive'),

  plans: () => request('/api/v1/billing/plans'),
  subscription: () => request('/api/v1/billing/subscription'),
  changePlan: (plan) => request('/api/v1/billing/subscription', { method: 'POST', body: { plan } }),

  seedDemo: () => request('/api/v1/demo/seed', { method: 'POST' }),
}
