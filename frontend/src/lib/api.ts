/**
 * BridgeGuardian AI — API Base URL Utility
 *
 * Resolves the correct API base URL for the current deployment mode:
 *
 * - Docker Compose (Nginx):  VITE_API_BASE_URL is empty → relative paths used
 *                            Nginx proxies /api and /static to the backend container
 *
 * - Vercel + Render (split): VITE_API_BASE_URL = https://your-backend.onrender.com
 *                            All API calls are made directly to the Render backend
 *
 * - Local Development:       VITE_API_BASE_URL is empty → Vite proxy handles /api
 *
 * Usage:
 *   import { apiUrl } from '../lib/api'
 *   fetch(apiUrl('/api/v1/health'))
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

/**
 * Returns a fully-qualified URL for the given API path.
 * @param path - API path starting with '/' e.g. '/api/v1/health'
 */
export function apiUrl(path: string): string {
  // Trim trailing slash from base, ensure path starts with /
  const base = API_BASE.replace(/\/$/, '')
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${base}${normalizedPath}`
}

export default apiUrl
