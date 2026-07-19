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
 *   import { apiUrl, API_BASE, getStaticUrl } from '../lib/api'
 */

const VITE_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

export const API_BASE = VITE_API_BASE_URL
  ? `${VITE_API_BASE_URL.replace(/\/$/, '')}/api/v1`
  : '/api/v1'

/**
 * Returns a fully-qualified URL for the given API path.
 * @param path - API path starting with '/' e.g. '/api/v1/health'
 */
export function apiUrl(path: string): string {
  const cleanPath = path.startsWith('/api/v1') ? path.substring(7) : path
  const normalizedPath = cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`
  return `${API_BASE}${normalizedPath}`
}

/**
 * Returns a fully-qualified URL for a static file/resource.
 * @param path - Static path starting with '/' (e.g., '/static/uploads/file.png')
 */
export function getStaticUrl(path: string): string {
  if (!path) return ''
  if (path.startsWith('data:') || path.startsWith('blob:') || path.startsWith('http:') || path.startsWith('https:')) {
    return path
  }
  const base = VITE_API_BASE_URL.replace(/\/$/, '')
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${base}${normalizedPath}`
}

export default apiUrl

