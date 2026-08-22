/**
 * BridgeGuardian AI — API Base URL Utility
 *
 * Resolves the correct API base URL for the current deployment mode:
 *
 * - Vercel All-in-One (Serverless): VITE_API_BASE_URL is empty → relative paths (/api/v1) used.
 *                                    Vercel routes /api calls to /api/index.py via vercel.json rewrites.
 *
 * - Vercel / Netlify + Remote API:  VITE_API_BASE_URL set (e.g. https://your-backend.onrender.com or koyeb)
 *
 * - Local Development (Vite Dev):   VITE_API_BASE_URL is empty → relative paths proxied to 127.0.0.1:8000 by Vite
 *
 * Usage:
 *   import { apiUrl, API_BASE, getStaticUrl } from '../lib/api'
 */

const envApiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim()
const envKoyebUrl = (import.meta.env.VITE_KOYEB_URL as string | undefined)?.trim()

// If explicitly provided via VITE_API_BASE_URL or VITE_KOYEB_URL, use that.
// Otherwise, default to empty string "" to use relative path /api/v1 (works for Vercel All-In-One & Vite Dev Proxy).
const VITE_API_BASE_URL = envApiBase || envKoyebUrl || ''

export const API_BASE = VITE_API_BASE_URL ? `${VITE_API_BASE_URL.replace(/\/$/, '')}/api/v1` : '/api/v1'

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
