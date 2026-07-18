import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],
  // In production builds, VITE_API_BASE_URL is baked in via import.meta.env
  // In Docker Compose, nginx proxies /api to backend — leave base as '/'
  base: '/',
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  server: {
    port: 3000,
    // Dev proxy: routes /api calls to the backend during local development
    // In production (Vercel), frontend calls VITE_API_BASE_URL directly
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/static': {
        target: process.env.VITE_API_BASE_URL || 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
}))
