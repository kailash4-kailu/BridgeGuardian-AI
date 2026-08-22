# BridgeGuardian AI — Vercel Deployment Guide

This guide explains how to deploy **BridgeGuardian AI** to Vercel as an All-in-One full-stack application (Vite React SPA + FastAPI Python Serverless Backend).

---

## Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code** to GitHub, GitLab, or Bitbucket.
2. Go to [Vercel Dashboard](https://vercel.com/new) and click **"Add New Project"**.
3. Import your **BridgeGuardian-AI** repository.
4. **Configure Project Settings**:
   - **Framework Preset**: Select `Vite` (or `Other`).
   - **Root Directory**: `./` (Leave as repository root).
   - **Build Command**: `npm run build` (or `npm --prefix frontend install && npm --prefix frontend run build`).
   - **Output Directory**: `frontend/dist`.
5. **Environment Variables** *(Optional)*:
   - `APP_ENV`: `production`
   - `SECRET_KEY`: *Your custom secret key*
   - `JWT_SECRET_KEY`: *Your custom JWT secret key*
   - `DATABASE_URL`: *(Optional)* Managed PostgreSQL string (e.g. Neon, Supabase, Railway). If omitted, SQLite `/tmp/bridgeguardian.db` will be used.
   - `VITE_API_BASE_URL`: *(Optional)* Leave blank for Vercel Serverless Backend, or set to external backend URL if hosted separately (e.g. Render/Koyeb).
6. Click **Deploy**.

---

## Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```
2. **Log in to Vercel**:
   ```bash
   vercel login
   ```
3. **Deploy from project root**:
   ```bash
   vercel
   ```
4. **Deploy to production**:
   ```bash
   vercel --prod
   ```

---

## Architecture Overview on Vercel

```
[ Client Browser ]
        │
        ├──────► Static Frontend Routes (Vite SPA) ──► frontend/dist/index.html
        │
        └──────► /api/* API Requests ────────────────► api/index.py (FastAPI Serverless Function)
```

- **Frontend**: Vite + React 18 + TailwindCSS, compiled into `frontend/dist`.
- **Backend**: FastAPI app entrypoint at `api/index.py`, executing in Vercel Python Serverless environment.
- **Rewrites**: Handled automatically by `vercel.json`.
- **Package Limits**: `requirements.txt` is optimized under Vercel's 250 MB lambda uncompressed limit.
