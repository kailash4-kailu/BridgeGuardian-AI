# BridgeGuardian AI — Production Deployment Guide

## Deployment Topologies

BridgeGuardian AI supports multi-cloud production deployments (Render + Vercel) as well as self-hosted multi-container Docker Compose environments.

---

## 1. Docker Compose Deployment (Self-Hosted)

### Prerequisites
- Docker Engine 24.0+
- Docker Compose v2.20+

### Steps
1. Clone repository:
   ```bash
   git clone https://github.com/kailash4-kailu/BridgeGuardian-AI.git
   cd BridgeGuardian-AI
   ```
2. Configure `.env` environment variables:
   ```env
   APP_ENV=production
   SECRET_KEY=your-production-secret-key
   DATABASE_URL=postgresql://bridge_user:bridge_password_secure_123@db:5432/bridgeguardian
   REDIS_URL=redis://redis:6379/0
   CELERY_BROKER_URL=redis://redis:6379/1
   CELERY_RESULT_BACKEND=redis://redis:6379/2
   ```
3. Launch container stack:
   ```bash
   docker-compose up --build -d
   ```
4. Verify deployment health:
   ```bash
   curl http://localhost:8000/api/v1/health
   curl http://localhost:8000/metrics
   ```

---

## 2. Render & Vercel Multi-Cloud Deployment

### Backend (Render Web Service)
- **Runtime:** Docker (`Dockerfile.backend`)
- **Build Command:** Built automatically via Render Blueprint (`render.yaml`)
- **Environment Variables:**
  - `DATABASE_URL`: Deployed PostgreSQL connection string
  - `REDIS_URL`: Render Redis instance URL
  - `JWT_SECRET_KEY`: Production JWT signing key
  - `CORS_ORIGINS`: `https://bridge-guardian-ai.vercel.app`

### Frontend (Vercel SPA)
- **Framework Preset:** Vite
- **Root Directory:** `frontend`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Environment Variable:** `VITE_API_BASE_URL=https://bridgeguardian-backend.onrender.com`
