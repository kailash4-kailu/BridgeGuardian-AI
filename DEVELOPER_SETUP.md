# BridgeGuardian AI — Developer Onboarding & Local Setup Guide

## Quick Start Guide

### Prerequisites
- Python 3.12+
- Node.js 20+ & npm 10+
- PostgreSQL 16 (Optional, SQLite fallback supported)
- Redis 7 (Optional, memory fallback supported)

---

## Backend Setup (Python / FastAPI)

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (or rely on default local fallback):
   ```bash
   cp .env.example .env
   ```

4. Start local FastAPI dev server:
   ```bash
   python -m backend.main
   ```
   The backend server will start at `http://localhost:8000`. OpenAPI docs available at `http://localhost:8000/docs`.

---

## Frontend Setup (React 19 / Vite 8)

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start Vite dev server:
   ```bash
   npm run dev
   ```
   The frontend web application will start at `http://localhost:5173`.

---

## Running Automated Tests

Run the backend pytest suite:
```bash
pytest backend/tests -v
```
