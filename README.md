# BridgeGuardian AI — Production Ready Deployment Setup

BridgeGuardian AI is an enterprise-quality, explainable predictive maintenance and structural health monitoring (SHM) application for bridge infrastructure. It integrates machine learning (Scikit-Learn, XGBoost, LightGBM, CatBoost) with computer vision (YOLOv11 and SAM2 simulations) to analyze sensor telemetry and drone inspections.

---

## Project Structure

```text
├── backend/                       # FastAPI application
│   ├── api/                       # API endpoints and routers
│   ├── core/                      # Core configuration, database, and logging
│   ├── ml/                        # ML pipeline, RUL estimation, computer vision, and reporting
│   ├── static/                    # Configurable static assets (uploads, processed files, PDFs)
│   └── main.py                    # Application entry point
├── frontend/                      # React / TypeScript / Vite web interface
├── config/                        # Modular deployment configuration YAMLs
│   ├── development.yaml           # Dev environment settings
│   ├── production.yaml            # Prod environment settings
│   ├── prediction.yaml            # Feature thresholds, targets, and ML configs
│   ├── vision.yaml                # Vision parameters & limits
│   └── report.yaml                # PDF report layouts and styling
├── models/                        # Pre-trained ML model binaries (*.joblib)
├── dataset/                       # Source datasets for local training
├── Dockerfile.backend             # Production multi-worker FastAPI Dockerfile
├── Dockerfile.frontend            # Production multi-stage Nginx-served React Dockerfile
├── nginx.conf                     # Custom Nginx gateway configuration
├── docker-compose.yml             # Service orchestration configuration
├── .env.example                   # Environment variables template
└── README.md                      # General documentation
```

---

## Getting Started

### Prerequisites
- [Docker](https://www.docker.com/) (with Docker Compose support)
- *Or locally for dev:* Python 3.11+ and Node.js 20+

---

## 🚀 Docker Compose Production Deployment

The entire application compiles and launches using a single command. It orchestrates the frontend served via Nginx and the backend served via Gunicorn + Uvicorn.

### 1. Configure the Environment
Copy `.env.example` to `.env` in the root directory:
```bash
cp .env.example .env
```
*(The default settings are preconfigured for instant container launch using SQLite and Simulation Demo Mode).*

### 2. Launch Containers
```bash
docker compose up --build -d
```

Once execution finishes:
- **Frontend Dashboard:** Access at [http://localhost](http://localhost) (port 80)
- **Backend API Docs:** Access interactive Swagger docs at [http://localhost/docs](http://localhost/docs) (or [http://localhost:8000/docs](http://localhost:8000/docs))
- **SQLite Database:** Automatically initialized and written to the container volume.
- **Application Logs:** Streamed to stdout and saved directly to `logs/app.log`.

---

## 🛠️ Local Development Setup

If you prefer running the components directly on your host machine for development:

### 1. Backend Server Setup
From the project root:
1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
2. Install python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database and launch the development server (runs on port 8000 with auto-reload):
   ```bash
   python -m backend.main
   ```

### 2. Frontend Setup
From the `frontend/` directory:
1. Install dependencies:
   ```bash
   npm install
   ```
2. Run development dev server (runs on port 3000):
   ```bash
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## ⚙️ Environment Variables Config

| Key | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Active deployment stage (`production` or `development`). |
| `DATABASE_URL` | `sqlite:///./bridgeguardian.db` | Connection URI. Supports SQLite, PostgreSQL, MySQL. |
| `CORS_ORIGINS` | `http://localhost` | Allowed request origins separated by commas. |
| `LOG_LEVEL` | `WARNING` | Console/File logging filter (`INFO`, `WARNING`, `ERROR`). |
| `GUNICORN_WORKERS` | `4` | Worker process count for FastAPI server inside container. |
| `DEMO_MODE` | `true` | When `true`, activates seeded computer vision simulations if weights are absent. |
| `MAX_UPLOAD_SIZE` | `10485760` | Max file upload body payload size limit (bytes). |

---

## 🧪 Verification & Testing

### Run Pytest Suite
```bash
pytest backend/tests/
```

### Validate Frontend Compilation
From the `frontend/` directory:
```bash
npm run build
```

---

## 📄 Cloud Deployment Guide
For detailed guidance on launching to VM instances (AWS EC2, Azure VM, Google Cloud VM), Railway, DigitalOcean, or Hostinger VPS, please refer to [deployment.md](deployment.md).

