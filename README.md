# BridgeGuardian AI — Structural Health Monitoring & Predictive Maintenance

[![Production Status](https://img.shields.io/badge/Production-Live-success?style=for-the-badge)](https://bridge-guardian-ai.vercel.app)
[![Backend Framework](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://bridgeguardian-backend.onrender.com/docs)
[![Frontend Client](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react)](https://bridge-guardian-ai.vercel.app)

BridgeGuardian AI is an enterprise-quality, explainable predictive maintenance and structural health monitoring (SHM) application for bridge infrastructure. It integrates advanced machine learning models (Scikit-Learn, XGBoost, LightGBM, CatBoost) with computer vision to analyze sensor telemetry and segment drone inspection findings.

---

# Live Demo

The application is deployed across multi-cloud environments in a split architecture for maximum performance and cost efficiency:

*   **Frontend Dashboard:** [https://bridge-guardian-ai.vercel.app](https://bridge-guardian-ai.vercel.app) (Hosted on **Vercel**)
*   **Backend REST API:** [https://bridgeguardian-backend.onrender.com](https://bridgeguardian-backend.onrender.com) (Hosted on **Render**)
*   **API Health Endpoint:** [https://bridgeguardian-backend.onrender.com/api/v1/health](https://bridgeguardian-backend.onrender.com/api/v1/health)

### Health Endpoint Check
The API health endpoint returns a real-time status check including:
- `status`: Overall application state (`healthy` or `degraded`).
- `version`: Deployed application version.
- `model_ready`: Verification of memory-loaded machine learning models.
- `database_ok`: Connectivity status of the SQLite/relational database.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_ready": true,
  "database_ok": true
}
```

---

## Deployment Architecture

```text
       User Browser
            │
            ▼ [Vite Dev Proxy / REST Client]
     Vercel Frontend (React + Vite)
            │
            ▼ [CORS Enabled HTTPS Calls under /api/v1]
  FastAPI Backend Server (Render + Docker)
      │               │
      ▼               ▼
SQLite DB       ML inference models
                (Loaded into RAM at startup)
```

- **Frontend Client**: React + Vite SPA optimized for production.
- **Backend API**: FastAPI application served by Gunicorn + Uvicorn workers inside Docker.
- **ML Engine**: Pre-trained prediction pipelines and SHAP explainers loaded into RAM at startup.
- **Router Prefixing**: Communication occurs entirely through REST APIs under the `/api/v1` namespace.

---

## ⚠️ Important Notes

### Render Free Tier Cold Start
The backend is hosted on the Render Free Tier. Render automatically puts web services to sleep after 15 minutes of inactivity. 
- The **first request** to the application after sleeping may take **30–60 seconds** to complete.
- During this wakeup period, the frontend may temporarily report messages like `Offline`, `Loading`, or `Waiting for backend`.
- This is normal, expected behavior on the free tier. Once awake, all subsequent requests respond instantly.

### Model & Explainer Initialization
At backend startup, the server performs a self-check before accepting requests:
1. Loads Scikit-Learn, XGBoost, LightGBM, and CatBoost models from disk.
2. Initializes SHAP explainers to calculate local feature attributions.
3. The health checker marks `model_ready = true` and `database_ok = true`. If either check fails, the API status reports `degraded`.

---

## Features

- **Structural Health Prediction**: Calculates overall bridge structural health indexes (SHI) from live telemetry.
- **Failure Probability Prediction**: Evaluates real-time risk level (%) for failure modes.
- **Remaining Useful Life (RUL) Prediction**: Estimates time-to-maintenance in days with a detailed confidence range.
- **Drone Image Inspection**: Single-image upload, defect highlighting, and defect categorization.
- **Campaign Based Inspection**: Batch-upload multi-image campaigns with background processing queue and progress tracking.
- **SHAP Explainability**: Outputs feature attributions showing positive and negative sensor influences on prediction scores.
- **PDF Report Generation**: Compiles structured, inspection-campaign-wide PDF reports for offline archiving.
- **Inspection History**: Paginated repository of past sensor measurements and drone-derived analysis logs.
- **Live Dashboard**: Modern UI containing dark-mode charts, tabular inputs, and interactive visualization views.
- **REST API & Model Health Monitoring**: Full OpenAPI endpoint compliance with self-healing checks.

---

## Technology Stack

### Frontend
- **Framework & Runtime:** React 19, TypeScript, Vite 8
- **Styling:** Vanilla CSS, Tailwind CSS 4, Lucide React icons

### Backend
- **Framework:** FastAPI, Pydantic, Gunicorn, Uvicorn
- **ORM & Database:** SQLAlchemy, SQLite

### Machine Learning & Data Science
- **Tabular Models:** Scikit-Learn, XGBoost, CatBoost, LightGBM
- **Explainability:** SHAP
- **Data Manipulation:** Pandas, NumPy
- **Image Processing:** OpenCV, Pillow (PIL)

### Deployment & Infrastructure
- **Hosting Platforms:** Vercel (Frontend), Render (Backend)
- **Containerization:** Docker, Docker Compose

---

## API Endpoints Reference

All API endpoints are version-prefixed under `/api/v1`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/api/v1/health` | Returns server health status, database connection, and model readiness. |
| **GET** | `/api/v1/model-info` | Obtains loaded model metadata, feature columns, and training results. |
| **GET** | `/api/v1/history` | Retrieves a paginated history of past sensor telemetry predictions. |
| **POST** | `/api/v1/predict` | Computes health scores, failure probabilities, and RUL estimations. |
| **POST** | `/api/v1/explain` | Generates a SHAP explanation detailing feature contributions for a prediction. |
| **POST** | `/api/v1/vision/upload-image` | Uploads a drone inspection image to static storage. |
| **POST** | `/api/v1/vision/vision-predict` | Runs image inspection for defect detection and morphological segmentations. |
| **POST** | `/api/v1/inspection/upload-images` | Uploads multiple campaign-wide images to the server. |
| **POST** | `/api/v1/inspection/run-inspection` | Triggers a background campaign inspection for multiple images. |
| **GET** | `/api/v1/inspection/{id}` | Polls progress and retrieves aggregated metrics of a campaign inspection. |
| **GET** | `/api/v1/inspection/report/{id}` | Downloads the compiled PDF inspection report. |

---

## Environment Variables

### Frontend Environment Variables
Set the following environment variable in the Vercel Dashboard configuration:

- `VITE_API_BASE_URL`: The origin URL of the deployed FastAPI backend.
  - **Example:** `https://bridgeguardian-backend.onrender.com`
  - **Note:** The frontend automatically appends the `/api/v1` namespace internally. **Do not** include `/api/v1` at the end of this environment variable.

---

## Deployment Notes & Resolved Challenges

During the migration of this application from local development to production-ready cloud deployment, the following deployment challenges were successfully resolved:

1. **Docker Dependency Conflicts**: Resolved issues with dynamic library linking for image-processing libraries.
2. **NumPy & OpenCV Compatibility**: Configured explicit runtime packages (e.g. `opencv-python-headless`) to build inside lightweight Linux containers.
3. **Pandas Model Serialization**: Upgraded local package versions to ensure pickling/unpickling compatibility for ML pipelines.
4. **Model Loading Validation**: Structured a synchronous lifecyle validator at server start to guarantee models are in RAM before handling queries.
5. **CORS Configuration**: Patched FastAPI CORSMiddleware to securely route requests from Vercel origins to Render endpoints.
6. **Frontend API Routing**: Refactored the HTTP request client to support seamless transitions between dev proxies (local localhost:8000) and remote API URLs without path duplication.

---

## Usage Instructions

Follow this quick guide to run the live application:
1. Open the [Frontend Dashboard](https://bridge-guardian-ai.vercel.app).
2. Wait 30–60 seconds for the backend to wake up if this is the first interaction after a period of inactivity (due to Render's Free Tier sleep mode).
3. **Sensor Telemetry Analysis**: Fill out the sensor data forms and click **Analyze Bridge Health** to fetch tabular predictions and SHAP explainability charts.
4. **Drone Visual Analysis**: Go to the Drone Inspection tab and upload an inspection image to segment and extract cracks, rust, or bboxes.
5. **Campaign Upload**: Launch a multi-image inspection campaign and watch real-time background progress bars complete.
6. **Report Generation**: Click **Download Report** once visual segments have compiled to download an official PDF report.

---

## Production Status

- **Frontend Client:** ✅ Live on Vercel
- **Backend API:** ✅ Live on Render
- **Model Loading:** ✅ Operational
- **Database:** ✅ Connected
- **Health Endpoint:** ✅ Healthy
- **REST API:** ✅ Operational
