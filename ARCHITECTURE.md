# BridgeGuardian AI — System Architecture & Design Document

## System Architecture Diagram

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        React 19 + Vite 8 SPA Client                         │  │
│  │     • Recharts Analytics    • Lucide Icons     • Drone Inspection UI        │  │
│  └──────────────────────────────────────┬──────────────────────────────────────┘  │
└─────────────────────────────────────────┼─────────────────────────────────────────┘
                                          │ HTTPS / WSS
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                API GATEWAY LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Application Server (Docker)                    │  │
│  │  • OAuth2 / JWT Auth (Bcrypt)   • RBAC (Admin, Inspector, Viewer)           │  │
│  │  • Rate Limiting Middleware     • Security Headers & Request Tracing        │  │
│  │  • Prometheus Metrics (/metrics)• Real-Time WebSockets (/ws/campaigns)     │  │
│  └──────────────────────────────────────┬──────────────────────────────────────┘  │
└─────────────────────────────────────────┼─────────────────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
┌───────────────────────────┐ ┌───────────────────────┐ ┌───────────────────────────┐
│     DATA & CACHE LAYER    │ │    INFERENCE ENGINE   │ │     BACKGROUND WORKER     │
│ ┌───────────────────────┐ │ │ ┌───────────────────┐ │ │ ┌───────────────────────┐ │
│ │ PostgreSQL Database   │ │ │ │ RAM Loaded Models │ │ │ │ Celery Async Workers  │ │
│ │ • Users & Sessions    │ │ │ │ • XGBoost         │ │ │ │ • PDF Compilation     │ │
│ │ • Prediction Records  │ │ │ │ • LightGBM        │ │ │ │ • Multi-Image Campaign│ │
│ │ • Model Registry      │ │ │ │ • CatBoost        │ │ │ │ • Model Re-Training   │ │
│ └───────────────────────┘ │ │ │ • SHAP Explainers │ │ │ └───────────────────────┘ │
│ ┌───────────────────────┐ │ │ └───────────────────┘ │ └─────────────┬─────────────┘
│ │ Redis Cache           │ │ └───────────────────────┘               │
│ │ • Session Storage     │ │                                         │
│ │ • API Response Cache  │ │◀────────────────────────────────────────┘
│ └───────────────────────┘ │
└───────────────────────────┘
```

## Component Architecture

1. **Frontend Layer (Vercel)**: Single Page Application built with React 19, Vite 8, and Tailwind CSS 4. Handles interactive sensor forms, real-time WebSocket stage tracking, SHAP visualization bars, and OpenCV overlay visualizers.
2. **API Gateway Layer (Render / Docker)**: FastAPI web service providing REST endpoints under `/api/v1`. Implements JWT token authentication, RBAC authorization, IP rate limiting, security headers, and Prometheus telemetry metrics.
3. **Machine Learning Engine**: Singleton memory-loaded pipeline (`InferencePipeline`) evaluating sensor telemetry through XGBoost, LightGBM, CatBoost, and Scikit-Learn ensembles. Computes SHAP attributions and remaining useful life (RUL).
4. **Data Persistence Layer**: PostgreSQL relational database with SQLAlchemy ORM and Alembic migrations. Stores prediction records, model metadata, inspection campaign logs, and user credentials.
5. **Caching & Asynchronous Processing**: Redis memory store caching API data and managing Celery worker queues for long-running PDF report compilation and multi-image drone campaigns.
6. **Cloud Storage**: Cloudinary / AWS S3 abstraction driver (`StorageService`) storing high-resolution drone inspection images with public URL resolution.
