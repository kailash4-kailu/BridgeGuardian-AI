# BridgeGuardian AI — Codebase Folder Structure

```text
BridgeGuardian-AI/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD automation pipeline
├── backend/                      # FastAPI Python Backend Application
│   ├── alembic/                  # Database migration scripts & environment
│   ├── api/                      # REST & WebSocket API layer
│   │   ├── routes/
│   │   │   ├── auth.py           # JWT Authentication & user management
│   │   │   ├── campaigns.py      # Drone campaign CRUD endpoints
│   │   │   ├── drift.py          # Data drift monitoring endpoints
│   │   │   ├── explain.py        # SHAP feature explainability routes
│   │   │   ├── inspection.py     # Campaign inspections & PDF routes
│   │   │   ├── misc.py           # Model info and history pagination routes
│   │   │   ├── models.py         # Model Registry & rollback routes
│   │   │   ├── predict.py        # Tabular sensor health prediction routes
│   │   │   ├── train.py          # Background model re-training routes
│   │   │   ├── vision.py         # Computer Vision defect segmentation routes
│   │   │   └── websocket.py      # Real-time WebSocket stage streaming
│   │   ├── deps.py               # Dependency injection & RBAC roles
│   │   └── router.py             # API route aggregator
│   ├── app/                      # Application core services & workers
│   │   ├── core/                 # Exceptions & telemetry
│   │   ├── middleware/           # Security headers, tracing, rate limiting
│   │   ├── services/             # CacheService & StorageService drivers
│   │   └── tasks/                # Celery async task implementations
│   ├── core/                     # Configuration, database, models, Redis, Celery
│   │   ├── celery_app.py         # Celery instance configuration
│   │   ├── config.py             # Application settings & environment variables
│   │   ├── database.py           # SQLAlchemy database setup & connection pool
│   │   ├── logging_config.py     # Structured logging configuration
│   │   ├── models.py             # SQLAlchemy ORM database models
│   │   └── redis.py              # Redis client connection manager
│   ├── ml/                       # Inference pipeline loader & predictor
│   ├── schemas/                  # Pydantic schema envelopes & common models
│   ├── tests/                    # Pytest test suite (auth, predict, inspection, DB)
│   └── main.py                   # FastAPI application entrypoint
├── frontend/                     # React 19 + Vite 8 SPA Client
│   ├── src/
│   │   ├── components/           # DroneInspection visualizer component
│   │   ├── lib/                  # API client helper & static asset resolver
│   │   ├── App.tsx               # Main Dashboard SPA UI
│   │   └── index.css             # Design tokens & Tailwind CSS 4 setup
│   ├── vite.config.ts            # Vite bundler configuration
│   └── vercel.json               # Vercel deployment routing configuration
├── ml_pipeline/                  # ML training script and feature engineering
├── models/                       # Pre-trained ML models & SHAP explainer artifacts
├── config/                       # YAML configuration files
├── dataset/                      # Sensor dataset archives
├── Dockerfile.backend            # Production backend Dockerfile
├── Dockerfile.frontend           # Production frontend Dockerfile
├── docker-compose.yml            # Multi-container local execution manifest
├── requirements.txt              # Backend dependencies
├── ARCHITECTURE.md               # Architecture design & ASCII system diagram
├── DEPLOYMENT.md                 # Production deployment guide
├── API_DOCUMENTATION.md          # REST & WebSocket API specification
├── FOLDER_STRUCTURE.md           # Folder structure document
├── DEVELOPER_SETUP.md            # Developer setup instructions
└── README.md                     # Main project README
```
