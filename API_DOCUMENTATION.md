# BridgeGuardian AI — REST & WebSocket API Specification

All REST endpoints are version-prefixed under `/api/v1`. OpenAPI interactive swagger documentation is available live at `/docs`.

---

## Endpoint Summary Table

| Category | Method | Route | Description | Auth / RBAC |
| :--- | :--- | :--- | :--- | :--- |
| **System** | `GET` | `/api/v1/health` | Operational health check probe | Public |
| **System** | `GET` | `/metrics` | Prometheus metrics text stream | Public |
| **Auth** | `POST` | `/api/v1/auth/register` | Register new user account | Public |
| **Auth** | `POST` | `/api/v1/auth/login` | Authenticate credentials and receive JWT tokens | Public |
| **Auth** | `POST` | `/api/v1/auth/refresh` | Exchange refresh token for new access token | Public |
| **Auth** | `POST` | `/api/v1/auth/logout` | Revoke current user session JWT token | Bearer Token |
| **Auth** | `GET` | `/api/v1/auth/me` | Fetch authenticated user profile details | Bearer Token |
| **Model Registry** | `GET` | `/api/v1/models/registry` | List registered model versions and metrics | Viewer / Inspector |
| **Model Registry** | `POST` | `/api/v1/models/rollback` | Roll back production engine to target version | Admin Only |
| **Prediction** | `POST` | `/api/v1/predict` | Predict structural health, failure risk, and RUL | Inspector / Admin |
| **Explainability** | `POST` | `/api/v1/explain` | Generate SHAP feature attributions | Inspector / Admin |
| **Vision** | `POST` | `/api/v1/vision/vision-predict` | OpenCV defect segmentation and defect measurements | Inspector / Admin |
| **Inspection** | `POST` | `/api/v1/inspection/upload-images` | Upload campaign drone image files | Inspector / Admin |
| **Inspection** | `POST` | `/api/v1/inspection/run-inspection` | Trigger async drone campaign processing | Inspector / Admin |
| **Inspection** | `GET` | `/api/v1/inspection/{id}` | Fetch campaign progress & aggregated scores | Viewer / Inspector |
| **Inspection** | `GET` | `/api/v1/inspection/report/{id}` | Download compiled PDF inspection report | Viewer / Inspector |
| **Real-Time** | `WS` | `/api/v1/ws/campaigns/{id}` | WebSocket multi-stage inspection stream | Public / Token |
