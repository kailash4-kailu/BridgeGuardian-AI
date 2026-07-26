"""BridgeGuardian AI — API Router Aggregator"""
from fastapi import APIRouter
from backend.api.routes import predict, explain, train, misc, vision, inspection, auth, drift, campaigns, websocket, models

api_router = APIRouter()
api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(models.router, tags=["Model Registry"])
api_router.include_router(drift.router, tags=["ML Governance"])
api_router.include_router(campaigns.router, tags=["Inspection Campaigns"])
api_router.include_router(websocket.router, tags=["Real-Time Streaming"])
api_router.include_router(predict.router)
api_router.include_router(explain.router)
api_router.include_router(train.router)
api_router.include_router(misc.router)
api_router.include_router(vision.router)
api_router.include_router(inspection.router)
