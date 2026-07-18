"""BridgeGuardian AI — API Router"""
from fastapi import APIRouter
from backend.api.routes import predict, explain, train, misc, vision, inspection

api_router = APIRouter()
api_router.include_router(predict.router)
api_router.include_router(explain.router)
api_router.include_router(train.router)
api_router.include_router(misc.router)
api_router.include_router(vision.router)
api_router.include_router(inspection.router)
