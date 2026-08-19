"""
API v1 router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.prediction import router as prediction_router


router = APIRouter(
    prefix="/api/v1"
)

router.include_router(
    prediction_router
)