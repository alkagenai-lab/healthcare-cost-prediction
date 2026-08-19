"""
Health check endpoint.
"""

from fastapi import APIRouter


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
        "service": "healthcare-cost-prediction-api",
    }