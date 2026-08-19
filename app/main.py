"""
Main FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.v1.router import router as api_v1_router
from app.core.logger import logger
from app.core.exceptions import HealthcarePredictionException

app = FastAPI(
    title="Healthcare Treatment Cost Prediction API",
    description=(
        "Production-style API for predicting "
        "healthcare treatment costs."
    ),
    version="1.0.0",
)
logger.info("Healthcare Treatment Cost Prediction API started")
@app.exception_handler(HealthcarePredictionException)
async def healthcare_prediction_exception_handler(
    request: Request,
    exc: HealthcarePredictionException,
):
    logger.error(
        "Application error on %s: %s",
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": exc.message,
        },
    )

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

app.include_router(
    health_router
)


# ---------------------------------------------------------
# API v1
# ---------------------------------------------------------

app.include_router(
    api_v1_router
)


@app.get(
    "/",
    tags=["Root"],
)
def root():
    """
    Root endpoint.
    """

    return {
        "message": "Healthcare Treatment Cost Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }