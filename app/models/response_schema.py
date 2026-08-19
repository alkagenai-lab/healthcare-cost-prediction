"""
API response schemas.
"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response returned after healthcare cost prediction.
    """

    predicted_cost: float = Field(
        ...,
        ge=0,
        description="Predicted healthcare treatment cost.",
    )

    currency: str = Field(
        default="USD",
        description="Currency of predicted cost.",
    )