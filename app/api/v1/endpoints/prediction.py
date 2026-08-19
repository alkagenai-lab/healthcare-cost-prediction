"""
Healthcare cost prediction API endpoint.
"""

from fastapi import APIRouter

from app.models.request_schema import PredictionRequest
from app.models.response_schema import PredictionResponse
from app.services.predictor import HealthcareCostPredictor


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


predictor = HealthcareCostPredictor()


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict_cost(
    request: PredictionRequest,
):
    """
    Predict healthcare treatment cost.
    """

    predicted_cost = predictor.predict(
        age=request.age,
        sex=request.sex.value,
        bmi=request.bmi,
        children=request.children,
        smoker=request.smoker.value,
        region=request.region.value,
    )

    return PredictionResponse(
        predicted_cost=round(
            predicted_cost,
            2,
        ),
        currency="USD",
    )