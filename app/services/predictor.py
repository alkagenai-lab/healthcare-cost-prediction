"""
Prediction service.

Loads the production model from MLflow Model Registry
using the "champion" alias and generates predictions
for new patients.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd

from app.core.exceptions import (
    ModelNotFoundException,
    PredictionException,
)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

MLFLOW_DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "mlflow.db",
    )
)

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"
)

REGISTERED_MODEL_NAME = "HealthcareCostModel"

MODEL_ALIAS = "champion"


# ---------------------------------------------------------
# Configure MLflow
# ---------------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


class HealthcareCostPredictor:
    """
    Service responsible for loading the production model
    from MLflow Model Registry and generating predictions.
    """

    def __init__(self):
        """
        Load the champion model from MLflow.
        """

        try:

            print("=" * 60)
            print("Loading Healthcare Cost Prediction Model")
            print("=" * 60)

            print(
                f"MLflow Tracking URI: "
                f"{MLFLOW_TRACKING_URI}"
            )

            print(
                f"Registered Model: "
                f"{REGISTERED_MODEL_NAME}"
            )

            print(
                f"Model Alias: "
                f"{MODEL_ALIAS}"
            )

            # -------------------------------------------------
            # Load model from MLflow Model Registry
            # -------------------------------------------------

            model_uri = (
                f"models:/{REGISTERED_MODEL_NAME}@"
                f"{MODEL_ALIAS}"
            )

            print(
                f"Model URI: {model_uri}"
            )

            self.model = mlflow.sklearn.load_model(
                model_uri
            )

            print(
                "Model loaded successfully from MLflow."
            )

            print("=" * 60)

        except Exception as exc:

            print(
                "ERROR: Unable to load model from MLflow."
            )

            raise ModelNotFoundException(
                "Production model could not be loaded "
                "from MLflow Model Registry."
            ) from exc

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ) -> float:
        """
        Predict healthcare treatment cost.
        """

        try:

            # -------------------------------------------------
            # Create input DataFrame
            # -------------------------------------------------

            input_data = pd.DataFrame(
                [
                    {
                        "age": age,
                        "sex": sex,
                        "bmi": bmi,
                        "children": children,
                        "smoker": smoker,
                        "region": region,
                    }
                ]
            )

            # -------------------------------------------------
            # Generate prediction
            # -------------------------------------------------

            prediction = self.model.predict(
                input_data
            )

            return float(
                prediction[0]
            )

        except Exception as exc:

            raise PredictionException(
                "Unable to generate healthcare cost prediction."
            ) from exc


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nStarting local prediction test...\n")

    predictor = HealthcareCostPredictor()

    result = predictor.predict(
        age=35,
        sex="female",
        bmi=28.5,
        children=2,
        smoker="no",
        region="southeast",
    )

    print("=" * 60)
    print("Healthcare Cost Prediction Test")
    print("=" * 60)

    print(
        f"Predicted healthcare cost: "
        f"${result:,.2f}"
    )

    print("=" * 60)