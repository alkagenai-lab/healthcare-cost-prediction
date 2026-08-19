"""
Evaluate the trained Healthcare Treatment Cost Prediction model.
"""

from pathlib import Path

import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from pipeline import get_features_and_target


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "model.pkl"


def evaluate_model():

    print("=" * 60)
    print("Healthcare Treatment Cost Prediction")
    print("Model Evaluation")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Check model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------

    print("\n1. Loading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # 3. Load dataset
    # ---------------------------------------------------------

    print("\n2. Loading dataset...")

    X, y = get_features_and_target()

    # ---------------------------------------------------------
    # 4. Create same test split
    # ---------------------------------------------------------

    print("\n3. Creating test dataset...")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------------------------------
    # 5. Predict
    # ---------------------------------------------------------

    print("\n4. Generating predictions...")

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 6. Calculate metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred,
    )

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    print("\n5. Evaluation Results")
    print("-" * 40)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()