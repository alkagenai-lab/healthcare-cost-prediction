"""
Healthcare Treatment Cost Prediction
-------------------------------------

Complete ML training + MLflow tracking + Model Registry pipeline.

Model:
    Linear Regression

Outputs:
    ml/artifacts/model.pkl
    ml/artifacts/preprocessor.pkl

MLflow:
    Experiment:
        healthcare-cost-prediction

    Registered Model:
        HealthcareCostModel

    Alias:
        champion
"""

from pathlib import Path
import sys

import joblib
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ML_DIR = BASE_DIR / "ml"

ARTIFACTS_DIR = ML_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


# ============================================================
# MLflow CONFIGURATION
# ============================================================

# SQLite database-backed MLflow tracking store.
# This avoids the MLflow 3.x file-store maintenance error.

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

EXPERIMENT_NAME = "healthcare-cost-prediction"

REGISTERED_MODEL_NAME = "HealthcareCostModel"

MODEL_ALIAS = "champion"


# ============================================================
# IMPORT PROJECT PIPELINE
# ============================================================

# ml/train.py is executed from the project root:
#
# python ml/train.py
#
# Therefore ml is added to sys.path so that pipeline.py
# can be imported reliably.

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


from pipeline import (  # noqa: E402
    create_preprocessing_pipeline,
    get_features_and_target,
)


# ============================================================
# MLflow SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():
    """
    Train the healthcare treatment cost prediction model,
    evaluate it, save local artifacts, and log/register
    the model with MLflow.
    """

    print("=" * 70)
    print("Healthcare Treatment Cost Prediction")
    print("Production ML Training Pipeline")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n1. Loading dataset...")

    X, y = get_features_and_target()

    print(f"Dataset shape : {X.shape}")
    print(f"Target shape  : {y.shape}")

    # --------------------------------------------------------
    # 2. Train-test split
    # --------------------------------------------------------

    print("\n2. Splitting dataset...")

    TEST_SIZE = 0.20
    RANDOM_STATE = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------------------------
    # 3. Create preprocessing pipeline
    # --------------------------------------------------------

    print("\n3. Creating preprocessing pipeline...")

    preprocessor = create_preprocessing_pipeline()

    print("Preprocessing pipeline created successfully.")

    # --------------------------------------------------------
    # 4. Create ML pipeline
    # --------------------------------------------------------

    print("\n4. Creating Linear Regression pipeline...")

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    print("Linear Regression pipeline created successfully.")

    # --------------------------------------------------------
    # 5. Start MLflow run
    # --------------------------------------------------------

    print("\n5. Starting MLflow run...")

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(f"MLflow Run ID: {run_id}")

        # ----------------------------------------------------
        # 6. Log parameters
        # ----------------------------------------------------

        print("\n6. Logging parameters to MLflow...")

        mlflow.log_params(
            {
                "model_type": "LinearRegression",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "feature_count": X.shape[1],
            }
        )

        # ----------------------------------------------------
        # 7. Train model
        # ----------------------------------------------------

        print("\n7. Training model...")

        model_pipeline.fit(
            X_train,
            y_train,
        )

        print("Model trained successfully.")

        # ----------------------------------------------------
        # 8. Make predictions
        # ----------------------------------------------------

        print("\n8. Making predictions...")

        y_pred = model_pipeline.predict(X_test)

        print("Predictions generated successfully.")

        # ----------------------------------------------------
        # 9. Calculate metrics
        # ----------------------------------------------------

        print("\n9. Model Evaluation")
        print("-" * 50)

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

        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")

        # ----------------------------------------------------
        # 10. Log metrics to MLflow
        # ----------------------------------------------------

        print("\n10. Logging metrics to MLflow...")

        mlflow.log_metrics(
            {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        )

        # ----------------------------------------------------
        # 11. Log useful tags
        # ----------------------------------------------------

        print("\n11. Logging MLflow tags...")

        mlflow.set_tags(
            {
                "project": "healthcare-cost-prediction",
                "model_type": "LinearRegression",
                "framework": "scikit-learn",
                "environment": "local",
                "stage": "training",
            }
        )

        # ----------------------------------------------------
        # 12. Create artifacts directory
        # ----------------------------------------------------

        print("\n12. Creating artifacts directory...")

        ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # 13. Save complete model pipeline
        # ----------------------------------------------------

        print("\n13. Saving model...")

        joblib.dump(
            model_pipeline,
            MODEL_PATH,
        )

        print(
            f"Model saved to:\n{MODEL_PATH}"
        )

        # ----------------------------------------------------
        # 14. Save preprocessor separately
        # ----------------------------------------------------

        print("\n14. Saving preprocessor...")

        joblib.dump(
            preprocessor,
            PREPROCESSOR_PATH,
        )

        print(
            f"Preprocessor saved to:\n{PREPROCESSOR_PATH}"
        )

        # ----------------------------------------------------
        # 15. Log local artifacts
        # ----------------------------------------------------

        print("\n15. Logging local artifacts to MLflow...")

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="local_artifacts",
        )

        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="local_artifacts",
        )

        # ----------------------------------------------------
        # 16. Create MLflow model signature
        # ----------------------------------------------------

        print("\n16. Creating MLflow model signature...")

        # Use a small example rather than the entire dataset.

        input_example = X_train.head(5)

        example_predictions = model_pipeline.predict(
            input_example
        )

        signature = infer_signature(
            input_example,
            example_predictions,
        )

        print("Model signature created successfully.")

        # ----------------------------------------------------
        # 17. Log and register model
        # ----------------------------------------------------

        print("\n17. Logging model to MLflow Model Registry...")

        print(
            f"Registered Model Name: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Model Alias: "
            f"{MODEL_ALIAS}"
        )

        # MLflow 3.x uses the skops serialization format
        # by default. Your previous run showed that numpy.dtype
        # needed to be explicitly trusted.

        model_info = mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="healthcare_cost_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "numpy.dtype",
            ],
            metadata={
                "project": "healthcare-cost-prediction",
                "model_type": "LinearRegression",
                "metric_mae": str(round(mae, 2)),
                "metric_rmse": str(round(rmse, 2)),
                "metric_r2": str(round(r2, 4)),
            },
        )

        print("Model logged successfully.")

        # ----------------------------------------------------
        # 18. Get registered model version
        # ----------------------------------------------------

        print("\n18. Finding registered model version...")

        client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
        )

        model_versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        # Find the model version belonging to this run.

        current_version = None

        for version in model_versions:

            if version.run_id == run_id:
                current_version = version.version
                break

        if current_version is None:

            raise RuntimeError(
                "The model was registered, but the "
                "current model version could not be found."
            )

        print(
            f"Registered model version: "
            f"{current_version}"
        )

        # ----------------------------------------------------
        # 19. Assign champion alias
        # ----------------------------------------------------

        print(
            "\n19. Assigning 'champion' alias..."
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
            str(current_version),
        )

        print(
            f"Alias '{MODEL_ALIAS}' now points to "
            f"version {current_version}."
        )

        # ----------------------------------------------------
        # 20. Log final training summary
        # ----------------------------------------------------

        mlflow.set_tag(
            "registered_model",
            REGISTERED_MODEL_NAME,
        )

        mlflow.set_tag(
            "registered_model_version",
            str(current_version),
        )

        mlflow.set_tag(
            "model_alias",
            MODEL_ALIAS,
        )

        print("\n20. MLflow Summary")
        print("-" * 50)

        print(
            f"Experiment : {EXPERIMENT_NAME}"
        )

        print(
            f"Run ID     : {run_id}"
        )

        print(
            f"Model      : {REGISTERED_MODEL_NAME}"
        )

        print(
            f"Version    : {current_version}"
        )

        print(
            f"Alias      : {MODEL_ALIAS}"
        )

        print(
            f"MAE        : {mae:.2f}"
        )

        print(
            f"MSE        : {mse:.2f}"
        )

        print(
            f"RMSE       : {rmse:.2f}"
        )

        print(
            f"R²         : {r2:.4f}"
        )

        print("\n" + "=" * 70)
        print("TRAINING + MLFLOW REGISTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return {
            "run_id": run_id,
            "model_version": current_version,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()