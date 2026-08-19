"""
Machine Learning preprocessing pipeline
for Healthcare Treatment Cost Prediction.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "insurance.csv"


# Dataset columns
TARGET_COLUMN = "charges"

NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "children",
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoker",
    "region",
]


def load_data() -> pd.DataFrame:
    """
    Load the raw insurance dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def get_features_and_target():
    """
    Separate input features (X) and target (y).
    """

    df = load_data()

    X = df[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ]

    y = df[TARGET_COLUMN]

    return X, y


if __name__ == "__main__":

    print("Loading dataset...")

    X, y = get_features_and_target()

    print(f"Dataset loaded successfully.")
    print(f"Number of rows: {len(X)}")
    print(f"Number of features: {X.shape[1]}")

    print("\nFeatures:")
    print(list(X.columns))

    print("\nTarget:")
    print(TARGET_COLUMN)

    preprocessor = create_preprocessing_pipeline()

    print("\nPreprocessing pipeline created successfully.")
    print(preprocessor)