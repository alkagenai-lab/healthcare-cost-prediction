from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration settings."""

    APP_NAME: str = "Healthcare Treatment Cost Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    MODEL_PATH: str = str(
        BASE_DIR / "ml" / "artifacts" / "model.pkl"
    )

    PREPROCESSOR_PATH: str = str(
        BASE_DIR / "ml" / "artifacts" / "preprocessor.pkl"
    )

    LOG_DIR: str = str(
        BASE_DIR / "logs"
    )

    API_PREFIX: str = "/api/v1"
    API_URL: str = "http://127.0.0.1:8000/api/v1/predict"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()