"""
Custom application exceptions.
"""


class HealthcarePredictionException(Exception):
    """
    Base exception for the healthcare prediction application.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundException(HealthcarePredictionException):
    """
    Raised when the trained model cannot be found.
    """

    def __init__(self, message: str = "Prediction model not found."):
        super().__init__(message)


class PredictionException(HealthcarePredictionException):
    """
    Raised when prediction fails.
    """

    def __init__(self, message: str = "Healthcare cost prediction failed."):
        super().__init__(message)