"""
SynFinance API Module

Production-ready API for real-time fraud detection with comprehensive
feature engineering, model loading, and batch processing capabilities.
"""

from .fraud_detection_api import (
    FraudDetectionAPI,
    BatchDetectionAPI,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

__all__ = [
    'FraudDetectionAPI',
    'BatchDetectionAPI',
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
]
