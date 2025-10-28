"""
FastAPI Server for Fraud Detection

Production-ready REST API server with comprehensive endpoints for
real-time fraud detection, batch processing, and monitoring.

Author: SynFinance ML Team
Date: October 28, 2025
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import logging
from contextlib import asynccontextmanager

from .fraud_detection_api import (
    FraudDetectionAPI,
    BatchDetectionAPI,
    PredictionRequest,
    BatchPredictionRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global API instances (will be initialized in lifespan)
fraud_api: Optional[FraudDetectionAPI] = None
batch_api: Optional[BatchDetectionAPI] = None

# API metrics
api_metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'batch_requests': 0,
    'start_time': datetime.now().isoformat()
}


# Pydantic models for request/response validation
class TransactionInput(BaseModel):
    """Single transaction input schema"""
    transaction_id: str = Field(..., description="Unique transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    merchant_id: str = Field(..., description="Merchant ID")
    merchant_name: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    city: str = Field(..., description="Transaction city")
    timestamp: str = Field(..., description="Transaction timestamp (ISO format)")
    payment_mode: str = Field(..., description="Payment mode (UPI, Credit Card, etc.)")
    customer_id: Optional[str] = Field(None, description="Customer ID")
    device_id: Optional[str] = Field(None, description="Device ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    merchant_reputation: Optional[float] = Field(None, ge=0, le=1)
    distance_from_home: Optional[float] = Field(None, ge=0)
    hour: Optional[int] = Field(None, ge=0, lt=24)
    is_weekend: Optional[bool] = None
    
    @validator('payment_mode')
    def validate_payment_mode(cls, v):
        valid_modes = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash', 'Wallet']
        if v not in valid_modes:
            raise ValueError(f"Payment mode must be one of {valid_modes}")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Timestamp must be in ISO format")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_20251028_000001",
                "amount": 1500.0,
                "merchant_id": "M001",
                "merchant_name": "Amazon India",
                "category": "Shopping",
                "city": "Mumbai",
                "timestamp": "2025-10-28T14:30:00",
                "payment_mode": "UPI",
                "customer_id": "CUST0000001",
                "merchant_reputation": 0.95,
                "distance_from_home": 5.2,
                "hour": 14,
                "is_weekend": False
            }
        }


class BatchInput(BaseModel):
    """Batch prediction input schema"""
    transactions: List[Dict[str, Any]] = Field(..., description="List of transactions")
    batch_id: Optional[str] = Field(None, description="Batch ID")
    parallel: bool = Field(True, description="Enable parallel processing")
    chunk_size: int = Field(100, gt=0, description="Chunk size for processing")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if not v:
            raise ValueError("Transactions list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Maximum 10,000 transactions per batch")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN001",
                        "amount": 1500.0,
                        "merchant_id": "M001",
                        "merchant_name": "Amazon",
                        "category": "Shopping",
                        "city": "Mumbai",
                        "timestamp": "2025-10-28T14:30:00",
                        "payment_mode": "UPI"
                    }
                ],
                "batch_id": "BATCH001",
                "parallel": True,
                "chunk_size": 100
            }
        }


class PredictionOutput(BaseModel):
    """Prediction output schema"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    fraud_type: Optional[str] = None
    confidence: float
    risk_score: float
    processing_time_ms: float
    model_version: str
    timestamp: str
    features_used: Optional[List[str]] = None
    anomaly_detected: bool = False
    anomaly_score: Optional[float] = None
    recommendation: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_20251028_000001",
                "is_fraud": True,
                "fraud_probability": 0.85,
                "fraud_type": "high_value",
                "confidence": 0.85,
                "risk_score": 85.0,
                "processing_time_ms": 45.2,
                "model_version": "1.0.0",
                "timestamp": "2025-10-28T14:30:00",
                "recommendation": "REVIEW - Manual review required"
            }
        }


class BatchOutput(BaseModel):
    """Batch prediction output schema"""
    batch_id: str
    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float
    timestamp: str


class ModelInfo(BaseModel):
    """Model information schema"""
    version: str
    threshold: float
    feature_count: int
    feature_names: List[str]
    model_loaded: bool
    metadata: Dict[str, Any]


class HealthStatus(BaseModel):
    """Health check schema"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class APIMetrics(BaseModel):
    """API metrics schema"""
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    batch_requests: int
    start_time: str
    uptime_seconds: float
    fraud_detection_metrics: Dict[str, Any]


# Application lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global fraud_api, batch_api
    
    # Startup
    logger.info("Starting SynFinance Fraud Detection API...")
    
    # Initialize APIs (model will be loaded when path is provided)
    fraud_api = FraudDetectionAPI(
        model_path=None,  # Will be set via configuration or environment
        threshold=0.5,
        feature_engineering=True
    )
    
    batch_api = BatchDetectionAPI(
        model_path=None,
        threshold=0.5,
        max_workers=4
    )
    
    logger.info("API initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="SynFinance Fraud Detection API",
    description="Production-ready API for real-time fraud detection with comprehensive ML capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SynFinance Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "model_info": "/model_info",
            "health": "/health",
            "metrics": "/metrics"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud for a single transaction
    
    - **transaction_id**: Unique transaction ID
    - **amount**: Transaction amount (must be positive)
    - **merchant_id**: Merchant identifier
    - **merchant_name**: Merchant name
    - **category**: Transaction category
    - **city**: Transaction city
    - **timestamp**: Transaction timestamp (ISO format)
    - **payment_mode**: Payment mode (UPI, Credit Card, etc.)
    
    Returns fraud prediction with probability, risk score, and recommendation.
    Target response time: < 100ms
    """
    global api_metrics, fraud_api
    
    api_metrics['total_requests'] += 1
    
    try:
        # Convert to PredictionRequest
        request = PredictionRequest(**transaction.dict())
        
        # Predict
        response = fraud_api.predict(request)
        
        api_metrics['successful_predictions'] += 1
        
        # Convert to Pydantic model
        return PredictionOutput(**response.to_dict())
        
    except Exception as e:
        api_metrics['failed_predictions'] += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchOutput, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Predict fraud for batch of transactions
    
    Processes multiple transactions in parallel for improved performance.
    Maximum 10,000 transactions per batch.
    
    - **transactions**: List of transaction dictionaries
    - **batch_id**: Optional batch identifier
    - **parallel**: Enable parallel processing (default: True)
    - **chunk_size**: Processing chunk size (default: 100)
    
    Returns batch results with fraud statistics and individual predictions.
    """
    global api_metrics, batch_api
    
    api_metrics['total_requests'] += 1
    api_metrics['batch_requests'] += 1
    
    try:
        # Convert to BatchPredictionRequest
        request = BatchPredictionRequest(**batch.dict())
        
        # Predict
        response = batch_api.predict_batch(request)
        
        api_metrics['successful_predictions'] += len(response.predictions)
        
        # Convert predictions to Pydantic models
        predictions = [PredictionOutput(**p.to_dict()) for p in response.predictions]
        
        return BatchOutput(
            batch_id=response.batch_id,
            predictions=predictions,
            total_transactions=response.total_transactions,
            fraud_count=response.fraud_count,
            fraud_rate=response.fraud_rate,
            processing_time_ms=response.processing_time_ms,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        api_metrics['failed_predictions'] += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model_info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get model information
    
    Returns model version, configuration, and metadata.
    """
    global fraud_api
    
    try:
        info = fraud_api.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/health", response_model=HealthStatus, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint
    
    Returns API health status and model availability.
    """
    global fraud_api
    
    try:
        health = fraud_api.health_check()
        return HealthStatus(**health)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthStatus(
            status="unhealthy",
            model_loaded=False,
            version="unknown",
            timestamp=datetime.now().isoformat()
        )


@app.get("/metrics", response_model=APIMetrics, tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics
    
    Returns request statistics, fraud detection metrics, and performance data.
    """
    global api_metrics, fraud_api
    
    try:
        # Calculate uptime
        start_time = datetime.fromisoformat(api_metrics['start_time'])
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        # Get fraud detection metrics
        fraud_metrics = fraud_api.get_metrics()
        
        return APIMetrics(
            total_requests=api_metrics['total_requests'],
            successful_predictions=api_metrics['successful_predictions'],
            failed_predictions=api_metrics['failed_predictions'],
            batch_requests=api_metrics['batch_requests'],
            start_time=api_metrics['start_time'],
            uptime_seconds=uptime_seconds,
            fraud_detection_metrics=fraud_metrics
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
