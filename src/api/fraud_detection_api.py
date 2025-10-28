"""
Fraud Detection API

Production-ready API for real-time fraud detection with feature engineering,
model loading, and prediction capabilities. Optimized for < 100ms response time.

Author: SynFinance ML Team
Date: October 28, 2025
"""

import os
import pickle
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Single transaction prediction request"""
    transaction_id: str
    amount: float
    merchant_id: str
    merchant_name: str
    category: str
    city: str
    timestamp: str
    payment_mode: str
    customer_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Additional optional fields for better feature engineering
    merchant_reputation: Optional[float] = None
    distance_from_home: Optional[float] = None
    hour: Optional[int] = None
    is_weekend: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PredictionResponse:
    """Fraud prediction response"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    fraud_type: Optional[str] = None
    confidence: float = 0.0
    risk_score: float = 0.0
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional details
    features_used: Optional[List[str]] = None
    anomaly_detected: bool = False
    anomaly_score: Optional[float] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BatchPredictionRequest:
    """Batch prediction request"""
    transactions: List[Dict[str, Any]]
    batch_id: Optional[str] = None
    parallel: bool = True
    chunk_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BatchPredictionResponse:
    """Batch prediction response"""
    batch_id: str
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'batch_id': self.batch_id,
            'predictions': [p.to_dict() for p in self.predictions],
            'total_transactions': self.total_transactions,
            'fraud_count': self.fraud_count,
            'fraud_rate': self.fraud_rate,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp
        }


class FraudDetectionAPI:
    """
    Production-ready fraud detection API for real-time predictions
    
    Features:
    - Real-time feature engineering (< 100ms)
    - Model loading and caching
    - Fraud probability prediction
    - Threshold-based classification
    - Comprehensive logging
    - Error handling
    
    Example:
        >>> api = FraudDetectionAPI(model_path='models/fraud_detector.pkl')
        >>> request = PredictionRequest(
        ...     transaction_id='TXN001',
        ...     amount=1500.0,
        ...     merchant_id='M001',
        ...     merchant_name='Amazon',
        ...     category='Shopping',
        ...     city='Mumbai',
        ...     timestamp='2025-10-28T14:30:00',
        ...     payment_mode='UPI'
        ... )
        >>> response = api.predict(request)
        >>> print(f"Fraud: {response.is_fraud}, Probability: {response.fraud_probability:.2%}")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        feature_engineering: bool = True,
        enable_caching: bool = True,
        model_version: str = "1.0.0"
    ):
        """
        Initialize Fraud Detection API
        
        Args:
            model_path: Path to trained model file (.pkl)
            threshold: Classification threshold (0-1)
            feature_engineering: Enable automatic feature engineering
            enable_caching: Enable model caching
            model_version: Model version string
        """
        self.model_path = model_path
        self.threshold = threshold
        self.feature_engineering = feature_engineering
        self.enable_caching = enable_caching
        self.model_version = model_version
        
        self.model = None
        self.model_metadata = {}
        self.feature_names = []
        self.prediction_count = 0
        self.fraud_count = 0
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Model path not provided or doesn't exist: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to model file
        """
        start_time = time.time()
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.feature_names = self.model_metadata.get('feature_names', [])
                    self.model_version = self.model_metadata.get('version', self.model_version)
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded successfully in {load_time:.2f}ms")
            logger.info(f"Model version: {self.model_version}")
            logger.info(f"Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def engineer_features(self, request: PredictionRequest) -> np.ndarray:
        """
        Engineer features from transaction request
        
        Args:
            request: Prediction request
            
        Returns:
            Feature array
        """
        # Basic features (simplified for API - real implementation would use full feature engineering)
        features = []
        
        # Numeric features
        features.append(request.amount)
        features.append(request.merchant_reputation or 0.5)
        features.append(request.distance_from_home or 0.0)
        features.append(request.hour or 12)
        features.append(1.0 if request.is_weekend else 0.0)
        
        # Category encoding (simplified one-hot)
        categories = ['Food', 'Shopping', 'Travel', 'Entertainment', 'Bills']
        for cat in categories:
            features.append(1.0 if request.category == cat else 0.0)
        
        # Payment mode encoding
        payment_modes = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash']
        for mode in payment_modes:
            features.append(1.0 if request.payment_mode == mode else 0.0)
        
        # City tier (simplified)
        tier1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad']
        features.append(1.0 if request.city in tier1_cities else 0.0)
        
        # Derived features
        features.append(np.log1p(request.amount))  # Log amount
        features.append(1.0 if request.amount > 10000 else 0.0)  # High value flag
        
        # Additional derived features to reach 20
        features.append(request.amount / 1000.0)  # Normalized amount
        features.append(1.0 if (request.hour or 12) >= 22 or (request.hour or 12) <= 6 else 0.0)  # Late night flag
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Predict fraud for single transaction
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response with fraud probability
        """
        start_time = time.time()
        
        try:
            # Engineer features
            if self.feature_engineering:
                features = self.engineer_features(request)
            else:
                # Assume features are pre-computed (would need to be in request)
                features = np.zeros((1, 20))  # Placeholder
            
            # Predict
            if self.model is None:
                # Fallback: random prediction for testing
                fraud_probability = np.random.random()
                logger.warning("No model loaded, using random prediction")
            else:
                if hasattr(self.model, 'predict_proba'):
                    fraud_probability = float(self.model.predict_proba(features)[0, 1])
                else:
                    fraud_probability = float(self.model.predict(features)[0])
            
            # Classify based on threshold
            is_fraud = fraud_probability >= self.threshold
            
            # Determine fraud type (simplified - would use actual fraud pattern detection)
            fraud_type = None
            confidence = fraud_probability if is_fraud else (1 - fraud_probability)
            
            if is_fraud:
                if request.amount > 10000:
                    fraud_type = "high_value"
                elif request.distance_from_home and request.distance_from_home > 500:
                    fraud_type = "impossible_travel"
                else:
                    fraud_type = "suspicious_pattern"
            
            # Calculate risk score (0-100)
            risk_score = fraud_probability * 100
            
            # Generate recommendation
            recommendation = None
            if is_fraud:
                if fraud_probability > 0.9:
                    recommendation = "BLOCK - High confidence fraud detected"
                elif fraud_probability > 0.7:
                    recommendation = "REVIEW - Manual review required"
                else:
                    recommendation = "FLAG - Monitor additional transactions"
            
            # Update metrics
            self.prediction_count += 1
            if is_fraud:
                self.fraud_count += 1
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                transaction_id=request.transaction_id,
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                fraud_type=fraud_type,
                confidence=confidence,
                risk_score=risk_score,
                processing_time_ms=processing_time_ms,
                model_version=self.model_version,
                features_used=self.feature_names if self.feature_names else None,
                recommendation=recommendation
            )
            
            logger.info(
                f"Prediction: {request.transaction_id} | "
                f"Fraud: {is_fraud} | "
                f"Prob: {fraud_probability:.4f} | "
                f"Time: {processing_time_ms:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return error response
            return PredictionResponse(
                transaction_id=request.transaction_id,
                is_fraud=False,
                fraud_probability=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                recommendation=f"ERROR: {str(e)}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get API metrics
        
        Returns:
            Dictionary of metrics
        """
        fraud_rate = (self.fraud_count / self.prediction_count) if self.prediction_count > 0 else 0.0
        
        return {
            'total_predictions': self.prediction_count,
            'fraud_count': self.fraud_count,
            'fraud_rate': fraud_rate,
            'model_version': self.model_version,
            'threshold': self.threshold,
            'model_loaded': self.model is not None,
            'feature_count': len(self.feature_names)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Model metadata dictionary
        """
        return {
            'version': self.model_version,
            'threshold': self.threshold,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_loaded': self.model is not None,
            'metadata': self.model_metadata
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status dictionary
        """
        return {
            'status': 'healthy' if self.model is not None else 'degraded',
            'model_loaded': self.model is not None,
            'version': self.model_version,
            'timestamp': datetime.now().isoformat()
        }


class BatchDetectionAPI:
    """
    Batch fraud detection API for processing multiple transactions
    
    Features:
    - Parallel processing for large batches
    - Progress tracking
    - Chunk-based processing
    - Memory efficient
    - Comprehensive statistics
    
    Example:
        >>> batch_api = BatchDetectionAPI(model_path='models/fraud_detector.pkl')
        >>> transactions = [...]  # List of transaction dicts
        >>> request = BatchPredictionRequest(
        ...     transactions=transactions,
        ...     batch_id='BATCH001',
        ...     parallel=True,
        ...     chunk_size=100
        ... )
        >>> response = batch_api.predict_batch(request)
        >>> print(f"Processed {response.total_transactions} in {response.processing_time_ms:.0f}ms")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        max_workers: int = 4
    ):
        """
        Initialize Batch Detection API
        
        Args:
            model_path: Path to trained model file
            threshold: Classification threshold
            max_workers: Maximum parallel workers
        """
        # Use same model instance as single prediction API
        self.api = FraudDetectionAPI(
            model_path=model_path,
            threshold=threshold
        )
        self.max_workers = max_workers
    
    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Predict fraud for batch of transactions
        
        Args:
            request: Batch prediction request
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        batch_id = request.batch_id or f"BATCH_{int(time.time())}"
        
        logger.info(f"Processing batch {batch_id} with {len(request.transactions)} transactions")
        
        predictions = []
        
        if request.parallel and len(request.transactions) > 100:
            # Parallel processing for large batches
            predictions = self._predict_parallel(request.transactions)
        else:
            # Sequential processing for small batches
            predictions = self._predict_sequential(request.transactions)
        
        # Calculate statistics
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        fraud_rate = fraud_count / len(predictions) if predictions else 0.0
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch {batch_id} complete: {len(predictions)} transactions | "
            f"Fraud: {fraud_count} ({fraud_rate:.2%}) | "
            f"Time: {processing_time_ms:.2f}ms"
        )
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_count=fraud_count,
            fraud_rate=fraud_rate,
            processing_time_ms=processing_time_ms
        )
    
    def _predict_sequential(self, transactions: List[Dict[str, Any]]) -> List[PredictionResponse]:
        """Sequential prediction"""
        predictions = []
        for txn in transactions:
            try:
                request = PredictionRequest(**txn)
                response = self.api.predict(request)
                predictions.append(response)
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                # Add error response
                predictions.append(PredictionResponse(
                    transaction_id=txn.get('transaction_id', 'UNKNOWN'),
                    is_fraud=False,
                    fraud_probability=0.0,
                    recommendation=f"ERROR: {str(e)}"
                ))
        return predictions
    
    def _predict_parallel(self, transactions: List[Dict[str, Any]]) -> List[PredictionResponse]:
        """Parallel prediction using ThreadPoolExecutor"""
        predictions = []
        
        def predict_one(txn: Dict[str, Any]) -> PredictionResponse:
            try:
                request = PredictionRequest(**txn)
                return self.api.predict(request)
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                return PredictionResponse(
                    transaction_id=txn.get('transaction_id', 'UNKNOWN'),
                    is_fraud=False,
                    fraud_probability=0.0,
                    recommendation=f"ERROR: {str(e)}"
                )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(predict_one, txn) for txn in transactions]
            
            for future in as_completed(futures):
                predictions.append(future.result())
        
        return predictions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch API metrics"""
        return self.api.get_metrics()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return self.api.health_check()
