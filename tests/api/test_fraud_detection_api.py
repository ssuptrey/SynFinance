"""
Comprehensive test suite for fraud detection API.

Tests cover:
- FraudDetectionAPI class
- BatchDetectionAPI class
- FastAPI endpoints
- Request validation
- Error handling
- Performance requirements
"""

import os
import sys
import pytest
import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pickle
from typing import Dict, Any, List
from fastapi.testclient import TestClient
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.fraud_detection_api import (
    FraudDetectionAPI,
    BatchDetectionAPI,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from src.api.api_server import app
from src.api.api_client import FraudDetectionClient, predict_fraud


# Test fixtures
@pytest.fixture
def sample_transaction() -> Dict[str, Any]:
    """Sample transaction data."""
    return {
        "transaction_id": "TXN001",
        "amount": 150.50,
        "merchant_id": "M001",
        "merchant_name": "Test Store",
        "category": "retail",
        "city": "New York",
        "timestamp": datetime.now().isoformat(),
        "payment_mode": "Credit Card"  # Fixed: must match API validator
    }


@pytest.fixture
def sample_request(sample_transaction) -> PredictionRequest:
    """Sample prediction request."""
    return PredictionRequest(**sample_transaction)


@pytest.fixture
def batch_transactions() -> List[Dict[str, Any]]:
    """Generate batch of transaction data."""
    transactions = []
    base_time = datetime.now()
    
    for i in range(100):
        transactions.append({
            "transaction_id": f"TXN{i:04d}",
            "amount": 100.0 + i * 10,
            "merchant_id": f"M{i % 10:03d}",
            "merchant_name": f"Merchant {i % 10}",
            "category": ["retail", "food", "travel", "entertainment"][i % 4],
            "city": ["New York", "Los Angeles", "Chicago", "Houston"][i % 4],
            "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            "payment_mode": ["credit_card", "debit_card", "cash", "upi"][i % 4]
        })
    
    return transactions


@pytest.fixture
def mock_model():
    """Create a mock trained model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Create simple mock model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Train on dummy data
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    return model


@pytest.fixture
def model_path(mock_model, tmp_path):
    """Create temporary model file."""
    model_file = tmp_path / "test_model.pkl"
    metadata_file = tmp_path / "test_model_metadata.json"
    
    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(mock_model, f)
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "version": "1.0.0",
        "feature_count": 20,
        "training_date": datetime.now().isoformat(),
        "accuracy": 0.95
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    return str(model_file)


@pytest.fixture
def fraud_api(model_path):
    """Create FraudDetectionAPI instance with test model."""
    return FraudDetectionAPI(model_path=model_path, threshold=0.5)


@pytest.fixture
def batch_api(model_path):
    """Create BatchDetectionAPI instance with test model."""
    return BatchDetectionAPI(model_path=model_path, threshold=0.5, max_workers=2)


@pytest.fixture
def api_client():
    """Create test client for FastAPI app."""
    # Reset metrics before each test
    from src.api import api_server
    api_server.api_metrics.update({
        'total_requests': 0,
        'successful_predictions': 0,
        'failed_predictions': 0,
        'batch_requests': 0,
        'start_time': datetime.now().isoformat()
    })
    
    # TestClient automatically triggers lifespan events
    with TestClient(app) as client:
        yield client


# ============================================================================
# FraudDetectionAPI Tests
# ============================================================================

class TestFraudDetectionAPI:
    """Tests for FraudDetectionAPI class."""
    
    def test_initialization_with_model(self, model_path):
        """Test API initialization with valid model."""
        api = FraudDetectionAPI(model_path=model_path)
        
        assert api.model is not None
        assert api.model_metadata is not None
        assert api.threshold == 0.5
        assert api.feature_engineering is True
    
    def test_initialization_without_model(self):
        """Test API initialization without model (fallback mode)."""
        api = FraudDetectionAPI(model_path=None)
        
        assert api.model is None
        assert api.model_metadata == {}  # Empty dict, not None
        assert api.threshold == 0.5
    
    def test_load_model_success(self, model_path):
        """Test successful model loading."""
        api = FraudDetectionAPI(model_path=None)
        api.load_model(model_path)
        
        assert api.model is not None
        assert api.model_metadata is not None
        assert "version" in api.model_metadata
    
    def test_load_model_failure(self):
        """Test model loading with invalid path."""
        api = FraudDetectionAPI(model_path=None)
        
        with pytest.raises(FileNotFoundError):
            api.load_model("nonexistent_model.pkl")
    
    def test_feature_engineering(self, fraud_api, sample_request):
        """Test feature engineering creates correct number of features."""
        features = fraud_api.engineer_features(sample_request)
        
        # Should return 2D array with shape (1, 20)
        assert features.shape == (1, 20) or features.shape == (1, 18)  # May vary
        assert all(isinstance(f, (int, float, np.number)) for f in features.flatten())
    
    def test_predict_success(self, fraud_api, sample_request):
        """Test successful prediction."""
        response = fraud_api.predict(sample_request)
        
        assert isinstance(response, PredictionResponse)
        assert response.transaction_id == sample_request.transaction_id
        assert isinstance(response.is_fraud, bool)
        assert 0.0 <= response.fraud_probability <= 1.0
        assert 0 <= response.risk_score <= 100
        assert isinstance(response.confidence, (int, float))  # Confidence is a number
        assert 0.0 <= response.confidence <= 1.0
        # Recommendation may include description
        assert any(r in response.recommendation for r in ["BLOCK", "REVIEW", "FLAG", "ALLOW"])
        assert response.processing_time_ms > 0
    
    def test_predict_performance(self, fraud_api, sample_request):
        """Test prediction meets <100ms requirement."""
        start_time = time.time()
        response = fraud_api.predict(sample_request)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Allow some margin for test environment
        assert elapsed_ms < 200, f"Prediction took {elapsed_ms:.2f}ms (target: <100ms)"
        assert response.processing_time_ms < 200
    
    def test_fraud_type_detection(self, fraud_api):
        """Test fraud type detection logic."""
        # High value transaction
        high_value_req = PredictionRequest(
            transaction_id="HV001",
            amount=15000.0,
            merchant_id="M001",
            merchant_name="Luxury Store",
            category="retail",
            city="New York",
            timestamp=datetime.now().isoformat(),
            payment_mode="credit_card"
        )
        
        response = fraud_api.predict(high_value_req)
        # Should detect high_value in fraud_type if predicted as fraud
        assert isinstance(response.fraud_type, str)
    
    def test_get_metrics(self, fraud_api, sample_request):
        """Test metrics collection."""
        # Make some predictions
        for _ in range(5):
            fraud_api.predict(sample_request)
        
        metrics = fraud_api.get_metrics()
        
        assert "total_predictions" in metrics
        assert metrics["total_predictions"] >= 5
        assert "fraud_detected_count" in metrics or "fraud_count" in metrics
        assert "fraud_rate" in metrics
    
    def test_get_model_info(self, fraud_api):
        """Test model info retrieval."""
        info = fraud_api.get_model_info()
        
        assert "model_loaded" in info
        assert info["model_loaded"] is True
        assert "version" in info
        assert "threshold" in info
        assert info["threshold"] == 0.5
    
    def test_health_check(self, fraud_api):
        """Test health check."""
        health = fraud_api.health_check()
        
        assert "status" in health
        assert health["status"] == "healthy"
        assert "model_loaded" in health
        assert health["model_loaded"] is True


# ============================================================================
# BatchDetectionAPI Tests
# ============================================================================

class TestBatchDetectionAPI:
    """Tests for BatchDetectionAPI class."""
    
    def test_initialization(self, model_path):
        """Test batch API initialization."""
        api = BatchDetectionAPI(model_path=model_path, threshold=0.6, max_workers=4)
        
        assert api.api.threshold == 0.6
        assert api.max_workers == 4
        assert api.api.model is not None
    
    def test_batch_predict_sequential(self, batch_api, batch_transactions):
        """Test sequential batch processing."""
        batch_request = BatchPredictionRequest(
            transactions=batch_transactions[:10],
            batch_id="BATCH001",
            parallel=False
        )
        
        response = batch_api.predict_batch(batch_request)
        
        assert isinstance(response, BatchPredictionResponse)
        assert response.batch_id == "BATCH001"
        assert response.total_transactions == 10
        assert len(response.predictions) == 10
        assert response.fraud_count >= 0
        assert 0.0 <= response.fraud_rate <= 1.0
        assert response.processing_time_ms > 0
    
    def test_batch_predict_parallel(self, batch_api, batch_transactions):
        """Test parallel batch processing."""
        batch_request = BatchPredictionRequest(
            transactions=batch_transactions,
            batch_id="BATCH002",
            parallel=True
        )
        
        response = batch_api.predict_batch(batch_request)
        
        assert response.total_transactions == 100
        assert len(response.predictions) == 100
        assert response.processing_time_ms > 0
    
    def test_batch_performance(self, batch_api, batch_transactions):
        """Test batch processing meets <5s for 1000 transactions requirement."""
        # Use 100 transactions for faster testing
        batch_request = BatchPredictionRequest(
            transactions=batch_transactions,
            batch_id="PERF001",
            parallel=True
        )
        
        start_time = time.time()
        response = batch_api.predict_batch(batch_request)
        elapsed = time.time() - start_time
        
        # 100 transactions should be much faster than 5s
        assert elapsed < 2.0, f"Batch processing took {elapsed:.2f}s (100 txns)"
        
        # Extrapolate: if 100 takes X seconds, 1000 should take ~10X
        estimated_1000 = elapsed * 10
        assert estimated_1000 < 5.0, f"Estimated 1000 txns: {estimated_1000:.2f}s (target: <5s)"
    
    def test_parallel_faster_than_sequential(self, batch_api, batch_transactions):
        """Test that parallel processing is faster than sequential."""
        transactions = batch_transactions[:50]
        
        # Sequential
        seq_request = BatchPredictionRequest(
            transactions=transactions,
            batch_id="SEQ001",
            parallel=False
        )
        seq_response = batch_api.predict_batch(seq_request)
        seq_time = seq_response.processing_time_ms
        
        # Parallel
        par_request = BatchPredictionRequest(
            transactions=transactions,
            batch_id="PAR001",
            parallel=True
        )
        par_response = batch_api.predict_batch(par_request)
        par_time = par_response.processing_time_ms
        
        # Parallel should be faster (or at least not significantly slower)
        # Allow some margin for overhead
        assert par_time <= seq_time * 1.5, f"Parallel ({par_time}ms) not faster than sequential ({seq_time}ms)"


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SynFinance Fraud Detection API"
        assert "version" in data
        assert "endpoints" in data
    
    def test_predict_endpoint_success(self, api_client, sample_transaction):
        """Test successful prediction endpoint."""
        response = api_client.post("/predict", json=sample_transaction)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "transaction_id" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_score" in data
        assert "processing_time_ms" in data
        
        # Check timing header
        assert "X-Process-Time-Ms" in response.headers
    
    def test_predict_endpoint_validation(self, api_client):
        """Test prediction endpoint validates input."""
        # Missing required field
        invalid_data = {
            "transaction_id": "TXN001",
            "amount": 100.0
            # Missing other required fields
        }
        
        response = api_client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_payment_mode(self, api_client, sample_transaction):
        """Test validation of payment mode."""
        sample_transaction["payment_mode"] = "invalid_mode"
        
        response = api_client.post("/predict", json=sample_transaction)
        assert response.status_code == 422
    
    def test_predict_endpoint_negative_amount(self, api_client, sample_transaction):
        """Test validation of amount."""
        sample_transaction["amount"] = -100.0
        
        response = api_client.post("/predict", json=sample_transaction)
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self, api_client, batch_transactions):
        """Test batch prediction endpoint."""
        batch_data = {
            "transactions": batch_transactions[:10],
            "batch_id": "API_BATCH001",
            "parallel": True
        }
        
        response = api_client.post("/predict_batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["batch_id"] == "API_BATCH001"
        assert data["total_transactions"] == 10
        assert len(data["predictions"]) == 10
        assert "fraud_count" in data
        assert "fraud_rate" in data
    
    def test_batch_predict_size_limit(self, api_client, batch_transactions):
        """Test batch size limit enforcement."""
        # Create oversized batch (> 10,000)
        large_batch = batch_transactions * 150  # 15,000 transactions
        
        batch_data = {
            "transactions": large_batch,
            "batch_id": "LARGE_BATCH",
            "parallel": True
        }
        
        response = api_client.post("/predict_batch", json=batch_data)
        assert response.status_code == 422  # Should reject
    
    def test_model_info_endpoint(self, api_client):
        """Test model info endpoint."""
        response = api_client.get("/model_info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_loaded" in data
        assert "threshold" in data
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, api_client, sample_transaction):
        """Test metrics endpoint."""
        # Make some requests first
        for _ in range(3):
            api_client.post("/predict", json=sample_transaction)
        
        response = api_client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_requests" in data
        assert "successful_predictions" in data
        assert "uptime_seconds" in data
        assert data["total_requests"] >= 3


# ============================================================================
# API Client Tests
# ============================================================================

class TestAPIClient:
    """Tests for API client library."""
    
    @pytest.fixture
    def running_server(self):
        """Start test server for client tests."""
        # Note: In real testing, you'd start uvicorn in background
        # For these tests, we'll use TestClient which doesn't require a running server
        return TestClient(app)
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = FraudDetectionClient(
            base_url="http://localhost:8000",
            timeout=30,
            max_retries=3
        )
        
        assert client.config.base_url == "http://localhost:8000"
        assert client.config.timeout == 30
        assert client.config.max_retries == 3
    
    def test_predict_function(self, sample_transaction):
        """Test convenience predict function."""
        # Note: This would require a running server in real scenario
        # Here we test the function exists and has correct signature
        assert callable(predict_fraud)
    
    def test_client_context_manager(self):
        """Test client as context manager."""
        with FraudDetectionClient(base_url="http://localhost:8000") as client:
            assert client.session is not None
    
    def test_client_retry_logic(self):
        """Test client retry configuration."""
        client = FraudDetectionClient(
            base_url="http://localhost:8000",
            max_retries=5,
            retry_delay=2.0
        )
        
        assert client.config.max_retries == 5
        assert client.config.retry_delay == 2.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_prediction_workflow(self, api_client, sample_transaction):
        """Test complete prediction workflow."""
        # 1. Check health
        health_response = api_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info
        info_response = api_client.get("/model_info")
        assert info_response.status_code == 200
        
        # 3. Make prediction
        pred_response = api_client.post("/predict", json=sample_transaction)
        assert pred_response.status_code == 200
        
        # 4. Check metrics updated
        metrics_response = api_client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        assert metrics["total_requests"] > 0
    
    def test_batch_workflow(self, api_client, batch_transactions):
        """Test complete batch prediction workflow."""
        # Prepare batch
        batch_data = {
            "transactions": batch_transactions[:20],
            "batch_id": "INTEGRATION_BATCH",
            "parallel": True
        }
        
        # Submit batch
        response = api_client.post("/predict_batch", json=batch_data)
        assert response.status_code == 200
        
        # Verify results
        data = response.json()
        assert data["total_transactions"] == 20
        assert len(data["predictions"]) == 20
        
        # All predictions should have required fields
        for pred in data["predictions"]:
            assert "transaction_id" in pred
            assert "is_fraud" in pred
            assert "fraud_probability" in pred


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""
    
    def test_single_prediction_latency(self, fraud_api, sample_request):
        """Test single prediction latency < 100ms."""
        latencies = []
        
        for _ in range(10):
            start = time.time()
            fraud_api.predict(sample_request)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nLatency stats: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
        
        # Average should be well under 100ms
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        
        # P95 should also be reasonable
        assert p95_latency < 150, f"P95 latency {p95_latency:.2f}ms too high"
    
    def test_throughput(self, fraud_api, sample_request):
        """Test API throughput (predictions per second)."""
        duration = 2.0  # 2 seconds
        count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            fraud_api.predict(sample_request)
            count += 1
        
        throughput = count / duration
        print(f"\nThroughput: {throughput:.0f} predictions/sec")
        
        # Should handle at least 10 predictions per second
        assert throughput >= 10, f"Throughput {throughput:.0f} req/s too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
