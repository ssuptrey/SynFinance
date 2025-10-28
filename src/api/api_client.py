"""
Fraud Detection API Client

Python client for easy integration with SynFinance Fraud Detection API.
Includes retry logic, timeout handling, and comprehensive error handling.

Author: SynFinance ML Team
Date: October 28, 2025
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Client configuration"""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None


class FraudDetectionClient:
    """
    Client for SynFinance Fraud Detection API
    
    Features:
    - Simple interface for predictions
    - Automatic retry logic
    - Timeout handling
    - Error handling with detailed messages
    - Batch prediction support
    - Health checking
    
    Example:
        >>> client = FraudDetectionClient(base_url="http://localhost:8000")
        >>> 
        >>> # Single prediction
        >>> transaction = {
        ...     "transaction_id": "TXN001",
        ...     "amount": 1500.0,
        ...     "merchant_id": "M001",
        ...     "merchant_name": "Amazon",
        ...     "category": "Shopping",
        ...     "city": "Mumbai",
        ...     "timestamp": "2025-10-28T14:30:00",
        ...     "payment_mode": "UPI"
        ... }
        >>> result = client.predict(transaction)
        >>> print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']:.2%}")
        >>>
        >>> # Batch prediction
        >>> transactions = [transaction1, transaction2, ...]
        >>> batch_result = client.predict_batch(transactions)
        >>> print(f"Fraud rate: {batch_result['fraud_rate']:.2%}")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize Fraud Detection Client
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            api_key: Optional API key for authentication
        """
        self.config = ClientConfig(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            api_key=api_key
        )
        
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SynFinance-Client/1.0.0'
        })
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data dictionary
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.config.timeout
                )
                
                # Raise for HTTP errors
                response.raise_for_status()
                
                # Log processing time if available
                process_time = response.headers.get('X-Process-Time-Ms')
                if process_time:
                    logger.debug(f"Request processed in {process_time}ms")
                
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                raise
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud for single transaction
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Prediction result dictionary
            
        Example:
            >>> transaction = {
            ...     "transaction_id": "TXN001",
            ...     "amount": 1500.0,
            ...     "merchant_id": "M001",
            ...     "merchant_name": "Amazon",
            ...     "category": "Shopping",
            ...     "city": "Mumbai",
            ...     "timestamp": "2025-10-28T14:30:00",
            ...     "payment_mode": "UPI"
            ... }
            >>> result = client.predict(transaction)
        """
        logger.info(f"Predicting fraud for transaction: {transaction.get('transaction_id')}")
        
        try:
            result = self._make_request('POST', '/predict', data=transaction)
            
            logger.info(
                f"Prediction complete: {transaction.get('transaction_id')} | "
                f"Fraud: {result.get('is_fraud')} | "
                f"Probability: {result.get('fraud_probability', 0):.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        parallel: bool = True,
        chunk_size: int = 100
    ) -> Dict[str, Any]:
        """
        Predict fraud for batch of transactions
        
        Args:
            transactions: List of transaction dictionaries
            batch_id: Optional batch identifier
            parallel: Enable parallel processing
            chunk_size: Processing chunk size
            
        Returns:
            Batch prediction result dictionary
            
        Example:
            >>> transactions = [txn1, txn2, txn3, ...]
            >>> result = client.predict_batch(transactions, batch_id="BATCH001")
            >>> print(f"Processed {result['total_transactions']} transactions")
            >>> print(f"Fraud rate: {result['fraud_rate']:.2%}")
        """
        logger.info(f"Predicting fraud for batch: {len(transactions)} transactions")
        
        try:
            data = {
                'transactions': transactions,
                'batch_id': batch_id,
                'parallel': parallel,
                'chunk_size': chunk_size
            }
            
            result = self._make_request('POST', '/predict_batch', data=data)
            
            logger.info(
                f"Batch prediction complete: {result.get('batch_id')} | "
                f"Total: {result.get('total_transactions')} | "
                f"Fraud: {result.get('fraud_count')} ({result.get('fraud_rate', 0):.2%}) | "
                f"Time: {result.get('processing_time_ms', 0):.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Model information dictionary
            
        Example:
            >>> info = client.get_model_info()
            >>> print(f"Model version: {info['version']}")
            >>> print(f"Features: {info['feature_count']}")
        """
        try:
            return self._make_request('GET', '/model_info')
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status dictionary
            
        Example:
            >>> health = client.health_check()
            >>> print(f"Status: {health['status']}")
            >>> print(f"Model loaded: {health['model_loaded']}")
        """
        try:
            return self._make_request('GET', '/health')
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get API metrics
        
        Returns:
            API metrics dictionary
            
        Example:
            >>> metrics = client.get_metrics()
            >>> print(f"Total requests: {metrics['total_requests']}")
            >>> print(f"Success rate: {metrics['successful_predictions'] / metrics['total_requests']:.2%}")
        """
        try:
            return self._make_request('GET', '/metrics')
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """
        Check if API is healthy
        
        Returns:
            True if healthy, False otherwise
            
        Example:
            >>> if client.is_healthy():
            ...     print("API is running")
            ... else:
            ...     print("API is down")
        """
        try:
            health = self.health_check()
            return health.get('status') == 'healthy'
        except:
            return False
    
    def wait_until_ready(self, timeout: int = 60, interval: int = 5) -> bool:
        """
        Wait until API is ready
        
        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            
        Returns:
            True if API becomes ready, False if timeout
            
        Example:
            >>> if client.wait_until_ready(timeout=60):
            ...     print("API is ready")
            ... else:
            ...     print("Timeout waiting for API")
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_healthy():
                logger.info("API is ready")
                return True
            
            logger.info(f"Waiting for API... ({int(time.time() - start_time)}s)")
            time.sleep(interval)
        
        logger.error(f"Timeout waiting for API after {timeout}s")
        return False
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function for quick predictions
def predict_fraud(
    transaction: Dict[str, Any],
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Quick fraud prediction function
    
    Args:
        transaction: Transaction data
        base_url: API base URL
        
    Returns:
        Prediction result
        
    Example:
        >>> from src.api.api_client import predict_fraud
        >>> 
        >>> result = predict_fraud({
        ...     "transaction_id": "TXN001",
        ...     "amount": 1500.0,
        ...     "merchant_id": "M001",
        ...     "merchant_name": "Amazon",
        ...     "category": "Shopping",
        ...     "city": "Mumbai",
        ...     "timestamp": "2025-10-28T14:30:00",
        ...     "payment_mode": "UPI"
        ... })
        >>> print(f"Fraud: {result['is_fraud']}")
    """
    with FraudDetectionClient(base_url=base_url) as client:
        return client.predict(transaction)
