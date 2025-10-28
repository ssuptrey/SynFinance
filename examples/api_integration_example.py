"""
API Integration Example
========================

Demonstrates how to integrate the SynFinance fraud detection API into your applications.

This example shows:
1. Making single predictions
2. Batch processing
3. Error handling
4. Rate limiting awareness
5. Authentication
6. Response parsing

Author: SynFinance Team
Version: 0.7.0
Date: October 28, 2025
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class SynFinanceAPIClient:
    """
    Client for interacting with the SynFinance Fraud Detection API.
    
    Usage:
        client = SynFinanceAPIClient("http://localhost:8000", api_key="your-key")
        result = client.predict_single(transaction_data)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set authentication header if API key provided
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        print(f"SynFinance API Client initialized")
        print(f"  Base URL: {self.base_url}")
        print(f"  Auth: {'Enabled' if api_key else 'Disabled'}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the deployed model.
        
        Returns:
            Model metadata dictionary
        """
        response = self.session.get(
            f"{self.base_url}/model-info",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict_single(
        self,
        transaction: Dict[str, Any],
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make a fraud prediction for a single transaction.
        
        Args:
            transaction: Transaction data dictionary
            return_features: Whether to return engineered features
        
        Returns:
            Prediction result with fraud probability and classification
        """
        payload = {
            "transaction": transaction,
            "return_features": return_features
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make fraud predictions for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            parallel: Use parallel processing (faster for large batches)
        
        Returns:
            List of prediction results
        """
        payload = {
            "transactions": transactions,
            "parallel": parallel
        }
        
        response = self.session.post(
            f"{self.base_url}/batch-predict",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get API performance metrics.
        
        Returns:
            Metrics dictionary with request counts, latencies, etc.
        """
        response = self.session.get(
            f"{self.base_url}/metrics",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


def example1_single_prediction():
    """Example 1: Make a single fraud prediction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Transaction Prediction")
    print("=" * 80)
    
    # Initialize client
    client = SynFinanceAPIClient()
    
    # Check API health
    health = client.health_check()
    print(f"\nAPI Health: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("⚠️  API is not healthy. Make sure the server is running:")
        print("  $ docker-compose up -d")
        return
    
    # Sample transaction (potentially fraudulent)
    transaction = {
        "transaction_id": "TXN_20251028_001",
        "amount": 15000.00,
        "payment_mode": "Credit Card",
        "merchant_name": "Online Electronics Store",
        "category": "Electronics",
        "city": "Mumbai",
        "customer_id": "CUST_001",
        "age": 25,
        "gender": "Male",
        "occupation": "Student",
        "transaction_date": datetime.now().isoformat()
    }
    
    print("\nTransaction Details:")
    print(json.dumps(transaction, indent=2))
    
    # Make prediction
    print("\nMaking prediction...")
    start_time = time.time()
    result = client.predict_single(transaction)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Prediction completed in {elapsed*1000:.2f}ms")
    print("\nPrediction Result:")
    print(f"  Is Fraud: {result.get('is_fraud', 'unknown')}")
    print(f"  Fraud Probability: {result.get('fraud_probability', 0):.2%}")
    print(f"  Confidence: {result.get('confidence', 'unknown')}")
    
    if 'fraud_type' in result:
        print(f"  Fraud Type: {result['fraud_type']}")
    
    if 'risk_factors' in result:
        print(f"  Risk Factors: {', '.join(result['risk_factors'][:3])}")


def example2_batch_predictions():
    """Example 2: Batch prediction for multiple transactions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Transaction Predictions")
    print("=" * 80)
    
    client = SynFinanceAPIClient()
    
    # Generate sample transactions
    transactions = [
        {
            "transaction_id": f"TXN_{i:04d}",
            "amount": 100.0 + i * 50,
            "payment_mode": "UPI" if i % 2 == 0 else "Credit Card",
            "merchant_name": f"Merchant_{i % 10}",
            "category": "Shopping",
            "city": "Mumbai",
            "customer_id": f"CUST_{i % 5:03d}",
            "age": 25 + i % 40,
            "gender": "Male" if i % 2 == 0 else "Female",
            "occupation": "Salaried",
            "transaction_date": datetime.now().isoformat()
        }
        for i in range(20)
    ]
    
    print(f"\nProcessing {len(transactions)} transactions...")
    
    # Make batch prediction
    start_time = time.time()
    results = client.predict_batch(transactions, parallel=True)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Batch prediction completed in {elapsed*1000:.2f}ms")
    print(f"  Average latency: {elapsed/len(transactions)*1000:.2f}ms per transaction")
    
    # Analyze results
    fraud_count = sum(1 for r in results if r.get('is_fraud', False))
    avg_probability = sum(r.get('fraud_probability', 0) for r in results) / len(results)
    
    print(f"\nBatch Results:")
    print(f"  Total Transactions: {len(results)}")
    print(f"  Fraudulent: {fraud_count} ({fraud_count/len(results):.1%})")
    print(f"  Average Fraud Probability: {avg_probability:.2%}")
    
    # Show sample results
    print(f"\nSample Results (first 5):")
    print("-" * 80)
    print(f"{'ID':<12} {'Amount':<10} {'Is Fraud':<10} {'Probability':<15}")
    print("-" * 80)
    for r in results[:5]:
        print(f"{r['transaction_id']:<12} ${r.get('amount', 0):>8.2f} {str(r.get('is_fraud', False)):<10} {r.get('fraud_probability', 0):>12.2%}")


def example3_error_handling():
    """Example 3: Proper error handling and retries."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Error Handling")
    print("=" * 80)
    
    client = SynFinanceAPIClient()
    
    # Invalid transaction (missing required fields)
    invalid_transaction = {
        "transaction_id": "INVALID_001",
        "amount": -100.0,  # Negative amount (invalid)
    }
    
    print("\nAttempting prediction with invalid transaction:")
    print(json.dumps(invalid_transaction, indent=2))
    
    try:
        result = client.predict_single(invalid_transaction)
        print("\n✓ Prediction result:", result)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Request failed with status {e.response.status_code}")
        try:
            error_detail = e.response.json()
            print("Error details:", json.dumps(error_detail, indent=2))
        except:
            print("Error response:", e.response.text)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {str(e)}")
    
    # Retry with valid transaction
    print("\n\nRetrying with valid transaction...")
    valid_transaction = {
        "transaction_id": "VALID_001",
        "amount": 500.0,
        "payment_mode": "UPI",
        "merchant_name": "Coffee Shop",
        "category": "Food & Dining",
        "city": "Bangalore",
        "customer_id": "CUST_002",
        "age": 30,
        "gender": "Female",
        "occupation": "Salaried",
        "transaction_date": datetime.now().isoformat()
    }
    
    try:
        result = client.predict_single(valid_transaction)
        print(f"✓ Prediction successful!")
        print(f"  Fraud Probability: {result.get('fraud_probability', 0):.2%}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


def example4_performance_monitoring():
    """Example 4: Monitor API performance."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Performance Monitoring")
    print("=" * 80)
    
    client = SynFinanceAPIClient()
    
    # Make several predictions
    print("\nMaking 10 predictions to generate metrics...")
    
    latencies = []
    for i in range(10):
        transaction = {
            "transaction_id": f"PERF_TEST_{i}",
            "amount": 1000.0,
            "payment_mode": "UPI",
            "merchant_name": "Test Merchant",
            "category": "Shopping",
            "city": "Delhi",
            "customer_id": f"CUST_{i}",
            "age": 25,
            "gender": "Male",
            "occupation": "Salaried",
            "transaction_date": datetime.now().isoformat()
        }
        
        start = time.time()
        try:
            client.predict_single(transaction)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"  {i+1}/10: {latency:.2f}ms")
        except Exception as e:
            print(f"  {i+1}/10: Error - {e}")
    
    if latencies:
        print(f"\nPerformance Statistics:")
        print(f"  Min Latency: {min(latencies):.2f}ms")
        print(f"  Max Latency: {max(latencies):.2f}ms")
        print(f"  Avg Latency: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  Median Latency: {sorted(latencies)[len(latencies)//2]:.2f}ms")
    
    # Get API metrics
    print("\nFetching API metrics...")
    try:
        metrics = client.get_metrics()
        print(f"\nAPI Metrics:")
        if 'total_requests' in metrics:
            print(f"  Total Requests: {metrics['total_requests']}")
        if 'average_latency_ms' in metrics:
            print(f"  Average Latency: {metrics['average_latency_ms']:.2f}ms")
        if 'requests_per_second' in metrics:
            print(f"  Throughput: {metrics['requests_per_second']:.2f} req/s")
    except Exception as e:
        print(f"  Metrics endpoint not available: {e}")


def example5_csv_batch_processing():
    """Example 5: Process transactions from CSV file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: CSV Batch Processing")
    print("=" * 80)
    
    client = SynFinanceAPIClient()
    
    # Create sample CSV
    print("\nCreating sample CSV file...")
    sample_data = pd.DataFrame([
        {
            "transaction_id": f"CSV_{i:04d}",
            "amount": 100 + i * 25,
            "payment_mode": "UPI" if i % 3 == 0 else "Credit Card",
            "merchant_name": f"Merchant_{i % 5}",
            "category": "Shopping",
            "city": "Mumbai" if i % 2 == 0 else "Delhi",
            "customer_id": f"CUST_{i % 10:03d}",
            "age": 25 + i % 30,
            "gender": "Male" if i % 2 == 0 else "Female",
            "occupation": "Salaried",
            "transaction_date": datetime.now().isoformat()
        }
        for i in range(50)
    ])
    
    csv_file = "output/sample_transactions.csv"
    sample_data.to_csv(csv_file, index=False)
    print(f"✓ Created CSV with {len(sample_data)} transactions: {csv_file}")
    
    # Read and process
    print("\nReading CSV and making predictions...")
    df = pd.read_csv(csv_file)
    transactions = df.to_dict('records')
    
    # Process in batches of 10
    batch_size = 10
    all_results = []
    
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(transactions)-1)//batch_size + 1}...")
        
        try:
            results = client.predict_batch(batch, parallel=True)
            all_results.extend(results)
        except Exception as e:
            print(f"    Error processing batch: {e}")
    
    print(f"\n✓ Processed {len(all_results)} transactions")
    
    # Add predictions to dataframe
    df['is_fraud'] = [r.get('is_fraud', False) for r in all_results]
    df['fraud_probability'] = [r.get('fraud_probability', 0) for r in all_results]
    
    # Save results
    output_file = "output/predictions_results.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved predictions to: {output_file}")
    
    # Summary statistics
    fraud_count = df['is_fraud'].sum()
    print(f"\nSummary:")
    print(f"  Total: {len(df)}")
    print(f"  Fraudulent: {fraud_count} ({fraud_count/len(df):.1%})")
    print(f"  Average Probability: {df['fraud_probability'].mean():.2%}")
    print(f"  High Risk (>50%): {(df['fraud_probability'] > 0.5).sum()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  SynFinance API Integration Examples")
    print("  Version: 0.7.0")
    print("=" * 80)
    
    print("\nThese examples demonstrate how to integrate the SynFinance API.")
    print("\n⚠️  Make sure the API server is running:")
    print("  $ docker-compose up -d")
    print("  $ curl http://localhost:8000/health")
    
    input("\nPress Enter to continue with examples...")
    
    try:
        # Run examples
        example1_single_prediction()
        
        input("\nPress Enter for next example...")
        example2_batch_predictions()
        
        input("\nPress Enter for next example...")
        example3_error_handling()
        
        input("\nPress Enter for next example...")
        example4_performance_monitoring()
        
        input("\nPress Enter for final example...")
        example5_csv_batch_processing()
        
        print("\n" + "=" * 80)
        print("  All examples completed successfully!")
        print("=" * 80)
        
        print("\nNext Steps:")
        print("  1. Explore the API documentation: http://localhost:8000/docs")
        print("  2. Review example source code for integration patterns")
        print("  3. Adapt examples to your specific use case")
        print("  4. Set up authentication with API keys")
        print("  5. Configure rate limiting for production")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
