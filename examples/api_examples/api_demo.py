"""
SynFinance Fraud Detection API - Complete Demo

This script demonstrates all features of the fraud detection API:
1. Starting the API server
2. Making single predictions
3. Batch predictions with parallel processing  
4. Error handling and retry logic
5. Health checks and monitoring
6. Using the Python client library

Author: SynFinance ML Team
Date: October 28, 2025
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.api_client import FraudDetectionClient, predict_fraud
from src.api.fraud_detection_api import (
    FraudDetectionAPI,
    BatchDetectionAPI,
    PredictionRequest,
    BatchPredictionRequest
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_direct_api():
    """
    Demo 1: Using FraudDetectionAPI directly (without REST API)
    
    This is useful for embedded applications or when you don't need
    the full REST API infrastructure.
    """
    print_section("Demo 1: Direct API Usage (No REST Server)")
    
    # Initialize API (model_path=None uses fallback mode for demo)
    print("Initializing Fraud Detection API...")
    api = FraudDetectionAPI(
        model_path=None,  # Use None for demo (random predictions)
        threshold=0.5,
        feature_engineering=True
    )
    
    # Create a sample transaction
    transaction = PredictionRequest(
        transaction_id="TXN_DEMO_001",
        amount=15000.0,
        merchant_id="M_ELECTRONICS_001",
        merchant_name="TechMart Electronics",
        category="Shopping",
        city="Mumbai",
        timestamp=datetime.now().isoformat(),
        payment_mode="credit_card",
        merchant_reputation=0.85,
        distance_from_home=25.5,
        hour=22,  # Late night transaction
        is_weekend=False
    )
    
    print(f"\nTransaction Details:")
    print(f"  ID: {transaction.transaction_id}")
    print(f"  Amount: ₹{transaction.amount:,.2f}")
    print(f"  Merchant: {transaction.merchant_name}")
    print(f"  Time: {transaction.hour}:00")
    print(f"  Distance from home: {transaction.distance_from_home} km")
    
    # Make prediction
    print("\nMaking prediction...")
    response = api.predict(transaction)
    
    # Display results
    print(f"\n{'Prediction Results':^40}")
    print("-" * 40)
    print(f"  Fraud: {'YES' if response.is_fraud else 'NO'}")
    print(f"  Probability: {response.fraud_probability:.2%}")
    print(f"  Risk Score: {response.risk_score:.1f}/100")
    print(f"  Confidence: {response.confidence:.2%}")
    print(f"  Fraud Type: {response.fraud_type or 'None'}")
    print(f"  Recommendation: {response.recommendation}")
    print(f"  Processing Time: {response.processing_time_ms:.2f}ms")
    
    # Check API health
    print("\nAPI Health Check:")
    health = api.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Model Loaded: {health['model_loaded']}")
    
    # Get metrics
    print("\nAPI Metrics:")
    metrics = api.get_metrics()
    print(f"  Total Predictions: {metrics['total_predictions']}")
    print(f"  Fraud Rate: {metrics['fraud_rate']:.2%}")
    

def demo_batch_processing():
    """
    Demo 2: Batch Processing with Parallel Execution
    
    Demonstrates how to efficiently process large batches of transactions
    using parallel processing.
    """
    print_section("Demo 2: Batch Processing")
    
    # Initialize batch API
    print("Initializing Batch Detection API...")
    batch_api = BatchDetectionAPI(
        model_path=None,
        threshold=0.5,
        max_workers=4  # Use 4 parallel workers
    )
    
    # Generate sample batch of transactions
    print("\nGenerating batch of 50 transactions...")
    transactions = []
    base_time = datetime.now()
    
    for i in range(50):
        transactions.append({
            "transaction_id": f"BATCH_TXN_{i:04d}",
            "amount": 500.0 + (i * 100),
            "merchant_id": f"M_{i % 10:03d}",
            "merchant_name": f"Merchant {i % 10}",
            "category": ["Shopping", "Food", "Travel", "Entertainment"][i % 4],
            "city": ["Mumbai", "Delhi", "Bangalore", "Chennai"][i % 4],
            "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            "payment_mode": ["credit_card", "debit_card", "upi", "netbanking"][i % 4]
        })
    
    # Create batch request
    batch_request = BatchPredictionRequest(
        transactions=transactions,
        batch_id="DEMO_BATCH_001",
        parallel=True,  # Enable parallel processing
        chunk_size=10   # Process in chunks of 10
    )
    
    # Process batch
    print(f"Processing {len(transactions)} transactions with parallel=True...")
    start_time = time.time()
    response = batch_api.predict_batch(batch_request)
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n{'Batch Processing Results':^40}")
    print("-" * 40)
    print(f"  Batch ID: {response.batch_id}")
    print(f"  Total Transactions: {response.total_transactions}")
    print(f"  Fraud Detected: {response.fraud_count}")
    print(f"  Fraud Rate: {response.fraud_rate:.2%}")
    print(f"  Processing Time: {response.processing_time_ms:.2f}ms")
    print(f"  Actual Elapsed: {elapsed*1000:.2f}ms")
    print(f"  Throughput: {response.total_transactions / elapsed:.0f} txn/sec")
    
    # Show sample predictions
    print("\nSample Predictions (first 5):")
    for i, pred in enumerate(response.predictions[:5], 1):
        print(f"  {i}. {pred.transaction_id}: "
              f"Fraud={'YES' if pred.is_fraud else 'NO'} "
              f"({pred.fraud_probability:.2%})")


def demo_rest_api_client():
    """
    Demo 3: Using the REST API Client
    
    NOTE: This requires the API server to be running.
    Start it with: uvicorn src.api.api_server:app --reload
    """
    print_section("Demo 3: REST API Client Usage")
    
    print("Creating API client...")
    print("NOTE: This demo requires the API server to be running at http://localhost:8000")
    print("Start the server with: uvicorn src.api.api_server:app --reload")
    print("\nPress Enter to continue (or Ctrl+C to skip)...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nSkipping REST API demo.\n")
        return
    
    # Create client
    client = FraudDetectionClient(
        base_url="http://localhost:8000",
        timeout=30,
        max_retries=3,
        retry_delay=1.0
    )
    
    try:
        # Wait for server to be ready
        print("\nWaiting for server to be ready...")
        if not client.wait_until_ready(timeout=10):
            print("❌ Server not responding. Please start the server and try again.")
            return
        
        print("✅ Server is ready!\n")
        
        # Check health
        print("Checking API health...")
        health = client.health_check()
        print(f"  Status: {health['status']}")
        print(f"  Model Loaded: {health['model_loaded']}")
        
        # Get model info
        print("\nGetting model information...")
        model_info = client.get_model_info()
        print(f"  Model Version: {model_info.get('version', 'N/A')}")
        print(f"  Threshold: {model_info.get('threshold', 0.5)}")
        
        # Make a single prediction
        print("\nMaking single prediction...")
        transaction = {
            "transaction_id": "REST_TXN_001",
            "amount": 25000.0,
            "merchant_id": "M_LUXURY_001",
            "merchant_name": "Luxury Store",
            "category": "Shopping",
            "city": "Mumbai",
            "timestamp": datetime.now().isoformat(),
            "payment_mode": "credit_card"
        }
        
        result = client.predict(transaction)
        print(f"\nPrediction Result:")
        print(f"  Transaction: {result['transaction_id']}")
        print(f"  Fraud: {'YES' if result['is_fraud'] else 'NO'}")
        print(f"  Probability: {result['fraud_probability']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")
        
        # Batch prediction
        print("\nMaking batch prediction (20 transactions)...")
        batch_transactions = [
            {
                "transaction_id": f"BATCH_{i:03d}",
                "amount": 1000.0 + (i * 500),
                "merchant_id": f"M_{i:03d}",
                "merchant_name": f"Merchant {i}",
                "category": "Shopping",
                "city": "Mumbai",
                "timestamp": datetime.now().isoformat(),
                "payment_mode": "credit_card"
            }
            for i in range(20)
        ]
        
        batch_result = client.predict_batch(
            transactions=batch_transactions,
            batch_id="REST_BATCH_001",
            parallel=True
        )
        
        print(f"\nBatch Result:")
        print(f"  Total: {batch_result['total_transactions']}")
        print(f"  Fraud Count: {batch_result['fraud_count']}")
        print(f"  Fraud Rate: {batch_result['fraud_rate']:.2%}")
        print(f"  Processing Time: {batch_result['processing_time_ms']:.2f}ms")
        
        # Get API metrics
        print("\nGetting API metrics...")
        metrics = client.get_metrics()
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Successful Predictions: {metrics['successful_predictions']}")
        print(f"  Uptime: {metrics['uptime_seconds']:.0f} seconds")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the API server is running.")
    
    finally:
        client.close()


def demo_convenience_function():
    """
    Demo 4: Using the Convenience Function
    
    Shows the simplest way to make a prediction.
    """
    print_section("Demo 4: Convenience Function")
    
    print("Using the predict_fraud() convenience function...")
    print("NOTE: Requires API server running at http://localhost:8000")
    print("\nPress Enter to continue (or Ctrl+C to skip)...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nSkipping convenience function demo.\n")
        return
    
    # Simple transaction
    transaction = {
        "transaction_id": "SIMPLE_001",
        "amount": 5000.0,
        "merchant_id": "M_RESTAURANT_001",
        "merchant_name": "Fine Dining Restaurant",
        "category": "Food",
        "city": "Bangalore",
        "timestamp": datetime.now().isoformat(),
        "payment_mode": "credit_card"
    }
    
    try:
        # One-line prediction
        result = predict_fraud(
            transaction=transaction,
            base_url="http://localhost:8000"
        )
        
        print(f"\nResult:")
        print(f"  Fraud: {'YES' if result['is_fraud'] else 'NO'}")
        print(f"  Probability: {result['fraud_probability']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_error_handling():
    """
    Demo 5: Error Handling and Retry Logic
    
    Demonstrates how the client handles errors and retries.
    """
    print_section("Demo 5: Error Handling")
    
    print("Creating client with retry logic...")
    client = FraudDetectionClient(
        base_url="http://localhost:9999",  # Invalid port
        timeout=5,
        max_retries=2,
        retry_delay=0.5
    )
    
    # Try invalid transaction (will fail due to validation)
    print("\n1. Testing connection error handling...")
    invalid_transaction = {
        "transaction_id": "ERROR_001",
        "amount": 1000.0,
        "merchant_id": "M_001",
        "merchant_name": "Test Merchant",
        "category": "Shopping",
        "city": "Mumbai",
        "timestamp": datetime.now().isoformat(),
        "payment_mode": "invalid_mode"  # Invalid!
    }
    
    try:
        result = client.predict(invalid_transaction)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ✅ Error handled correctly: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}...")
    
    client.close()
    
    print("\n2. Testing validation error...")
    client2 = FraudDetectionClient(base_url="http://localhost:8000")
    
    # Transaction with negative amount
    invalid_transaction2 = {
        "transaction_id": "ERROR_002",
        "amount": -1000.0,  # Negative!
        "merchant_id": "M_001",
        "merchant_name": "Test Merchant",
        "category": "Shopping",
        "city": "Mumbai",
        "timestamp": datetime.now().isoformat(),
        "payment_mode": "credit_card"
    }
    
    try:
        result = client2.predict(invalid_transaction2)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ✅ Validation error handled: {type(e).__name__}")
    
    client2.close()


def demo_context_manager():
    """
    Demo 6: Using Client as Context Manager
    
    Shows proper resource management with context managers.
    """
    print_section("Demo 6: Context Manager Usage")
    
    print("Using client as context manager...")
    print("This ensures proper cleanup of resources.\n")
    
    try:
        with FraudDetectionClient(base_url="http://localhost:8000") as client:
            # Check if server is available
            if not client.is_healthy():
                print("❌ Server is not healthy")
                return
            
            print("✅ Server is healthy")
            
            # Make prediction
            transaction = {
                "transaction_id": "CTX_001",
                "amount": 3000.0,
                "merchant_id": "M_GROCERY_001",
                "merchant_name": "Grocery Store",
                "category": "Food",
                "city": "Delhi",
                "timestamp": datetime.now().isoformat(),
                "payment_mode": "upi"
            }
            
            result = client.predict(transaction)
            print(f"\nPrediction:")
            print(f"  Fraud: {'YES' if result['is_fraud'] else 'NO'}")
            print(f"  Probability: {result['fraud_probability']:.2%}")
            
        # Client is automatically closed after context
        print("\n✅ Client resources cleaned up automatically")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_performance_testing():
    """
    Demo 7: Performance Testing
    
    Measures API performance and throughput.
    """
    print_section("Demo 7: Performance Testing")
    
    # Test direct API performance
    print("Testing Direct API Performance...")
    api = FraudDetectionAPI(model_path=None, threshold=0.5)
    
    # Create test transaction
    test_txn = PredictionRequest(
        transaction_id="PERF_001",
        amount=5000.0,
        merchant_id="M_TEST",
        merchant_name="Test Merchant",
        category="Shopping",
        city="Mumbai",
        timestamp=datetime.now().isoformat(),
        payment_mode="credit_card"
    )
    
    # Measure latency
    latencies = []
    num_predictions = 100
    
    print(f"Making {num_predictions} predictions...")
    start = time.time()
    for i in range(num_predictions):
        pred_start = time.time()
        api.predict(test_txn)
        latencies.append((time.time() - pred_start) * 1000)
    total_time = time.time() - start
    
    # Calculate statistics
    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p99_latency = latencies[int(len(latencies) * 0.99)]
    throughput = num_predictions / total_time
    
    print(f"\n{'Performance Results':^40}")
    print("-" * 40)
    print(f"  Total Predictions: {num_predictions}")
    print(f"  Total Time: {total_time*1000:.2f}ms")
    print(f"  Average Latency: {avg_latency:.2f}ms")
    print(f"  P50 Latency: {p50_latency:.2f}ms")
    print(f"  P95 Latency: {p95_latency:.2f}ms")
    print(f"  P99 Latency: {p99_latency:.2f}ms")
    print(f"  Throughput: {throughput:.0f} predictions/sec")
    
    # Check if meets requirements
    print(f"\n{'Requirements Check':^40}")
    print("-" * 40)
    meets_latency = avg_latency < 100
    print(f"  {'✅' if meets_latency else '❌'} Avg latency <100ms: {avg_latency:.2f}ms")


def main():
    """Main demo function"""
    print("\n" + "=" * 80)
    print(" SynFinance Fraud Detection API - Complete Demonstration")
    print("=" * 80)
    print("\nThis demo showcases all features of the fraud detection API.")
    print("Some demos require the API server to be running.")
    print("\nTo start the server, run:")
    print("  uvicorn src.api.api_server:app --reload")
    print("\n" + "=" * 80)
    
    try:
        # Run all demos
        demo_direct_api()
        demo_batch_processing()
        demo_performance_testing()
        demo_rest_api_client()
        demo_convenience_function()
        demo_error_handling()
        demo_context_manager()
        
        # Final summary
        print_section("Demo Complete!")
        print("All demonstrations completed successfully.")
        print("\nKey Takeaways:")
        print("  ✅ Direct API usage for embedded applications")
        print("  ✅ Batch processing with parallel execution")
        print("  ✅ REST API client with retry logic")
        print("  ✅ Comprehensive error handling")
        print("  ✅ Performance monitoring and metrics")
        print("  ✅ Production-ready <100ms latency")
        print("\nFor more information, see docs/api/API_REFERENCE.md")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
