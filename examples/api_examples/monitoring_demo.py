"""
Advanced Monitoring & Metrics Example

Demonstrates:
- Prometheus metrics collection
- Business metrics tracking (fraud, performance, data quality)
- FastAPI integration with metrics middleware
- Grafana dashboard integration
- Alert rule testing

Usage:
    python examples/monitoring_demo.py
"""

import asyncio
import time
import random
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from uvicorn import Config, Server

from src.monitoring.prometheus_exporter import get_metrics_exporter
from src.monitoring.business_metrics import (
    FraudDetectionMetrics,
    PerformanceMetrics,
    DataQualityMetrics
)
from src.monitoring.metrics_middleware import MetricsMiddleware
from src.data_generator import DataGenerator


class MonitoringDemo:
    """
    Demonstration of SynFinance monitoring system.
    
    Shows:
    - Real-time metrics collection
    - Business metrics tracking
    - API endpoint monitoring
    - Performance metrics
    - Data quality tracking
    """
    
    def __init__(self):
        """Initialize monitoring demo."""
        self.metrics_exporter = get_metrics_exporter("synfinance")
        self.fraud_metrics = FraudDetectionMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = DataQualityMetrics()
        self.data_generator = DataGenerator()
        
        print("=" * 60)
        print("SynFinance Advanced Monitoring & Metrics Demo")
        print("=" * 60)
        print()
    
    def demonstrate_prometheus_metrics(self):
        """Demonstrate Prometheus metrics collection."""
        print("\n1. PROMETHEUS METRICS EXPORTER")
        print("-" * 60)
        
        # Simulate API requests
        print("\nSimulating API requests...")
        for i in range(20):
            method = random.choice(["GET", "POST", "PUT"])
            endpoint = random.choice(["/api/transactions", "/api/detect", "/api/health"])
            status = random.choice([200, 200, 200, 400, 500])  # Mostly 200s
            latency = random.uniform(0.01, 0.5)
            
            self.metrics_exporter.record_request(
                method=method,
                endpoint=endpoint,
                status=status,
                latency=latency,
                request_size=random.randint(100, 5000),
                response_size=random.randint(500, 10000)
            )
            
            if status >= 400:
                error_type = "BadRequest" if status == 400 else "InternalError"
                self.metrics_exporter.record_error(error_type, endpoint)
        
        print(f"✓ Recorded 20 HTTP requests")
        print(f"✓ Methods: GET, POST, PUT")
        print(f"✓ Endpoints: /api/transactions, /api/detect, /api/health")
        print(f"✓ Status codes: 200, 400, 500")
        
        # Export metrics
        print("\nExporting metrics in Prometheus format...")
        metrics_output = self.metrics_exporter.export()
        
        # Show sample of exported metrics
        sample_lines = metrics_output.decode('utf-8').split('\n')[:20]
        print("\nSample exported metrics:")
        for line in sample_lines:
            if line and not line.startswith('#'):
                print(f"  {line}")
        
        print(f"\n✓ Total metrics output: {len(metrics_output)} bytes")
        print("✓ Available at /metrics endpoint")
    
    def demonstrate_fraud_metrics(self):
        """Demonstrate fraud detection metrics."""
        print("\n\n2. FRAUD DETECTION METRICS")
        print("-" * 60)
        
        print("\nSimulating fraud detection workflow...")
        
        # Process batch of transactions
        total_txns = 1000
        print(f"\nProcessing {total_txns} transactions...")
        
        for i in range(total_txns):
            is_fraud = random.random() < 0.03  # 3% fraud rate
            self.fraud_metrics.record_transaction(is_fraud=is_fraud)
        
        # Detect fraud cases
        fraud_types = ["card_cloning", "phishing", "account_takeover", "money_laundering"]
        severities = ["low", "medium", "high", "critical"]
        
        print("\nDetecting fraud...")
        for _ in range(30):
            fraud_type = random.choice(fraud_types)
            severity = random.choice(severities)
            confidence = random.uniform(0.6, 0.99)
            actual_fraud = random.random() < 0.9  # 90% precision
            
            self.fraud_metrics.record_fraud_detection(
                fraud_type=fraud_type,
                severity=severity,
                confidence=confidence,
                actual_fraud=actual_fraud
            )
        
        # Miss some fraud (false negatives)
        for _ in range(3):
            self.fraud_metrics.record_missed_fraud(random.choice(fraud_types))
        
        # Record normal transactions
        for _ in range(970):
            correctly_classified = random.random() < 0.95  # 95% accuracy
            self.fraud_metrics.record_normal_transaction(correctly_classified)
        
        # Get and display statistics
        stats = self.fraud_metrics.get_fraud_stats()
        
        print(f"\n✓ Fraud Detection Statistics:")
        print(f"  Total Transactions: {stats.total_transactions}")
        print(f"  Fraud Detected: {stats.total_fraud_detected}")
        print(f"  Fraud Rate: {stats.fraud_rate:.2%}")
        print(f"  True Positives: {stats.true_positives}")
        print(f"  False Positives: {stats.false_positives}")
        print(f"  False Negatives: {stats.false_negatives}")
        print(f"  True Negatives: {stats.true_negatives}")
        print(f"  Precision: {stats.precision:.2%}")
        print(f"  Recall: {stats.recall:.2%}")
        print(f"  F1 Score: {stats.f1_score:.2%}")
        print(f"  Avg Confidence: {stats.average_confidence:.2%}")
        
        print(f"\n✓ Fraud by Type:")
        for fraud_type, count in stats.fraud_by_type.items():
            print(f"  {fraud_type}: {count}")
        
        print(f"\n✓ Fraud by Severity:")
        for severity, count in stats.fraud_by_severity.items():
            print(f"  {severity}: {count}")
    
    def demonstrate_performance_metrics(self):
        """Demonstrate performance metrics."""
        print("\n\n3. PERFORMANCE METRICS")
        print("-" * 60)
        
        print("\nSimulating transaction generation...")
        
        # Generate transactions
        batch_sizes = [1000, 5000, 10000]
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Simulate generation
            time.sleep(random.uniform(0.1, 0.3))
            
            elapsed = time.time() - start_time
            self.performance_metrics.record_generation(
                count=batch_size,
                elapsed_time=elapsed
            )
            
            rate = batch_size / elapsed if elapsed > 0 else 0
            print(f"  Generated {batch_size:,} txns in {elapsed:.3f}s ({rate:,.0f} txn/sec)")
        
        # Simulate feature engineering
        print("\nSimulating feature engineering...")
        for _ in range(100):
            elapsed = random.uniform(0.005, 0.05)  # 5-50ms
            self.performance_metrics.record_feature_engineering(elapsed)
        
        # Simulate predictions
        print("Simulating model predictions...")
        for _ in range(100):
            elapsed = random.uniform(0.001, 0.01)  # 1-10ms
            self.performance_metrics.record_prediction(elapsed)
        
        # Simulate cache operations
        print("Simulating cache operations...")
        cache_types = ["customer", "merchant", "features", "models"]
        for cache_type in cache_types:
            # 80-90% hit rate
            hits = random.randint(800, 900)
            misses = 1000 - hits
            
            for _ in range(hits):
                self.performance_metrics.record_cache_hit(cache_type)
            for _ in range(misses):
                self.performance_metrics.record_cache_miss(cache_type)
        
        # Get and display statistics
        stats = self.performance_metrics.get_performance_stats()
        
        print(f"\n✓ Performance Statistics:")
        print(f"  Total Transactions: {stats.transactions_generated:,}")
        print(f"  Total Generation Time: {stats.total_generation_time:.3f}s")
        print(f"  Avg Generation Time: {stats.avg_generation_time*1000:.2f}ms")
        print(f"  Generation Rate: {stats.generation_rate:,.0f} txn/sec")
        print(f"  Avg Feature Time: {stats.avg_feature_time*1000:.2f}ms")
        print(f"  Avg Prediction Time: {stats.avg_prediction_time*1000:.2f}ms")
        
        print(f"\n✓ Cache Performance:")
        for cache_type in cache_types:
            hit_rate = stats.get_cache_hit_rate(cache_type)
            print(f"  {cache_type}: {hit_rate:.1%} hit rate")
    
    def demonstrate_data_quality_metrics(self):
        """Demonstrate data quality metrics."""
        print("\n\n4. DATA QUALITY METRICS")
        print("-" * 60)
        
        print("\nSimulating data quality checks...")
        
        # Record dataset size
        dataset_size = 10000
        self.quality_metrics.record_dataset(dataset_size)
        print(f"  Dataset size: {dataset_size:,} records")
        
        # Check for missing values
        fields = ["email", "phone", "address", "ssn", "account_number"]
        print("\nChecking for missing values...")
        for field in fields:
            # Varying missing rates (0-5%)
            missing_count = int(dataset_size * random.uniform(0, 0.05))
            if missing_count > 0:
                self.quality_metrics.record_missing_values(field, missing_count)
                print(f"  {field}: {missing_count} missing ({missing_count/dataset_size:.1%})")
        
        # Check for outliers
        numeric_fields = ["amount", "balance", "transaction_count"]
        print("\nChecking for outliers...")
        for field in numeric_fields:
            # Varying outlier rates (0-3%)
            outlier_count = int(dataset_size * random.uniform(0, 0.03))
            if outlier_count > 0:
                self.quality_metrics.record_outliers(field, outlier_count)
                print(f"  {field}: {outlier_count} outliers ({outlier_count/dataset_size:.1%})")
        
        # Simulate schema violations
        violations = random.randint(0, 5)
        print(f"\nSchema violations: {violations}")
        for _ in range(violations):
            self.quality_metrics.record_schema_violation()
        
        # Simulate distribution drift
        drift_fields = random.sample(fields + numeric_fields, k=random.randint(0, 2))
        if drift_fields:
            print(f"\nDistribution drift detected in: {', '.join(drift_fields)}")
            for field in drift_fields:
                self.quality_metrics.record_distribution_drift(field)
        
        # Get and display statistics
        stats = self.quality_metrics.get_quality_stats()
        quality_score = self.quality_metrics.get_quality_score()
        
        print(f"\n✓ Data Quality Statistics:")
        print(f"  Total Records: {stats.total_records:,}")
        print(f"  Overall Missing Rate: {stats.overall_missing_rate:.2%}")
        print(f"  Schema Violations: {stats.schema_violations}")
        print(f"  Distribution Drifts: {stats.distribution_drifts}")
        print(f"  Quality Score: {quality_score:.1f}/100")
        
        if quality_score >= 90:
            status = "EXCELLENT"
        elif quality_score >= 70:
            status = "GOOD"
        elif quality_score >= 50:
            status = "FAIR"
        else:
            status = "POOR"
        
        print(f"  Quality Status: {status}")
    
    def create_monitoring_api(self) -> FastAPI:
        """
        Create FastAPI application with monitoring.
        
        Returns:
            FastAPI app with metrics middleware
        """
        app = FastAPI(
            title="SynFinance Monitoring Demo API",
            description="API demonstrating monitoring integration",
            version="0.8.0"
        )
        
        # Add metrics middleware
        app.add_middleware(MetricsMiddleware)
        
        @app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "SynFinance Monitoring Demo API",
                "version": "0.8.0",
                "endpoints": {
                    "/metrics": "Prometheus metrics",
                    "/health": "Health check",
                    "/simulate/fraud": "Simulate fraud detection",
                    "/simulate/generation": "Simulate data generation"
                }
            }
        
        @app.get("/metrics")
        async def metrics():
            """Export Prometheus metrics."""
            from starlette.responses import Response
            
            metrics_data = self.metrics_exporter.export()
            return Response(
                content=metrics_data,
                media_type=self.metrics_exporter.get_content_type()
            )
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "fraud_rate": self.fraud_metrics.stats.fraud_rate,
                    "generation_rate": self.performance_metrics.stats.generation_rate,
                    "quality_score": self.quality_metrics.get_quality_score()
                }
            }
        
        @app.post("/simulate/fraud")
        async def simulate_fraud(count: int = 100):
            """Simulate fraud detection."""
            for _ in range(count):
                fraud_type = random.choice(["card_cloning", "phishing", "account_takeover"])
                severity = random.choice(["low", "medium", "high", "critical"])
                confidence = random.uniform(0.6, 0.99)
                
                self.fraud_metrics.record_fraud_detection(
                    fraud_type=fraud_type,
                    severity=severity,
                    confidence=confidence,
                    actual_fraud=True
                )
            
            return {
                "simulated": count,
                "fraud_rate": self.fraud_metrics.stats.fraud_rate
            }
        
        @app.post("/simulate/generation")
        async def simulate_generation(count: int = 1000):
            """Simulate transaction generation."""
            start_time = time.time()
            
            # Simulate work
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            elapsed = time.time() - start_time
            self.performance_metrics.record_generation(count, elapsed)
            
            return {
                "generated": count,
                "elapsed": f"{elapsed:.3f}s",
                "rate": f"{count/elapsed:.0f} txn/sec"
            }
        
        return app
    
    def run_api_server(self):
        """Run the monitoring API server."""
        print("\n\n5. MONITORING API SERVER")
        print("-" * 60)
        print("\nStarting FastAPI server with metrics middleware...")
        
        app = self.create_monitoring_api()
        
        print("\n✓ Server configuration:")
        print("  Host: 0.0.0.0")
        print("  Port: 8000")
        print("  Metrics endpoint: http://localhost:8000/metrics")
        print("  Health check: http://localhost:8000/health")
        print("\nEndpoints available:")
        print("  GET  /              - API info")
        print("  GET  /metrics       - Prometheus metrics")
        print("  GET  /health        - Health check")
        print("  POST /simulate/fraud - Simulate fraud detection")
        print("  POST /simulate/generation - Simulate data generation")
        
        print("\nTo start the server, run:")
        print("  uvicorn examples.monitoring_demo:app --host 0.0.0.0 --port 8000")
        
        print("\nTo view metrics in Prometheus:")
        print("  1. Start monitoring stack: docker-compose --profile monitoring up -d")
        print("  2. Access Prometheus: http://localhost:9090")
        print("  3. Access Grafana: http://localhost:3000 (admin/admin123)")
        
        return app


def main():
    """Run monitoring demonstration."""
    demo = MonitoringDemo()
    
    try:
        # 1. Demonstrate Prometheus metrics
        demo.demonstrate_prometheus_metrics()
        
        # 2. Demonstrate fraud metrics
        demo.demonstrate_fraud_metrics()
        
        # 3. Demonstrate performance metrics
        demo.demonstrate_performance_metrics()
        
        # 4. Demonstrate data quality metrics
        demo.demonstrate_data_quality_metrics()
        
        # 5. Show API server configuration
        demo.run_api_server()
        
        print("\n" + "=" * 60)
        print("MONITORING DEMO COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start monitoring stack:")
        print("   docker-compose --profile monitoring up -d")
        print("\n2. Access dashboards:")
        print("   Prometheus: http://localhost:9090")
        print("   Grafana: http://localhost:3000")
        print("\n3. View pre-configured dashboards in Grafana:")
        print("   - System Overview")
        print("   - Fraud Detection")
        print("   - Performance Analytics")
        print("   - Data Quality")
        print("\n4. Trigger alerts by simulating high load:")
        print("   POST http://localhost:8000/simulate/fraud?count=1000")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


# Create app instance for uvicorn
demo_instance = MonitoringDemo()
app = demo_instance.create_monitoring_api()


if __name__ == "__main__":
    main()
