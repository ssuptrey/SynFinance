"""
Observability Framework Demo

Demonstrates the complete observability framework for SynFinance.
Shows structured logging, profiling, and debugging capabilities.
"""

import time
from datetime import datetime

from src.observability import (
    # Logging
    get_logger,
    configure_logging,
    LogLevel,
    LogCategory,
    
    # Profiling
    PerformanceProfiler,
    profile_operation,
    
    # Inspection
    DebugInspector,
    inspect_object
)


def demo_structured_logging():
    """Demonstrate structured logging capabilities"""
    print("\n" + "=" * 80)
    print("STRUCTURED LOGGING DEMO")
    print("=" * 80)
    
    # Get logger
    logger = get_logger("demo")
    
    # Basic logging
    logger.info("Application started")
    logger.debug("Debug message", extra={"config": "loaded"})
    logger.warning("Warning message", category=LogCategory.SYSTEM)
    
    # Context-based logging
    with logger.context(
        request_id="req-123",
        user_id="user-456",
        customer_id="cust-789"
    ):
        logger.info("Processing fraud detection request")
        logger.info("Feature extraction complete", extra={"feature_count": 25})
        logger.info("Model prediction complete", extra={"fraud_score": 0.85})
    
    # Operation timing
    with logger.operation("data_generation"):
        time.sleep(0.1)  # Simulate work
        logger.info("Generated 1000 transactions")
    
    # Performance stats
    stats = logger.get_performance_stats()
    print(f"\nPerformance Stats: {stats}")
    
    # API request logging
    logger.log_api_request(
        method="POST",
        path="/api/v1/fraud/detect",
        status_code=200,
        duration_ms=45.3,
        extra={"request_size": 1024}
    )
    
    # Security event logging
    logger.log_security_event(
        event_type="authentication",
        severity=LogLevel.WARNING,
        message="Failed login attempt",
        user_id="user-999",
        ip="192.168.1.100",
        attempts=3
    )


def demo_cpu_profiling():
    """Demonstrate CPU profiling"""
    print("\n" + "=" * 80)
    print("CPU PROFILING DEMO")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    def fibonacci(n):
        """Compute fibonacci number (inefficiently for demo)"""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    with profiler.profile("fibonacci_calculation", enable_memory=False, enable_io=False) as result:
        fib_result = fibonacci(20)
        print(f"Fibonacci(20) = {fib_result}")
    
    # Print profiling report
    report = profiler.generate_report(result)
    print(report)


def demo_memory_profiling():
    """Demonstrate memory profiling"""
    print("\n" + "=" * 80)
    print("MEMORY PROFILING DEMO")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    with profiler.profile("memory_allocation", enable_cpu=False, enable_io=False) as result:
        # Allocate memory
        data = []
        for i in range(100):
            data.append([0] * 10000)
        
        profiler.memory_profiler.snapshot("after_allocation")
        
        # Clear memory
        data.clear()
        
        profiler.memory_profiler.snapshot("after_clear")
    
    print(f"Memory Delta: {result.memory_delta_mb:+.2f} MB")
    if 'memory_snapshots' in result.metadata:
        print("\nMemory Snapshots:")
        for label, memory_mb in result.metadata['memory_snapshots']:
            print(f"  {label}: {memory_mb:.2f} MB")


@profile_operation("decorated_function")
def demo_profile_decorator():
    """Demonstrate profiling decorator"""
    print("\n" + "=" * 80)
    print("PROFILING DECORATOR DEMO")
    print("=" * 80)
    
    # Do some work
    total = sum(range(1000000))
    
    # Allocate some memory
    data = [i ** 2 for i in range(10000)]
    
    return total


def demo_object_inspection():
    """Demonstrate object inspection"""
    print("\n" + "=" * 80)
    print("OBJECT INSPECTION DEMO")
    print("=" * 80)
    
    # Create sample transaction
    class Transaction:
        def __init__(self):
            self.transaction_id = "tx-12345"
            self.customer_id = "cust-67890"
            self.merchant_id = "merch-111"
            self.amount = 150.75
            self.timestamp = datetime.now()
            self.category = "retail"
            self.is_fraud = False
            self.fraud_score = 0.12
        
        def validate(self):
            """Validate transaction"""
            return self.amount > 0
    
    tx = Transaction()
    
    # Inspect transaction
    inspector = DebugInspector()
    result = inspector.inspect(tx, "sample_transaction")
    
    inspector.print_report(result)


def demo_feature_inspection():
    """Demonstrate feature inspection"""
    print("\n" + "=" * 80)
    print("FEATURE INSPECTION DEMO")
    print("=" * 80)
    
    # Create sample features
    features = {
        "transaction_amount": 150.75,
        "merchant_category": "retail",
        "hour_of_day": 14,
        "day_of_week": 3,
        "is_weekend": False,
        "customer_age": 35,
        "avg_transaction_amount_7d": 120.50,
        "transaction_velocity_1h": 3,
        "distance_from_home_km": 5.2,
        "merchant_risk_score": 0.15
    }
    
    # Inspect features
    from src.observability import FeatureInspector
    inspector = FeatureInspector()
    result = inspector.inspect_features(features, "fraud_features")
    
    print(f"Feature Count: {result.metadata['feature_count']}")
    print(f"Feature Names: {result.metadata['feature_names']}")
    print(f"\nAnalysis:")
    for key, value in result.metadata['analysis'].items():
        print(f"  {key}: {value}")


def demo_complete_workflow():
    """Demonstrate complete observability workflow"""
    print("\n" + "=" * 80)
    print("COMPLETE OBSERVABILITY WORKFLOW")
    print("=" * 80)
    
    logger = get_logger("workflow")
    profiler = PerformanceProfiler()
    inspector = DebugInspector()
    
    # Start workflow with context
    with logger.context(
        request_id="req-workflow-001",
        operation="fraud_detection_pipeline"
    ):
        logger.info("Starting fraud detection pipeline")
        
        # Profile the entire operation
        with profiler.profile("fraud_detection") as result:
            # Step 1: Feature extraction
            with logger.operation("feature_extraction"):
                features = {
                    "amount": 250.0,
                    "merchant": "online_retailer",
                    "hour": 22,
                    "velocity": 5
                }
                logger.info("Features extracted", extra={"count": len(features)})
            
            # Step 2: Inspect features
            feature_result = inspector.inspect(features, "extracted_features")
            logger.debug(
                "Feature inspection complete",
                extra={"feature_count": feature_result.metadata['feature_count']}
            )
            
            # Step 3: Model prediction (simulated)
            with logger.operation("model_prediction"):
                time.sleep(0.05)  # Simulate prediction
                fraud_score = 0.78
                logger.info("Prediction complete", extra={"fraud_score": fraud_score})
            
            # Step 4: Decision
            is_fraud = fraud_score > 0.5
            logger.info(
                "Fraud detection complete",
                category=LogCategory.MODEL,
                extra={"is_fraud": is_fraud, "score": fraud_score}
            )
        
        # Log completion
        logger.info(
            "Pipeline complete",
            extra={"duration_seconds": result.duration_seconds}
        )
        
        # Print performance stats
        stats = logger.get_performance_stats()
        print(f"\nOperation Performance:")
        for op_name, op_stats in stats.items():
            print(f"  {op_name}:")
            print(f"    Executions: {op_stats['count']}")
            print(f"    Mean Time: {op_stats['mean']:.3f}s")
            print(f"    Total Time: {op_stats['total']:.3f}s")


if __name__ == "__main__":
    # Configure logging for demo
    configure_logging(level=LogLevel.INFO)
    
    print("\n" + "=" * 80)
    print("SYNFINANCE OBSERVABILITY FRAMEWORK DEMONSTRATION")
    print("Week 7 Day 4: Enhanced Observability")
    print("=" * 80)
    
    # Run demos
    demo_structured_logging()
    demo_cpu_profiling()
    demo_memory_profiling()
    demo_profile_decorator()
    demo_object_inspection()
    demo_feature_inspection()
    demo_complete_workflow()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
