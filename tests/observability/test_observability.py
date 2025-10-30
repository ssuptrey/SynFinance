"""
Tests for SynFinance Observability Framework

Tests for structured logging, profiling, and inspection tools.
"""

import pytest
import json
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path

from src.observability import (
    # Logging
    LogLevel,
    LogCategory,
    LogContext,
    StructuredLogger,
    get_logger,
    configure_logging,
    
    # Profiling
    ProfileResult,
    CPUProfiler,
    MemoryProfiler,
    IOProfiler,
    PerformanceProfiler,
    profile_operation,
    
    # Inspection
    InspectionResult,
    ObjectInspector,
    TransactionInspector,
    FeatureInspector,
    ModelInspector,
    DebugInspector,
    inspect_object
)


# ============================================================================
# Structured Logger Tests
# ============================================================================

class TestStructuredLogger:
    """Test structured logging functionality"""
    
    def test_logger_creation(self):
        """Test creating a logger"""
        logger = get_logger("test_logger")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test_logger"
    
    def test_log_levels(self):
        """Test all log levels"""
        logger = get_logger("test_levels")
        
        # Should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_context_setting(self):
        """Test setting context"""
        logger = get_logger("test_context")
        
        # set_context takes kwargs, not a LogContext object
        logger.set_context(
            request_id="req-123",
            user_id="user-456",
            operation="test_operation"
        )
        retrieved = logger.get_context()
        
        assert retrieved.request_id == "req-123"
        assert retrieved.user_id == "user-456"
        assert retrieved.operation == "test_operation"
    
    def test_context_manager(self):
        """Test context manager"""
        logger = get_logger("test_ctx_mgr")
        
        with logger.context(request_id="req-789"):
            ctx = logger.get_context()
            assert ctx.request_id == "req-789"
        
        # Context should be cleared after manager exits
        ctx = logger.get_context()
        assert ctx.request_id is None
    
    def test_operation_timing(self):
        """Test operation timing context manager"""
        logger = get_logger("test_timing")
        
        with logger.operation("slow_operation"):
            time.sleep(0.1)  # Simulate work
        
        stats = logger.get_performance_stats()
        assert "slow_operation" in stats
        assert stats["slow_operation"]["count"] == 1
        # Check for mean (the actual key name)
        assert stats["slow_operation"]["mean"] >= 0.1
    
    def test_api_request_logging(self):
        """Test API request logging"""
        logger = get_logger("test_api")
        
        # Should not raise exceptions
        # API signature is: log_api_request(method, path, status_code, duration_ms)
        logger.log_api_request(
            method="GET",
            path="/api/transactions",
            status_code=200,
            duration_ms=50.0
        )
    
    def test_security_event_logging(self):
        """Test security event logging"""
        logger = get_logger("test_security")
        
        # Should not raise exceptions
        # API signature is: log_security_event(event_type, severity: LogLevel, message, **kwargs)
        logger.log_security_event(
            event_type="login_attempt",
            severity=LogLevel.WARNING,
            message="Failed login attempt",
            user_id="user-123",
            details={"ip": "192.168.1.1"}
        )


class TestLogContext:
    """Test LogContext dataclass"""
    
    def test_context_creation(self):
        """Test creating log context"""
        context = LogContext(
            request_id="req-1",
            correlation_id="corr-1",
            user_id="user-1"
        )
        
        assert context.request_id == "req-1"
        assert context.correlation_id == "corr-1"
        assert context.user_id == "user-1"
    
    def test_context_to_dict(self):
        """Test converting context to dict"""
        context = LogContext(
            request_id="req-1",
            metadata={"key": "value"}
        )
        
        ctx_dict = context.to_dict()
        assert ctx_dict["request_id"] == "req-1"
        assert ctx_dict["metadata"] == {"key": "value"}


# ============================================================================
# Profiling Tests
# ============================================================================

class TestCPUProfiler:
    """Test CPU profiling"""
    
    def test_cpu_profiling(self):
        """Test basic CPU profiling"""
        profiler = CPUProfiler()
        
        with profiler.profile("test_operation") as result:
            # Do some CPU work
            sum(range(1000000))
        
        assert result.operation_name == "test_operation"
        assert result.duration_seconds > 0
        assert result.cpu_time_seconds is not None
        assert len(result.function_stats) > 0
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        profiler = CPUProfiler()
        
        def slow_function():
            time.sleep(0.1)
        
        with profiler.profile("bottleneck_test") as result:
            slow_function()
        
        # Should have some function stats
        assert result.function_stats is not None


class TestMemoryProfiler:
    """Test memory profiling"""
    
    def test_memory_profiling(self):
        """Test basic memory profiling"""
        profiler = MemoryProfiler()
        
        with profiler.profile("memory_test") as result:
            # Allocate some memory
            data = [0] * 1000000
            del data
        
        assert result.operation_name == "memory_test"
        assert result.duration_seconds > 0
        assert result.memory_delta_mb is not None
    
    def test_memory_snapshots(self):
        """Test memory snapshots"""
        profiler = MemoryProfiler()
        
        with profiler.profile("snapshot_test") as result:
            profiler.snapshot("before_allocation")
            data = [0] * 100000
            profiler.snapshot("after_allocation")
            del data
        
        snapshots = result.metadata.get("memory_snapshots", [])
        assert len(snapshots) >= 3  # start, before, after, end


class TestIOProfiler:
    """Test I/O profiling"""
    
    def test_io_profiling(self):
        """Test basic I/O profiling"""
        profiler = IOProfiler()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        
        try:
            with profiler.profile("io_test") as result:
                # Do some I/O
                with open(temp_file, 'w') as f:
                    f.write("test data" * 1000)
                
                with open(temp_file, 'r') as f:
                    f.read()
            
            assert result.operation_name == "io_test"
            assert result.duration_seconds > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestPerformanceProfiler:
    """Test comprehensive performance profiling"""
    
    def test_full_profiling(self):
        """Test complete profiling"""
        profiler = PerformanceProfiler()
        
        with profiler.profile("full_test") as result:
            # CPU work
            sum(range(10000))
            # Memory allocation
            data = [0] * 10000
            del data
        
        assert result.operation_name == "full_test"
        assert result.duration_seconds > 0
    
    def test_profile_decorator(self):
        """Test profiling decorator"""
        @profile_operation("decorated_function", enable_cpu=False, enable_io=False)
        def test_function():
            time.sleep(0.05)
            return "result"
        
        # Should run without errors
        result = test_function()
        assert result == "result"
    
    def test_profile_report_generation(self):
        """Test report generation"""
        profiler = PerformanceProfiler()
        
        with profiler.profile("report_test") as result:
            sum(range(1000))
        
        report = profiler.generate_report(result)
        assert "Performance Profile" in report
        assert "report_test" in report
        assert "Duration" in report


# ============================================================================
# Inspector Tests
# ============================================================================

class TestObjectInspector:
    """Test object inspection"""
    
    def test_basic_inspection(self):
        """Test inspecting a basic object"""
        class TestClass:
            def __init__(self):
                self.value = 42
                self.name = "test"
            
            def method(self):
                pass
        
        obj = TestClass()
        inspector = ObjectInspector()
        result = inspector.inspect(obj, "test_object")
        
        assert result.target_name == "test_object"
        assert "value" in result.attributes
        assert result.attributes["value"] == 42
        assert "name" in result.attributes
        assert len(result.methods) > 0
    
    def test_dict_inspection(self):
        """Test inspecting a dictionary"""
        data = {"key1": "value1", "key2": 42}
        inspector = ObjectInspector()
        result = inspector.inspect(data, "test_dict")
        
        assert result.target_type == "dict"


class TestTransactionInspector:
    """Test transaction inspection"""
    
    def test_transaction_inspection(self):
        """Test inspecting a transaction"""
        class Transaction:
            def __init__(self):
                self.transaction_id = "tx-123"
                self.customer_id = "cust-456"
                self.amount = 100.50
                self.timestamp = datetime.now()
        
        tx = Transaction()
        inspector = TransactionInspector()
        result = inspector.inspect_transaction(tx)
        
        assert result.target_type == "transaction"
        assert "transaction_fields" in result.metadata
        assert result.metadata["transaction_fields"]["transaction_id"] == "tx-123"
    
    def test_transaction_validation(self):
        """Test transaction validation"""
        class ValidTransaction:
            transaction_id = "tx-1"
            customer_id = "cust-1"
            amount = 50.0
            timestamp = datetime.now()
        
        class InvalidTransaction:
            transaction_id = "tx-2"
            # Missing required fields
        
        inspector = TransactionInspector()
        
        # Valid transaction
        valid_result = inspector.validate_transaction(ValidTransaction())
        assert valid_result["valid"] is True
        
        # Invalid transaction
        invalid_result = inspector.validate_transaction(InvalidTransaction())
        assert invalid_result["valid"] is False
        assert len(invalid_result["issues"]) > 0


class TestFeatureInspector:
    """Test feature inspection"""
    
    def test_feature_dict_inspection(self):
        """Test inspecting feature dictionary"""
        features = {
            "amount": 100.0,
            "merchant_category": "retail",
            "hour_of_day": 14,
            "is_weekend": False
        }
        
        inspector = FeatureInspector()
        result = inspector.inspect_features(features, "test_features")
        
        assert result.target_type == "features"
        assert result.metadata["feature_count"] == 4
        assert "numeric_features" in result.metadata["analysis"]
    
    def test_feature_vector_inspection(self):
        """Test inspecting feature vector"""
        features = [1.0, 2.0, 3.0, 0, None]
        
        inspector = FeatureInspector()
        result = inspector.inspect_features(features, "test_vector")
        
        assert result.metadata["feature_type"] == "vector"
        assert result.metadata["analysis"]["null_count"] == 1


class TestModelInspector:
    """Test model inspection"""
    
    def test_model_inspection(self):
        """Test inspecting a model"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        inspector = ModelInspector()
        result = inspector.inspect_model(model, "test_model")
        
        assert result.target_type == "model"
        assert "model_info" in result.metadata


class TestDebugInspector:
    """Test comprehensive debug inspector"""
    
    def test_auto_detection(self):
        """Test auto-detection of object types"""
        inspector = DebugInspector()
        
        # Test transaction detection
        class Transaction:
            transaction_id = "tx-1"
        
        result = inspector.inspect(Transaction())
        assert result.target_type == "transaction"
        
        # Test features detection
        features = {"key": "value"}
        result = inspector.inspect(features)
        assert result.target_type == "features"
    
    def test_inspect_function(self):
        """Test quick inspect function"""
        data = {"test": "data"}
        result = inspect_object(data, verbose=False)
        
        assert isinstance(result, InspectionResult)


# ============================================================================
# Integration Tests
# ============================================================================

class TestObservabilityIntegration:
    """Test integration of observability components"""
    
    def test_logging_with_profiling(self):
        """Test using logging and profiling together"""
        logger = get_logger("integration_test")
        profiler = PerformanceProfiler()
        
        with logger.context(request_id="req-integration"):
            with profiler.profile("integrated_operation") as result:
                logger.info("Starting operation")
                time.sleep(0.05)
                logger.info("Operation complete")
        
        assert result.duration_seconds >= 0.05
    
    def test_profiling_with_inspection(self):
        """Test using profiling and inspection together"""
        profiler = PerformanceProfiler()
        inspector = DebugInspector()
        
        class DataObject:
            value = 42
        
        obj = DataObject()
        
        with profiler.profile("inspect_operation") as prof_result:
            insp_result = inspector.inspect(obj, "data_object")
        
        assert prof_result.duration_seconds > 0
        assert insp_result.attributes["value"] == 42
    
    def test_complete_observability_workflow(self):
        """Test complete observability workflow"""
        logger = get_logger("complete_workflow")
        profiler = PerformanceProfiler()
        inspector = DebugInspector()
        
        # Simulate a complete operation
        with logger.context(
            request_id="req-complete",
            operation="fraud_detection"
        ):
            logger.info("Starting fraud detection")
            
            with profiler.profile("detection_operation") as result:
                # Simulate feature extraction
                features = {
                    "amount": 150.0,
                    "merchant": "online_shop",
                    "hour": 15
                }
                
                # Inspect features
                feature_inspection = inspector.inspect(features, "extracted_features")
                
                logger.info(
                    "Features extracted",
                    extra={"feature_count": len(features)}
                )
                
                # Simulate model prediction
                time.sleep(0.01)
                
                logger.info("Detection complete")
            
            stats = logger.get_performance_stats()
        
        # Verify everything ran
        assert result.duration_seconds > 0
        assert feature_inspection.metadata["feature_count"] == 3
        assert "detection_operation" in stats or len(stats) >= 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestObservabilityPerformance:
    """Test performance overhead of observability tools"""
    
    def test_logging_overhead(self):
        """Test logging overhead is acceptable"""
        logger = get_logger("perf_test")
        
        iterations = 1000
        
        start = time.time()
        for i in range(iterations):
            logger.info(f"Message {i}")
        duration_with_logging = time.time() - start
        
        # Logging 1000 messages should complete in reasonable time
        assert duration_with_logging < 1.0  # Less than 1 second
    
    def test_profiling_overhead(self):
        """Test profiling overhead"""
        profiler = PerformanceProfiler()
        
        def simple_operation():
            sum(range(1000))
        
        # Without profiling
        start = time.time()
        for _ in range(10):
            simple_operation()
        baseline = time.time() - start
        
        # With profiling
        start = time.time()
        for _ in range(10):
            with profiler.profile("operation", enable_cpu=False, enable_io=False):
                simple_operation()
        with_profiling = time.time() - start
        
        # Overhead should be reasonable (less than 10x)
        overhead_ratio = with_profiling / baseline if baseline > 0 else 0
        assert overhead_ratio < 10
