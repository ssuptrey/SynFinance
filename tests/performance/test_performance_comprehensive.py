"""
Comprehensive tests for performance optimization modules.

Tests QueryOptimizer, Profiler, MetricsCollector, LoadTester, and Optimizer.
"""

import pytest
import time
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

from src.performance import (
    QueryOptimizer, QueryAnalysis, IndexRecommendation,
    Profiler, ProfileResult,
    MetricsCollector, MetricSnapshot, Alert,
    LoadTester, LoadTestResult,
    Optimizer, BatchProcessor, AsyncProcessor, ParallelProcessor,
    DataFrameOptimizer, LazyLoader
)


# ============================================================================
# QueryOptimizer Tests
# ============================================================================

class TestQueryOptimizer:
    """Test QueryOptimizer functionality."""
    
    def test_query_analysis_initialization(self):
        """Test QueryAnalysis object creation."""
        query = "SELECT * FROM transactions WHERE customer_id = '123'"
        analysis = QueryAnalysis(query)
        
        assert analysis.query == query
        assert analysis.execution_time_ms == 0.0
        assert analysis.rows_scanned == 0
        assert isinstance(analysis.recommendations, list)
    
    def test_index_recommendation_creation(self):
        """Test IndexRecommendation creation."""
        rec = IndexRecommendation(
            table='transactions',
            columns=['customer_id', 'timestamp'],
            reason='Slow query optimization',
            estimated_improvement=0.5
        )
        
        assert rec.table == 'transactions'
        assert len(rec.columns) == 2
        assert rec.estimated_improvement == 0.5
        assert isinstance(rec.to_dict(), dict)
    
    def test_query_optimizer_initialization(self):
        """Test QueryOptimizer initialization with SQLite."""
        # Use in-memory SQLite for testing
        optimizer = QueryOptimizer('sqlite:///:memory:')
        
        assert optimizer.engine is not None
        assert isinstance(optimizer.slow_queries, list)
        assert isinstance(optimizer.query_stats, dict)
    
    def test_slow_query_detection(self):
        """Test slow query detection."""
        optimizer = QueryOptimizer('sqlite:///:memory:')
        
        # Manually add a slow query
        analysis = QueryAnalysis("SELECT * FROM users")
        analysis.execution_time_ms = 150.0
        optimizer.slow_queries.append(analysis)
        
        slow = optimizer.detect_slow_queries(threshold_ms=100)
        
        assert len(slow) == 1
        assert slow[0].execution_time_ms > 100
    
    def test_pool_stats(self):
        """Test connection pool statistics."""
        optimizer = QueryOptimizer('sqlite:///:memory:')
        stats = optimizer.get_pool_stats()
        
        assert 'size' in stats
        assert 'checked_in' in stats
        assert 'total_connections' in stats


# ============================================================================
# Profiler Tests
# ============================================================================

class TestProfiler:
    """Test Profiler functionality."""
    
    def test_profiler_initialization(self):
        """Test Profiler initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            assert profiler.output_dir == tmpdir
            assert isinstance(profiler.profiles, list)
            assert os.path.exists(tmpdir)
    
    def test_cpu_profiling(self):
        """Test CPU profiling context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            # Profile a simple function
            with profiler.profile_cpu('test_function'):
                result = sum(range(1000))
            
            assert len(profiler.profiles) == 1
            assert profiler.profiles[0].name == 'test_function'
            assert profiler.profiles[0].profile_type == 'cpu'
            assert profiler.profiles[0].duration_seconds > 0
    
    def test_memory_profiling(self):
        """Test memory profiling context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            # Profile memory usage
            with profiler.profile_memory('test_memory'):
                data = [i for i in range(10000)]
            
            assert len(profiler.profiles) == 1
            assert profiler.profiles[0].profile_type == 'memory'
            assert 'current_mb' in profiler.profiles[0].stats
            assert 'peak_mb' in profiler.profiles[0].stats
    
    def test_profile_function_decorator(self):
        """Test profile_function decorator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            @profiler.profile_function
            def expensive_function():
                return sum(range(1000))
            
            result = expensive_function()
            
            assert result == sum(range(1000))
            assert len(profiler.profiles) == 1
            assert profiler.profiles[0].name == 'expensive_function'
    
    def test_profiling_summary(self):
        """Test profiling summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            with profiler.profile_cpu('test1'):
                time.sleep(0.01)
            
            with profiler.profile_memory('test2'):
                data = [i for i in range(1000)]
            
            summary = profiler.get_profiling_summary()
            
            assert summary['total_profiles'] == 2
            assert summary['cpu_profiles'] == 1
            assert summary['memory_profiles'] == 1
    
    def test_hotspot_extraction(self):
        """Test CPU hotspot extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            with profiler.profile_cpu('hotspot_test'):
                # Create some CPU-intensive work
                for i in range(100):
                    _ = sum(range(100))
            
            assert len(profiler.profiles[0].hotspots) > 0
            hotspot = profiler.profiles[0].hotspots[0]
            assert 'function' in hotspot
            assert 'total_time' in hotspot


# ============================================================================
# MetricsCollector Tests
# ============================================================================

class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(window_size=3600)
        
        assert collector.window_size == 3600
        assert isinstance(collector.metrics, dict)
        assert isinstance(collector.request_metrics, dict)
        assert isinstance(collector.alerts, list)
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()
        
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert 'network' in metrics
        assert 'timestamp' in metrics
    
    def test_application_metrics_collection(self):
        """Test application metrics collection."""
        collector = MetricsCollector()
        
        custom_metrics = {'custom_metric': 42.0}
        metrics = collector.collect_application_metrics(custom_metrics)
        
        assert 'process' in metrics
        assert 'custom' in metrics
        assert metrics['custom']['custom_metric'] == 42.0
    
    def test_request_recording(self):
        """Test request metrics recording."""
        collector = MetricsCollector()
        
        collector.record_request('/api/test', 50.0, 200)
        collector.record_request('/api/test', 75.0, 200)
        collector.record_request('/api/test', 100.0, 500)
        
        assert '/api/test' in collector.request_metrics
        assert len(collector.request_metrics['/api/test']['requests']) == 3
        assert len(collector.request_metrics['/api/test']['errors']) == 1
    
    def test_percentile_calculation(self):
        """Test percentile calculation."""
        collector = MetricsCollector()
        
        # Store some metric values
        for i in range(100):
            collector._store_metric('test_metric', float(i))
        
        p50 = collector.calculate_percentile('test_metric', 50)
        p95 = collector.calculate_percentile('test_metric', 95)
        
        assert 40 < p50 < 60
        assert 90 < p95 < 100
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        collector = MetricsCollector()
        
        # Record some requests
        for i in range(10):
            collector.record_request('/api/test', 50.0 + i, 200)
        
        summary = collector.get_metrics_summary(time_window=60)
        
        assert 'system' in summary
        assert 'requests' in summary
        assert '/api/test' in summary['requests']
        assert 'total_requests' in summary['requests']['/api/test']
    
    def test_alert_checking(self):
        """Test alert threshold checking."""
        collector = MetricsCollector()
        
        # Store high CPU metric
        collector._store_metric('cpu_percent', 90.0)
        
        thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0
        }
        
        alerts = collector.check_alerts(thresholds)
        
        assert len(alerts) == 1
        assert alerts[0].metric == 'cpu_percent'
        assert alerts[0].value == 90.0
        assert alerts[0].threshold == 80.0
    
    def test_prometheus_export(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        
        collector._store_metric('cpu_percent', 50.0)
        collector.record_request('/api/test', 25.0, 200)
        
        prometheus_output = collector.export_prometheus()
        
        assert 'synfinance_cpu_percent' in prometheus_output
        assert 'synfinance_requests_total' in prometheus_output
        assert 'HELP' in prometheus_output
        assert 'TYPE' in prometheus_output
    
    def test_uptime_tracking(self):
        """Test uptime calculation."""
        collector = MetricsCollector()
        time.sleep(0.1)
        
        uptime = collector.get_uptime()
        
        assert 'uptime_seconds' in uptime
        assert uptime['uptime_seconds'] > 0
        assert 'start_time' in uptime


# ============================================================================
# LoadTester Tests
# ============================================================================

class TestLoadTester:
    """Test LoadTester functionality."""
    
    def test_load_tester_initialization(self):
        """Test LoadTester initialization."""
        tester = LoadTester(base_url='http://localhost:8000')
        
        assert tester.base_url == 'http://localhost:8000'
        assert isinstance(tester.results, list)
        assert os.path.exists(tester.output_dir)
    
    def test_load_test_result_creation(self):
        """Test LoadTestResult object creation."""
        result = LoadTestResult('test_scenario')
        
        assert result.test_name == 'test_scenario'
        assert isinstance(result.timestamp, datetime)
        assert result.total_requests == 0
        assert isinstance(result.to_dict(), dict)
    
    def test_simulated_load_test(self):
        """Test simulated load test (when Locust not available)."""
        tester = LoadTester()
        
        # Use the simulated version directly
        result = tester._simulate_load_test(users=100, spawn_rate=10, duration=10)
        
        assert result.user_count == 100
        assert result.spawn_rate == 10
        assert result.total_requests > 0
        assert result.error_rate >= 0
    
    def test_analyze_results(self):
        """Test load test result analysis."""
        tester = LoadTester()
        
        result = LoadTestResult('test')
        result.error_rate = 0.005
        result.response_times = {'p95': 75.0, 'p99': 120.0}
        result.requests_per_second = 500.0
        result.user_count = 100
        
        analysis = tester.analyze_results(result)
        
        assert 'performance_grade' in analysis
        assert 'issues' in analysis
        assert 'recommendations' in analysis
    
    def test_compare_results(self):
        """Test comparison of two load test results."""
        tester = LoadTester()
        
        result1 = LoadTestResult('baseline')
        result1.requests_per_second = 100.0
        result1.error_rate = 0.01
        result1.response_times = {'p95': 100.0}
        
        result2 = LoadTestResult('improved')
        result2.requests_per_second = 150.0
        result2.error_rate = 0.005
        result2.response_times = {'p95': 80.0}
        
        comparison = tester.compare_results(result1, result2)
        
        assert 'improvements' in comparison
        assert 'regressions' in comparison
        assert len(comparison['improvements']) > 0


# ============================================================================
# Optimizer Tests
# ============================================================================

class TestOptimizer:
    """Test Optimizer functionality."""
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(batch_size=50)
        
        assert processor.batch_size == 50
        assert isinstance(processor.stats, dict)
    
    def test_batch_processing(self):
        """Test batch processing."""
        processor = BatchProcessor(batch_size=10)
        
        items = list(range(25))
        
        def batch_func(batch):
            return [x * 2 for x in batch]
        
        results = processor.process_batches(items, batch_func)
        
        assert len(results) == 25
        assert results[0] == 0
        assert results[24] == 48
        assert processor.stats['total_batches'] == 3
    
    def test_async_processor(self):
        """Test async processing."""
        processor = AsyncProcessor(max_concurrent=10)
        
        items = list(range(20))
        
        async def async_func(item):
            await asyncio.sleep(0.01)
            return item * 2
        
        results = processor.run_async(items, async_func)
        
        assert len(results) == 20
        assert 0 in results
        assert 38 in results
    
    def test_parallel_processor_threads(self):
        """Test parallel processing with threads."""
        processor = ParallelProcessor(max_workers=4)
        
        items = list(range(10))
        
        def process_func(item):
            return item * 2
        
        results = processor.process_parallel_threads(items, process_func)
        
        assert len(results) == 10
        processor.cleanup()
    
    def test_dataframe_optimizer_dtypes(self):
        """Test DataFrame dtype optimization."""
        # Create larger DataFrame for meaningful optimization
        df = pd.DataFrame({
            'col1': np.random.rand(10000).astype('float64'),
            'col2': np.random.randint(0, 100, 10000).astype('int64'),
            'col3': ['A', 'B', 'C'] * 3333 + ['A']
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = DataFrameOptimizer.optimize_dtypes(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Check memory reduction (should reduce due to category conversion)
        assert optimized_memory <= original_memory
    
    def test_lazy_loader(self):
        """Test lazy loading."""
        loader = LazyLoader()
        
        call_count = 0
        
        def expensive_load():
            nonlocal call_count
            call_count += 1
            return {'data': 'expensive'}
        
        # First call should load
        result1 = loader.get('test_key', expensive_load)
        assert call_count == 1
        assert result1['data'] == 'expensive'
        
        # Second call should use cache
        result2 = loader.get('test_key', expensive_load)
        assert call_count == 1  # Not incremented
        assert result2 == result1
    
    def test_optimizer_integration(self):
        """Test integrated Optimizer."""
        optimizer = Optimizer(max_workers=4, batch_size=10)
        
        items = list(range(25))
        
        def batch_func(batch):
            return [x * 2 for x in batch]
        
        results = optimizer.batch_process(items, batch_func)
        
        assert len(results) == 25
        assert optimizer.stats['operations'] == 1
        assert optimizer.stats['items_processed'] == 25
        
        optimizer.cleanup()
    
    def test_batch_optimized_decorator(self):
        """Test batch_optimized decorator."""
        from src.performance.optimizer import batch_optimized
        
        @batch_optimized(batch_size=10)
        def process_items(items):
            return [x * 2 for x in items]
        
        items = list(range(25))
        results = process_items(items)
        
        assert len(results) == 25
        assert results[0] == 0
        assert results[24] == 48


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for performance modules."""
    
    def test_end_to_end_performance_optimization(self):
        """Test complete performance optimization workflow."""
        # 1. Create optimizer
        optimizer = Optimizer()
        
        # 2. Setup metrics collector
        collector = MetricsCollector()
        
        # 3. Profile a batch operation
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(output_dir=tmpdir)
            
            with profiler.profile_cpu('batch_operation'):
                items = list(range(100))
                
                def process_batch(batch):
                    return [x * 2 for x in batch]
                
                results = optimizer.batch_process(items, process_batch, batch_size=10)
            
            # 4. Collect metrics
            metrics = collector.collect_system_metrics()
            
            # 5. Verify results
            assert len(results) == 100
            assert len(profiler.profiles) == 1
            assert 'cpu' in metrics
            
            # 6. Get stats
            opt_stats = optimizer.get_stats()
            prof_summary = profiler.get_profiling_summary()
            
            assert opt_stats['operations'] == 1
            assert prof_summary['cpu_profiles'] == 1
        
        optimizer.cleanup()
    
    def test_performance_monitoring_workflow(self):
        """Test performance monitoring workflow."""
        collector = MetricsCollector()
        
        # Simulate API requests
        for i in range(50):
            latency = 25.0 + (i % 20)
            status = 200 if i < 48 else 500
            collector.record_request('/api/test', latency, status)
        
        # Get summary
        summary = collector.get_metrics_summary(time_window=60)
        
        assert '/api/test' in summary['requests']
        assert summary['requests']['/api/test']['total_requests'] == 50
        assert summary['requests']['/api/test']['error_count'] == 2
        
        # Check alerts
        thresholds = {'p95_latency_ms': 30.0}
        alerts = collector.check_alerts(thresholds)
        
        assert len(alerts) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
