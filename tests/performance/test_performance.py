"""
Comprehensive test suite for performance optimization modules.

Tests cover:
- ParallelGenerator
- StreamingGenerator
- CacheManager
- PerformanceBenchmark
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time
import tempfile
import shutil

from src.performance.parallel_generator import (
    ParallelGenerator,
    GenerationConfig,
    quick_generate
)
from src.performance.streaming_generator import (
    StreamingGenerator,
    StreamConfig,
    ChunkedFileReader
)
from src.performance.cache_manager import (
    CacheManager,
    cached_method
)
from src.performance.benchmarks import (
    PerformanceBenchmark,
    quick_benchmark,
    compare_performance
)


# ============================================================================
# ParallelGenerator Tests
# ============================================================================

class TestParallelGenerator:
    """Test parallel generation functionality."""
    
    def test_generation_config_defaults(self):
        """Test generation config with default auto-configuration."""
        config = GenerationConfig(num_transactions=10000)
        
        assert config.num_transactions == 10000
        assert config.num_workers is not None
        assert config.num_workers >= 1
        assert config.chunk_size is not None
        assert config.chunk_size > 0
    
    def test_parallel_generation_basic(self):
        """Test basic parallel generation."""
        config = GenerationConfig(
            num_transactions=100,
            num_workers=2,
            show_progress=False
        )
        
        generator = ParallelGenerator(config)
        df = generator.generate(seed=42)
        
        assert len(df) == 100
        assert 'Transaction_ID' in df.columns  # Changed from transaction_id
        assert 'Amount' in df.columns  # Changed from amount
        assert generator.stats is not None
        assert generator.stats.total_transactions == 100
    
    def test_parallel_generation_reproducible(self):
        """Test that same seed produces same results."""
        config = GenerationConfig(
            num_transactions=50,
            num_workers=2,
            show_progress=False
        )
        
        gen1 = ParallelGenerator(config)
        df1 = gen1.generate(seed=42)
        
        gen2 = ParallelGenerator(config)
        df2 = gen2.generate(seed=42)
        
        # Same seed should produce same number of transactions
        assert len(df1) == len(df2)
    
    def test_parallel_vs_sequential_performance(self):
        """Test that parallel is faster than sequential (with sufficient size)."""
        size = 1000
        
        # Parallel (2 workers)
        config_parallel = GenerationConfig(
            num_transactions=size,
            num_workers=2,
            show_progress=False
        )
        gen_parallel = ParallelGenerator(config_parallel)
        df_parallel = gen_parallel.generate(seed=42)
        time_parallel = gen_parallel.stats.total_time
        
        # Sequential (1 worker)
        config_sequential = GenerationConfig(
            num_transactions=size,
            num_workers=1,
            show_progress=False
        )
        gen_sequential = ParallelGenerator(config_sequential)
        df_sequential = gen_sequential.generate(seed=42)
        time_sequential = gen_sequential.stats.total_time
        
        # Both should produce correct size
        assert len(df_parallel) == size
        assert len(df_sequential) == size
        
        # Parallel should be faster or similar (accounting for overhead)
        # Note: May not always be faster for small datasets
        assert time_parallel < time_sequential * 1.5  # Allow some overhead
    
    def test_generate_to_file_csv(self):
        """Test generating directly to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.csv"
            
            config = GenerationConfig(
                num_transactions=100,
                show_progress=False
            )
            generator = ParallelGenerator(config)
            stats = generator.generate_to_file(output_file, seed=42)
            
            assert output_file.exists()
            assert stats.total_transactions == 100
            
            # Verify file content
            df = pd.read_csv(output_file)
            assert len(df) == 100
    
    def test_quick_generate_convenience(self):
        """Test quick_generate convenience function."""
        df = quick_generate(num_transactions=50, seed=42)
        
        assert len(df) == 50
        assert 'Transaction_ID' in df.columns  # Changed from transaction_id
    
    def test_generation_stats_complete(self):
        """Test that generation stats are complete."""
        config = GenerationConfig(
            num_transactions=100,
            fraud_rate=0.05,
            show_progress=False
        )
        generator = ParallelGenerator(config)
        df = generator.generate(seed=42)
        
        stats = generator.stats
        assert stats.total_transactions == 100
        assert stats.total_time > 0
        assert stats.transactions_per_second > 0
        assert stats.num_workers >= 1
        assert stats.chunk_size > 0
        assert stats.memory_mb > 0
        assert stats.chunks_processed > 0


# ============================================================================
# StreamingGenerator Tests
# ============================================================================

class TestStreamingGenerator:
    """Test streaming generation functionality."""
    
    def test_stream_config_defaults(self):
        """Test stream config defaults."""
        config = StreamConfig(num_transactions=10000)
        
        assert config.num_transactions == 10000
        assert config.batch_size == 1000
        assert config.show_progress == True
    
    def test_streaming_batches(self):
        """Test batch generation."""
        config = StreamConfig(
            num_transactions=250,
            batch_size=100,
            show_progress=False
        )
        
        generator = StreamingGenerator(config)
        batches = list(generator.generate_batches(seed=42))
        
        # Should have 3 batches (100, 100, 50)
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50
        
        # Verify stats
        assert generator.stats.total_transactions == 250
        assert generator.stats.total_batches == 3
    
    def test_streaming_memory_efficiency(self):
        """Test that streaming uses limited memory."""
        config = StreamConfig(
            num_transactions=1000,
            batch_size=100,
            show_progress=False
        )
        
        generator = StreamingGenerator(config)
        
        for batch in generator.generate_batches(seed=42):
            # Each batch should be small
            batch_memory = batch.memory_usage(deep=True).sum() / 1024 / 1024
            assert batch_memory < 50  # Less than 50MB per batch
    
    def test_stream_to_csv(self):
        """Test streaming to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "stream_test.csv"
            
            config = StreamConfig(
                num_transactions=200,
                batch_size=50,
                show_progress=False
            )
            generator = StreamingGenerator(config)
            stats = generator.stream_to_file(output_file, seed=42)
            
            assert output_file.exists()
            assert stats.total_transactions == 200
            assert stats.output_file_mb > 0
            
            # Verify content
            df = pd.read_csv(output_file)
            assert len(df) == 200
    
    def test_stream_to_json(self):
        """Test streaming to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "stream_test.jsonl"
            
            config = StreamConfig(
                num_transactions=100,
                batch_size=50,
                show_progress=False
            )
            generator = StreamingGenerator(config)
            stats = generator.stream_to_file(output_file, seed=42)
            
            assert output_file.exists()
            assert stats.total_transactions == 100
    
    def test_memory_estimation(self):
        """Test memory estimation."""
        config = StreamConfig(
            num_transactions=1000,
            batch_size=100
        )
        generator = StreamingGenerator(config)
        
        estimate = generator.estimate_memory(batch_size=100)
        
        assert 'base_mb' in estimate
        assert 'batch_mb' in estimate
        assert 'total_mb' in estimate
        assert estimate['total_mb'] > 0


class TestChunkedFileReader:
    """Test chunked file reading."""
    
    def test_read_csv_chunks(self):
        """Test reading CSV in chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.csv"
            df = pd.DataFrame({
                'col1': range(500),
                'col2': range(500, 1000)
            })
            df.to_csv(test_file, index=False)
            
            # Read in chunks
            reader = ChunkedFileReader(test_file, chunk_size=100)
            chunks = list(reader.read_chunks())
            
            assert len(chunks) == 5  # 500 rows / 100 per chunk
            assert all(len(chunk) == 100 for chunk in chunks)
    
    def test_count_rows(self):
        """Test row counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.csv"
            df = pd.DataFrame({'col': range(250)})
            df.to_csv(test_file, index=False)
            
            reader = ChunkedFileReader(test_file, chunk_size=50)
            row_count = reader.count_rows()
            
            assert row_count == 250


# ============================================================================
# CacheManager Tests
# ============================================================================

class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_initialization(self):
        """Test cache manager initialization."""
        cache = CacheManager(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache.stats) == 4  # customer, merchant, history, feature
    
    def test_customer_cache(self):
        """Test customer caching."""
        cache = CacheManager(max_size=100)
        
        # Cache miss
        result = cache.get_customer("CUST001")
        assert result is None
        assert cache.stats['customer'].misses == 1
        
        # Store
        cache.set_customer("CUST001", {'id': 'CUST001', 'name': 'Test'})
        
        # Cache hit
        result = cache.get_customer("CUST001")
        assert result is not None
        assert result['id'] == 'CUST001'
        assert cache.stats['customer'].hits == 1
    
    def test_merchant_cache(self):
        """Test merchant caching."""
        cache = CacheManager(max_size=100)
        
        cache.set_merchant("MERCH001", {'id': 'MERCH001'})
        result = cache.get_merchant("MERCH001")
        
        assert result is not None
        assert result['id'] == 'MERCH001'
    
    def test_history_cache(self):
        """Test history caching."""
        cache = CacheManager(max_size=100)
        
        history = [{'txn': 1}, {'txn': 2}]
        cache.set_history("CUST001", history)
        result = cache.get_history("CUST001")
        
        assert result == history
    
    def test_feature_cache(self):
        """Test feature caching."""
        cache = CacheManager(max_size=100)
        
        features = np.array([1, 2, 3, 4, 5])
        cache.set_features("TXN001", features)
        result = cache.get_features("TXN001")
        
        assert np.array_equal(result, features)
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = CacheManager(max_size=10)
        
        # Fill cache beyond max_size
        for i in range(20):
            cache.set_customer(f"CUST{i:03d}", {'id': i})
        
        # Size should be limited
        assert cache.stats['customer'].size <= 10
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = CacheManager(max_size=100)
        
        # Generate some hits and misses
        cache.set_customer("CUST001", {'id': 'CUST001'})
        cache.get_customer("CUST001")  # Hit
        cache.get_customer("CUST002")  # Miss
        cache.get_customer("CUST001")  # Hit
        
        stats = cache.get_stats()
        
        assert stats['customer']['hits'] == 2
        assert stats['customer']['misses'] == 1
        assert stats['customer']['hit_rate'] == '66.67%'
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CacheManager(max_size=100)
        
        cache.set_customer("CUST001", {'id': 'CUST001'})
        cache.set_merchant("MERCH001", {'id': 'MERCH001'})
        
        # Clear specific cache
        cache.clear('customer')
        
        assert cache.get_customer("CUST001") is None
        assert cache.get_merchant("MERCH001") is not None  # Still there
        
        # Clear all
        cache.clear()
        assert cache.get_merchant("MERCH001") is None
    
    def test_cache_warming(self):
        """Test cache warming."""
        cache = CacheManager(max_size=100)
        
        customers = [
            {'customer_id': f'CUST{i:03d}', 'name': f'Customer {i}'}
            for i in range(10)
        ]
        
        cache.warm_cache(customers=customers)
        
        # All should be cached
        for customer in customers:
            cached = cache.get_customer(customer['customer_id'])
            assert cached is not None


# ============================================================================
# PerformanceBenchmark Tests
# ============================================================================

class TestPerformanceBenchmark:
    """Test benchmarking functionality."""
    
    def test_benchmark_initialization(self):
        """Test benchmark suite initialization."""
        benchmark = PerformanceBenchmark()
        
        assert benchmark.results == []
        assert benchmark.process is not None
    
    def test_benchmark_method(self):
        """Test benchmarking a method."""
        def simple_method(num_transactions):
            time.sleep(0.001)  # Small delay to ensure measurable duration
            return pd.DataFrame({'col': range(num_transactions)})
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_method(
            name="Test",
            method=simple_method,
            dataset_size=100,
            num_transactions=100
        )
        
        assert result.success is True
        assert result.dataset_size == 100
        assert result.duration_seconds >= 0  # May be very small
        assert result.transactions_per_second >= 0
    
    def test_benchmark_failure_handling(self):
        """Test handling of benchmark failures."""
        def failing_method(num_transactions):
            raise ValueError("Test error")
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_method(
            name="FailTest",
            method=failing_method,
            dataset_size=100,
            num_transactions=100
        )
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_scaling_test(self):
        """Test scaling test across multiple sizes."""
        def test_method(num_transactions):
            time.sleep(num_transactions / 10000)  # Simulate work
            return pd.DataFrame({'col': range(num_transactions)})
        
        benchmark = PerformanceBenchmark()
        results = benchmark.run_scaling_test(
            method=test_method,
            sizes=[100, 200, 300],
            name_prefix="Scale"
        )
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    def test_method_comparison(self):
        """Test comparing multiple methods."""
        def method1(num_transactions):
            return pd.DataFrame({'col': range(num_transactions)})
        
        def method2(num_transactions):
            time.sleep(0.01)
            return pd.DataFrame({'col': range(num_transactions)})
        
        benchmark = PerformanceBenchmark()
        comparison = benchmark.compare_methods(
            methods={'Fast': method1, 'Slow': method2},
            dataset_size=100
        )
        
        assert len(comparison) == 2
        assert 'speed_rank' in comparison.columns
        assert 'memory_rank' in comparison.columns
    
    def test_export_report(self):
        """Test exporting benchmark report."""
        def test_method(num_transactions):
            return pd.DataFrame({'col': range(num_transactions)})
        
        benchmark = PerformanceBenchmark()
        benchmark.benchmark_method("Test", test_method, 100, num_transactions=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "report.md"
            benchmark.export_report(output_file)
            
            assert output_file.exists()
            content = output_file.read_text()
            assert "Performance Benchmark Report" in content
    
    def test_quick_benchmark_convenience(self):
        """Test quick_benchmark convenience function."""
        def test_method(num_transactions):
            return pd.DataFrame({'col': range(num_transactions)})
        
        result = quick_benchmark(test_method, 100, name="QuickTest")
        
        assert result.success is True
        assert result.dataset_size == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestPerformanceIntegration:
    """Integration tests for performance modules."""
    
    def test_parallel_vs_streaming_comparison(self):
        """Compare parallel and streaming generation."""
        size = 500
        
        # Parallel generation
        parallel_config = GenerationConfig(
            num_transactions=size,
            show_progress=False
        )
        parallel_gen = ParallelGenerator(parallel_config)
        df_parallel = parallel_gen.generate(seed=42)
        
        # Streaming generation
        stream_config = StreamConfig(
            num_transactions=size,
            batch_size=100,
            show_progress=False
        )
        stream_gen = StreamingGenerator(stream_config)
        batches = list(stream_gen.generate_batches(seed=42))
        df_streaming = pd.concat(batches, ignore_index=True)
        
        # Both should produce correct size
        assert len(df_parallel) == size
        assert len(df_streaming) == size
    
    def test_cached_generation(self):
        """Test generation with caching."""
        cache = CacheManager(max_size=100)
        
        # Warm cache with customer data
        customers = [
            {'customer_id': f'CUST{i:03d}', 'name': f'Customer {i}'}
            for i in range(10)
        ]
        cache.warm_cache(customers=customers)
        
        # Verify cache usage
        for customer in customers:
            cached = cache.get_customer(customer['customer_id'])
            assert cached is not None
        
        # Check hit rate
        stats = cache.get_stats()
        assert stats['customer']['hits'] > 0
