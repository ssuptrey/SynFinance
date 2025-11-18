"""
Performance optimization module for SynFinance.

This module provides high-performance data generation and processing capabilities:
- ParallelGenerator: Multi-core parallel processing
- StreamingGenerator: Memory-efficient streaming
- CacheManager: Intelligent caching
- Benchmarking utilities

Week 10 Day 5 additions:
- QueryOptimizer: Database query optimization and indexing
- Profiler: CPU and memory profiling
- MetricsCollector: Real-time performance metrics
- LoadTester: Load testing with Locust
- Optimizer: Batch processing and async optimizations
"""

from .parallel_generator import ParallelGenerator, GenerationConfig, GenerationStats, quick_generate
from .streaming_generator import StreamingGenerator, StreamConfig, StreamStats
from .cache_manager import CacheManager, CacheStats
from .benchmarks import PerformanceBenchmark, BenchmarkResult, quick_benchmark

# Week 10 Day 5: Performance optimization modules
from .query_optimizer import QueryOptimizer, QueryAnalysis, IndexRecommendation
from .profiler import Profiler, ProfileResult
from .metrics_collector import MetricsCollector, MetricSnapshot, Alert
from .load_tester import LoadTester, LoadTestResult, FraudDetectionUser, TransactionProcessingUser
from .optimizer import (
    Optimizer, BatchProcessor, AsyncProcessor, ParallelProcessor,
    DataFrameOptimizer, LazyLoader, batch_optimized, parallel_optimized
)

__all__ = [
    # Original exports
    'ParallelGenerator',
    'GenerationConfig',
    'GenerationStats',
    'quick_generate',
    'StreamingGenerator',
    'StreamConfig',
    'StreamStats',
    'CacheManager',
    'CacheStats',
    'PerformanceBenchmark',
    'BenchmarkResult',
    'quick_benchmark',
    
    # Week 10 Day 5 exports
    'QueryOptimizer',
    'QueryAnalysis',
    'IndexRecommendation',
    'Profiler',
    'ProfileResult',
    'MetricsCollector',
    'MetricSnapshot',
    'Alert',
    'LoadTester',
    'LoadTestResult',
    'FraudDetectionUser',
    'TransactionProcessingUser',
    'Optimizer',
    'BatchProcessor',
    'AsyncProcessor',
    'ParallelProcessor',
    'DataFrameOptimizer',
    'LazyLoader',
    'batch_optimized',
    'parallel_optimized',
]
