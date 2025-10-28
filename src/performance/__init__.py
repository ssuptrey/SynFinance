"""
Performance optimization module for SynFinance.

This module provides high-performance data generation and processing capabilities:
- ParallelGenerator: Multi-core parallel processing
- StreamingGenerator: Memory-efficient streaming
- CacheManager: Intelligent caching
- Benchmarking utilities
"""

from .parallel_generator import ParallelGenerator, GenerationConfig, GenerationStats, quick_generate
from .streaming_generator import StreamingGenerator, StreamConfig, StreamStats
from .cache_manager import CacheManager, CacheStats
from .benchmarks import PerformanceBenchmark, BenchmarkResult, quick_benchmark

__all__ = [
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
]
