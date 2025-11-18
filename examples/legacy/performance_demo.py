"""
Performance Optimization Demo - SynFinance

This example demonstrates all performance optimization features:
1. Parallel generation for speed
2. Streaming generation for memory efficiency
3. Caching for repeated operations
4. Benchmarking for performance analysis

Run with: python examples/performance_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.performance import (
    ParallelGenerator,
    StreamingGenerator,
    CacheManager,
    PerformanceBenchmark,
    GenerationConfig,
    StreamConfig,
    quick_generate,
    quick_benchmark
)
from src.data_generator import generate_realistic_dataset
import pandas as pd
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_parallel_generation():
    """Demo 1: Fast parallel generation for large datasets."""
    print_section("Demo 1: Parallel Generation (Speed Optimized)")
    
    print("Generating 50,000 transactions using 4 parallel workers...")
    
    config = GenerationConfig(
        num_transactions=50_000,
        num_workers=4,
        chunk_size=12_500,
        show_progress=True
    )
    
    generator = ParallelGenerator(config)
    start_time = time.time()
    df = generator.generate(seed=42)
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n[OK] Generated {len(df):,} transactions in {elapsed:.2f} seconds")
    print(f"  - Rate: {len(df)/elapsed:,.0f} transactions/second")
    print(f"  - Speedup: {generator.stats.speedup_factor:.2f}x over sequential")
    print(f"  - Peak memory: {generator.stats.peak_memory_mb:.1f} MB")
    
    return df


def demo_2_streaming_generation():
    """Demo 2: Memory-efficient streaming for massive datasets."""
    print_section("Demo 2: Streaming Generation (Memory Optimized)")
    
    print("Generating 100,000 transactions in streaming batches...")
    
    config = StreamConfig(
        total_transactions=100_000,
        batch_size=10_000,
        num_customers=10_000
    )
    
    generator = StreamingGenerator(config)
    
    # Estimate memory first
    estimated_mb = generator.estimate_memory()
    print(f"Estimated memory usage: {estimated_mb:.1f} MB")
    
    # Stream to CSV file
    output_path = Path("output/performance_demo_streaming.csv")
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nStreaming to file: {output_path}")
    generator.stream_to_file(output_path, format="csv")
    
    print(f"\n[OK] Generated {generator.stats.total_transactions:,} transactions")
    print(f"  - Batches: {generator.stats.batches_generated}")
    print(f"  - Peak memory: {generator.stats.peak_memory_mb:.1f} MB")
    
    return output_path


def demo_3_caching():
    """Demo 3: Use caching to speed up repeated operations."""
    print_section("Demo 3: Intelligent Caching")
    
    cache = CacheManager(
        max_customer_size=1_000,
        max_merchant_size=500,
        enable_disk_cache=False
    )
    
    print("Simulating repeated customer lookups...")
    
    # Simulate customer data
    def generate_customer(customer_id: str) -> dict:
        """Simulate expensive customer generation."""
        time.sleep(0.01)  # Simulate database query
        return {
            'id': customer_id,
            'name': f'Customer {customer_id}',
            'age': 25 + int(customer_id[-3:]) % 50
        }
    
    # First pass - cache misses
    print("\nFirst pass (cold cache):")
    start_time = time.time()
    for i in range(100):
        customer_id = f"CUST{i % 20:04d}"  # Only 20 unique customers
        customer = cache.get_customer(customer_id)
        if customer is None:
            customer = generate_customer(customer_id)
            cache.set_customer(customer_id, customer)
    cold_time = time.time() - start_time
    
    stats = cache.get_stats()
    print(f"  Time: {cold_time:.3f}s")
    print(f"  Hit rate: {stats.overall_hit_rate:.1%}")
    
    # Second pass - cache hits
    print("\nSecond pass (warm cache):")
    start_time = time.time()
    for i in range(100):
        customer_id = f"CUST{i % 20:04d}"
        customer = cache.get_customer(customer_id)
        if customer is None:
            customer = generate_customer(customer_id)
            cache.set_customer(customer_id, customer)
    warm_time = time.time() - start_time
    
    stats = cache.get_stats()
    print(f"  Time: {warm_time:.3f}s")
    print(f"  Hit rate: {stats.overall_hit_rate:.1%}")
    
    print(f"\n[OK] Speedup with cache: {cold_time/warm_time:.1f}x faster")
    
    return cache


def demo_4_benchmarking():
    """Demo 4: Benchmark different generation methods."""
    print_section("Demo 4: Performance Benchmarking")
    
    benchmark = PerformanceBenchmark()
    
    # Define methods to compare
    def method_standard():
        return generate_realistic_dataset(
            num_customers=1_000,
            transactions_per_customer=10
        )
    
    def method_parallel():
        return quick_generate(
            num_transactions=10_000,
            num_workers=4,
            seed=42
        )
    
    print("Benchmarking two generation methods (3 iterations each)...")
    
    # Compare methods
    comparison = benchmark.compare_methods(
        methods=[
            ("Standard Generation", method_standard),
            ("Parallel Generation", method_parallel)
        ],
        iterations=3
    )
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 80)
    for i, result in enumerate(comparison.results, 1):
        print(f"\n{i}. {result.method_name}:")
        print(f"   - Average time: {result.avg_time:.3f}s +/- {result.std_time:.3f}s")
        print(f"   - Throughput: {result.transactions_per_second:,.0f} txns/sec")
        print(f"   - Peak memory: {result.peak_memory_mb:.1f} MB")
    
    # Show ranking
    print(f"\n[OK] Best performing method: {comparison.results[0].method_name}")
    
    return comparison


def main():
    """Run all performance demos."""
    print("=" * 80)
    print("  SynFinance Performance Optimization Demo")
    print("  Parallel Processing | Streaming | Caching | Benchmarking")
    print("=" * 80)
    
    try:
        # Run demos
        demo_1_parallel_generation()
        demo_2_streaming_generation()
        demo_3_caching()
        demo_4_benchmarking()
        
        print_section("Demo Complete!")
        print("All performance features demonstrated successfully.")
        print("\nKey Takeaways:")
        print("  1. Use parallel generation for speed (< 100K transactions)")
        print("  2. Use streaming for memory efficiency (> 100K transactions)")
        print("  3. Cache frequently accessed data (70%+ speedup)")
        print("  4. Benchmark to validate performance improvements")
        print("\nFor more details, see: WEEK6_DAY5_COMPLETE.md")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
