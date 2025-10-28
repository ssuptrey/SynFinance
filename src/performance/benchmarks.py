"""
Comprehensive benchmarking suite for performance testing.

This module provides tools to measure and compare performance:
- Generation speed
- Memory usage
- Scaling characteristics
- Comparison reports
"""

import logging
import time
import psutil
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    dataset_size: int
    duration_seconds: float
    transactions_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_percent: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'dataset_size': self.dataset_size,
            'duration_seconds': round(self.duration_seconds, 2),
            'transactions_per_second': round(self.transactions_per_second, 0),
            'peak_memory_mb': round(self.peak_memory_mb, 1),
            'avg_memory_mb': round(self.avg_memory_mb, 1),
            'cpu_percent': round(self.cpu_percent, 1),
            'success': self.success,
            'error_message': self.error_message,
            **self.metadata
        }


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite.
    
    Features:
    - Memory profiling
    - CPU profiling
    - Scaling tests (1K to 1M)
    - Method comparison
    - Statistical analysis
    
    Example:
        >>> benchmark = PerformanceBenchmark()
        >>> 
        >>> # Run scaling test
        >>> results = benchmark.run_scaling_test(
        >>>     method=generate_data,
        >>>     sizes=[1000, 10000, 100000]
        >>> )
        >>>
        >>> # Export report
        >>> benchmark.export_report('benchmark_report.md')
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process(os.getpid())
    
    def benchmark_method(
        self,
        name: str,
        method: Callable,
        dataset_size: int,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single method.
        
        Args:
            name: Benchmark name
            method: Method to benchmark
            dataset_size: Size of dataset
            *args, **kwargs: Arguments to pass to method
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking: {name} (size={dataset_size:,})")
        
        # Track memory
        memory_samples = []
        
        # Start monitoring
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run method
            result = method(*args, **kwargs)
            
            # Measure
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            peak_memory = end_memory
            avg_memory = (start_memory + end_memory) / 2
            cpu_percent = self.process.cpu_percent()
            
            # Calculate throughput
            txns_per_sec = dataset_size / duration if duration > 0 else 0
            
            benchmark_result = BenchmarkResult(
                name=name,
                dataset_size=dataset_size,
                duration_seconds=duration,
                transactions_per_second=txns_per_sec,
                peak_memory_mb=peak_memory,
                avg_memory_mb=avg_memory,
                cpu_percent=cpu_percent,
                success=True
            )
            
            self.results.append(benchmark_result)
            
            logger.info(f"✓ {name}: {duration:.2f}s, {txns_per_sec:.0f} txns/sec, {peak_memory:.1f} MB")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            
            benchmark_result = BenchmarkResult(
                name=name,
                dataset_size=dataset_size,
                duration_seconds=0,
                transactions_per_second=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message=str(e)
            )
            
            self.results.append(benchmark_result)
            return benchmark_result
    
    def run_scaling_test(
        self,
        method: Callable,
        sizes: List[int] = None,
        name_prefix: str = "Scaling"
    ) -> List[BenchmarkResult]:
        """
        Run scaling test across multiple dataset sizes.
        
        Args:
            method: Method to test (should accept num_transactions parameter)
            sizes: List of dataset sizes (default: [1K, 10K, 100K])
            name_prefix: Prefix for benchmark names
            
        Returns:
            List of benchmark results
        """
        if sizes is None:
            sizes = [1000, 10000, 100000]
        
        results = []
        
        for size in sizes:
            result = self.benchmark_method(
                name=f"{name_prefix}_{size:,}",
                method=method,
                dataset_size=size,
                num_transactions=size
            )
            results.append(result)
        
        # Analyze scaling
        self._analyze_scaling(results)
        
        return results
    
    def _analyze_scaling(self, results: List[BenchmarkResult]):
        """Analyze scaling characteristics."""
        if len(results) < 2:
            return
        
        sizes = [r.dataset_size for r in results]
        times = [r.duration_seconds for r in results]
        
        # Calculate scaling factor
        # Perfect linear scaling: O(n) - time doubles when size doubles
        # Sub-linear: O(log n) - time increases slower than size
        # Super-linear: O(n^2) - time increases faster than size
        
        size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        
        avg_scaling = np.mean([t/s for t, s in zip(time_ratios, size_ratios)])
        
        if avg_scaling < 0.8:
            scaling_type = "Sub-linear (excellent)"
        elif avg_scaling < 1.2:
            scaling_type = "Linear (good)"
        else:
            scaling_type = "Super-linear (needs optimization)"
        
        logger.info(f"Scaling Analysis: {scaling_type} (factor={avg_scaling:.2f})")
    
    def compare_methods(
        self,
        methods: Dict[str, Callable],
        dataset_size: int
    ) -> pd.DataFrame:
        """
        Compare multiple methods on same dataset size.
        
        Args:
            methods: Dictionary of {name: method}
            dataset_size: Size to test
            
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for name, method in methods.items():
            result = self.benchmark_method(
                name=name,
                method=method,
                dataset_size=dataset_size,
                num_transactions=dataset_size
            )
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        
        # Add rankings
        if len(df) > 0:
            df['speed_rank'] = df['transactions_per_second'].rank(ascending=False)
            df['memory_rank'] = df['peak_memory_mb'].rank(ascending=True)
            df['overall_rank'] = (df['speed_rank'] + df['memory_rank']) / 2
        
        return df.sort_values('overall_rank')
    
    def get_results_df(self) -> pd.DataFrame:
        """Get all benchmark results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def export_report(
        self,
        output_file: Path,
        include_plots: bool = False
    ):
        """
        Export comprehensive benchmark report.
        
        Args:
            output_file: Path to output markdown file
            include_plots: Include performance plots
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.get_results_df()
        
        if df.empty:
            logger.warning("No benchmark results to export")
            return
        
        # Generate report
        report = self._generate_markdown_report(df, include_plots)
        
        # Save
        output_file.write_text(report, encoding='utf-8')
        logger.info(f"Saved benchmark report to {output_file}")
    
    def _generate_markdown_report(
        self,
        df: pd.DataFrame,
        include_plots: bool
    ) -> str:
        """Generate markdown report."""
        lines = [
            "# Performance Benchmark Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Benchmarks**: {len(df)}",
            f"**Successful**: {df['success'].sum()}",
            f"**Failed**: {(~df['success']).sum()}",
            "\n## Summary Statistics\n",
        ]
        
        # Summary stats
        if df['success'].any():
            successful = df[df['success']]
            lines.extend([
                f"- **Fastest**: {successful['transactions_per_second'].max():.0f} txns/sec",
                f"- **Slowest**: {successful['transactions_per_second'].min():.0f} txns/sec",
                f"- **Lowest Memory**: {successful['peak_memory_mb'].min():.1f} MB",
                f"- **Highest Memory**: {successful['peak_memory_mb'].max():.1f} MB",
                f"- **Average CPU**: {successful['cpu_percent'].mean():.1f}%",
                "\n## Detailed Results\n",
            ])
        
        # Results table
        lines.append("| Benchmark | Size | Duration (s) | Throughput (txn/s) | Memory (MB) | CPU % | Status |")
        lines.append("|-----------|------|--------------|-------------------|-------------|-------|--------|")
        
        for _, row in df.iterrows():
            status = "✓" if row['success'] else "✗"
            lines.append(
                f"| {row['name']} | {row['dataset_size']:,} | {row['duration_seconds']:.2f} | "
                f"{row['transactions_per_second']:.0f} | {row['peak_memory_mb']:.1f} | "
                f"{row['cpu_percent']:.1f} | {status} |"
            )
        
        # Recommendations
        lines.extend([
            "\n## Recommendations\n",
        ])
        
        if df['success'].any():
            successful = df[df['success']]
            
            # Find best method
            best_speed = successful.loc[successful['transactions_per_second'].idxmax()]
            best_memory = successful.loc[successful['peak_memory_mb'].idxmin()]
            
            lines.extend([
                f"- **For Speed**: Use `{best_speed['name']}` ({best_speed['transactions_per_second']:.0f} txns/sec)",
                f"- **For Memory**: Use `{best_memory['name']}` ({best_memory['peak_memory_mb']:.1f} MB)",
            ])
            
            # Scaling analysis
            if len(successful) >= 3:
                # Check if this looks like a scaling test
                sizes = sorted(successful['dataset_size'].unique())
                if len(sizes) >= 3:
                    scaling_results = successful[successful['dataset_size'].isin(sizes[:3])]
                    times = scaling_results.groupby('dataset_size')['duration_seconds'].mean()
                    
                    if len(times) >= 2:
                        size_increase = sizes[1] / sizes[0]
                        time_increase = times.iloc[1] / times.iloc[0]
                        scaling_factor = time_increase / size_increase
                        
                        if scaling_factor < 1.2:
                            lines.append(f"- **Scaling**: Linear or better (factor={scaling_factor:.2f})")
                        else:
                            lines.append(f"- **Scaling**: Needs optimization (factor={scaling_factor:.2f})")
        
        return "\n".join(lines)
    
    def clear_results(self):
        """Clear all benchmark results."""
        self.results.clear()
        logger.info("Cleared benchmark results")


# Convenience functions

def quick_benchmark(
    method: Callable,
    dataset_size: int,
    name: str = "Quick Benchmark"
) -> BenchmarkResult:
    """
    Quick single benchmark.
    
    Args:
        method: Method to benchmark
        dataset_size: Dataset size
        name: Benchmark name
        
    Returns:
        Benchmark result
        
    Example:
        >>> result = quick_benchmark(generate_data, 10000)
        >>> print(f"Speed: {result.transactions_per_second:.0f} txns/sec")
    """
    benchmark = PerformanceBenchmark()
    return benchmark.benchmark_method(
        name=name,
        method=method,
        dataset_size=dataset_size,
        num_transactions=dataset_size
    )


def compare_performance(
    methods: Dict[str, Callable],
    dataset_size: int = 10000
) -> pd.DataFrame:
    """
    Quick method comparison.
    
    Args:
        methods: Dictionary of {name: method}
        dataset_size: Size to test
        
    Returns:
        Comparison DataFrame
        
    Example:
        >>> results = compare_performance({
        >>>     'Serial': serial_generate,
        >>>     'Parallel': parallel_generate
        >>> })
        >>> print(results)
    """
    benchmark = PerformanceBenchmark()
    return benchmark.compare_methods(methods, dataset_size)


if __name__ == "__main__":
    # Demo: Benchmark suite
    logging.basicConfig(level=logging.INFO)
    
    print("=== Performance Benchmark Demo ===\n")
    
    # Mock generation function
    def mock_generate(num_transactions):
        """Mock data generation for demo."""
        import time
        import numpy as np
        
        # Simulate work
        time.sleep(num_transactions / 100000)  # Simulate processing time
        
        # Create dummy data
        data = np.random.rand(num_transactions, 10)
        return pd.DataFrame(data)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    
    print("Running scaling test...")
    results = benchmark.run_scaling_test(
        method=mock_generate,
        sizes=[1000, 5000, 10000],
        name_prefix="MockGenerate"
    )
    
    # Print results
    print("\n=== Results ===")
    df = benchmark.get_results_df()
    print(df[['name', 'dataset_size', 'duration_seconds', 'transactions_per_second', 'peak_memory_mb']])
    
    # Export report
    output_file = Path("output/benchmark_demo.md")
    benchmark.export_report(output_file)
    print(f"\n✓ Report saved to {output_file}")
