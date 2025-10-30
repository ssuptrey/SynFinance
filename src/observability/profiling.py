"""
Profiling Tools for SynFinance

Comprehensive profiling utilities for performance analysis:
- CPU profiling (cProfile integration)
- Memory profiling (memory_profiler integration)
- I/O profiling
- Function call tracing
- Bottleneck detection
- Performance visualization

Week 7 Day 4: Enhanced Observability
"""

import cProfile
import pstats
import io
import time
import sys
import os
import traceback
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import threading


@dataclass
class ProfileResult:
    """
    Results from profiling operation
    
    Attributes:
        operation_name: Name of profiled operation
        duration_seconds: Total duration
        cpu_time_seconds: CPU time used
        memory_delta_mb: Memory change in MB
        function_stats: Per-function statistics
        bottlenecks: Identified bottlenecks
        timestamp: When profiling occurred
    """
    operation_name: str
    duration_seconds: float
    cpu_time_seconds: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    function_stats: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'operation_name': self.operation_name,
            'duration_seconds': self.duration_seconds,
            'cpu_time_seconds': self.cpu_time_seconds,
            'memory_delta_mb': self.memory_delta_mb,
            'function_stats': self.function_stats,
            'bottlenecks': self.bottlenecks,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class FunctionStats:
    """Statistics for a single function"""
    function_name: str
    file_name: str
    line_number: int
    call_count: int
    total_time: float
    cumulative_time: float
    time_per_call: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'function': self.function_name,
            'file': self.file_name,
            'line': self.line_number,
            'calls': self.call_count,
            'total_time': round(self.total_time, 6),
            'cumulative_time': round(self.cumulative_time, 6),
            'time_per_call': round(self.time_per_call, 6)
        }


class CPUProfiler:
    """
    CPU profiling using cProfile
    
    Tracks function call times and identifies CPU bottlenecks.
    """
    
    def __init__(self):
        self.profiler: Optional[cProfile.Profile] = None
        self._profiling = False
    
    @contextmanager
    def profile(self, operation_name: str = "operation"):
        """
        Context manager for CPU profiling
        
        Args:
            operation_name: Name of operation being profiled
        
        Yields:
            ProfileResult object
        
        Example:
            >>> profiler = CPUProfiler()
            >>> with profiler.profile("data_generation") as result:
            ...     generate_data()
            >>> print(result.duration_seconds)
        """
        start_time = time.time()
        
        # Start profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self._profiling = True
        
        result = ProfileResult(operation_name=operation_name, duration_seconds=0)
        
        try:
            yield result
        finally:
            # Stop profiling
            self.profiler.disable()
            self._profiling = False
            
            # Calculate duration
            duration = time.time() - start_time
            result.duration_seconds = duration
            
            # Extract statistics
            stats = self._extract_stats()
            result.function_stats = stats
            result.cpu_time_seconds = sum(s['total_time'] for s in stats.values())
            
            # Identify bottlenecks
            result.bottlenecks = self._identify_bottlenecks(stats)
    
    def _extract_stats(self, top_n: int = 50) -> Dict[str, Dict[str, Any]]:
        """Extract function statistics from profiler"""
        if not self.profiler:
            return {}
        
        # Create StringIO to capture stats output
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Extract raw statistics
        function_stats = {}
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            # func is (filename, lineno, function_name)
            filename, lineno, func_name = func
            
            # Skip system/library functions
            if '/lib/python' in filename or '/lib64/python' in filename:
                continue
            
            # Calculate time per call
            time_per_call = tt / cc if cc > 0 else 0
            
            key = f"{filename}:{lineno}:{func_name}"
            function_stats[key] = {
                'function': func_name,
                'file': os.path.basename(filename),
                'line': lineno,
                'calls': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': time_per_call
            }
        
        # Return top N by cumulative time
        sorted_stats = sorted(
            function_stats.items(),
            key=lambda x: x[1]['cumulative_time'],
            reverse=True
        )
        
        return dict(sorted_stats[:top_n])
    
    def _identify_bottlenecks(
        self,
        stats: Dict[str, Dict[str, Any]],
        threshold_percent: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks
        
        Args:
            stats: Function statistics
            threshold_percent: Minimum % of total time to be a bottleneck
        
        Returns:
            List of bottleneck descriptions
        """
        if not stats:
            return []
        
        total_time = sum(s['cumulative_time'] for s in stats.values())
        if total_time == 0:
            return []
        
        bottlenecks = []
        
        for key, stat in stats.items():
            percent = (stat['cumulative_time'] / total_time) * 100
            
            if percent >= threshold_percent:
                bottlenecks.append({
                    'function': stat['function'],
                    'file': stat['file'],
                    'line': stat['line'],
                    'cumulative_time': stat['cumulative_time'],
                    'percent_of_total': percent,
                    'calls': stat['calls'],
                    'time_per_call': stat['time_per_call']
                })
        
        return bottlenecks
    
    def print_stats(self, top_n: int = 20) -> None:
        """Print profiling statistics"""
        if not self.profiler:
            print("No profiling data available")
            return
        
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)
        print(stream.getvalue())


class MemoryProfiler:
    """
    Memory profiling for tracking memory usage
    
    Tracks memory allocations and identifies memory bottlenecks.
    """
    
    def __init__(self):
        self._start_memory: Optional[float] = None
        self._memory_snapshots: List[Tuple[str, float]] = []
    
    @contextmanager
    def profile(self, operation_name: str = "operation"):
        """
        Context manager for memory profiling
        
        Args:
            operation_name: Name of operation being profiled
        
        Yields:
            ProfileResult object
        """
        import psutil
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        
        # Record starting memory
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self._start_memory = start_memory
        self._memory_snapshots = [("start", start_memory)]
        
        result = ProfileResult(operation_name=operation_name, duration_seconds=0)
        
        try:
            yield result
        finally:
            # Record ending memory
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            self._memory_snapshots.append(("end", end_memory))
            
            # Calculate metrics
            duration = time.time() - start_time
            memory_delta = end_memory - start_memory
            
            result.duration_seconds = duration
            result.memory_delta_mb = memory_delta
            result.metadata['memory_snapshots'] = self._memory_snapshots
            result.metadata['peak_memory_mb'] = max(s[1] for s in self._memory_snapshots)
    
    def snapshot(self, label: str) -> None:
        """Take a memory snapshot"""
        import psutil
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self._memory_snapshots.append((label, current_memory))
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)


class IOProfiler:
    """
    I/O profiling for tracking disk and network operations
    """
    
    def __init__(self):
        self._io_counters_start = None
        self._io_counters_end = None
    
    @contextmanager
    def profile(self, operation_name: str = "operation"):
        """
        Context manager for I/O profiling
        
        Args:
            operation_name: Name of operation being profiled
        
        Yields:
            ProfileResult object
        """
        import psutil
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        
        # Record starting I/O
        try:
            self._io_counters_start = process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            # I/O counters not available on all platforms
            self._io_counters_start = None
        
        result = ProfileResult(operation_name=operation_name, duration_seconds=0)
        
        try:
            yield result
        finally:
            # Record ending I/O
            try:
                self._io_counters_end = process.io_counters()
            except (AttributeError, psutil.AccessDenied):
                self._io_counters_end = None
            
            duration = time.time() - start_time
            result.duration_seconds = duration
            
            # Calculate I/O metrics
            if self._io_counters_start and self._io_counters_end:
                read_bytes = self._io_counters_end.read_bytes - self._io_counters_start.read_bytes
                write_bytes = self._io_counters_end.write_bytes - self._io_counters_start.write_bytes
                read_count = self._io_counters_end.read_count - self._io_counters_start.read_count
                write_count = self._io_counters_end.write_count - self._io_counters_start.write_count
                
                result.metadata['io'] = {
                    'read_mb': read_bytes / (1024 * 1024),
                    'write_mb': write_bytes / (1024 * 1024),
                    'read_count': read_count,
                    'write_count': write_count,
                    'read_mb_per_sec': (read_bytes / (1024 * 1024)) / duration if duration > 0 else 0,
                    'write_mb_per_sec': (write_bytes / (1024 * 1024)) / duration if duration > 0 else 0
                }


class PerformanceProfiler:
    """
    Comprehensive performance profiler
    
    Combines CPU, memory, and I/O profiling for complete performance analysis.
    """
    
    def __init__(self):
        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler()
        self.io_profiler = IOProfiler()
    
    @contextmanager
    def profile(self, operation_name: str = "operation", enable_cpu: bool = True,
                enable_memory: bool = True, enable_io: bool = True):
        """
        Context manager for comprehensive profiling
        
        Args:
            operation_name: Name of operation being profiled
            enable_cpu: Enable CPU profiling
            enable_memory: Enable memory profiling
            enable_io: Enable I/O profiling
        
        Yields:
            ProfileResult object with combined metrics
        
        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile("generate_data") as result:
            ...     data = generate_large_dataset()
            >>> print(f"Duration: {result.duration_seconds}s")
            >>> print(f"Memory: {result.memory_delta_mb}MB")
        """
        start_time = time.time()
        result = ProfileResult(operation_name=operation_name, duration_seconds=0)
        
        # Start profilers
        cpu_ctx = self.cpu_profiler.profile(operation_name) if enable_cpu else None
        mem_ctx = self.memory_profiler.profile(operation_name) if enable_memory else None
        io_ctx = self.io_profiler.profile(operation_name) if enable_io else None
        
        # Enter contexts
        cpu_result = cpu_ctx.__enter__() if cpu_ctx else None
        mem_result = mem_ctx.__enter__() if mem_ctx else None
        io_result = io_ctx.__enter__() if io_ctx else None
        
        try:
            yield result
        finally:
            # Exit contexts
            if cpu_ctx:
                cpu_ctx.__exit__(None, None, None)
            if mem_ctx:
                mem_ctx.__exit__(None, None, None)
            if io_ctx:
                io_ctx.__exit__(None, None, None)
            
            # Combine results
            duration = time.time() - start_time
            result.duration_seconds = duration
            
            if cpu_result:
                result.cpu_time_seconds = cpu_result.cpu_time_seconds
                result.function_stats = cpu_result.function_stats
                result.bottlenecks = cpu_result.bottlenecks
            
            if mem_result:
                result.memory_delta_mb = mem_result.memory_delta_mb
                result.metadata.update(mem_result.metadata)
            
            if io_result:
                result.metadata.update(io_result.metadata)
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator for profiling functions
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function that profiles execution
        
        Example:
            >>> profiler = PerformanceProfiler()
            >>> @profiler.profile_function
            ... def slow_function():
            ...     time.sleep(1)
            >>> slow_function()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile(func.__name__) as result:
                output = func(*args, **kwargs)
            
            # Print summary
            print(f"\nProfile: {func.__name__}")
            print(f"  Duration: {result.duration_seconds:.3f}s")
            if result.cpu_time_seconds:
                print(f"  CPU Time: {result.cpu_time_seconds:.3f}s")
            if result.memory_delta_mb is not None:
                print(f"  Memory Delta: {result.memory_delta_mb:+.2f}MB")
            
            if result.bottlenecks:
                print(f"  Bottlenecks: {len(result.bottlenecks)}")
                for b in result.bottlenecks[:3]:
                    print(f"    - {b['function']} ({b['percent_of_total']:.1f}% of time)")
            
            return output
        
        return wrapper
    
    def generate_report(self, result: ProfileResult) -> str:
        """
        Generate human-readable profiling report
        
        Args:
            result: Profile result to report on
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            f"Performance Profile: {result.operation_name}",
            "=" * 80,
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {result.duration_seconds:.3f}s",
        ]
        
        if result.cpu_time_seconds:
            lines.append(f"CPU Time: {result.cpu_time_seconds:.3f}s")
        
        if result.memory_delta_mb is not None:
            lines.append(f"Memory Delta: {result.memory_delta_mb:+.2f}MB")
        
        if 'io' in result.metadata:
            io_stats = result.metadata['io']
            lines.extend([
                "",
                "I/O Statistics:",
                f"  Read: {io_stats['read_mb']:.2f}MB ({io_stats['read_count']} operations)",
                f"  Write: {io_stats['write_mb']:.2f}MB ({io_stats['write_count']} operations)",
                f"  Read Rate: {io_stats['read_mb_per_sec']:.2f}MB/s",
                f"  Write Rate: {io_stats['write_mb_per_sec']:.2f}MB/s",
            ])
        
        if result.bottlenecks:
            lines.extend([
                "",
                "Performance Bottlenecks:",
                "-" * 80,
            ])
            for i, b in enumerate(result.bottlenecks[:10], 1):
                lines.append(
                    f"{i}. {b['function']} ({b['file']}:{b['line']}) "
                    f"- {b['percent_of_total']:.1f}% of time "
                    f"({b['calls']} calls, {b['time_per_call']*1000:.2f}ms/call)"
                )
        
        if result.function_stats:
            lines.extend([
                "",
                "Top Functions by Cumulative Time:",
                "-" * 80,
            ])
            sorted_funcs = sorted(
                result.function_stats.items(),
                key=lambda x: x[1]['cumulative_time'],
                reverse=True
            )
            for i, (key, stat) in enumerate(sorted_funcs[:10], 1):
                lines.append(
                    f"{i}. {stat['function']} ({stat['file']}:{stat['line']}) "
                    f"- {stat['cumulative_time']:.3f}s "
                    f"({stat['calls']} calls)"
                )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def profile_operation(
    operation_name: str,
    enable_cpu: bool = True,
    enable_memory: bool = True,
    enable_io: bool = True
):
    """
    Decorator for profiling operations
    
    Args:
        operation_name: Name of operation
        enable_cpu: Enable CPU profiling
        enable_memory: Enable memory profiling
        enable_io: Enable I/O profiling
    
    Returns:
        Decorator function
    
    Example:
        >>> @profile_operation("data_generation")
        ... def generate_data():
        ...     return create_dataset(1000)
    """
    def decorator(func: Callable) -> Callable:
        profiler = PerformanceProfiler()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(
                operation_name,
                enable_cpu=enable_cpu,
                enable_memory=enable_memory,
                enable_io=enable_io
            ) as result:
                output = func(*args, **kwargs)
            
            # Print report
            print(profiler.generate_report(result))
            
            return output
        
        return wrapper
    
    return decorator
