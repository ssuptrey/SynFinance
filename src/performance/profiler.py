"""
Profiler

CPU and memory profiling tools for performance analysis.
Provides cProfile integration, memory tracking, and flamegraph generation.
"""

import cProfile
import pstats
import io
import os
import tracemalloc
import time
from contextlib import contextmanager
from typing import Callable, Dict, Any, List, Optional
from functools import wraps
from datetime import datetime
import json


class ProfileResult:
    """Profile execution result."""
    
    def __init__(self, name: str, profile_type: str = 'cpu'):
        self.name = name
        self.profile_type = profile_type
        self.timestamp = datetime.now()
        self.duration_seconds: float = 0.0
        self.stats: Dict[str, Any] = {}
        self.hotspots: List[Dict[str, Any]] = []
        self.output_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'profile_type': self.profile_type,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'hotspots': self.hotspots,
            'output_file': self.output_file
        }


class Profiler:
    """CPU and memory profiling tools."""
    
    def __init__(self, output_dir: str = 'profiling_results'):
        """
        Initialize profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir
        self.profiles: List[ProfileResult] = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Profiling state
        self._cpu_profiler: Optional[cProfile.Profile] = None
        self._memory_snapshot = None
    
    @contextmanager
    def profile_cpu(self, name: str = 'profile'):
        """
        Profile CPU usage for code block.
        
        Usage:
            with profiler.profile_cpu('my_function'):
                # Code to profile
                expensive_operation()
        
        Args:
            name: Profile name for identification
        """
        pr = cProfile.Profile()
        pr.enable()
        
        result = ProfileResult(name, 'cpu')
        start_time = time.time()
        
        try:
            yield pr
        finally:
            pr.disable()
            result.duration_seconds = time.time() - start_time
            
            # Generate stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            
            result.stats = {'summary': s.getvalue()}
            
            # Extract hotspots
            result.hotspots = self._extract_hotspots(pr)
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir,
                f"{name}_{timestamp}.prof"
            )
            pr.dump_stats(output_file)
            result.output_file = output_file
            
            # Save JSON summary
            json_file = output_file.replace('.prof', '.json')
            with open(json_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            self.profiles.append(result)
    
    def _extract_hotspots(self, profiler: cProfile.Profile, 
                          top_n: int = 20) -> List[Dict[str, Any]]:
        """Extract CPU hotspots from profiler."""
        hotspots = []
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            filename, line, func_name = func
            
            hotspots.append({
                'function': func_name,
                'filename': filename,
                'line': line,
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            })
        
        return hotspots
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution.
        
        Usage:
            @profiler.profile_function
            def expensive_operation():
                # Code to profile
                pass
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_cpu(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    
    def analyze_hotspots(self, profile_file: str, 
                        top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Identify CPU hotspots from saved profile file.
        
        Args:
            profile_file: Path to .prof file
            top_n: Number of top hotspots to return
            
        Returns:
            List of hotspot dictionaries
        """
        stats = pstats.Stats(profile_file)
        stats.sort_stats('cumulative')
        
        hotspots = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            filename, line, func_name = func
            
            hotspots.append({
                'function': func_name,
                'filename': filename,
                'line': line,
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0,
                'percent_time': (ct / stats.total_tt * 100) if stats.total_tt > 0 else 0
            })
        
        return hotspots
    
    @contextmanager
    def profile_memory(self, name: str = 'memory_profile'):
        """
        Profile memory usage for code block.
        
        Usage:
            with profiler.profile_memory('my_function'):
                # Code to profile
                data = load_large_dataset()
        
        Args:
            name: Profile name for identification
        """
        tracemalloc.start()
        
        result = ProfileResult(name, 'memory')
        start_time = time.time()
        
        try:
            yield
        finally:
            result.duration_seconds = time.time() - start_time
            
            # Take snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Extract top memory consumers
            result.hotspots = []
            for stat in top_stats[:20]:
                result.hotspots.append({
                    'filename': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                })
            
            result.stats = {
                'current_mb': tracemalloc.get_traced_memory()[0] / 1024 / 1024,
                'peak_mb': tracemalloc.get_traced_memory()[1] / 1024 / 1024
            }
            
            tracemalloc.stop()
            
            # Save JSON summary
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(
                self.output_dir,
                f"{name}_memory_{timestamp}.json"
            )
            with open(json_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            result.output_file = json_file
            self.profiles.append(result)
    
    def detect_memory_leaks(self, threshold_mb: float = 100,
                           snapshots: int = 3) -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks by taking multiple snapshots.
        
        Args:
            threshold_mb: Memory growth threshold in MB
            snapshots: Number of snapshots to take
            
        Returns:
            List of potential memory leaks
        """
        leaks = []
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        initial_snapshot = tracemalloc.take_snapshot()
        initial_memory = sum(stat.size for stat in initial_snapshot.statistics('filename'))
        
        # Wait and take another snapshot
        time.sleep(1)
        
        for i in range(snapshots - 1):
            time.sleep(1)
            current_snapshot = tracemalloc.take_snapshot()
            current_memory = sum(stat.size for stat in current_snapshot.statistics('filename'))
            
            growth_mb = (current_memory - initial_memory) / 1024 / 1024
            
            if growth_mb > threshold_mb:
                # Compare snapshots to find leaks
                top_stats = current_snapshot.compare_to(initial_snapshot, 'lineno')
                
                for stat in top_stats[:10]:
                    if stat.size_diff > 1024 * 1024:  # 1MB+
                        leaks.append({
                            'snapshot': i + 1,
                            'filename': str(stat.traceback),
                            'size_diff_mb': stat.size_diff / 1024 / 1024,
                            'count_diff': stat.count_diff
                        })
        
        tracemalloc.stop()
        return leaks
    
    def generate_flamegraph(self, profile_file: str,
                           output_svg: str = 'flamegraph.svg') -> bool:
        """
        Generate flamegraph visualization from profile.
        
        Note: Requires flamegraph.pl or py-spy installed.
        
        Args:
            profile_file: Path to .prof file
            output_svg: Output SVG filename
            
        Returns:
            True if generated successfully
        """
        # This is a placeholder - actual implementation would require
        # flamegraph.pl or py-spy to be installed
        
        # For now, just create a simple text-based representation
        stats = pstats.Stats(profile_file)
        stats.sort_stats('cumulative')
        
        output_file = os.path.join(self.output_dir, output_svg.replace('.svg', '.txt'))
        
        with open(output_file, 'w') as f:
            f.write("Flamegraph (Text Representation)\n")
            f.write("=" * 80 + "\n\n")
            
            for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:30]:
                filename, line, func_name = func
                percent = (ct / stats.total_tt * 100) if stats.total_tt > 0 else 0
                
                # Create visual bar
                bar_length = int(percent / 2)
                bar = '#' * bar_length
                
                f.write(f"{func_name:40} {bar} {percent:5.1f}%\n")
        
        return True
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling runs."""
        cpu_profiles = [p for p in self.profiles if p.profile_type == 'cpu']
        memory_profiles = [p for p in self.profiles if p.profile_type == 'memory']
        
        return {
            'total_profiles': len(self.profiles),
            'cpu_profiles': len(cpu_profiles),
            'memory_profiles': len(memory_profiles),
            'latest_cpu_profile': cpu_profiles[-1].to_dict() if cpu_profiles else None,
            'latest_memory_profile': memory_profiles[-1].to_dict() if memory_profiles else None,
            'output_dir': self.output_dir
        }
    
    def compare_profiles(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """
        Compare two CPU profiles to identify performance changes.
        
        Args:
            profile1: Path to first .prof file (baseline)
            profile2: Path to second .prof file (current)
            
        Returns:
            Comparison results
        """
        stats1 = pstats.Stats(profile1)
        stats2 = pstats.Stats(profile2)
        
        # Get top functions from both
        funcs1 = {func: (cc, nc, tt, ct) for func, (cc, nc, tt, ct, callers) 
                 in stats1.stats.items()}
        funcs2 = {func: (cc, nc, tt, ct) for func, (cc, nc, tt, ct, callers) 
                 in stats2.stats.items()}
        
        improvements = []
        regressions = []
        
        for func in funcs1:
            if func in funcs2:
                _, _, _, ct1 = funcs1[func]
                _, _, _, ct2 = funcs2[func]
                
                change_percent = ((ct2 - ct1) / ct1 * 100) if ct1 > 0 else 0
                
                filename, line, func_name = func
                
                if change_percent < -10:  # 10%+ improvement
                    improvements.append({
                        'function': func_name,
                        'change_percent': change_percent,
                        'time_before': ct1,
                        'time_after': ct2
                    })
                elif change_percent > 10:  # 10%+ regression
                    regressions.append({
                        'function': func_name,
                        'change_percent': change_percent,
                        'time_before': ct1,
                        'time_after': ct2
                    })
        
        return {
            'improvements': sorted(improvements, key=lambda x: x['change_percent'])[:10],
            'regressions': sorted(regressions, key=lambda x: -x['change_percent'])[:10],
            'total_time_before': stats1.total_tt,
            'total_time_after': stats2.total_tt,
            'total_change_percent': ((stats2.total_tt - stats1.total_tt) / stats1.total_tt * 100) if stats1.total_tt > 0 else 0
        }
    
    def profile_async(self, async_func: Callable, *args, **kwargs) -> ProfileResult:
        """
        Profile async function execution.
        
        Args:
            async_func: Async function to profile
            *args, **kwargs: Function arguments
            
        Returns:
            Profile result
        """
        import asyncio
        
        result = ProfileResult(async_func.__name__, 'cpu')
        
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        
        try:
            # Run async function
            asyncio.run(async_func(*args, **kwargs))
        finally:
            pr.disable()
            result.duration_seconds = time.time() - start_time
            
            # Extract hotspots
            result.hotspots = self._extract_hotspots(pr)
            
            # Save profile
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir,
                f"{async_func.__name__}_async_{timestamp}.prof"
            )
            pr.dump_stats(output_file)
            result.output_file = output_file
            
            self.profiles.append(result)
        
        return result
