"""
Optimizer

Performance optimizations for batch processing, async I/O, and parallelization.
Provides batch operations, async processing, and parallel execution capabilities.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Iterable
from functools import wraps
import pandas as pd
import numpy as np
import time
from datetime import datetime


class BatchProcessor:
    """Batch processing utilities."""
    
    def __init__(self, batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
        """
        self.batch_size = batch_size
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'avg_batch_time': 0.0
        }
    
    def process_batches(self, items: List[Any], processor: Callable,
                       batch_size: Optional[int] = None) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor: Function to process each batch
            batch_size: Batch size (uses default if not specified)
            
        Returns:
            List of processed results
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        start_time = time.time()
        num_batches = 0
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = processor(batch)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            
            num_batches += 1
        
        total_time = time.time() - start_time
        
        # Update stats
        self.stats['total_items'] += len(items)
        self.stats['total_batches'] += num_batches
        self.stats['total_time'] += total_time
        self.stats['avg_batch_time'] = total_time / num_batches if num_batches > 0 else 0
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.stats.copy()


class AsyncProcessor:
    """Async I/O processing utilities."""
    
    def __init__(self, max_concurrent: int = 100):
        """
        Initialize async processor.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_async(self, items: List[Any],
                           async_processor: Callable) -> List[Any]:
        """
        Process items concurrently using asyncio.
        
        Args:
            items: List of items to process
            async_processor: Async function to process each item
            
        Returns:
            List of processed results
        """
        async def bounded_process(item):
            async with self.semaphore:
                return await async_processor(item)
        
        tasks = [bounded_process(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]
    
    def run_async(self, items: List[Any], async_processor: Callable) -> List[Any]:
        """
        Run async processing (convenience wrapper for sync code).
        
        Args:
            items: List of items to process
            async_processor: Async function to process each item
            
        Returns:
            List of processed results
        """
        return asyncio.run(self.process_async(items, async_processor))


class ParallelProcessor:
    """Parallel processing utilities."""
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum worker threads/processes
        """
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    def process_parallel_threads(self, items: List[Any],
                                 processor: Callable) -> List[Any]:
        """
        Process items in parallel using threads (I/O-bound tasks).
        
        Args:
            items: List of items to process
            processor: Function to process each item
            
        Returns:
            List of processed results
        """
        futures = [self.thread_pool.submit(processor, item) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
        
        return results
    
    def process_parallel_processes(self, items: List[Any],
                                   processor: Callable) -> List[Any]:
        """
        Process items in parallel using processes (CPU-bound tasks).
        
        Args:
            items: List of items to process
            processor: Function to process each item
            
        Returns:
            List of processed results
        """
        futures = [self.process_pool.submit(processor, item) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
        
        return results
    
    def map_parallel_threads(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map function over items using thread pool."""
        return list(self.thread_pool.map(func, items))
    
    def map_parallel_processes(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map function over items using process pool."""
        return list(self.process_pool.map(func, items))
    
    def cleanup(self) -> None:
        """Cleanup thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class DataFrameOptimizer:
    """DataFrame memory and performance optimization."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting dtypes.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert to categorical for low-cardinality strings
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_memory / original_memory) * 100
        
        print(f"Memory reduced by {reduction:.1f}% ({original_memory:.1f}MB -> {optimized_memory:.1f}MB)")
        
        return df
    
    @staticmethod
    def optimize_sparse(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """
        Convert columns to sparse arrays if mostly null.
        
        Args:
            df: DataFrame to optimize
            threshold: Null percentage threshold (0-1)
            
        Returns:
            Optimized DataFrame
        """
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            
            if null_pct > threshold:
                df[col] = df[col].astype(pd.SparseDtype(df[col].dtype, fill_value=np.nan))
        
        return df
    
    @staticmethod
    def chunk_process(filepath: str, processor: Callable,
                     chunksize: int = 10000) -> pd.DataFrame:
        """
        Process large CSV file in chunks.
        
        Args:
            filepath: Path to CSV file
            processor: Function to process each chunk
            chunksize: Number of rows per chunk
            
        Returns:
            Concatenated processed DataFrame
        """
        chunks = []
        
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            processed_chunk = processor(chunk)
            chunks.append(processed_chunk)
        
        return pd.concat(chunks, ignore_index=True)


class LazyLoader:
    """Lazy loading utilities."""
    
    def __init__(self):
        """Initialize lazy loader."""
        self._cache: Dict[str, Any] = {}
    
    def lazy_load(self, key: str, loader: Callable) -> Callable:
        """
        Create lazy-loading function.
        
        Args:
            key: Cache key
            loader: Function to load resource
            
        Returns:
            Lazy-loading wrapper function
        """
        @wraps(loader)
        def wrapper(*args, **kwargs):
            if key not in self._cache:
                self._cache[key] = loader(*args, **kwargs)
            return self._cache[key]
        
        return wrapper
    
    def get(self, key: str, loader: Callable) -> Any:
        """
        Get value with lazy loading.
        
        Args:
            key: Cache key
            loader: Function to load if not cached
            
        Returns:
            Loaded value
        """
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            key: Specific key to clear (clears all if None)
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


class Optimizer:
    """Main performance optimizer with all optimization utilities."""
    
    def __init__(self, max_workers: int = 10, batch_size: int = 100):
        """
        Initialize optimizer.
        
        Args:
            max_workers: Maximum workers for parallel processing
            batch_size: Default batch size
        """
        self.batch_processor = BatchProcessor(batch_size)
        self.async_processor = AsyncProcessor(max_concurrent=max_workers * 2)
        self.parallel_processor = ParallelProcessor(max_workers)
        self.df_optimizer = DataFrameOptimizer()
        self.lazy_loader = LazyLoader()
        
        self.stats = {
            'operations': 0,
            'total_time': 0.0,
            'items_processed': 0
        }
    
    def batch_process(self, items: List[Any], processor: Callable,
                     batch_size: Optional[int] = None) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            processor: Batch processor function
            batch_size: Batch size
            
        Returns:
            Processed results
        """
        start_time = time.time()
        results = self.batch_processor.process_batches(items, processor, batch_size)
        
        self.stats['operations'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['items_processed'] += len(items)
        
        return results
    
    async def async_batch_process(self, items: List[Any],
                                  async_processor: Callable) -> List[Any]:
        """
        Process items asynchronously.
        
        Args:
            items: Items to process
            async_processor: Async processor function
            
        Returns:
            Processed results
        """
        start_time = time.time()
        results = await self.async_processor.process_async(items, async_processor)
        
        self.stats['operations'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['items_processed'] += len(items)
        
        return results
    
    def parallel_process_threads(self, items: List[Any],
                                processor: Callable) -> List[Any]:
        """
        Process items in parallel using threads.
        
        Args:
            items: Items to process
            processor: Processor function
            
        Returns:
            Processed results
        """
        start_time = time.time()
        results = self.parallel_processor.process_parallel_threads(items, processor)
        
        self.stats['operations'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['items_processed'] += len(items)
        
        return results
    
    def parallel_process_processes(self, items: List[Any],
                                  processor: Callable) -> List[Any]:
        """
        Process items in parallel using processes.
        
        Args:
            items: Items to process
            processor: Processor function
            
        Returns:
            Processed results
        """
        start_time = time.time()
        results = self.parallel_processor.process_parallel_processes(items, processor)
        
        self.stats['operations'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['items_processed'] += len(items)
        
        return results
    
    def optimize_dataframe(self, df: pd.DataFrame,
                          sparse_threshold: float = 0.8) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            sparse_threshold: Threshold for sparse conversion
            
        Returns:
            Optimized DataFrame
        """
        df = self.df_optimizer.optimize_dtypes(df)
        df = self.df_optimizer.optimize_sparse(df, sparse_threshold)
        return df
    
    def lazy_load_model(self, model_path: str) -> Callable:
        """
        Create lazy-loading function for ML model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Lazy-loading function
        """
        def load_model():
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        return self.lazy_loader.lazy_load(model_path, load_model)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        stats = self.stats.copy()
        stats['batch_stats'] = self.batch_processor.get_stats()
        stats['avg_operation_time'] = (
            stats['total_time'] / stats['operations']
            if stats['operations'] > 0 else 0
        )
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.parallel_processor.cleanup()
        self.lazy_loader.clear()


# Utility decorators
def batch_optimized(batch_size: int = 100):
    """
    Decorator to automatically batch process function inputs.
    
    Usage:
        @batch_optimized(batch_size=50)
        def process_items(items):
            return [item * 2 for item in items]
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            optimizer = Optimizer()
            
            def batch_func(batch):
                return func(batch, *args, **kwargs)
            
            return optimizer.batch_process(items, batch_func, batch_size)
        
        return wrapper
    
    return decorator


def parallel_optimized(max_workers: int = 10, use_processes: bool = False):
    """
    Decorator to automatically parallelize function execution.
    
    Usage:
        @parallel_optimized(max_workers=5)
        def process_item(item):
            return expensive_operation(item)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            optimizer = Optimizer(max_workers=max_workers)
            
            def process_func(item):
                return func(item, *args, **kwargs)
            
            if use_processes:
                return optimizer.parallel_process_processes(items, process_func)
            else:
                return optimizer.parallel_process_threads(items, process_func)
        
        return wrapper
    
    return decorator
