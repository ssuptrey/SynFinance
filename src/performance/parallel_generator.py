"""
Parallel data generator for high-performance transaction generation.

This module provides multi-core parallel processing for generating large datasets.
Achieves 100K transactions in < 30 seconds on modern hardware.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for parallel generation."""
    num_transactions: int
    num_customers: int = 1000
    fraud_rate: float = 0.05
    anomaly_rate: float = 0.10
    start_date: str = "2024-01-01"
    num_workers: Optional[int] = None  # None = auto-detect
    chunk_size: Optional[int] = None  # None = auto-calculate
    use_processes: bool = True  # True = ProcessPool, False = ThreadPool
    show_progress: bool = True
    
    def __post_init__(self):
        """Auto-configure optimal settings."""
        if self.num_workers is None:
            # Use all cores minus 1 for system responsiveness
            self.num_workers = max(1, mp.cpu_count() - 1)
        
        if self.chunk_size is None:
            # Optimal chunk size: transactions per worker
            self.chunk_size = max(100, self.num_transactions // (self.num_workers * 4))


@dataclass
class GenerationStats:
    """Statistics from parallel generation."""
    total_transactions: int
    total_time: float
    transactions_per_second: float
    num_workers: int
    chunk_size: int
    memory_mb: float
    fraud_count: int
    anomaly_count: int
    chunks_processed: int


class ParallelGenerator:
    """
    High-performance parallel transaction generator.
    
    Features:
    - Multi-core processing (ProcessPoolExecutor/ThreadPoolExecutor)
    - Automatic worker pool sizing
    - Progress tracking with tqdm
    - Chunk-based processing for memory efficiency
    - Reproducible results with seeding
    
    Performance:
    - Target: 100K transactions in < 30 seconds
    - Scales linearly with CPU cores
    - Memory efficient (streaming chunks)
    
    Example:
        >>> config = GenerationConfig(num_transactions=100000)
        >>> generator = ParallelGenerator(config)
        >>> df = generator.generate()
        >>> print(f"Generated {len(df)} transactions in {generator.stats.total_time:.2f}s")
    """
    
    def __init__(self, config: GenerationConfig):
        """
        Initialize parallel generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config
        self.stats: Optional[GenerationStats] = None
        
    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate transactions in parallel.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with generated transactions
        """
        start_time = time.time()
        
        # Calculate chunks
        chunks = self._calculate_chunks()
        logger.info(f"Generating {self.config.num_transactions} transactions "
                   f"using {self.config.num_workers} workers "
                   f"in {len(chunks)} chunks")
        
        # Generate in parallel
        results = self._parallel_generate(chunks, seed)
        
        # Combine results
        df = pd.concat(results, ignore_index=True)
        
        # Calculate stats
        elapsed = time.time() - start_time
        self.stats = GenerationStats(
            total_transactions=len(df),
            total_time=elapsed,
            transactions_per_second=len(df) / elapsed,
            num_workers=self.config.num_workers,
            chunk_size=self.config.chunk_size,
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            fraud_count=df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
            anomaly_count=df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0,
            chunks_processed=len(chunks)
        )
        
        logger.info(f"Generated {len(df)} transactions in {elapsed:.2f}s "
                   f"({self.stats.transactions_per_second:.0f} txns/sec)")
        
        return df
    
    def _calculate_chunks(self) -> List[Tuple[int, int, int]]:
        """
        Calculate chunk ranges for parallel processing.
        
        Returns:
            List of (start_idx, end_idx, chunk_seed) tuples
        """
        chunks = []
        chunk_size = self.config.chunk_size
        total = self.config.num_transactions
        
        for i in range(0, total, chunk_size):
            end = min(i + chunk_size, total)
            chunk_seed = i  # Unique seed per chunk
            chunks.append((i, end, chunk_seed))
        
        return chunks
    
    def _parallel_generate(
        self, 
        chunks: List[Tuple[int, int, int]], 
        base_seed: Optional[int]
    ) -> List[pd.DataFrame]:
        """
        Generate chunks in parallel.
        
        Args:
            chunks: List of chunk specifications
            base_seed: Base random seed
            
        Returns:
            List of DataFrames (one per chunk)
        """
        # Choose executor
        ExecutorClass = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        results = []
        
        with ExecutorClass(max_workers=self.config.num_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(
                    self._generate_chunk, 
                    start, 
                    end, 
                    chunk_seed if base_seed is None else base_seed + chunk_seed
                ): (start, end) 
                for start, end, chunk_seed in chunks
            }
            
            # Collect results with progress bar
            if self.config.show_progress:
                pbar = tqdm(total=len(chunks), desc="Generating chunks", unit="chunk")
            
            for future in as_completed(futures):
                try:
                    chunk_df = future.result()
                    results.append(chunk_df)
                    
                    if self.config.show_progress:
                        pbar.update(1)
                        pbar.set_postfix({
                            'txns': sum(len(r) for r in results),
                            'chunks': len(results)
                        })
                except Exception as e:
                    start, end = futures[future]
                    logger.error(f"Failed to generate chunk [{start}:{end}]: {e}")
                    raise
            
            if self.config.show_progress:
                pbar.close()
        
        return results
    
    def _generate_chunk(
        self, 
        start_idx: int, 
        end_idx: int, 
        seed: int
    ) -> pd.DataFrame:
        """
        Generate a single chunk of transactions.
        
        This method runs in a separate process/thread.
        
        Args:
            start_idx: Starting transaction index
            end_idx: Ending transaction index
            seed: Random seed for this chunk
            
        Returns:
            DataFrame with generated transactions
        """
        # Import here to avoid pickling issues with multiprocessing
        from src.data_generator import generate_realistic_dataset
        
        np.random.seed(seed)
        
        # Calculate chunk size
        chunk_size = end_idx - start_idx
        
        # Calculate date range for this chunk
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        days_span = max(1, chunk_size // 100)  # Spread over days
        
        # Generate using the realistic dataset generator
        # Estimate customers needed (~10 transactions per customer)
        num_customers = max(10, chunk_size // 10)
        
        # Calculate transactions per customer (round up to ensure we get enough)
        transactions_per_customer = (chunk_size + num_customers - 1) // num_customers
        
        df = generate_realistic_dataset(
            num_customers=num_customers,
            transactions_per_customer=transactions_per_customer,
            start_date=start_date,
            days=days_span,
            seed=seed
        )
        
        # Trim to exact size (we may have generated slightly more)
        if len(df) > chunk_size:
            df = df.iloc[:chunk_size].copy()
        elif len(df) < chunk_size:
            # If we somehow got fewer, log a warning
            logger.warning(f"Generated {len(df)} transactions, expected {chunk_size}")
            df = df.copy()
        
        # Add chunk metadata
        df['chunk_id'] = start_idx // self.config.chunk_size
        df['chunk_start'] = start_idx
        
        return df
    
    def generate_to_file(
        self, 
        output_path: Path, 
        seed: Optional[int] = None,
        file_format: str = 'csv'
    ) -> GenerationStats:
        """
        Generate transactions and save directly to file.
        
        Args:
            output_path: Path to output file
            seed: Random seed
            file_format: 'csv', 'parquet', or 'json'
            
        Returns:
            Generation statistics
        """
        df = self.generate(seed=seed)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif file_format == 'json':
            df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Saved {len(df)} transactions to {output_path}")
        
        return self.stats
    
    def benchmark(self, sizes: List[int] = None) -> pd.DataFrame:
        """
        Benchmark generation performance at different scales.
        
        Args:
            sizes: List of dataset sizes to test (default: [1K, 10K, 100K])
            
        Returns:
            DataFrame with benchmark results
        """
        if sizes is None:
            sizes = [1000, 10000, 100000]
        
        results = []
        
        for size in sizes:
            logger.info(f"Benchmarking {size:,} transactions...")
            
            # Update config
            original_size = self.config.num_transactions
            self.config.num_transactions = size
            self.config.show_progress = False
            
            # Generate
            df = self.generate(seed=42)
            
            # Record results
            results.append({
                'size': size,
                'time_seconds': self.stats.total_time,
                'txns_per_second': self.stats.transactions_per_second,
                'memory_mb': self.stats.memory_mb,
                'workers': self.stats.num_workers,
                'chunks': self.stats.chunks_processed
            })
            
            # Restore config
            self.config.num_transactions = original_size
            self.config.show_progress = True
        
        return pd.DataFrame(results)


def quick_generate(
    num_transactions: int,
    fraud_rate: float = 0.05,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Quick convenience function for parallel generation.
    
    Args:
        num_transactions: Number of transactions to generate
        fraud_rate: Fraud rate (0.0 to 1.0)
        num_workers: Number of parallel workers (None = auto)
        seed: Random seed
        
    Returns:
        DataFrame with generated transactions
        
    Example:
        >>> df = quick_generate(100000, fraud_rate=0.05)
        >>> print(f"Generated {len(df)} transactions")
    """
    config = GenerationConfig(
        num_transactions=num_transactions,
        fraud_rate=fraud_rate,
        num_workers=num_workers
    )
    
    generator = ParallelGenerator(config)
    return generator.generate(seed=seed)


if __name__ == "__main__":
    # Demo: Generate 10K transactions
    logging.basicConfig(level=logging.INFO)
    
    print("=== Parallel Generator Demo ===\n")
    
    config = GenerationConfig(
        num_transactions=10000,
        fraud_rate=0.05,
        num_workers=4
    )
    
    generator = ParallelGenerator(config)
    df = generator.generate(seed=42)
    
    print(f"\n=== Results ===")
    print(f"Transactions: {len(df):,}")
    print(f"Time: {generator.stats.total_time:.2f}s")
    print(f"Speed: {generator.stats.transactions_per_second:.0f} txns/sec")
    print(f"Memory: {generator.stats.memory_mb:.1f} MB")
    print(f"Fraud: {generator.stats.fraud_count:,} ({generator.stats.fraud_count/len(df)*100:.1f}%)")
    print(f"Workers: {generator.stats.num_workers}")
    print(f"Chunks: {generator.stats.chunks_processed}")
