"""
Streaming data generator for memory-efficient large dataset generation.

This module provides generator-based streaming for processing 1M+ transactions
with minimal memory footprint (<500MB).
"""

import logging
import csv
import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming generation."""
    num_transactions: int
    batch_size: int = 1000  # Transactions per batch
    num_customers: int = 1000
    fraud_rate: float = 0.05
    anomaly_rate: float = 0.10
    start_date: str = "2024-01-01"
    output_format: str = 'csv'  # csv, json, parquet
    buffer_size: int = 10000  # Write buffer size
    show_progress: bool = True


@dataclass
class StreamStats:
    """Statistics from streaming generation."""
    total_transactions: int
    total_batches: int
    total_time: float
    transactions_per_second: float
    peak_memory_mb: float
    fraud_count: int
    anomaly_count: int
    output_file_mb: Optional[float] = None


class StreamingGenerator:
    """
    Memory-efficient streaming transaction generator.
    
    Features:
    - Generator-based iteration (minimal memory)
    - Batch processing for efficiency
    - Direct-to-disk streaming
    - Support for 1M+ transactions with <500MB RAM
    - Multiple output formats (CSV, JSON, Parquet)
    
    Memory Usage:
    - Base: ~50MB (customer data + generator state)
    - Per batch: ~10-20MB (configurable)
    - Total: <500MB for 1M transactions
    
    Example:
        >>> config = StreamConfig(num_transactions=1_000_000)
        >>> generator = StreamingGenerator(config)
        >>> 
        >>> # Stream to file
        >>> generator.stream_to_file('output.csv')
        >>>
        >>> # Or iterate batches
        >>> for batch in generator.generate_batches():
        >>>     process_batch(batch)
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize streaming generator.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.stats: Optional[StreamStats] = None
        self._customers = None
        self._data_gen = None
        
    def _initialize_generators(self, seed: Optional[int] = None):
        """Initialize customer and data generators (lazy loading)."""
        if self._customers is None:
            from src.data_generator import generate_realistic_dataset
            
            if seed is not None:
                np.random.seed(seed)
            
            # Store the generator function for later use
            self._data_gen = generate_realistic_dataset
            self._customers = True  # Flag that we've initialized
            
            logger.info(f"Initialized generators for streaming")
    
    def generate_batches(
        self, 
        seed: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Generate transactions in batches (streaming).
        
        Yields DataFrames of size batch_size until num_transactions is reached.
        Memory efficient - only one batch in memory at a time.
        
        Args:
            seed: Random seed for reproducibility
            
        Yields:
            DataFrame batches
            
        Example:
            >>> for batch in generator.generate_batches(seed=42):
            >>>     print(f"Processing {len(batch)} transactions")
            >>>     # Process batch...
        """
        import time
        from tqdm import tqdm
        
        self._initialize_generators(seed)
        
        start_time = time.time()
        total_generated = 0
        fraud_count = 0
        anomaly_count = 0
        peak_memory = 0
        
        # Calculate number of batches
        num_batches = (self.config.num_transactions + self.config.batch_size - 1) // self.config.batch_size
        
        # Setup progress bar
        pbar = tqdm(
            total=self.config.num_transactions, 
            desc="Streaming batches",
            unit="txn",
            disable=not self.config.show_progress
        )
        
        # Calculate date range
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        days_span = max(1, self.config.num_transactions // 1000)
        end_date = start_date + timedelta(days=days_span)
        
        for batch_idx in range(num_batches):
            # Calculate batch size (last batch may be smaller)
            remaining = self.config.num_transactions - total_generated
            batch_size = min(self.config.batch_size, remaining)
            
            if batch_size <= 0:
                break
            
            # Generate batch using realistic dataset
            num_customers = max(5, batch_size // 10)
            batch_df = self._data_gen(
                num_customers=num_customers,
                transactions_per_customer=batch_size // num_customers,
                start_date=start_date,
                days=days_span,
                seed=seed + batch_idx if seed else None
            )
            
            # Trim to exact size if needed
            if len(batch_df) > batch_size:
                batch_df = batch_df.iloc[:batch_size]
            
            # Add batch metadata
            batch_df['batch_id'] = batch_idx
            
            # Update stats
            total_generated += len(batch_df)
            if 'is_fraud' in batch_df.columns:
                fraud_count += batch_df['is_fraud'].sum()
            if 'is_anomaly' in batch_df.columns:
                anomaly_count += batch_df['is_anomaly'].sum()
            
            # Track memory
            batch_memory = batch_df.memory_usage(deep=True).sum() / 1024 / 1024
            peak_memory = max(peak_memory, batch_memory)
            
            # Update progress
            pbar.update(len(batch_df))
            pbar.set_postfix({
                'batch': batch_idx + 1,
                'mem_mb': f'{batch_memory:.1f}'
            })
            
            yield batch_df
        
        pbar.close()
        
        # Store stats
        elapsed = time.time() - start_time
        self.stats = StreamStats(
            total_transactions=total_generated,
            total_batches=num_batches,
            total_time=elapsed,
            transactions_per_second=total_generated / elapsed if elapsed > 0 else 0,
            peak_memory_mb=peak_memory,
            fraud_count=fraud_count,
            anomaly_count=anomaly_count
        )
        
        logger.info(f"Streamed {total_generated:,} transactions in {elapsed:.2f}s "
                   f"(peak memory: {peak_memory:.1f} MB)")
    
    def stream_to_file(
        self, 
        output_path: Path,
        seed: Optional[int] = None,
        append: bool = False
    ) -> StreamStats:
        """
        Stream transactions directly to file (most memory efficient).
        
        Args:
            output_path: Path to output file
            seed: Random seed
            append: Append to existing file (CSV/JSON only)
            
        Returns:
            Generation statistics
        """
        import time
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension if not specified
        if output_path.suffix == '.parquet':
            format_type = 'parquet'
        elif output_path.suffix == '.json' or output_path.suffix == '.jsonl':
            format_type = 'json'
        else:
            format_type = 'csv'
        
        logger.info(f"Streaming {self.config.num_transactions:,} transactions to {output_path}")
        
        start_time = time.time()
        
        if format_type == 'csv':
            self._stream_to_csv(output_path, seed, append)
        elif format_type == 'json':
            self._stream_to_json(output_path, seed, append)
        elif format_type == 'parquet':
            self._stream_to_parquet(output_path, seed)
        
        # Update stats with file size
        if self.stats:
            self.stats.output_file_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"Output file size: {self.stats.output_file_mb:.1f} MB")
        
        return self.stats
    
    def _stream_to_csv(
        self, 
        output_path: Path, 
        seed: Optional[int],
        append: bool
    ):
        """Stream to CSV file."""
        mode = 'a' if append else 'w'
        write_header = not (append and output_path.exists())
        
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = None
            
            for batch_df in self.generate_batches(seed):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=batch_df.columns)
                    if write_header:
                        writer.writeheader()
                
                # Write batch
                for _, row in batch_df.iterrows():
                    writer.writerow(row.to_dict())
    
    def _stream_to_json(
        self, 
        output_path: Path, 
        seed: Optional[int],
        append: bool
    ):
        """Stream to JSON Lines file."""
        mode = 'a' if append else 'w'
        
        with open(output_path, mode, encoding='utf-8') as f:
            for batch_df in self.generate_batches(seed):
                for _, row in batch_df.iterrows():
                    json.dump(row.to_dict(), f, default=str)
                    f.write('\n')
    
    def _stream_to_parquet(
        self, 
        output_path: Path, 
        seed: Optional[int]
    ):
        """Stream to Parquet file (requires collecting all batches)."""
        # Parquet doesn't support true streaming, so we collect batches
        batches = list(self.generate_batches(seed))
        
        if batches:
            df = pd.concat(batches, ignore_index=True)
            df.to_parquet(output_path, index=False, engine='auto')
    
    def estimate_memory(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate memory usage for streaming.
        
        Args:
            batch_size: Batch size to estimate (uses config if None)
            
        Returns:
            Dictionary with memory estimates in MB
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Generate small sample to estimate per-transaction memory
        self._initialize_generators(seed=42)
        
        sample_df = self._data_gen(
            num_customers=10,
            transactions_per_customer=10,
            start_date=datetime.strptime(self.config.start_date, "%Y-%m-%d"),
            days=1,
            seed=42
        )
        
        # Calculate memory per transaction
        mem_per_txn = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        
        # Estimates
        base_memory = 50  # Customer data + generator state
        batch_memory = (mem_per_txn * batch_size) / 1024 / 1024
        total_memory = base_memory + batch_memory
        
        return {
            'base_mb': base_memory,
            'batch_mb': batch_memory,
            'total_mb': total_memory,
            'per_transaction_bytes': mem_per_txn
        }


class ChunkedFileReader:
    """
    Read large transaction files in chunks (memory efficient).
    
    Example:
        >>> reader = ChunkedFileReader('large_file.csv', chunk_size=10000)
        >>> for chunk in reader.read_chunks():
        >>>     process(chunk)
    """
    
    def __init__(self, file_path: Path, chunk_size: int = 10000):
        """
        Initialize chunked reader.
        
        Args:
            file_path: Path to file
            chunk_size: Rows per chunk
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Read file in chunks.
        
        Yields:
            DataFrame chunks
        """
        if self.file_path.suffix == '.csv':
            yield from pd.read_csv(self.file_path, chunksize=self.chunk_size)
        elif self.file_path.suffix == '.parquet':
            # Parquet doesn't support native chunking, read full file
            df = pd.read_parquet(self.file_path)
            for i in range(0, len(df), self.chunk_size):
                yield df.iloc[i:i + self.chunk_size]
        elif self.file_path.suffix in ['.json', '.jsonl']:
            # Read JSON lines in chunks
            chunks = []
            with open(self.file_path, 'r') as f:
                for i, line in enumerate(f):
                    chunks.append(json.loads(line))
                    if len(chunks) >= self.chunk_size:
                        yield pd.DataFrame(chunks)
                        chunks = []
                
                if chunks:
                    yield pd.DataFrame(chunks)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
    
    def count_rows(self) -> int:
        """Count total rows without loading into memory."""
        total = 0
        for chunk in self.read_chunks():
            total += len(chunk)
        return total


if __name__ == "__main__":
    # Demo: Stream 100K transactions
    logging.basicConfig(level=logging.INFO)
    
    print("=== Streaming Generator Demo ===\n")
    
    config = StreamConfig(
        num_transactions=100000,
        batch_size=5000,
        fraud_rate=0.05
    )
    
    generator = StreamingGenerator(config)
    
    # Estimate memory
    mem_est = generator.estimate_memory()
    print(f"Memory Estimate:")
    print(f"  Base: {mem_est['base_mb']:.1f} MB")
    print(f"  Per Batch: {mem_est['batch_mb']:.1f} MB")
    print(f"  Total: {mem_est['total_mb']:.1f} MB")
    
    # Stream to file
    print(f"\nStreaming to file...")
    output_file = Path("output/streaming_demo.csv")
    stats = generator.stream_to_file(output_file, seed=42)
    
    print(f"\n=== Results ===")
    print(f"Transactions: {stats.total_transactions:,}")
    print(f"Batches: {stats.total_batches}")
    print(f"Time: {stats.total_time:.2f}s")
    print(f"Speed: {stats.transactions_per_second:.0f} txns/sec")
    print(f"Peak Memory: {stats.peak_memory_mb:.1f} MB")
    print(f"Output File: {stats.output_file_mb:.1f} MB")
    print(f"Fraud: {stats.fraud_count:,} ({stats.fraud_count/stats.total_transactions*100:.1f}%)")
