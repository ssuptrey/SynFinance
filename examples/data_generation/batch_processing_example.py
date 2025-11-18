"""
Batch Processing Example
=========================

Demonstrates high-performance batch processing of large transaction datasets
for fraud detection using SynFinance.

This example shows:
1. Streaming large datasets efficiently
2. Parallel processing for speed
3. Memory management techniques
4. Progress tracking
5. Result aggregation
6. Performance optimization

Author: SynFinance Team
Version: 0.7.0
Date: October 28, 2025
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Iterator
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.performance.streaming_generator import StreamingGenerator, ChunkedFileReader
from src.performance.parallel_generator import ParallelGenerator
from src.performance.cache_manager import CacheManager
from src.generators.combined_ml_features import CombinedMLFeatureGenerator


class BatchProcessor:
    """
    High-performance batch processor for fraud detection.
    
    Handles large datasets efficiently using:
    - Streaming to minimize memory usage
    - Parallel processing for speed
    - Caching for frequently accessed data
    - Progress tracking and reporting
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        num_workers: int = None,
        use_cache: bool = True,
        output_dir: str = "output/batch_processing"
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of transactions per batch
            num_workers: Number of parallel workers (default: CPU count)
            use_cache: Enable caching for performance
            output_dir: Directory for output files
        """
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cache_manager = CacheManager() if use_cache else None
        self.feature_generator = CombinedMLFeatureGenerator()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'processing_time': 0,
            'batches_processed': 0
        }
        
        print("=" * 80)
        print("SynFinance Batch Processor")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Batch Size: {self.batch_size:,}")
        print(f"  Workers: {self.num_workers}")
        print(f"  Caching: {'Enabled' if use_cache else 'Disabled'}")
        print(f"  Output Dir: {self.output_dir}")
        print()
    
    def generate_large_dataset(
        self,
        num_transactions: int = 100000,
        fraud_rate: float = 0.05
    ) -> str:
        """
        Generate a large dataset for batch processing demonstration.
        
        Args:
            num_transactions: Number of transactions to generate
            fraud_rate: Fraud rate (0-1)
        
        Returns:
            Path to generated CSV file
        """
        print(f"Generating {num_transactions:,} transactions...")
        start_time = time.time()
        
        # Use parallel generation for speed
        parallel_gen = ParallelGenerator(num_workers=self.num_workers)
        transactions = parallel_gen.generate(
            num_transactions=num_transactions,
            fraud_rate=fraud_rate
        )
        
        # Save to CSV
        output_file = self.output_dir / f"dataset_{num_transactions}.csv"
        df = pd.DataFrame(transactions)
        df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        fraud_count = sum(1 for t in transactions if t.get('is_fraud', False))
        
        print(f"✓ Generated {len(transactions):,} transactions in {elapsed:.2f}s")
        print(f"  - Fraud: {fraud_count:,} ({fraud_count/len(transactions):.1%})")
        print(f"  - File: {output_file}")
        print(f"  - Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(output_file)
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of transactions.
        
        Args:
            batch: List of transactions to process
            history: Transaction history for context
        
        Returns:
            List of processed results
        """
        results = []
        
        for i, txn in enumerate(batch):
            # Generate features
            txn_history = history + batch[:i]
            features = self.feature_generator.generate_features(txn, txn_history)
            
            # Simple fraud detection (in production, use trained model)
            fraud_score = self._calculate_fraud_score(features)
            is_fraud = fraud_score > 0.5
            
            result = {
                'transaction_id': txn.get('transaction_id', f"TXN_{i}"),
                'amount': txn.get('amount', 0),
                'is_fraud': is_fraud,
                'fraud_probability': fraud_score,
                'risk_level': self._get_risk_level(fraud_score)
            }
            results.append(result)
            
            if is_fraud:
                self.stats['fraud_detected'] += 1
        
        self.stats['total_processed'] += len(batch)
        self.stats['batches_processed'] += 1
        
        return results
    
    def _calculate_fraud_score(self, features) -> float:
        """Calculate fraud score from features (simplified)."""
        # In production, use trained ML model
        # This is a simplified example using key features
        
        score = 0.0
        
        # Check velocity features
        if hasattr(features, 'fraud_velocity_1h'):
            score += min(features.fraud_velocity_1h / 10.0, 0.3)
        
        # Check amount features
        if hasattr(features, 'fraud_amount_deviation'):
            score += min(abs(features.fraud_amount_deviation) / 5.0, 0.2)
        
        # Check geographic features
        if hasattr(features, 'fraud_distance_from_home'):
            score += min(features.fraud_distance_from_home / 1000.0, 0.2)
        
        # Check anomaly indicators
        if hasattr(features, 'anomaly_severity_max'):
            score += min(features.anomaly_severity_max / 10.0, 0.3)
        
        return min(score, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level from fraud score."""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def process_file_streaming(
        self,
        input_file: str,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Process a file using streaming for memory efficiency.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)
        
        Returns:
            Processing statistics
        """
        print("\n" + "=" * 80)
        print("STREAMING BATCH PROCESSING")
        print("=" * 80)
        
        if output_file is None:
            output_file = self.output_dir / "streaming_results.csv"
        
        print(f"\nInput: {input_file}")
        print(f"Output: {output_file}")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'processing_time': 0,
            'batches_processed': 0
        }
        
        start_time = time.time()
        
        # Stream and process in chunks
        reader = ChunkedFileReader(chunk_size=self.batch_size)
        history = []
        all_results = []
        
        print(f"\nProcessing in batches of {self.batch_size:,}...")
        
        for i, chunk_df in enumerate(reader.read_csv_chunks(input_file)):
            batch = chunk_df.to_dict('records')
            
            # Process batch
            results = self.process_batch(batch, history)
            all_results.extend(results)
            
            # Update history (keep last 1000 for context)
            history.extend(batch)
            if len(history) > 1000:
                history = history[-1000:]
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = self.stats['total_processed'] / elapsed
                print(f"  Batch {i+1}: {self.stats['total_processed']:,} processed "
                      f"({speed:.0f} txns/sec), {self.stats['fraud_detected']} fraud detected")
        
        elapsed = time.time() - start_time
        self.stats['processing_time'] = elapsed
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✓ Streaming processing completed!")
        print(f"  Total Processed: {self.stats['total_processed']:,}")
        print(f"  Fraud Detected: {self.stats['fraud_detected']} ({self.stats['fraud_detected']/self.stats['total_processed']:.1%})")
        print(f"  Processing Time: {elapsed:.2f}s")
        print(f"  Throughput: {self.stats['total_processed']/elapsed:.0f} txns/sec")
        print(f"  Output Saved: {output_file}")
        
        return self.stats.copy()
    
    def process_file_parallel(
        self,
        input_file: str,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Process a file using parallel processing for speed.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)
        
        Returns:
            Processing statistics
        """
        print("\n" + "=" * 80)
        print("PARALLEL BATCH PROCESSING")
        print("=" * 80)
        
        if output_file is None:
            output_file = self.output_dir / "parallel_results.csv"
        
        print(f"\nInput: {input_file}")
        print(f"Output: {output_file}")
        print(f"Workers: {self.num_workers}")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'processing_time': 0,
            'batches_processed': 0
        }
        
        start_time = time.time()
        
        # Read entire file (for parallel processing)
        print(f"\nReading input file...")
        df = pd.read_csv(input_file)
        transactions = df.to_dict('records')
        print(f"✓ Loaded {len(transactions):,} transactions")
        
        # Split into batches
        batches = [
            transactions[i:i+self.batch_size]
            for i in range(0, len(transactions), self.batch_size)
        ]
        
        print(f"\nProcessing {len(batches)} batches in parallel...")
        
        # Process batches in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            # Note: History is simplified for parallel processing
            process_func = partial(self.process_batch, history=[])
            batch_results = pool.map(process_func, batches)
        
        # Flatten results
        all_results = [r for batch in batch_results for r in batch]
        
        elapsed = time.time() - start_time
        self.stats['processing_time'] = elapsed
        self.stats['total_processed'] = len(all_results)
        self.stats['fraud_detected'] = sum(1 for r in all_results if r['is_fraud'])
        self.stats['batches_processed'] = len(batches)
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✓ Parallel processing completed!")
        print(f"  Total Processed: {self.stats['total_processed']:,}")
        print(f"  Fraud Detected: {self.stats['fraud_detected']} ({self.stats['fraud_detected']/self.stats['total_processed']:.1%})")
        print(f"  Processing Time: {elapsed:.2f}s")
        print(f"  Throughput: {self.stats['total_processed']/elapsed:.0f} txns/sec")
        print(f"  Speedup: {self.num_workers:.1f}x (theoretical)")
        print(f"  Output Saved: {output_file}")
        
        return self.stats.copy()
    
    def compare_methods(self, input_file: str):
        """
        Compare streaming vs parallel processing methods.
        
        Args:
            input_file: Path to input CSV file
        """
        print("\n" + "=" * 80)
        print("METHOD COMPARISON")
        print("=" * 80)
        
        # Test streaming
        print("\n1. Testing Streaming Method...")
        streaming_stats = self.process_file_streaming(input_file)
        
        # Test parallel
        print("\n2. Testing Parallel Method...")
        parallel_stats = self.process_file_parallel(input_file)
        
        # Compare
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        print(f"\n{'Metric':<25} {'Streaming':<20} {'Parallel':<20}")
        print("-" * 65)
        print(f"{'Processing Time':<25} {streaming_stats['processing_time']:>10.2f}s       {parallel_stats['processing_time']:>10.2f}s")
        print(f"{'Throughput':<25} {streaming_stats['total_processed']/streaming_stats['processing_time']:>10.0f} txns/s   {parallel_stats['total_processed']/parallel_stats['processing_time']:>10.0f} txns/s")
        print(f"{'Total Processed':<25} {streaming_stats['total_processed']:>10,}         {parallel_stats['total_processed']:>10,}")
        print(f"{'Fraud Detected':<25} {streaming_stats['fraud_detected']:>10}           {parallel_stats['fraud_detected']:>10}")
        
        speedup = streaming_stats['processing_time'] / parallel_stats['processing_time']
        print(f"\n✓ Parallel is {speedup:.2f}x faster than streaming")
        
        print("\nRecommendations:")
        if speedup > 2:
            print("  → Use PARALLEL processing for maximum speed")
        else:
            print("  → Use STREAMING for better memory efficiency")
        print("  → Consider dataset size and available RAM")
        print("  → Streaming is better for datasets > 1M transactions")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("  SynFinance Batch Processing Demonstration")
    print("  Version: 0.7.0")
    print("=" * 80)
    
    # Initialize processor
    processor = BatchProcessor(
        batch_size=1000,
        num_workers=4,
        use_cache=True
    )
    
    # Generate sample dataset
    print("\n[Step 1] Generating sample dataset...")
    input_file = processor.generate_large_dataset(
        num_transactions=50000,  # 50K transactions
        fraud_rate=0.05           # 5% fraud
    )
    
    # Wait for user
    input("\nPress Enter to start batch processing...")
    
    # Compare methods
    processor.compare_methods(input_file)
    
    print("\n" + "=" * 80)
    print("  Batch Processing Demonstration Complete!")
    print("=" * 80)
    
    print("\nKey Takeaways:")
    print("  1. Streaming minimizes memory usage for large datasets")
    print("  2. Parallel processing maximizes speed with multiple CPUs")
    print("  3. Caching improves performance for repeated operations")
    print("  4. Batch size affects memory/speed tradeoff")
    print("  5. Choose method based on dataset size and resources")
    
    print(f"\nResults saved to: {processor.output_dir}")


if __name__ == "__main__":
    main()
