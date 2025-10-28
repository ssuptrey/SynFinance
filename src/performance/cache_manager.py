"""
Intelligent caching system for performance optimization.

This module provides LRU caching for frequently accessed data:
- Customer profiles
- Merchant data
- Transaction history
- Feature calculations
"""

import logging
from functools import lru_cache, wraps
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import pickle
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    avg_lookup_time_ms: float = 0.0
    total_lookups: int = 0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        self.total_lookups = self.hits + self.misses
        if self.total_lookups > 0:
            self.hit_rate = self.hits / self.total_lookups
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_lookups': self.total_lookups,
            'hit_rate': f'{self.hit_rate:.2%}',
            'size': self.size,
            'max_size': self.max_size,
            'avg_lookup_time_ms': f'{self.avg_lookup_time_ms:.3f}'
        }


class CacheManager:
    """
    Intelligent multi-level cache manager.
    
    Features:
    - LRU eviction policy
    - Automatic cache warming
    - Hit/miss tracking
    - Persistent disk cache
    - TTL (time-to-live) support
    
    Caches:
    - Customer profiles
    - Merchant data
    - Transaction history aggregates
    - Feature calculations
    
    Example:
        >>> cache = CacheManager(max_size=1000)
        >>> 
        >>> # Cache customer data
        >>> customer = cache.get_customer(customer_id)
        >>> if customer is None:
        >>>     customer = load_customer(customer_id)
        >>>     cache.set_customer(customer_id, customer)
        >>>
        >>> # Check stats
        >>> print(cache.get_stats())
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        enable_disk_cache: bool = False,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum items in memory cache
            enable_disk_cache: Enable persistent disk caching
            cache_dir: Directory for disk cache
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("output/cache")
        self.ttl_seconds = ttl_seconds
        
        # Memory caches (LRU)
        self._customer_cache: Dict[str, Tuple[Any, float]] = {}
        self._merchant_cache: Dict[str, Tuple[Any, float]] = {}
        self._history_cache: Dict[str, Tuple[Any, float]] = {}
        self._feature_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Cache statistics
        self.stats = {
            'customer': CacheStats(max_size=max_size),
            'merchant': CacheStats(max_size=max_size),
            'history': CacheStats(max_size=max_size),
            'feature': CacheStats(max_size=max_size)
        }
        
        # Lookup times
        self._lookup_times: List[float] = []
        
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk cache enabled at {self.cache_dir}")
    
    def get_customer(self, customer_id: str) -> Optional[Any]:
        """
        Get customer from cache.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Customer data or None if not cached
        """
        return self._get('customer', customer_id, self._customer_cache)
    
    def set_customer(self, customer_id: str, customer_data: Any):
        """
        Store customer in cache.
        
        Args:
            customer_id: Customer ID
            customer_data: Customer data to cache
        """
        self._set('customer', customer_id, customer_data, self._customer_cache)
    
    def get_merchant(self, merchant_id: str) -> Optional[Any]:
        """Get merchant from cache."""
        return self._get('merchant', merchant_id, self._merchant_cache)
    
    def set_merchant(self, merchant_id: str, merchant_data: Any):
        """Store merchant in cache."""
        self._set('merchant', merchant_id, merchant_data, self._merchant_cache)
    
    def get_history(self, customer_id: str) -> Optional[Any]:
        """Get transaction history from cache."""
        return self._get('history', customer_id, self._history_cache)
    
    def set_history(self, customer_id: str, history_data: Any):
        """Store transaction history in cache."""
        self._set('history', customer_id, history_data, self._history_cache)
    
    def get_features(self, transaction_id: str) -> Optional[Any]:
        """Get pre-computed features from cache."""
        return self._get('feature', transaction_id, self._feature_cache)
    
    def set_features(self, transaction_id: str, features: Any):
        """Store pre-computed features in cache."""
        self._set('feature', transaction_id, features, self._feature_cache)
    
    def _get(
        self, 
        cache_type: str, 
        key: str, 
        cache_dict: Dict
    ) -> Optional[Any]:
        """
        Generic cache get with TTL and disk fallback.
        
        Args:
            cache_type: Type of cache ('customer', 'merchant', etc.)
            key: Cache key
            cache_dict: Cache dictionary
            
        Returns:
            Cached value or None
        """
        start_time = time.time()
        
        # Check memory cache
        if key in cache_dict:
            value, timestamp = cache_dict[key]
            
            # Check TTL
            if self.ttl_seconds is None or (time.time() - timestamp) < self.ttl_seconds:
                self.stats[cache_type].hits += 1
                self.stats[cache_type].update_hit_rate()
                self._record_lookup_time(start_time)
                return value
            else:
                # Expired - remove
                del cache_dict[key]
        
        # Check disk cache
        if self.enable_disk_cache:
            disk_value = self._load_from_disk(cache_type, key)
            if disk_value is not None:
                # Restore to memory cache
                cache_dict[key] = (disk_value, time.time())
                self._evict_if_needed(cache_dict)
                
                self.stats[cache_type].hits += 1
                self.stats[cache_type].update_hit_rate()
                self._record_lookup_time(start_time)
                return disk_value
        
        # Cache miss
        self.stats[cache_type].misses += 1
        self.stats[cache_type].update_hit_rate()
        self._record_lookup_time(start_time)
        return None
    
    def _set(
        self, 
        cache_type: str, 
        key: str, 
        value: Any, 
        cache_dict: Dict
    ):
        """
        Generic cache set with LRU eviction.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Value to cache
            cache_dict: Cache dictionary
        """
        # Store in memory
        cache_dict[key] = (value, time.time())
        
        # Evict if needed (LRU)
        self._evict_if_needed(cache_dict)
        
        # Update stats
        self.stats[cache_type].size = len(cache_dict)
        
        # Store to disk if enabled
        if self.enable_disk_cache:
            self._save_to_disk(cache_type, key, value)
    
    def _evict_if_needed(self, cache_dict: Dict):
        """
        Evict oldest entries if cache exceeds max_size (LRU).
        
        Args:
            cache_dict: Cache dictionary to evict from
        """
        if len(cache_dict) > self.max_size:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(cache_dict.items(), key=lambda x: x[1][1])
            
            # Remove oldest 10%
            num_to_remove = len(cache_dict) - self.max_size
            for key, _ in sorted_items[:num_to_remove]:
                del cache_dict[key]
    
    def _save_to_disk(self, cache_type: str, key: str, value: Any):
        """Save cache entry to disk."""
        try:
            cache_file = self._get_cache_file(cache_type, key)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, cache_type: str, key: str) -> Optional[Any]:
        """Load cache entry from disk."""
        try:
            cache_file = self._get_cache_file(cache_type, key)
            
            if cache_file.exists():
                # Check TTL
                if self.ttl_seconds is not None:
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age > self.ttl_seconds:
                        cache_file.unlink()  # Delete expired
                        return None
                
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
        
        return None
    
    def _get_cache_file(self, cache_type: str, key: str) -> Path:
        """Get cache file path for a key."""
        # Hash key to create valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / cache_type / f"{key_hash}.pkl"
    
    def _record_lookup_time(self, start_time: float):
        """Record lookup time for statistics."""
        elapsed_ms = (time.time() - start_time) * 1000
        self._lookup_times.append(elapsed_ms)
        
        # Keep only last 1000 lookups
        if len(self._lookup_times) > 1000:
            self._lookup_times = self._lookup_times[-1000:]
        
        # Update average
        if self._lookup_times:
            avg_time = sum(self._lookup_times) / len(self._lookup_times)
            for stat in self.stats.values():
                stat.avg_lookup_time_ms = avg_time
    
    def clear(self, cache_type: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            cache_type: Specific cache to clear (None = all)
        """
        if cache_type is None or cache_type == 'customer':
            self._customer_cache.clear()
            self.stats['customer'] = CacheStats(max_size=self.max_size)
        
        if cache_type is None or cache_type == 'merchant':
            self._merchant_cache.clear()
            self.stats['merchant'] = CacheStats(max_size=self.max_size)
        
        if cache_type is None or cache_type == 'history':
            self._history_cache.clear()
            self.stats['history'] = CacheStats(max_size=self.max_size)
        
        if cache_type is None or cache_type == 'feature':
            self._feature_cache.clear()
            self.stats['feature'] = CacheStats(max_size=self.max_size)
        
        logger.info(f"Cleared cache: {cache_type or 'all'}")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with stats for each cache type
        """
        return {
            cache_type: stats.to_dict()
            for cache_type, stats in self.stats.items()
        }
    
    def warm_cache(
        self, 
        customers: List[Any] = None,
        merchants: List[Any] = None
    ):
        """
        Pre-populate cache with frequently accessed data.
        
        Args:
            customers: List of customers to cache
            merchants: List of merchants to cache
        """
        if customers:
            for customer in customers:
                customer_id = customer.get('customer_id') or customer.get('id')
                if customer_id:
                    self.set_customer(customer_id, customer)
            
            logger.info(f"Warmed customer cache with {len(customers)} entries")
        
        if merchants:
            for merchant in merchants:
                merchant_id = merchant.get('merchant_id') or merchant.get('id')
                if merchant_id:
                    self.set_merchant(merchant_id, merchant)
            
            logger.info(f"Warmed merchant cache with {len(merchants)} entries")
    
    def export_stats_report(self, output_file: Optional[Path] = None) -> str:
        """
        Export detailed statistics report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            "CACHE PERFORMANCE REPORT",
            "=" * 60,
            ""
        ]
        
        for cache_type, stats in self.stats.items():
            report_lines.extend([
                f"{cache_type.upper()} CACHE:",
                f"  Hits: {stats.hits:,}",
                f"  Misses: {stats.misses:,}",
                f"  Hit Rate: {stats.hit_rate:.2%}",
                f"  Size: {stats.size:,} / {stats.max_size:,}",
                f"  Avg Lookup: {stats.avg_lookup_time_ms:.3f} ms",
                ""
            ])
        
        report_lines.extend([
            "OVERALL:",
            f"  Total Lookups: {sum(s.total_lookups for s in self.stats.values()):,}",
            f"  Average Hit Rate: {sum(s.hit_rate for s in self.stats.values()) / len(self.stats):.2%}",
            f"  Disk Cache: {'Enabled' if self.enable_disk_cache else 'Disabled'}",
            f"  TTL: {self.ttl_seconds}s" if self.ttl_seconds else "  TTL: None",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report)
            logger.info(f"Saved cache report to {output_file}")
        
        return report


def cached_method(cache_manager: CacheManager, cache_type: str):
    """
    Decorator for caching method results.
    
    Args:
        cache_manager: CacheManager instance
        cache_type: Type of cache to use
        
    Example:
        >>> cache = CacheManager()
        >>> 
        >>> @cached_method(cache, 'feature')
        >>> def expensive_calculation(transaction_id):
        >>>     # ... expensive work ...
        >>>     return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try cache
            result = cache_manager.get_features(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.set_features(key, result)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo: Cache usage
    logging.basicConfig(level=logging.INFO)
    
    print("=== Cache Manager Demo ===\n")
    
    cache = CacheManager(max_size=1000, enable_disk_cache=True)
    
    # Simulate cache usage
    print("Simulating cache operations...")
    
    # Store some data
    for i in range(100):
        cache.set_customer(f"CUST{i:03d}", {'id': f"CUST{i:03d}", 'name': f"Customer {i}"})
        cache.set_merchant(f"MERCH{i:03d}", {'id': f"MERCH{i:03d}", 'name': f"Merchant {i}"})
    
    # Simulate lookups (mix of hits and misses)
    for i in range(200):
        customer_id = f"CUST{i%150:03d}"  # Some will miss
        customer = cache.get_customer(customer_id)
    
    # Print stats
    print("\n=== Cache Statistics ===")
    for cache_type, stats in cache.get_stats().items():
        print(f"\n{cache_type.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Export report
    print("\n" + cache.export_stats_report())
