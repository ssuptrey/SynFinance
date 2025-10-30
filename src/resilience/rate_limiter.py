"""
Rate Limiter using Token Bucket Algorithm

Controls request rates to prevent API abuse and system overload.

Week 7 Day 7: Final Integration
"""

import time
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
from functools import wraps

try:
    from src.observability import get_logger, LogCategory
    logger = get_logger(__name__)
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    OBSERVABILITY_AVAILABLE = False
    LogCategory = None


@dataclass
class RateLimitConfig:
    """Rate limiter configuration"""
    rate: float = 10.0  # Requests per second
    capacity: int = 10  # Maximum burst capacity
    per_user: bool = False  # Rate limit per user vs global


class RateLimiter:
    """
    Token Bucket Rate Limiter
    
    Implements token bucket algorithm for smooth rate limiting:
    - Tokens are added at a constant rate
    - Each request consumes a token
    - Requests are blocked when bucket is empty
    - Supports burst traffic up to bucket capacity
    
    Usage:
        limiter = RateLimiter(rate=10.0, capacity=10)
        
        @limiter
        def api_endpoint():
            # Your code here
            pass
    """
    
    def __init__(
        self,
        rate: float = 10.0,
        capacity: int = 10,
        per_user: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize rate limiter
        
        Args:
            rate: Requests per second allowed
            capacity: Maximum burst capacity (bucket size)
            per_user: If True, maintain separate buckets per user
            name: Optional name for logging
        """
        self.config = RateLimitConfig(
            rate=rate,
            capacity=capacity,
            per_user=per_user
        )
        self.name = name or "RateLimiter"
        
        # Token bucket state
        self._buckets: dict[str, dict] = {}
        self._global_bucket = {
            'tokens': float(capacity),
            'last_update': time.time()
        }
        self._lock = threading.RLock()
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Rate limiter '{self.name}' initialized",
                category=LogCategory.SYSTEM,
                context={
                    'rate': rate,
                    'capacity': capacity,
                    'per_user': per_user
                }
            )
    
    def _get_bucket(self, user_id: Optional[str] = None) -> dict:
        """Get or create bucket for user"""
        if not self.config.per_user or user_id is None:
            return self._global_bucket
        
        if user_id not in self._buckets:
            self._buckets[user_id] = {
                'tokens': float(self.config.capacity),
                'last_update': time.time()
            }
        
        return self._buckets[user_id]
    
    def _refill_bucket(self, bucket: dict) -> None:
        """Refill bucket based on time elapsed"""
        now = time.time()
        time_elapsed = now - bucket['last_update']
        
        # Add tokens based on rate and time elapsed
        tokens_to_add = time_elapsed * self.config.rate
        bucket['tokens'] = min(
            self.config.capacity,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_update'] = now
    
    def acquire(
        self,
        tokens: int = 1,
        user_id: Optional[str] = None,
        blocking: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire tokens from the bucket
        
        Args:
            tokens: Number of tokens to acquire
            user_id: User identifier for per-user limiting
            blocking: If True, wait until tokens are available
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if tokens acquired, False if rate limited
            
        Raises:
            RateLimitError: If rate limit exceeded and blocking=False
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                bucket = self._get_bucket(user_id)
                self._refill_bucket(bucket)
                
                if bucket['tokens'] >= tokens:
                    # Tokens available, acquire them
                    bucket['tokens'] -= tokens
                    return True
                
                if not blocking:
                    # Non-blocking and no tokens available
                    if OBSERVABILITY_AVAILABLE:
                        logger.warning(
                            f"Rate limit exceeded for '{self.name}'",
                            category=LogCategory.SYSTEM,
                            context={
                                'user_id': user_id,
                                'tokens_requested': tokens,
                                'tokens_available': bucket['tokens']
                            }
                        )
                    
                    raise RateLimitError(
                        f"Rate limit exceeded for '{self.name}'. "
                        f"Try again in {1.0 / self.config.rate:.2f}s"
                    )
                
                # Calculate wait time until next token
                tokens_needed = tokens - bucket['tokens']
                wait_time = tokens_needed / self.config.rate
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed + wait_time > timeout:
                        return False
                    wait_time = min(wait_time, timeout - elapsed)
            
            # Wait outside of lock
            time.sleep(wait_time)
    
    def call(
        self,
        func: Callable,
        *args,
        tokens: int = 1,
        user_id: Optional[str] = None,
        blocking: bool = True,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with rate limiting
        
        Args:
            func: Function to call
            *args: Positional arguments
            tokens: Number of tokens to acquire
            user_id: User identifier for per-user limiting
            blocking: If True, wait until tokens are available
            timeout: Maximum time to wait
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RateLimitError: If rate limit exceeded
        """
        if not self.acquire(tokens=tokens, user_id=user_id, blocking=blocking, timeout=timeout):
            raise RateLimitError(f"Rate limit exceeded for '{self.name}'")
        
        return func(*args, **kwargs)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator usage
        
        @limiter
        def my_function():
            pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper
    
    def reset(self, user_id: Optional[str] = None):
        """Reset rate limiter state"""
        with self._lock:
            if user_id is None:
                # Reset global bucket
                self._global_bucket = {
                    'tokens': float(self.config.capacity),
                    'last_update': time.time()
                }
                # Reset all user buckets
                self._buckets.clear()
            elif user_id in self._buckets:
                # Reset specific user bucket
                self._buckets[user_id] = {
                    'tokens': float(self.config.capacity),
                    'last_update': time.time()
                }
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Rate limiter '{self.name}' reset",
                category=LogCategory.SYSTEM,
                context={'user_id': user_id}
            )
    
    def get_stats(self, user_id: Optional[str] = None) -> dict:
        """Get rate limiter statistics"""
        with self._lock:
            bucket = self._get_bucket(user_id)
            self._refill_bucket(bucket)
            
            return {
                'name': self.name,
                'user_id': user_id,
                'tokens_available': bucket['tokens'],
                'capacity': self.config.capacity,
                'rate': self.config.rate,
                'utilization': 1.0 - (bucket['tokens'] / self.config.capacity)
            }
    
    def get_all_stats(self) -> list[dict]:
        """Get statistics for all users"""
        with self._lock:
            stats = [self.get_stats()]  # Global stats
            
            if self.config.per_user:
                for user_id in self._buckets:
                    stats.append(self.get_stats(user_id))
            
            return stats


class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""
    pass


# Global rate limiter registry
_rate_limiters: dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(
    name: str,
    rate: float = 10.0,
    capacity: int = 10,
    per_user: bool = False
) -> RateLimiter:
    """
    Get or create a named rate limiter
    
    Args:
        name: Unique name for rate limiter
        rate: Requests per second
        capacity: Maximum burst capacity
        per_user: Rate limit per user vs global
        
    Returns:
        RateLimiter instance
    """
    with _registry_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = RateLimiter(
                rate=rate,
                capacity=capacity,
                per_user=per_user,
                name=name
            )
        
        return _rate_limiters[name]


def reset_all_rate_limiters():
    """Reset all registered rate limiters"""
    with _registry_lock:
        for limiter in _rate_limiters.values():
            limiter.reset()


def get_all_rate_limiter_stats() -> list[dict]:
    """Get statistics for all rate limiters"""
    with _registry_lock:
        stats = []
        for limiter in _rate_limiters.values():
            stats.extend(limiter.get_all_stats())
        return stats
