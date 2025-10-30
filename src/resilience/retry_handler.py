"""
Retry Handler with Exponential Backoff

Automatically retries failed operations with intelligent backoff strategies.

Week 7 Day 7: Final Integration
"""

import time
import random
from typing import Callable, Any, Optional, Type
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
class RetryConfig:
    """Retry handler configuration"""
    max_retries: int = 3  # Maximum retry attempts
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retry_exceptions: tuple = (Exception,)  # Exception types to retry


class RetryHandler:
    """
    Retry Handler with Exponential Backoff and Jitter
    
    Automatically retries failed operations with intelligent backoff:
    - Exponential backoff: delay = base_delay * (exponential_base ^ attempt)
    - Jitter: Adds randomness to prevent thundering herd problem
    - Configurable exceptions: Only retry specific exception types
    
    Usage:
        retry = RetryHandler(max_retries=3, base_delay=1.0)
        
        @retry
        def unreliable_operation():
            # Your code here
            pass
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
        name: Optional[str] = None
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum retry attempts (0 = no retries)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Exponential backoff multiplier
            jitter: Add random jitter to delays
            retry_exceptions: Exception types to retry
            name: Optional name for logging
        """
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retry_exceptions=retry_exceptions
        )
        self.name = name or "RetryHandler"
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Retry handler '{self.name}' initialized",
                category=LogCategory.SYSTEM,
                context={
                    'max_retries': max_retries,
                    'base_delay': base_delay,
                    'max_delay': max_delay
                }
            )
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt
        
        Args:
            attempt: Current retry attempt (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            # Full jitter: random value between 0 and calculated delay
            delay = random.uniform(0, delay)
        
        return delay
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried
        
        Args:
            exception: Exception that occurred
            attempt: Current retry attempt (0-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max retries
        if attempt >= self.config.max_retries:
            return False
        
        # Check if exception type should be retried
        return isinstance(exception, self.config.retry_exceptions)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Success!
                if attempt > 0:
                    if OBSERVABILITY_AVAILABLE:
                        logger.info(
                            f"Retry handler '{self.name}' succeeded after {attempt} retries",
                            category=LogCategory.SYSTEM,
                            context={'function': func.__name__}
                        )
                    else:
                        logger.info(f"Retry handler '{self.name}' succeeded after {attempt} retries")
                
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    # Don't retry, re-raise
                    if OBSERVABILITY_AVAILABLE:
                        logger.error(
                            f"Retry handler '{self.name}' failed permanently",
                            category=LogCategory.SYSTEM,
                            context={
                                'function': func.__name__,
                                'attempts': attempt + 1,
                                'exception': str(e)
                            }
                        )
                    else:
                        logger.error(f"Retry handler '{self.name}' failed after {attempt + 1} attempts: {e}")
                    
                    raise
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                
                if OBSERVABILITY_AVAILABLE:
                    logger.warning(
                        f"Retry handler '{self.name}' attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                        category=LogCategory.SYSTEM,
                        context={
                            'function': func.__name__,
                            'exception': str(e),
                            'next_delay': delay
                        }
                    )
                else:
                    logger.warning(
                        f"Retry handler '{self.name}' attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                
                time.sleep(delay)
        
        # All retries exhausted, re-raise last exception
        raise last_exception
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator usage
        
        @retry
        def my_function():
            pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper


def retry_on_exception(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff multiplier
        jitter: Add random jitter to delays
        retry_exceptions: Exception types to retry
        
    Returns:
        Decorator function
    
    Usage:
        @retry_on_exception(max_retries=3, base_delay=1.0)
        def my_function():
            pass
    """
    handler = RetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions
    )
    
    return handler


# Convenience functions for common retry patterns
def retry_network_errors(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    """Retry on common network errors"""
    return retry_on_exception(
        max_retries=max_retries,
        base_delay=base_delay,
        retry_exceptions=(ConnectionError, TimeoutError, OSError)
    )


def retry_database_errors(max_retries: int = 3, base_delay: float = 2.0) -> Callable:
    """Retry on database errors"""
    try:
        from sqlalchemy.exc import OperationalError, DBAPIError
        retry_exceptions = (OperationalError, DBAPIError)
    except ImportError:
        retry_exceptions = (Exception,)
    
    return retry_on_exception(
        max_retries=max_retries,
        base_delay=base_delay,
        retry_exceptions=retry_exceptions
    )


def retry_api_errors(max_retries: int = 5, base_delay: float = 0.5) -> Callable:
    """Retry on API rate limiting and server errors"""
    return retry_on_exception(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=30.0,
        retry_exceptions=(ConnectionError, TimeoutError)
    )
