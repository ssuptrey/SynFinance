"""
Circuit Breaker Pattern

Prevents cascading failures by stopping calls to failing services.

Week 7 Day 7: Final Integration
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from dataclasses import dataclass

try:
    from src.observability import get_logger, LogCategory
    logger = get_logger(__name__)
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    OBSERVABILITY_AVAILABLE = False
    LogCategory = None


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Too many failures, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Successful calls to close from half-open
    timeout: float = 60.0  # Seconds to wait before trying half-open
    expected_exception: type = Exception  # Exception type to catch


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them time to recover.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests fail fast without calling service
    - HALF_OPEN: Testing recovery, limited requests pass through
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        @breaker
        def risky_operation():
            # Your code here
            pass
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Successful calls needed to close from half-open
            timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to catch
            name: Optional name for logging
        """
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )
        self.name = name or "CircuitBreaker"
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Circuit breaker '{self.name}' initialized",
                category=LogCategory.SYSTEM,
                context={
                    'failure_threshold': failure_threshold,
                    'success_threshold': success_threshold,
                    'timeout': timeout
                }
            )
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current state"""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if (self._state == CircuitBreakerState.OPEN and 
                self._last_failure_time is not None and
                time.time() - self._last_failure_time >= self.config.timeout):
                self._transition_to_half_open()
            
            return self._state
    
    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self._state = CircuitBreakerState.HALF_OPEN
        self._success_count = 0
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Circuit breaker '{self.name}' -> HALF_OPEN",
                category=LogCategory.SYSTEM,
                context={'previous_failures': self._failure_count}
            )
        else:
            logger.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                
                if self._success_count >= self.config.success_threshold:
                    # Recovered! Close the circuit
                    self._state = CircuitBreakerState.CLOSED
                    self._success_count = 0
                    
                    if OBSERVABILITY_AVAILABLE:
                        logger.info(
                            f"Circuit breaker '{self.name}' -> CLOSED (recovered)",
                            category=LogCategory.SYSTEM
                        )
                    else:
                        logger.info(f"Circuit breaker '{self.name}' -> CLOSED")
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failed during recovery, go back to OPEN
                self._state = CircuitBreakerState.OPEN
                self._success_count = 0
                
                if OBSERVABILITY_AVAILABLE:
                    logger.warning(
                        f"Circuit breaker '{self.name}' -> OPEN (recovery failed)",
                        category=LogCategory.SYSTEM,
                        context={'exception': str(exception)}
                    )
                else:
                    logger.warning(f"Circuit breaker '{self.name}' -> OPEN: {exception}")
            
            elif (self._state == CircuitBreakerState.CLOSED and 
                  self._failure_count >= self.config.failure_threshold):
                # Too many failures, open the circuit
                self._state = CircuitBreakerState.OPEN
                
                if OBSERVABILITY_AVAILABLE:
                    logger.error(
                        f"Circuit breaker '{self.name}' -> OPEN (threshold reached)",
                        category=LogCategory.SYSTEM,
                        context={
                            'failure_count': self._failure_count,
                            'threshold': self.config.failure_threshold
                        }
                    )
                else:
                    logger.error(f"Circuit breaker '{self.name}' -> OPEN after {self._failure_count} failures")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        current_state = self.state
        
        if current_state == CircuitBreakerState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Will retry after {self.config.timeout}s"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure(e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator usage
        
        @breaker
        def my_function():
            pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            
            if OBSERVABILITY_AVAILABLE:
                logger.info(
                    f"Circuit breaker '{self.name}' reset",
                    category=LogCategory.SYSTEM
                )
            else:
                logger.info(f"Circuit breaker '{self.name}' reset")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout
                }
            }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """
    Get or create a named circuit breaker
    
    Args:
        name: Unique name for circuit breaker
        failure_threshold: Number of failures before opening
        success_threshold: Successful calls to close from half-open
        timeout: Seconds before trying half-open
        expected_exception: Exception type to catch
        
    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
                name=name
            )
        
        return _circuit_breakers[name]


def reset_all_circuit_breakers():
    """Reset all registered circuit breakers"""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()


def get_all_circuit_breaker_stats() -> list[dict]:
    """Get statistics for all circuit breakers"""
    with _registry_lock:
        return [breaker.get_stats() for breaker in _circuit_breakers.values()]
