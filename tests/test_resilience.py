"""
Tests for Resilience Framework

Week 7 Day 7: Final Integration
"""

import time
import pytest
import threading
from unittest.mock import Mock, patch

from src.resilience import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerError,
    RetryHandler,
    retry_on_exception,
    RateLimiter,
    RateLimitError,
    HealthChecker,
    HealthStatus,
    get_circuit_breaker,
    get_rate_limiter,
    get_health_checker
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state"""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3, name="test")
        
        def failing_function():
            raise ValueError("Test error")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_function)
        
        assert breaker.state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_fails_fast_when_open(self):
        """Test circuit breaker fails fast when OPEN"""
        breaker = CircuitBreaker(failure_threshold=2, name="test")
        
        def failing_function():
            raise ValueError("Test error")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_function)
        
        # Should fail fast now
        with pytest.raises(CircuitBreakerError):
            breaker.call(failing_function)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit transitions to HALF_OPEN after timeout"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=0.1,  # Very short timeout
            name="test"
        )
        
        def failing_function():
            raise ValueError("Test error")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_function)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should transition to HALF_OPEN
        assert breaker.state == CircuitBreakerState.HALF_OPEN
    
    def test_circuit_breaker_recovery(self):
        """Test circuit closes after successful recovery"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1,
            name="test"
        )
        
        call_count = [0]
        
        def sometimes_failing():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Failing")
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_failing)
        
        # Wait for HALF_OPEN
        time.sleep(0.2)
        
        # Successful calls to close circuit
        result1 = breaker.call(sometimes_failing)
        result2 = breaker.call(sometimes_failing)
        
        assert result1 == "success"
        assert result2 == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator"""
        breaker = CircuitBreaker(failure_threshold=2, name="test")
        
        @breaker
        def my_function():
            return "success"
        
        result = my_function()
        assert result == "success"
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset"""
        breaker = CircuitBreaker(failure_threshold=2, name="test")
        
        def failing_function():
            raise ValueError("Test error")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_function)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Reset
        breaker.reset()
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_get_circuit_breaker_registry(self):
        """Test circuit breaker registry"""
        breaker1 = get_circuit_breaker("test1")
        breaker2 = get_circuit_breaker("test1")
        breaker3 = get_circuit_breaker("test2")
        
        assert breaker1 is breaker2  # Same instance
        assert breaker1 is not breaker3  # Different instances


class TestRetryHandler:
    """Test retry handler functionality"""
    
    def test_retry_handler_success_no_retry(self):
        """Test successful call doesn't retry"""
        retry = RetryHandler(max_retries=3, base_delay=0.1)
        
        call_count = [0]
        
        def successful_function():
            call_count[0] += 1
            return "success"
        
        result = retry.call(successful_function)
        
        assert result == "success"
        assert call_count[0] == 1  # Called only once
    
    def test_retry_handler_retries_on_failure(self):
        """Test retry on failures"""
        retry = RetryHandler(max_retries=3, base_delay=0.01)
        
        call_count = [0]
        
        def failing_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retry.call(failing_function)
        
        assert result == "success"
        assert call_count[0] == 3  # Retried 2 times
    
    def test_retry_handler_max_retries(self):
        """Test max retries exceeded"""
        retry = RetryHandler(max_retries=2, base_delay=0.01)
        
        def always_failing():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            retry.call(always_failing)
    
    def test_retry_handler_exponential_backoff(self):
        """Test exponential backoff timing"""
        retry = RetryHandler(
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )
        
        call_times = []
        
        def failing_function():
            call_times.append(time.time())
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            retry.call(failing_function)
        
        # Check delays increased exponentially
        assert len(call_times) == 4  # 1 initial + 3 retries
        
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]
        
        # Second delay should be ~2x first delay (with tolerance for timing)
        assert 0.08 < delay1 < 0.25  # ~0.1s base delay
        assert 0.15 < delay2 < 0.35  # ~0.2s (2^1 * 0.1)
        assert 0.35 < delay3 < 0.55  # ~0.4s (2^2 * 0.1)
    
    def test_retry_handler_decorator(self):
        """Test retry handler as decorator"""
        retry = RetryHandler(max_retries=2, base_delay=0.01)
        
        call_count = [0]
        
        @retry
        def my_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Fail once")
            return "success"
        
        result = my_function()
        assert result == "success"
        assert call_count[0] == 2
    
    def test_retry_on_exception_decorator(self):
        """Test retry_on_exception decorator"""
        call_count = [0]
        
        @retry_on_exception(max_retries=2, base_delay=0.01)
        def my_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Fail once")
            return "success"
        
        result = my_function()
        assert result == "success"
        assert call_count[0] == 2
    
    def test_retry_specific_exceptions(self):
        """Test retry only on specific exceptions"""
        retry = RetryHandler(
            max_retries=3,
            base_delay=0.01,
            retry_exceptions=(ValueError,)
        )
        
        # Should retry ValueError
        call_count = [0]
        
        def value_error_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Retry this")
            return "success"
        
        result = retry.call(value_error_func)
        assert result == "success"
        
        # Should NOT retry TypeError
        def type_error_func():
            raise TypeError("Don't retry this")
        
        with pytest.raises(TypeError):
            retry.call(type_error_func)


class TestRateLimiter:
    """Test rate limiter functionality"""
    
    def test_rate_limiter_allows_within_limit(self):
        """Test requests within limit are allowed"""
        limiter = RateLimiter(rate=10.0, capacity=5)
        
        # Should allow 5 requests immediately
        for _ in range(5):
            assert limiter.acquire(blocking=False)
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test requests over limit are blocked"""
        limiter = RateLimiter(rate=10.0, capacity=5)
        
        # Consume all tokens
        for _ in range(5):
            limiter.acquire(blocking=False)
        
        # Next request should fail
        with pytest.raises(RateLimitError):
            limiter.acquire(blocking=False)
    
    def test_rate_limiter_refills_tokens(self):
        """Test tokens refill over time"""
        limiter = RateLimiter(rate=10.0, capacity=5)
        
        # Consume all tokens
        for _ in range(5):
            limiter.acquire(blocking=False)
        
        # Wait for refill (0.1s should add 1 token)
        time.sleep(0.15)
        
        # Should succeed now
        assert limiter.acquire(blocking=False)
    
    def test_rate_limiter_blocking_wait(self):
        """Test blocking wait for tokens"""
        limiter = RateLimiter(rate=10.0, capacity=2)
        
        # Consume all tokens
        limiter.acquire(tokens=2, blocking=False)
        
        # Should wait and succeed
        start = time.time()
        limiter.acquire(tokens=1, blocking=True)
        duration = time.time() - start
        
        # Should have waited ~0.1s for 1 token at 10/s
        assert 0.05 < duration < 0.25
    
    def test_rate_limiter_decorator(self):
        """Test rate limiter as decorator"""
        limiter = RateLimiter(rate=10.0, capacity=2)
        
        call_count = [0]
        
        @limiter
        def my_function():
            call_count[0] += 1
            return f"success {call_count[0]}"
        
        # First two calls should succeed
        result1 = my_function()
        assert "success" in result1
        
        result2 = my_function()
        assert "success" in result2
        
        # Third should wait (blocking by default) or fail with explicit blocking=False
        # Test non-blocking behavior
        def non_blocking_call():
            return limiter.call(lambda: "fail", blocking=False)
        
        with pytest.raises(RateLimitError):
            non_blocking_call()
    
    def test_rate_limiter_per_user(self):
        """Test per-user rate limiting"""
        limiter = RateLimiter(rate=10.0, capacity=2, per_user=True)
        
        # User 1 consumes their tokens
        limiter.acquire(user_id="user1", tokens=2, blocking=False)
        
        # User 2 should still have tokens
        assert limiter.acquire(user_id="user2", tokens=1, blocking=False)
        
        # User 1 should be rate limited
        with pytest.raises(RateLimitError):
            limiter.acquire(user_id="user1", blocking=False)
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset"""
        limiter = RateLimiter(rate=10.0, capacity=2)
        
        # Consume all tokens
        limiter.acquire(tokens=2, blocking=False)
        
        # Reset
        limiter.reset()
        
        # Should have tokens again
        assert limiter.acquire(blocking=False)
    
    def test_get_rate_limiter_registry(self):
        """Test rate limiter registry"""
        limiter1 = get_rate_limiter("test1", rate=10.0)
        limiter2 = get_rate_limiter("test1", rate=10.0)
        limiter3 = get_rate_limiter("test2", rate=20.0)
        
        assert limiter1 is limiter2  # Same instance
        assert limiter1 is not limiter3  # Different instances


class TestHealthChecker:
    """Test health checker functionality"""
    
    def test_health_checker_initial_state(self):
        """Test health checker initial state"""
        checker = HealthChecker(name="test")
        health = checker.check_health()
        
        assert health['status'] == HealthStatus.UNKNOWN.value
        assert health['name'] == "test"
        assert len(health['components']) == 0
    
    def test_health_checker_register_check(self):
        """Test registering health checks"""
        checker = HealthChecker(name="test")
        
        @checker.register_check("test_component")
        def check_component():
            return True
        
        health = checker.check_health()
        
        assert health['status'] == HealthStatus.HEALTHY.value
        assert len(health['components']) == 1
        assert health['components'][0]['name'] == "test_component"
        assert health['components'][0]['status'] == HealthStatus.HEALTHY.value
    
    def test_health_checker_unhealthy_component(self):
        """Test unhealthy component detection"""
        checker = HealthChecker(name="test")
        
        @checker.register_check("failing_component", required=True)
        def check_failing():
            return False
        
        health = checker.check_health()
        
        assert health['status'] == HealthStatus.UNHEALTHY.value
        assert health['components'][0]['status'] == HealthStatus.UNHEALTHY.value
    
    def test_health_checker_degraded_state(self):
        """Test degraded state with optional failures"""
        checker = HealthChecker(name="test")
        
        @checker.register_check("required_component", required=True)
        def check_required():
            return True
        
        @checker.register_check("optional_component", required=False)
        def check_optional():
            return False
        
        health = checker.check_health()
        
        # Should be degraded (required ok, optional failed)
        assert health['status'] == HealthStatus.DEGRADED.value
    
    def test_health_checker_exception_handling(self):
        """Test health check exception handling"""
        checker = HealthChecker(name="test")
        
        @checker.register_check("error_component")
        def check_error():
            raise RuntimeError("Check failed")
        
        health = checker.check_health()
        
        assert health['components'][0]['status'] == HealthStatus.UNHEALTHY.value
        assert 'error' in health['components'][0]['details']
    
    def test_health_checker_probes(self):
        """Test readiness, liveness, and startup probes"""
        checker = HealthChecker(name="test")
        
        @checker.register_check("component", required=True)
        def check_component():
            return True
        
        # Liveness should be true (not unhealthy)
        assert checker.liveness_probe()
        
        # Readiness should be true (healthy or degraded)
        assert checker.readiness_probe()
        
        # Startup should be false initially
        assert not checker.startup_probe()
        
        # Mark startup complete
        checker.mark_startup_complete()
        assert checker.startup_probe()
    
    def test_health_checker_caching(self):
        """Test health check result caching"""
        checker = HealthChecker(name="test")
        
        call_count = [0]
        
        @checker.register_check("cached_component")
        def check_cached():
            call_count[0] += 1
            return True
        
        # First call
        checker.check_health()
        assert call_count[0] == 1
        
        # Second call should use cache
        checker.check_health(use_cache=True)
        assert call_count[0] == 1  # Not called again
        
        # Disable cache
        checker.check_health(use_cache=False)
        assert call_count[0] == 2  # Called again
    
    def test_get_health_checker_singleton(self):
        """Test health checker singleton"""
        checker1 = get_health_checker("SynFinance")
        checker2 = get_health_checker("SynFinance")
        
        assert checker1 is checker2  # Same instance


class TestResilienceIntegration:
    """Test integration between resilience components"""
    
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker with retry handler"""
        breaker = CircuitBreaker(failure_threshold=5, name="test")
        retry = RetryHandler(max_retries=2, base_delay=0.01)
        
        call_count = [0]
        
        @breaker
        @retry
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count[0] == 2
    
    def test_rate_limiter_with_health_check(self):
        """Test rate limiter with health checking"""
        limiter = RateLimiter(rate=10.0, capacity=5)
        checker = HealthChecker(name="test")
        
        @checker.register_check("rate_limiter")
        def check_rate_limiter():
            stats = limiter.get_stats()
            return stats['utilization'] < 0.9  # Not too utilized
        
        # Should be healthy initially
        health = checker.check_health()
        assert health['status'] == HealthStatus.HEALTHY.value
        
        # Consume most tokens
        for _ in range(4):
            limiter.acquire(blocking=False)
        
        # Clear cache and check again
        checker.clear_cache()
        health = checker.check_health(use_cache=False)
        
        # Should still be healthy (80% utilization)
        assert health['status'] == HealthStatus.HEALTHY.value
