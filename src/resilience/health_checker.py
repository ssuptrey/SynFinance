"""
Health Checker System

Monitors system health with readiness, liveness, and startup probes.

Week 7 Day 7: Final Integration
"""

import time
import threading
from enum import Enum
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field

try:
    from src.observability import get_logger, LogCategory
    logger = get_logger(__name__)
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    OBSERVABILITY_AVAILABLE = False
    LogCategory = None


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a component"""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    check_duration: float = 0.0


class HealthChecker:
    """
    System Health Checker
    
    Implements Kubernetes-style health probes:
    - Readiness: Can the service handle requests?
    - Liveness: Is the service alive and running?
    - Startup: Has the service finished starting up?
    
    Components can register health check functions that are called
    to determine overall system health.
    
    Usage:
        checker = HealthChecker()
        
        @checker.register_check("database")
        def check_database():
            # Return True if healthy, False if unhealthy
            return db.is_connected()
        
        health = checker.check_health()
    """
    
    def __init__(self, name: str = "SynFinance"):
        """
        Initialize health checker
        
        Args:
            name: System name for health reporting
        """
        self.name = name
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._cached_results: Dict[str, ComponentHealth] = {}
        self._cache_duration = 5.0  # Cache health checks for 5 seconds
        self._lock = threading.RLock()
        self._startup_complete = False
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Health checker initialized for '{name}'",
                category=LogCategory.SYSTEM
            )
    
    def register_check(
        self,
        name: str,
        check_func: Optional[Callable[[], bool]] = None,
        required: bool = True
    ) -> Callable:
        """
        Register a health check function
        
        Args:
            name: Component name
            check_func: Function that returns True if healthy
            required: If True, system is unhealthy if this check fails
            
        Returns:
            Decorator function if check_func is None
        
        Usage:
            @checker.register_check("database")
            def check_db():
                return True
        """
        def decorator(func: Callable[[], bool]) -> Callable:
            with self._lock:
                self._checks[name] = {
                    'func': func,
                    'required': required
                }
            
            if OBSERVABILITY_AVAILABLE:
                logger.info(
                    f"Registered health check: {name}",
                    category=LogCategory.SYSTEM,
                    context={'required': required}
                )
            
            return func
        
        if check_func is None:
            return decorator
        else:
            return decorator(check_func)
    
    def _execute_check(self, name: str, check_info: dict) -> ComponentHealth:
        """Execute a single health check"""
        start_time = time.time()
        
        try:
            func = check_info['func']
            result = func()
            
            if result:
                status = HealthStatus.HEALTHY
                message = f"{name} is healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"{name} is unhealthy"
            
            details = {}
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"{name} check failed: {str(e)}"
            details = {'error': str(e), 'type': type(e).__name__}
            
            if OBSERVABILITY_AVAILABLE:
                logger.error(
                    f"Health check failed: {name}",
                    category=LogCategory.SYSTEM,
                    context={'error': str(e)}
                )
        
        duration = time.time() - start_time
        
        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            details=details,
            last_check=start_time,
            check_duration=duration
        )
    
    def check_component(
        self,
        name: str,
        use_cache: bool = True
    ) -> ComponentHealth:
        """
        Check health of a specific component
        
        Args:
            name: Component name
            use_cache: Use cached result if available
            
        Returns:
            ComponentHealth instance
        """
        with self._lock:
            # Check cache
            if use_cache and name in self._cached_results:
                cached = self._cached_results[name]
                if time.time() - cached.last_check < self._cache_duration:
                    return cached
            
            # Execute check
            if name not in self._checks:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"No health check registered for {name}"
                )
            
            result = self._execute_check(name, self._checks[name])
            self._cached_results[name] = result
            
            return result
    
    def check_health(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Check overall system health
        
        Args:
            use_cache: Use cached component results
            
        Returns:
            Dictionary with health status and component details
        """
        with self._lock:
            components = []
            unhealthy_required = []
            
            # Check all components
            for name, check_info in self._checks.items():
                component = self.check_component(name, use_cache=use_cache)
                components.append(component)
                
                if (component.status == HealthStatus.UNHEALTHY and 
                    check_info['required']):
                    unhealthy_required.append(name)
            
            # Determine overall status
            if unhealthy_required:
                status = HealthStatus.UNHEALTHY
                message = f"System unhealthy: {', '.join(unhealthy_required)} failed"
            elif any(c.status == HealthStatus.UNHEALTHY for c in components):
                status = HealthStatus.DEGRADED
                message = "System degraded: some optional checks failed"
            elif components:
                status = HealthStatus.HEALTHY
                message = "All systems operational"
            else:
                status = HealthStatus.UNKNOWN
                message = "No health checks configured"
            
            return {
                'name': self.name,
                'status': status.value,
                'message': message,
                'timestamp': time.time(),
                'startup_complete': self._startup_complete,
                'components': [
                    {
                        'name': c.name,
                        'status': c.status.value,
                        'message': c.message,
                        'details': c.details,
                        'last_check': c.last_check,
                        'check_duration': c.check_duration
                    }
                    for c in components
                ]
            }
    
    def readiness_probe(self) -> bool:
        """
        Readiness probe - Can service handle traffic?
        
        Returns:
            True if ready to serve traffic
        """
        health = self.check_health()
        return health['status'] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]
    
    def liveness_probe(self) -> bool:
        """
        Liveness probe - Is service alive?
        
        Returns:
            True if service is alive
        """
        health = self.check_health()
        return health['status'] != HealthStatus.UNHEALTHY.value
    
    def startup_probe(self) -> bool:
        """
        Startup probe - Has service finished starting?
        
        Returns:
            True if startup is complete
        """
        return self._startup_complete
    
    def mark_startup_complete(self):
        """Mark startup as complete"""
        self._startup_complete = True
        
        if OBSERVABILITY_AVAILABLE:
            logger.info(
                f"Startup complete for '{self.name}'",
                category=LogCategory.SYSTEM
            )
    
    def clear_cache(self):
        """Clear cached health check results"""
        with self._lock:
            self._cached_results.clear()


# Global health checker instance
_health_checker: Optional[HealthChecker] = None
_checker_lock = threading.Lock()


def get_health_checker(name: str = "SynFinance") -> HealthChecker:
    """
    Get global health checker instance
    
    Args:
        name: System name
        
    Returns:
        HealthChecker instance
    """
    global _health_checker
    
    with _checker_lock:
        if _health_checker is None:
            _health_checker = HealthChecker(name=name)
        
        return _health_checker


def register_database_health_check():
    """Register database health check"""
    checker = get_health_checker()
    
    @checker.register_check("database", required=True)
    def check_database() -> bool:
        try:
            from src.database import get_db_manager
            manager = get_db_manager()
            return manager.health_check()
        except Exception:
            return False
    
    return check_database


def register_monitoring_health_check():
    """Register monitoring health check"""
    checker = get_health_checker()
    
    @checker.register_check("monitoring", required=False)
    def check_monitoring() -> bool:
        try:
            from src.monitoring import PrometheusExporter
            # Check if metrics are being collected
            return True  # Simplified check
        except Exception:
            return False
    
    return check_monitoring


def register_system_health_checks():
    """Register system resource health checks"""
    checker = get_health_checker()
    
    @checker.register_check("disk_space", required=True)
    def check_disk_space() -> bool:
        try:
            import psutil
            disk = psutil.disk_usage('.')
            return disk.percent < 90  # Less than 90% full
        except Exception:
            return False
    
    @checker.register_check("memory", required=False)
    def check_memory() -> bool:
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% used
        except Exception:
            return False
    
    @checker.register_check("cpu", required=False)
    def check_cpu() -> bool:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            return cpu < 90  # Less than 90% usage
        except Exception:
            return False
    
    return check_disk_space, check_memory, check_cpu


def register_all_health_checks():
    """Register all default health checks"""
    register_database_health_check()
    register_monitoring_health_check()
    register_system_health_checks()
    
    if OBSERVABILITY_AVAILABLE:
        logger.info(
            "All health checks registered",
            category=LogCategory.SYSTEM
        )
