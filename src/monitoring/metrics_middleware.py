"""
Middleware for automatic metrics collection in FastAPI.

Provides:
- Automatic request/response metrics
- Error tracking
- Performance monitoring
- Custom business metrics integration
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .prometheus_exporter import get_metrics_exporter

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.
    
    Automatically tracks:
    - Request counts by method, endpoint, and status
    - Request latency distributions
    - Active request gauges
    - Request/response sizes
    - Error rates
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize metrics middleware.
        
        Args:
            app: ASGI application
        """
        super().__init__(app)
        self.exporter = get_metrics_exporter()
        logger.info("Initialized MetricsMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Normalize path (remove IDs, UUIDs, etc.)
        endpoint = self._normalize_path(path)
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        # Track active requests
        self.exporter.active_requests.labels(endpoint=endpoint).inc()
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            latency = time.time() - start_time
            status = response.status_code
            response_size = int(response.headers.get('content-length', 0))
            
            # Record metrics
            self.exporter.record_request(
                method=method,
                endpoint=endpoint,
                status=status,
                latency=latency,
                request_size=request_size,
                response_size=response_size
            )
            
            logger.debug(
                f"{method} {endpoint} - {status} - {latency*1000:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            # Record error
            error_type = type(e).__name__
            self.exporter.record_error(error_type, endpoint)
            
            logger.error(
                f"Error processing {method} {endpoint}: {error_type}",
                exc_info=True
            )
            
            # Re-raise
            raise
            
        finally:
            # Decrement active requests
            self.exporter.active_requests.labels(endpoint=endpoint).dec()
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path by removing variable components.
        
        Examples:
            /api/transactions/12345 -> /api/transactions/{id}
            /customers/abc-123/details -> /customers/{id}/details
        
        Args:
            path: Request path
            
        Returns:
            Normalized path
        """
        import re
        
        # Remove UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path,
            flags=re.IGNORECASE
        )
        
        # Remove pure numeric IDs
        path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
        
        # Remove alphanumeric IDs only if they contain a hyphen (e.g., abc-123)
        # This preserves static paths like /api/health
        path = re.sub(r'/[a-zA-Z0-9]+-[a-zA-Z0-9\-_]+(?=/|$)', '/{id}', path)
        
        return path
