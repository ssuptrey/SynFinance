"""
Middleware for request tracking and observability.

Provides:
- Request ID generation and propagation
- Request logging with context
- Trace context extraction
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable
import time

from .logging_config import (
    set_request_context,
    clear_request_context,
    generate_request_id,
    get_logger
)

logger = get_logger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track requests with unique IDs and log request/response.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add tracking."""
        # Generate or extract request ID
        request_id = request.headers.get('X-Request-ID', generate_request_id())
        
        # Extract trace context from headers (W3C Trace Context)
        trace_id = request.headers.get('traceparent', '').split('-')[1] if 'traceparent' in request.headers else None
        
        # Extract user/tenant context (if available)
        user_id = request.headers.get('X-User-ID')
        tenant_id = request.headers.get('X-Tenant-ID')
        
        # Set context for logging
        set_request_context(
            request_id=request_id,
            trace_id=trace_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'http_method': request.method,
                'http_url': str(request.url),
                'http_path': request.url.path,
                'http_query': str(request.url.query),
                'client_ip': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent'),
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Add request ID to response headers
            response.headers['X-Request-ID'] = request_id
            
            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    'http_method': request.method,
                    'http_path': request.url.path,
                    'http_status_code': response.status_code,
                    'duration_seconds': round(duration, 3),
                    'duration_ms': round(duration * 1000, 2)
                }
            )
            
            return response
        
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    'http_method': request.method,
                    'http_path': request.url.path,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'duration_seconds': round(duration, 3)
                },
                exc_info=True
            )
            
            raise
        
        finally:
            # Clear context
            clear_request_context()
