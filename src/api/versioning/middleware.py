"""
API Versioning Middleware

Middleware for automatic version detection and injection into request state.
"""

from typing import Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.api.versioning.negotiation import VersionNegotiator
from src.api.versioning.deprecation import add_sunset_headers


class VersionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to detect and inject API version into requests.
    
    Automatically detects version from URL, headers, or query parameters
    and makes it available via request.state.api_version.
    
    Also adds deprecation/sunset headers for deprecated versions.
    """
    
    def __init__(
        self,
        app: Any,
        default_version: Optional[str] = None,
        allow_query_param: bool = True,
        require_version: bool = False,
    ):
        """
        Initialize version middleware.
        
        Args:
            app: FastAPI application
            default_version: Default version if none specified
            allow_query_param: Allow version in query parameters
            require_version: Raise error if no version specified
        """
        super().__init__(app)
        self.negotiator = VersionNegotiator(
            default_version=default_version,
            allow_query_param=allow_query_param,
            require_version=require_version,
        )
    
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Process request and inject version.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with version headers
        """
        # Detect version
        version = self.negotiator.detect(request)
        
        # Inject into request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add version header to response
        response.headers["X-API-Version"] = version
        
        # Add sunset headers if deprecated
        add_sunset_headers(response, version)
        
        return response


def create_version_middleware(
    default_version: Optional[str] = None,
    allow_query_param: bool = True,
    require_version: bool = False,
) -> type[VersionMiddleware]:
    """
    Factory function to create version middleware with configuration.
    
    Args:
        default_version: Default version if none specified
        allow_query_param: Allow version in query parameters
        require_version: Raise error if no version specified
        
    Returns:
        Configured middleware class
    """
    class ConfiguredVersionMiddleware(VersionMiddleware):
        def __init__(self, app: Any):
            super().__init__(
                app,
                default_version=default_version,
                allow_query_param=allow_query_param,
                require_version=require_version,
            )
    
    return ConfiguredVersionMiddleware
