"""
API Deprecation Management

Provides tools for managing API deprecation:
- Deprecation decorators for endpoints and fields
- Sunset header generation
- Deprecation warnings in responses
- Timeline tracking
"""

import warnings
from datetime import date
from typing import Optional, Callable, Any
from functools import wraps
from fastapi import Response, Request
from src.api.versioning.registry import get_version


class DeprecationWarning(UserWarning):
    """Custom warning for API deprecation"""
    pass


def deprecated(
    message: str,
    sunset_date: Optional[date] = None,
    replacement: Optional[str] = None,
    version: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark API endpoints as deprecated.
    
    Adds deprecation warnings to response headers and logs warnings.
    
    Args:
        message: Deprecation message
        sunset_date: When this endpoint will be removed
        replacement: Suggested replacement endpoint
        version: Version where this was deprecated
        
    Example:
        @router.get("/old-endpoint")
        @deprecated(
            message="Use /new-endpoint instead",
            sunset_date=date(2026, 6, 1),
            replacement="/api/v2/new-endpoint"
        )
        async def old_endpoint():
            return {"data": "..."}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Get response from kwargs (FastAPI dependency injection)
            response = None
            for arg in args:
                if isinstance(arg, Response):
                    response = arg
                    break
            
            if response:
                add_deprecation_headers(
                    response=response,
                    message=message,
                    sunset_date=sunset_date,
                    replacement=replacement,
                )
            
            # Log warning
            warnings.warn(
                f"Endpoint {func.__name__} is deprecated: {message}",
                DeprecationWarning,
                stacklevel=2
            )
            
            return result
        
        # Mark function as deprecated
        wrapper.__deprecated__ = True  # type: ignore
        wrapper.__deprecation_info__ = {  # type: ignore
            "message": message,
            "sunset_date": sunset_date,
            "replacement": replacement,
            "version": version,
        }
        
        return wrapper
    
    return decorator


def deprecate_field(
    field_name: str,
    message: str,
    replacement: Optional[str] = None,
) -> dict:
    """
    Mark a response field as deprecated.
    
    Returns metadata to include in API documentation.
    
    Args:
        field_name: Name of deprecated field
        message: Deprecation message
        replacement: Suggested replacement field
        
    Returns:
        Deprecation metadata dictionary
    """
    return {
        "deprecated": True,
        "deprecation_message": message,
        "replacement": replacement,
        "field_name": field_name,
    }


def add_deprecation_headers(
    response: Response,
    message: str,
    sunset_date: Optional[date] = None,
    replacement: Optional[str] = None,
) -> None:
    """
    Add deprecation headers to HTTP response.
    
    Headers added:
    - Deprecation: true (RFC 8594)
    - Sunset: <date> (RFC 8594)
    - X-API-Deprecation-Message: <message>
    - X-API-Replacement: <endpoint> (if provided)
    
    Args:
        response: FastAPI response object
        message: Deprecation message
        sunset_date: When endpoint will be removed
        replacement: Suggested replacement endpoint
    """
    # Standard deprecation header (RFC 8594)
    response.headers["Deprecation"] = "true"
    
    # Sunset header (RFC 8594)
    if sunset_date:
        response.headers["Sunset"] = sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    # Custom headers for additional info
    response.headers["X-API-Deprecation-Message"] = message
    
    if replacement:
        response.headers["X-API-Replacement"] = replacement


def add_sunset_headers(
    response: Response,
    version: str,
) -> None:
    """
    Add sunset headers based on version information.
    
    Args:
        response: FastAPI response object
        version: API version identifier
    """
    api_version = get_version(version)
    if not api_version:
        return
    
    if api_version.is_deprecated:
        response.headers["Deprecation"] = "true"
        
        if api_version.sunset_date:
            response.headers["Sunset"] = api_version.sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        if api_version.deprecation_date:
            response.headers["X-API-Deprecation-Date"] = api_version.deprecation_date.isoformat()
        
        # Days until sunset
        if api_version.days_until_sunset is not None:
            response.headers["X-API-Days-Until-Sunset"] = str(api_version.days_until_sunset)


def check_deprecation_status(version: str) -> dict:
    """
    Get deprecation status for a version.
    
    Args:
        version: Version identifier
        
    Returns:
        Dictionary with deprecation information
    """
    api_version = get_version(version)
    if not api_version:
        return {"deprecated": False}
    
    return {
        "deprecated": api_version.is_deprecated,
        "status": api_version.status.value,
        "deprecation_date": api_version.deprecation_date.isoformat() if api_version.deprecation_date else None,
        "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None,
        "days_until_sunset": api_version.days_until_sunset,
        "is_sunset": api_version.is_sunset,
    }


class DeprecationMiddleware:
    """
    Middleware to automatically add deprecation headers to responses.
    
    Detects API version from request and adds appropriate headers
    if the version is deprecated.
    """
    
    def __init__(self, app: Any):
        self.app = app
    
    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Process request and add deprecation headers"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create request to detect version
        from fastapi import Request
        request = Request(scope, receive)
        
        # Detect version from request
        from src.api.versioning.negotiation import detect_version_from_request
        try:
            version = detect_version_from_request(request)
        except Exception:
            # If version detection fails, just continue
            await self.app(scope, receive, send)
            return
        
        # Check if deprecated
        api_version = get_version(version)
        if not api_version or not api_version.is_deprecated:
            await self.app(scope, receive, send)
            return
        
        # Wrap send to add headers
        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                
                # Add deprecation headers
                headers.append((b"deprecation", b"true"))
                
                if api_version.sunset_date:
                    sunset = api_version.sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
                    headers.append((b"sunset", sunset.encode()))
                
                if api_version.days_until_sunset is not None:
                    headers.append((b"x-api-days-until-sunset", str(api_version.days_until_sunset).encode()))
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_with_headers)
