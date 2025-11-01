"""
API Version Negotiation

Handles detection and negotiation of API versions from various sources:
- URL path segments (/api/v2/...)
- Accept headers (application/vnd.synfinance.v2+json)
- Custom headers (X-API-Version: 2)
- Query parameters (?api_version=2)

Priority order: URL > Accept Header > Custom Header > Query Param > Default
"""

import re
from typing import Optional
from fastapi import Request, HTTPException
from src.api.versioning.registry import get_latest_version, get_version


class VersionNegotiator:
    """
    Handles API version negotiation from HTTP requests.
    
    Detects version from multiple sources with configurable priority.
    """
    
    def __init__(
        self,
        default_version: Optional[str] = None,
        allow_query_param: bool = True,
        require_version: bool = False,
    ):
        """
        Initialize version negotiator.
        
        Args:
            default_version: Default version if none specified (uses latest if None)
            allow_query_param: Allow version in query parameters
            require_version: Raise error if no version specified
        """
        self.default_version = default_version
        self.allow_query_param = allow_query_param
        self.require_version = require_version
    
    def detect(self, request: Request) -> str:
        """
        Detect API version from request.
        
        Priority order:
        1. URL path (/api/v2/...)
        2. Accept header (application/vnd.synfinance.v2+json)
        3. Custom header (X-API-Version: 2)
        4. Query parameter (?api_version=2) if allowed
        5. Default version or latest
        
        Args:
            request: FastAPI request object
            
        Returns:
            Version identifier (e.g., "v2")
            
        Raises:
            HTTPException: If version required but not found, or invalid version
        """
        # Try URL path first
        version = detect_version_from_url(request.url.path)
        if version:
            return self._validate_version(version)
        
        # Try Accept header
        accept = request.headers.get("Accept", "")
        version = detect_version_from_accept(accept)
        if version:
            return self._validate_version(version)
        
        # Try custom header
        version = request.headers.get("X-API-Version")
        if version:
            # Normalize to vX format
            if not version.startswith("v"):
                version = f"v{version}"
            return self._validate_version(version)
        
        # Try query parameter if allowed
        if self.allow_query_param:
            version = request.query_params.get("api_version")
            if version:
                if not version.startswith("v"):
                    version = f"v{version}"
                return self._validate_version(version)
        
        # Use default or latest
        if self.require_version:
            raise HTTPException(
                status_code=400,
                detail="API version must be specified in URL, headers, or query parameters"
            )
        
        if self.default_version:
            return self.default_version
        
        # Get latest version
        latest = get_latest_version()
        if not latest:
            raise HTTPException(
                status_code=500,
                detail="No API versions registered"
            )
        
        return latest.version
    
    def _validate_version(self, version: str) -> str:
        """
        Validate that version exists.
        
        Args:
            version: Version identifier
            
        Returns:
            Validated version identifier
            
        Raises:
            HTTPException: If version not found
        """
        api_version = get_version(version)
        if not api_version:
            raise HTTPException(
                status_code=400,
                detail=f"API version '{version}' not found"
            )
        
        # Check if sunset
        if api_version.is_sunset:
            raise HTTPException(
                status_code=410,  # Gone
                detail=f"API version '{version}' is no longer supported. Please migrate to a newer version."
            )
        
        return version


def detect_version_from_url(path: str) -> Optional[str]:
    """
    Extract version from URL path.
    
    Matches patterns like:
    - /api/v1/transactions
    - /api/v2/fraud-detection
    - /v3/analytics
    
    Args:
        path: URL path
        
    Returns:
        Version identifier or None if not found
    """
    # Match /api/vX/ or /vX/ patterns
    match = re.search(r'/(?:api/)?(v\d+)(?:/|$)', path)
    if match:
        return match.group(1)
    return None


def detect_version_from_accept(accept_header: str) -> Optional[str]:
    """
    Extract version from Accept header.
    
    Matches patterns like:
    - application/vnd.synfinance.v2+json
    - application/vnd.synfinance.v1+xml
    
    Args:
        accept_header: Accept header value
        
    Returns:
        Version identifier or None if not found
    """
    # Match vendor-specific media type with version
    match = re.search(r'application/vnd\.synfinance\.(v\d+)\+', accept_header)
    if match:
        return match.group(1)
    return None


def detect_version_from_header(request: Request) -> Optional[str]:
    """
    Extract version from custom header.
    
    Checks X-API-Version header.
    
    Args:
        request: FastAPI request
        
    Returns:
        Version identifier or None if not found
    """
    version = request.headers.get("X-API-Version")
    if version:
        if not version.startswith("v"):
            version = f"v{version}"
        return version
    return None


def detect_version_from_request(
    request: Request,
    default_version: Optional[str] = None,
) -> str:
    """
    Detect version from request using default negotiator.
    
    Convenience function for quick version detection.
    
    Args:
        request: FastAPI request
        default_version: Default version if none found
        
    Returns:
        Version identifier
    """
    negotiator = VersionNegotiator(default_version=default_version)
    return negotiator.detect(request)
