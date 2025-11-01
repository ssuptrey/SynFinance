"""
Tenant Isolation Middleware

This module provides middleware for FastAPI that automatically extracts
tenant information from requests and sets the tenant context for the
duration of the request.

Supports multiple tenant identification methods:
- X-Tenant-ID header
- JWT token claims
- Subdomain-based tenants (optional)
"""

from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import jwt
import logging

from .context import TenantContext, TenantContextError
from .models import Tenant, TenantStatus, TenantUser

logger = logging.getLogger(__name__)


class TenantMiddlewareConfig:
    """Configuration for tenant middleware"""
    
    def __init__(
        self,
        header_name: str = "X-Tenant-ID",
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        jwt_tenant_claim: str = "tenant_id",
        jwt_user_claim: str = "user_id",
        require_tenant: bool = True,
        enable_subdomain: bool = False,
        subdomain_suffix: Optional[str] = None,
        exempt_paths: Optional[list[str]] = None,
    ):
        """
        Initialize tenant middleware configuration
        
        Args:
            header_name: HTTP header name for tenant ID
            jwt_secret: Secret key for JWT validation
            jwt_algorithm: Algorithm for JWT decoding
            jwt_tenant_claim: Claim name for tenant ID in JWT
            jwt_user_claim: Claim name for user ID in JWT
            require_tenant: Whether to require tenant for all requests
            enable_subdomain: Whether to extract tenant from subdomain
            subdomain_suffix: Domain suffix for subdomain extraction
            exempt_paths: List of paths that don't require tenant context
        """
        self.header_name = header_name
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_tenant_claim = jwt_tenant_claim
        self.jwt_user_claim = jwt_user_claim
        self.require_tenant = require_tenant
        self.enable_subdomain = enable_subdomain
        self.subdomain_suffix = subdomain_suffix
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]


class TenantMiddleware:
    """
    Middleware for tenant isolation in multi-tenant applications
    
    Automatically extracts tenant information from requests and sets
    the tenant context for the duration of each request.
    """
    
    def __init__(
        self,
        config: TenantMiddlewareConfig,
        tenant_loader: Optional[Callable[[str], Optional[Tenant]]] = None,
        user_loader: Optional[Callable[[str, str], Optional[TenantUser]]] = None,
    ):
        """
        Initialize tenant middleware
        
        Args:
            config: Middleware configuration
            tenant_loader: Function to load tenant by ID (optional)
            user_loader: Function to load user by tenant_id and user_id (optional)
        """
        self.config = config
        self.tenant_loader = tenant_loader
        self.user_loader = user_loader
        logger.info(
            f"TenantMiddleware initialized with header={config.header_name}, "
            f"require_tenant={config.require_tenant}"
        )
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and inject tenant context
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response
            
        Raises:
            HTTPException: If tenant is required but not found or invalid
        """
        # Check if path is exempt from tenant requirement
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        try:
            # Extract tenant and user information
            tenant_id = self._extract_tenant_id(request)
            user_id = self._extract_user_id(request)
            
            # Validate tenant if loader is provided
            if self.tenant_loader and tenant_id:
                tenant = self.tenant_loader(tenant_id)
                if not tenant:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tenant not found: {tenant_id}"
                    )
                if not tenant.is_active():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Tenant is not active: {tenant_id}"
                    )
            
            # Validate user if loader is provided
            if self.user_loader and tenant_id and user_id:
                user = self.user_loader(tenant_id, user_id)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"User not found: {user_id}"
                    )
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"User is not active: {user_id}"
                    )
            
            # Require tenant if configured
            if self.config.require_tenant and not tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tenant ID is required"
                )
            
            # Set tenant context
            if tenant_id:
                # Convert string to UUID if it looks like a UUID
                from uuid import UUID
                try:
                    tenant_uuid = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
                    TenantContext.set_current_tenant(tenant_uuid)
                except ValueError:
                    # Not a valid UUID, skip setting context (tenant_id is still in request.state)
                    logger.warning(f"Tenant ID is not a valid UUID: {tenant_id}")
                logger.debug(f"Set tenant context: {tenant_id}")
            
            # Set user context
            if user_id:
                # Convert string to UUID if it looks like a UUID
                from uuid import UUID
                try:
                    user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
                    TenantContext.set_current_user(user_uuid)
                except ValueError:
                    # Not a valid UUID, skip setting context (user_id is still in request.state)
                    logger.warning(f"User ID is not a valid UUID: {user_id}")
                logger.debug(f"Set user context: {user_id}")
            
            # Add tenant and user to request state for easy access
            request.state.tenant_id = tenant_id
            request.state.user_id = user_id
            
            # Process request
            response = await call_next(request)
            
            # Add tenant ID to response headers for debugging
            if tenant_id:
                response.headers["X-Tenant-ID"] = tenant_id
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except TenantContextError as e:
            # Handle tenant context errors
            logger.error(f"Tenant context error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Tenant context error: {str(e)}"}
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in tenant middleware: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
            
        finally:
            # Always clear context after request
            TenantContext.clear_tenant()
            TenantContext.clear_user()
            logger.debug("Cleared tenant context")
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if path is exempt from tenant requirement
        
        Args:
            path: Request path
            
        Returns:
            True if path is exempt, False otherwise
        """
        for exempt_path in self.config.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False
    
    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """
        Extract tenant ID from request
        
        Tries multiple methods in order:
        1. X-Tenant-ID header
        2. JWT token
        3. Subdomain (if enabled)
        
        Args:
            request: HTTP request
            
        Returns:
            Tenant ID or None
        """
        # Try header first
        tenant_id = request.headers.get(self.config.header_name)
        if tenant_id:
            logger.debug(f"Extracted tenant from header: {tenant_id}")
            return tenant_id
        
        # Try JWT token
        if self.config.jwt_secret:
            tenant_id = self._extract_from_jwt(request, self.config.jwt_tenant_claim)
            if tenant_id:
                logger.debug(f"Extracted tenant from JWT: {tenant_id}")
                return tenant_id
        
        # Try subdomain
        if self.config.enable_subdomain:
            tenant_id = self._extract_from_subdomain(request)
            if tenant_id:
                logger.debug(f"Extracted tenant from subdomain: {tenant_id}")
                return tenant_id
        
        return None
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """
        Extract user ID from request
        
        Currently only supports JWT tokens
        
        Args:
            request: HTTP request
            
        Returns:
            User ID or None
        """
        if self.config.jwt_secret:
            user_id = self._extract_from_jwt(request, self.config.jwt_user_claim)
            if user_id:
                logger.debug(f"Extracted user from JWT: {user_id}")
                return user_id
        
        return None
    
    def _extract_from_jwt(self, request: Request, claim_name: str) -> Optional[str]:
        """
        Extract value from JWT token
        
        Args:
            request: HTTP request
            claim_name: Name of claim to extract
            
        Returns:
            Claim value or None
        """
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        # Extract token
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Extract claim
            return payload.get(claim_name)
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def _extract_from_subdomain(self, request: Request) -> Optional[str]:
        """
        Extract tenant ID from subdomain
        
        Example: tenant1.example.com -> tenant1
        
        Args:
            request: HTTP request
            
        Returns:
            Tenant ID or None
        """
        if not self.config.subdomain_suffix:
            return None
        
        # Get host from request
        host = request.url.hostname
        if not host:
            return None
        
        # Check if host ends with suffix
        if not host.endswith(self.config.subdomain_suffix):
            return None
        
        # Extract subdomain
        # Remove suffix and split by dots
        base = host[:-len(self.config.subdomain_suffix)].rstrip(".")
        parts = base.split(".")
        
        # Return first part as tenant ID
        if parts and parts[0]:
            return parts[0]
        
        return None


def create_tenant_middleware(
    header_name: str = "X-Tenant-ID",
    jwt_secret: Optional[str] = None,
    require_tenant: bool = True,
    exempt_paths: Optional[list[str]] = None,
    tenant_loader: Optional[Callable[[str], Optional[Tenant]]] = None,
    user_loader: Optional[Callable[[str, str], Optional[TenantUser]]] = None,
) -> TenantMiddleware:
    """
    Convenience function to create tenant middleware
    
    Args:
        header_name: HTTP header name for tenant ID
        jwt_secret: Secret key for JWT validation
        require_tenant: Whether to require tenant for all requests
        exempt_paths: List of paths that don't require tenant context
        tenant_loader: Function to load tenant by ID
        user_loader: Function to load user by tenant_id and user_id
        
    Returns:
        Configured TenantMiddleware instance
    """
    config = TenantMiddlewareConfig(
        header_name=header_name,
        jwt_secret=jwt_secret,
        require_tenant=require_tenant,
        exempt_paths=exempt_paths,
    )
    
    return TenantMiddleware(
        config=config,
        tenant_loader=tenant_loader,
        user_loader=user_loader,
    )
