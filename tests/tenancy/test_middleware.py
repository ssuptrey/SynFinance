"""
Tests for Tenant Isolation Middleware

Tests cover:
- Tenant extraction from headers
- Tenant extraction from JWT
- Tenant extraction from subdomain
- Tenant validation
- User validation
- Context cleanup
- Exempt paths
- Error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException, status
from fastapi.responses import Response, JSONResponse
import jwt
from datetime import datetime, timedelta

from src.tenancy.middleware import (
    TenantMiddleware,
    TenantMiddlewareConfig,
    create_tenant_middleware,
)
from src.tenancy.models import Tenant, TenantUser, TenantStatus, UserRole
from src.tenancy.context import TenantContext


class TestTenantMiddlewareConfig:
    """Test TenantMiddlewareConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TenantMiddlewareConfig()
        
        assert config.header_name == "X-Tenant-ID"
        assert config.jwt_secret is None
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_tenant_claim == "tenant_id"
        assert config.jwt_user_claim == "user_id"
        assert config.require_tenant is True
        assert config.enable_subdomain is False
        assert config.subdomain_suffix is None
        assert "/health" in config.exempt_paths
        assert "/docs" in config.exempt_paths
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = TenantMiddlewareConfig(
            header_name="X-Custom-Tenant",
            jwt_secret="secret",
            require_tenant=False,
            exempt_paths=["/api/public"],
        )
        
        assert config.header_name == "X-Custom-Tenant"
        assert config.jwt_secret == "secret"
        assert config.require_tenant is False
        assert "/api/public" in config.exempt_paths


class TestTenantMiddleware:
    """Test TenantMiddleware"""
    
    @pytest.fixture
    def config(self):
        """Create default config"""
        return TenantMiddlewareConfig(require_tenant=False)
    
    @pytest.fixture
    def middleware(self, config):
        """Create middleware instance"""
        return TenantMiddleware(config=config)
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.url.hostname = "localhost"
        request.headers = {}
        request.state = Mock()
        return request
    
    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next"""
        async def _call_next(request):
            response = Response(content="OK", status_code=200)
            return response
        return _call_next
    
    @pytest.mark.asyncio
    async def test_exempt_path_skips_tenant_check(self, middleware, mock_call_next):
        """Test that exempt paths skip tenant checking"""
        request = Mock(spec=Request)
        request.url.path = "/health"
        
        response = await middleware(request, mock_call_next)
        
        assert response.status_code == 200
        # Context should not be set
        assert TenantContext.get_current_tenant() is None
    
    @pytest.mark.asyncio
    async def test_extract_tenant_from_header(
        self, middleware, mock_request, mock_call_next
    ):
        """Test extracting tenant from header"""
        from uuid import uuid4
        tenant_id = str(uuid4())
        mock_request.headers = {"X-Tenant-ID": tenant_id}
        
        response = await middleware(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert response.headers.get("X-Tenant-ID") == tenant_id
        assert mock_request.state.tenant_id == tenant_id
    
    @pytest.mark.asyncio
    async def test_extract_tenant_from_jwt(self, mock_request, mock_call_next):
        """Test extracting tenant from JWT token"""
        from uuid import uuid4
        
        secret = "test-secret"
        tenant_id = str(uuid4())
        user_id = str(uuid4())
        
        # Create JWT token
        payload = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        # Configure middleware with JWT
        config = TenantMiddlewareConfig(jwt_secret=secret, require_tenant=False)
        middleware = TenantMiddleware(config=config)
        
        # Add token to request
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        
        response = await middleware(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert mock_request.state.tenant_id == tenant_id
        assert mock_request.state.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_extract_tenant_from_subdomain(self, mock_call_next):
        """Test extracting tenant from subdomain"""
        from uuid import uuid4
        tenant_slug = "tenant-xyz"  # Subdomain is slug, not UUID
        
        config = TenantMiddlewareConfig(
            enable_subdomain=True,
            subdomain_suffix=".example.com",
            require_tenant=False,
        )
        middleware = TenantMiddleware(config=config)
        
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.url.hostname = f"{tenant_slug}.example.com"
        request.headers = {}
        request.state = Mock()
        
        response = await middleware(request, mock_call_next)
        
        assert response.status_code == 200
        # Subdomain extraction works, but non-UUID tenants aren't set in context
        assert request.state.tenant_id == tenant_slug
    
    @pytest.mark.asyncio
    async def test_require_tenant_raises_error(self, mock_request, mock_call_next):
        """Test that missing tenant raises error when required"""
        config = TenantMiddlewareConfig(require_tenant=True)
        middleware = TenantMiddleware(config=config)
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware(mock_request, mock_call_next)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Tenant ID is required" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_invalid_tenant_raises_error(self, mock_request, mock_call_next):
        """Test that invalid tenant raises error"""
        def tenant_loader(tenant_id):
            return None  # Tenant not found
        
        config = TenantMiddlewareConfig(require_tenant=False)
        middleware = TenantMiddleware(
            config=config,
            tenant_loader=tenant_loader,
        )
        
        mock_request.headers = {"X-Tenant-ID": "invalid-tenant"}
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware(mock_request, mock_call_next)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Tenant not found" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_inactive_tenant_raises_error(self, mock_request, mock_call_next):
        """Test that inactive tenant raises error"""
        from uuid import uuid4
        tenant = Tenant(
            id=uuid4(),
            name="Test Tenant",
            slug="test-tenant",
            status=TenantStatus.SUSPENDED,  # Suspended tenant
        )
        
        def tenant_loader(tenant_id):
            return tenant
        
        config = TenantMiddlewareConfig(require_tenant=False)
        middleware = TenantMiddleware(
            config=config,
            tenant_loader=tenant_loader,
        )
        
        mock_request.headers = {"X-Tenant-ID": "tenant-123"}
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware(mock_request, mock_call_next)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Tenant is not active" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_inactive_user_raises_error(self, mock_request, mock_call_next):
        """Test that inactive user raises error"""
        from uuid import uuid4
        
        tenant_id = uuid4()
        user_id = uuid4()
        
        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            slug="test-tenant",
            status=TenantStatus.ACTIVE,
        )
        
        user = TenantUser(
            id=uuid4(),
            tenant_id=tenant_id,
            user_id=user_id,
            email="user@test.com",
            role=UserRole.ANALYST,
            is_active=False,  # Inactive user
        )
        
        def tenant_loader(tenant_id):
            return tenant
        
        def user_loader(tenant_id, user_id):
            return user
        
        secret = "test-secret"
        payload = {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        config = TenantMiddlewareConfig(jwt_secret=secret, require_tenant=False)
        middleware = TenantMiddleware(
            config=config,
            tenant_loader=tenant_loader,
            user_loader=user_loader,
        )
        
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware(mock_request, mock_call_next)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "User is not active" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_context_cleanup_after_request(
        self, middleware, mock_request, mock_call_next
    ):
        """Test that context is cleaned up after request"""
        mock_request.headers = {"X-Tenant-ID": "tenant-123"}
        
        # Context should be set during request
        response = await middleware(mock_request, mock_call_next)
        
        # Context should be cleared after request
        assert TenantContext.get_current_tenant() is None
        assert TenantContext.get_current_user() is None
    
    @pytest.mark.asyncio
    async def test_context_cleanup_on_error(
        self, middleware, mock_request
    ):
        """Test that context is cleaned up even on error"""
        mock_request.headers = {"X-Tenant-ID": "tenant-123"}
        
        async def error_call_next(request):
            raise Exception("Test error")
        
        # Should handle error gracefully
        response = await middleware(mock_request, error_call_next)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500
        
        # Context should still be cleared
        assert TenantContext.get_current_tenant() is None
    
    @pytest.mark.asyncio
    async def test_valid_tenant_and_user(self, mock_request, mock_call_next):
        """Test successful request with valid tenant and user"""
        from uuid import uuid4
        
        tenant_id = uuid4()
        user_id = uuid4()
        
        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant",
            slug="test-tenant",
            status=TenantStatus.ACTIVE,
        )
        
        user = TenantUser(
            id=uuid4(),
            tenant_id=tenant_id,
            user_id=user_id,
            email="user@test.com",
            role=UserRole.ANALYST,
            is_active=True,
        )
        
        def tenant_loader(tenant_id):
            return tenant
        
        def user_loader(tenant_id, user_id):
            return user
        
        secret = "test-secret"
        payload = {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        config = TenantMiddlewareConfig(jwt_secret=secret, require_tenant=False)
        middleware = TenantMiddleware(
            config=config,
            tenant_loader=tenant_loader,
            user_loader=user_loader,
        )
        
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        
        response = await middleware(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert str(mock_request.state.tenant_id) == str(tenant_id)
        assert str(mock_request.state.user_id) == str(user_id)
    
    @pytest.mark.asyncio
    async def test_header_takes_precedence_over_jwt(self, mock_request, mock_call_next):
        """Test that header takes precedence over JWT"""
        from uuid import uuid4
        
        header_tenant_id = str(uuid4())
        jwt_tenant_id = str(uuid4())
        
        secret = "test-secret"
        payload = {
            "tenant_id": jwt_tenant_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        config = TenantMiddlewareConfig(jwt_secret=secret, require_tenant=False)
        middleware = TenantMiddleware(config=config)
        
        mock_request.headers = {
            "X-Tenant-ID": header_tenant_id,
            "Authorization": f"Bearer {token}",
        }
        
        response = await middleware(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert mock_request.state.tenant_id == header_tenant_id


class TestCreateTenantMiddleware:
    """Test create_tenant_middleware convenience function"""
    
    def test_creates_middleware_with_defaults(self):
        """Test creating middleware with default settings"""
        middleware = create_tenant_middleware()
        
        assert isinstance(middleware, TenantMiddleware)
        assert middleware.config.header_name == "X-Tenant-ID"
        assert middleware.config.require_tenant is True
    
    def test_creates_middleware_with_custom_settings(self):
        """Test creating middleware with custom settings"""
        def tenant_loader(tenant_id):
            return None
        
        middleware = create_tenant_middleware(
            header_name="X-Custom-Tenant",
            jwt_secret="secret",
            require_tenant=False,
            tenant_loader=tenant_loader,
        )
        
        assert isinstance(middleware, TenantMiddleware)
        assert middleware.config.header_name == "X-Custom-Tenant"
        assert middleware.config.jwt_secret == "secret"
        assert middleware.config.require_tenant is False
        assert middleware.tenant_loader is tenant_loader
