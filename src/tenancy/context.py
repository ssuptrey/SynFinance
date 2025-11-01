"""
Tenant Context Management

Thread-local storage and context management for multi-tenant requests.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from contextvars import ContextVar
from typing import Optional, Any
from uuid import UUID
from types import TracebackType


# Context variable for current tenant ID
_current_tenant: ContextVar[Optional[UUID]] = ContextVar('current_tenant', default=None)

# Context variable for current user ID
_current_user: ContextVar[Optional[UUID]] = ContextVar('current_user', default=None)


class TenantContext:
    """
    Tenant context manager for request-scoped tenant isolation
    
    Uses context variables to maintain tenant context across async operations.
    """
    
    @staticmethod
    def set_current_tenant(tenant_id: Optional[UUID]) -> None:
        """
        Set the current tenant ID for this context
        
        Args:
            tenant_id: Tenant UUID or None to clear
        """
        _current_tenant.set(tenant_id)
    
    @staticmethod
    def get_current_tenant() -> Optional[UUID]:
        """
        Get the current tenant ID
        
        Returns:
            Current tenant UUID or None if not set
        """
        return _current_tenant.get()
    
    @staticmethod
    def require_tenant() -> UUID:
        """
        Get the current tenant ID, raising error if not set
        
        Returns:
            Current tenant UUID
        
        Raises:
            TenantContextError: If no tenant is set in context
        """
        tenant_id = _current_tenant.get()
        if tenant_id is None:
            raise TenantContextError("No tenant set in current context")
        return tenant_id
    
    @staticmethod
    def clear_tenant() -> None:
        """Clear the current tenant from context"""
        _current_tenant.set(None)
    
    @staticmethod
    def set_current_user(user_id: Optional[UUID]) -> None:
        """
        Set the current user ID for this context
        
        Args:
            user_id: User UUID or None to clear
        """
        _current_user.set(user_id)
    
    @staticmethod
    def get_current_user() -> Optional[UUID]:
        """
        Get the current user ID
        
        Returns:
            Current user UUID or None if not set
        """
        return _current_user.get()
    
    @staticmethod
    def require_user() -> UUID:
        """
        Get the current user ID, raising error if not set
        
        Returns:
            Current user UUID
        
        Raises:
            TenantContextError: If no user is set in context
        """
        user_id = _current_user.get()
        if user_id is None:
            raise TenantContextError("No user set in current context")
        return user_id
    
    @staticmethod
    def clear_user() -> None:
        """Clear the current user from context"""
        _current_user.set(None)
    
    @staticmethod
    def clear_all() -> None:
        """Clear all context variables"""
        _current_tenant.set(None)
        _current_user.set(None)


class TenantContextManager:
    """
    Context manager for temporarily setting tenant context
    
    Usage:
        with TenantContextManager(tenant_id):
            # Code runs with tenant context set
            pass
        # Tenant context is restored to previous value
    """
    
    def __init__(self, tenant_id: Optional[UUID], user_id: Optional[UUID] = None):
        """
        Initialize context manager
        
        Args:
            tenant_id: Tenant ID to set
            user_id: Optional user ID to set
        """
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.previous_tenant: Optional[UUID] = None
        self.previous_user: Optional[UUID] = None
    
    def __enter__(self):
        """Save current context and set new tenant"""
        self.previous_tenant = TenantContext.get_current_tenant()
        self.previous_user = TenantContext.get_current_user()
        
        TenantContext.set_current_tenant(self.tenant_id)
        if self.user_id is not None:
            TenantContext.set_current_user(self.user_id)
        
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> bool:
        """Restore previous context"""
        TenantContext.set_current_tenant(self.previous_tenant)
        TenantContext.set_current_user(self.previous_user)
        return False


class TenantContextError(Exception):
    """Raised when tenant context is required but not available"""
    pass


def get_tenant_id() -> Optional[UUID]:
    """
    Convenience function to get current tenant ID
    
    Returns:
        Current tenant UUID or None
    """
    return TenantContext.get_current_tenant()


def require_tenant_id() -> UUID:
    """
    Convenience function to require tenant ID
    
    Returns:
        Current tenant UUID
    
    Raises:
        TenantContextError: If no tenant in context
    """
    return TenantContext.require_tenant()


def get_user_id() -> Optional[UUID]:
    """
    Convenience function to get current user ID
    
    Returns:
        Current user UUID or None
    """
    return TenantContext.get_current_user()
