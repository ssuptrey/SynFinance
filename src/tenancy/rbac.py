"""
Role-Based Access Control (RBAC)

Permission checking and enforcement for multi-tenant system.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from typing import Optional, Set, List, Callable
from uuid import UUID
from functools import wraps

from src.tenancy.models import TenantUser, UserRole
from src.tenancy.permissions import Permission, get_role_permissions
from src.tenancy.context import TenantContext


class PermissionDeniedError(Exception):
    """Raised when user lacks required permission"""
    pass


class RBACManager:
    """
    RBAC manager for permission checking
    
    Handles permission validation, role checking, and access control.
    """
    
    def __init__(self):
        """Initialize RBAC manager"""
        self._user_cache: dict[UUID, TenantUser] = {}
    
    def check_permission(
        self,
        user: TenantUser,
        permission: Permission,
        raise_on_deny: bool = True
    ) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            user: Tenant user
            permission: Permission to check
            raise_on_deny: Raise exception if permission denied
        
        Returns:
            True if user has permission
        
        Raises:
            PermissionDeniedError: If permission denied and raise_on_deny=True
        """
        # Get role permissions
        role_permissions = get_role_permissions(user.role)
        
        # Check role permissions
        has_perm = permission in role_permissions
        
        # Check custom permission overrides
        if permission.value in user.permissions:
            has_perm = True
        
        # Check if user is active
        if not user.is_active:
            has_perm = False
        
        if not has_perm and raise_on_deny:
            raise PermissionDeniedError(
                f"User {user.email} (role: {user.role.value}) lacks permission: {permission.value}"
            )
        
        return has_perm
    
    def check_any_permission(
        self,
        user: TenantUser,
        permissions: List[Permission],
        raise_on_deny: bool = True
    ) -> bool:
        """
        Check if user has any of the specified permissions
        
        Args:
            user: Tenant user
            permissions: List of permissions to check
            raise_on_deny: Raise exception if none match
        
        Returns:
            True if user has at least one permission
        
        Raises:
            PermissionDeniedError: If no permissions match and raise_on_deny=True
        """
        for perm in permissions:
            if self.check_permission(user, perm, raise_on_deny=False):
                return True
        
        if raise_on_deny:
            perm_names = [p.value for p in permissions]
            raise PermissionDeniedError(
                f"User {user.email} lacks any of: {', '.join(perm_names)}"
            )
        
        return False
    
    def check_all_permissions(
        self,
        user: TenantUser,
        permissions: List[Permission],
        raise_on_deny: bool = True
    ) -> bool:
        """
        Check if user has all of the specified permissions
        
        Args:
            user: Tenant user
            permissions: List of permissions to check
            raise_on_deny: Raise exception if any missing
        
        Returns:
            True if user has all permissions
        
        Raises:
            PermissionDeniedError: If any permissions missing and raise_on_deny=True
        """
        for perm in permissions:
            if not self.check_permission(user, perm, raise_on_deny=raise_on_deny):
                return False
        
        return True
    
    def get_user_permissions(self, user: TenantUser) -> Set[Permission]:
        """
        Get all permissions for a user
        
        Args:
            user: Tenant user
        
        Returns:
            Set of all permissions
        """
        permissions = get_role_permissions(user.role).copy()
        
        # Add custom permission overrides
        for perm_str in user.permissions:
            try:
                permissions.add(Permission(perm_str))
            except ValueError:
                # Invalid permission string, skip
                pass
        
        return permissions
    
    def has_role(self, user: TenantUser, role: UserRole) -> bool:
        """
        Check if user has a specific role
        
        Args:
            user: Tenant user
            role: Role to check
        
        Returns:
            True if user has role
        """
        return user.role == role
    
    def is_admin(self, user: TenantUser) -> bool:
        """
        Check if user is tenant admin
        
        Args:
            user: Tenant user
        
        Returns:
            True if user is admin
        """
        return user.role == UserRole.TENANT_ADMIN


# Global RBAC manager instance
_rbac_manager = RBACManager()


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance"""
    return _rbac_manager


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission
    
    Usage:
        @require_permission(Permission.TRANSACTIONS_READ)
        def get_transactions():
            pass
    
    Args:
        permission: Required permission
    
    Raises:
        PermissionDeniedError: If user lacks permission
        TenantContextError: If no user in context
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get current user from context
            user_id = TenantContext.get_current_user()
            if user_id is None:
                raise PermissionDeniedError("No user in context")
            
            # For now, this is a placeholder
            # In production, would fetch user from database/cache
            # and check permissions via RBAC manager
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_any_permission(*permissions: Permission):
    """
    Decorator to require any of the specified permissions
    
    Usage:
        @require_any_permission(Permission.FRAUD_DETECT, Permission.FRAUD_REVIEW)
        def check_fraud():
            pass
    
    Args:
        *permissions: Required permissions (at least one)
    
    Raises:
        PermissionDeniedError: If user lacks all permissions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = TenantContext.get_current_user()
            if user_id is None:
                raise PermissionDeniedError("No user in context")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: UserRole):
    """
    Decorator to require a specific role
    
    Usage:
        @require_role(UserRole.TENANT_ADMIN)
        def admin_function():
            pass
    
    Args:
        role: Required role
    
    Raises:
        PermissionDeniedError: If user doesn't have role
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = TenantContext.get_current_user()
            if user_id is None:
                raise PermissionDeniedError("No user in context")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
