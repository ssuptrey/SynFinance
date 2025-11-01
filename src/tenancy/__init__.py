"""
Multi-tenancy Module

Provides multi-tenant architecture with tenant isolation, RBAC, and resource quotas.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from src.tenancy.models import (
    Tenant,
    TenantUser,
    TenantQuota,
    TenantStatus,
    TenantPlan,
    UserRole,
    create_tenant_from_plan,
    PLAN_CONFIGS,
)

from src.tenancy.context import (
    TenantContext,
    TenantContextManager,
    TenantContextError,
    get_tenant_id,
    require_tenant_id,
    get_user_id,
)

from src.tenancy.permissions import (
    Permission,
    get_role_permissions,
    get_permission_list,
    has_permission_for_role,
    PERMISSION_CATEGORIES,
)

from src.tenancy.rbac import (
    RBACManager,
    PermissionDeniedError,
    get_rbac_manager,
    require_permission,
    require_any_permission,
    require_role,
)

from src.tenancy.middleware import (
    TenantMiddleware,
    TenantMiddlewareConfig,
    create_tenant_middleware,
)

__all__ = [
    # Models
    "Tenant",
    "TenantUser",
    "TenantQuota",
    "TenantStatus",
    "TenantPlan",
    "UserRole",
    "create_tenant_from_plan",
    "PLAN_CONFIGS",
    
    # Context
    "TenantContext",
    "TenantContextManager",
    "TenantContextError",
    "get_tenant_id",
    "require_tenant_id",
    "get_user_id",
    
    # Permissions
    "Permission",
    "get_role_permissions",
    "get_permission_list",
    "has_permission_for_role",
    "PERMISSION_CATEGORIES",
    
    # RBAC
    "RBACManager",
    "PermissionDeniedError",
    "get_rbac_manager",
    "require_permission",
    "require_any_permission",
    "require_role",
    
    # Middleware
    "TenantMiddleware",
    "TenantMiddlewareConfig",
    "create_tenant_middleware",
]
