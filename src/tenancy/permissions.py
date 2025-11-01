"""
Permission System

Permission definitions and constants for RBAC.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from enum import Enum
from typing import Set, Dict, List
from src.tenancy.models import UserRole


class Permission(Enum):
    """Permission enumeration for RBAC"""
    
    # Transaction permissions
    TRANSACTIONS_READ = "transactions.read"
    TRANSACTIONS_CREATE = "transactions.create"
    TRANSACTIONS_UPDATE = "transactions.update"
    TRANSACTIONS_DELETE = "transactions.delete"
    
    # Fraud detection permissions
    FRAUD_DETECT = "fraud.detect"
    FRAUD_REVIEW = "fraud.review"
    FRAUD_OVERRIDE = "fraud.override"
    FRAUD_RULES_MANAGE = "fraud.rules_manage"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics.view"
    ANALYTICS_EXPORT = "analytics.export"
    ANALYTICS_ADVANCED = "analytics.advanced"
    
    # Customer permissions
    CUSTOMERS_READ = "customers.read"
    CUSTOMERS_CREATE = "customers.create"
    CUSTOMERS_UPDATE = "customers.update"
    CUSTOMERS_DELETE = "customers.delete"
    
    # Merchant permissions
    MERCHANTS_READ = "merchants.read"
    MERCHANTS_CREATE = "merchants.create"
    MERCHANTS_UPDATE = "merchants.update"
    
    # Settings permissions
    SETTINGS_READ = "settings.read"
    SETTINGS_UPDATE = "settings.update"
    
    # User management permissions
    USERS_READ = "users.read"
    USERS_CREATE = "users.create"
    USERS_UPDATE = "users.update"
    USERS_DELETE = "users.delete"
    
    # API permissions
    API_ACCESS = "api.access"
    API_WEBHOOKS = "api.webhooks"
    
    # Audit permissions
    AUDIT_VIEW = "audit.view"
    AUDIT_EXPORT = "audit.export"
    
    # Tenant management
    TENANT_SETTINGS = "tenant.settings"
    TENANT_BILLING = "tenant.billing"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.TENANT_ADMIN: {
        # Full access to everything
        Permission.TRANSACTIONS_READ,
        Permission.TRANSACTIONS_CREATE,
        Permission.TRANSACTIONS_UPDATE,
        Permission.TRANSACTIONS_DELETE,
        Permission.FRAUD_DETECT,
        Permission.FRAUD_REVIEW,
        Permission.FRAUD_OVERRIDE,
        Permission.FRAUD_RULES_MANAGE,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.ANALYTICS_ADVANCED,
        Permission.CUSTOMERS_READ,
        Permission.CUSTOMERS_CREATE,
        Permission.CUSTOMERS_UPDATE,
        Permission.CUSTOMERS_DELETE,
        Permission.MERCHANTS_READ,
        Permission.MERCHANTS_CREATE,
        Permission.MERCHANTS_UPDATE,
        Permission.SETTINGS_READ,
        Permission.SETTINGS_UPDATE,
        Permission.USERS_READ,
        Permission.USERS_CREATE,
        Permission.USERS_UPDATE,
        Permission.USERS_DELETE,
        Permission.API_ACCESS,
        Permission.API_WEBHOOKS,
        Permission.AUDIT_VIEW,
        Permission.AUDIT_EXPORT,
        Permission.TENANT_SETTINGS,
        Permission.TENANT_BILLING,
    },
    
    UserRole.ANALYST: {
        # Read-only analytics and data access
        Permission.TRANSACTIONS_READ,
        Permission.FRAUD_REVIEW,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.ANALYTICS_ADVANCED,
        Permission.CUSTOMERS_READ,
        Permission.MERCHANTS_READ,
        Permission.SETTINGS_READ,
        Permission.AUDIT_VIEW,
    },
    
    UserRole.OPERATOR: {
        # Operational access to transactions and fraud
        Permission.TRANSACTIONS_READ,
        Permission.TRANSACTIONS_CREATE,
        Permission.TRANSACTIONS_UPDATE,
        Permission.FRAUD_DETECT,
        Permission.FRAUD_REVIEW,
        Permission.FRAUD_OVERRIDE,
        Permission.ANALYTICS_VIEW,
        Permission.CUSTOMERS_READ,
        Permission.CUSTOMERS_CREATE,
        Permission.CUSTOMERS_UPDATE,
        Permission.MERCHANTS_READ,
        Permission.SETTINGS_READ,
        Permission.API_ACCESS,
    },
    
    UserRole.AUDITOR: {
        # Read-only access for audit purposes
        Permission.TRANSACTIONS_READ,
        Permission.FRAUD_REVIEW,
        Permission.ANALYTICS_VIEW,
        Permission.CUSTOMERS_READ,
        Permission.MERCHANTS_READ,
        Permission.SETTINGS_READ,
        Permission.AUDIT_VIEW,
        Permission.AUDIT_EXPORT,
    },
    
    UserRole.API_USER: {
        # Programmatic API access
        Permission.API_ACCESS,
        Permission.TRANSACTIONS_READ,
        Permission.TRANSACTIONS_CREATE,
        Permission.FRAUD_DETECT,
        Permission.CUSTOMERS_READ,
        Permission.MERCHANTS_READ,
    },
}


def get_role_permissions(role: UserRole) -> Set[Permission]:
    """
    Get all permissions for a role
    
    Args:
        role: User role
    
    Returns:
        Set of permissions for the role
    """
    return ROLE_PERMISSIONS.get(role, set())


def get_permission_list(role: UserRole) -> List[str]:
    """
    Get permission strings for a role
    
    Args:
        role: User role
    
    Returns:
        List of permission strings
    """
    permissions = get_role_permissions(role)
    return [p.value for p in permissions]


def has_permission_for_role(role: UserRole, permission: Permission) -> bool:
    """
    Check if a role has a specific permission
    
    Args:
        role: User role
        permission: Permission to check
    
    Returns:
        True if role has permission
    """
    return permission in ROLE_PERMISSIONS.get(role, set())


# Permission categories for organization
PERMISSION_CATEGORIES = {
    "Transactions": [
        Permission.TRANSACTIONS_READ,
        Permission.TRANSACTIONS_CREATE,
        Permission.TRANSACTIONS_UPDATE,
        Permission.TRANSACTIONS_DELETE,
    ],
    "Fraud Detection": [
        Permission.FRAUD_DETECT,
        Permission.FRAUD_REVIEW,
        Permission.FRAUD_OVERRIDE,
        Permission.FRAUD_RULES_MANAGE,
    ],
    "Analytics": [
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.ANALYTICS_ADVANCED,
    ],
    "Customers": [
        Permission.CUSTOMERS_READ,
        Permission.CUSTOMERS_CREATE,
        Permission.CUSTOMERS_UPDATE,
        Permission.CUSTOMERS_DELETE,
    ],
    "Merchants": [
        Permission.MERCHANTS_READ,
        Permission.MERCHANTS_CREATE,
        Permission.MERCHANTS_UPDATE,
    ],
    "Settings": [
        Permission.SETTINGS_READ,
        Permission.SETTINGS_UPDATE,
    ],
    "Users": [
        Permission.USERS_READ,
        Permission.USERS_CREATE,
        Permission.USERS_UPDATE,
        Permission.USERS_DELETE,
    ],
    "API": [
        Permission.API_ACCESS,
        Permission.API_WEBHOOKS,
    ],
    "Audit": [
        Permission.AUDIT_VIEW,
        Permission.AUDIT_EXPORT,
    ],
    "Tenant": [
        Permission.TENANT_SETTINGS,
        Permission.TENANT_BILLING,
    ],
}
