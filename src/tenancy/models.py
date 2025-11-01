"""
Tenant Models

Dataclasses and models for multi-tenant architecture.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
from uuid import UUID, uuid4


class TenantStatus(Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    DELETED = "deleted"


class TenantPlan(Enum):
    """Tenant subscription plan"""
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class UserRole(Enum):
    """User roles within a tenant"""
    TENANT_ADMIN = "tenant_admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    API_USER = "api_user"


@dataclass
class Tenant:
    """
    Tenant entity representing an organization
    
    Attributes:
        id: Unique tenant identifier
        name: Organization name
        slug: URL-friendly identifier
        status: Current tenant status
        plan: Subscription plan
        max_transactions: Monthly transaction limit
        max_users: User limit
        max_api_calls: Daily API call limit
        features: Enabled feature flags
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional tenant configuration
    """
    id: UUID
    name: str
    slug: str
    status: TenantStatus = TenantStatus.ACTIVE
    plan: TenantPlan = TenantPlan.FREE
    max_transactions: int = 10000
    max_users: int = 5
    max_api_calls: int = 1000
    features: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if tenant is active"""
        return self.status == TenantStatus.ACTIVE
    
    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a specific feature enabled"""
        return feature in self.features or "all" in self.features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "status": self.status.value,
            "plan": self.plan.value,
            "max_transactions": self.max_transactions,
            "max_users": self.max_users,
            "max_api_calls": self.max_api_calls,
            "features": self.features,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class TenantUser:
    """
    User associated with a tenant
    
    Attributes:
        id: Unique user ID within tenant
        tenant_id: Associated tenant
        user_id: Global user ID (from auth system)
        email: User email
        role: User role within tenant
        permissions: Custom permission overrides
        is_active: Whether user is active
        created_at: Creation timestamp
    """
    id: UUID
    tenant_id: UUID
    user_id: UUID
    email: str
    role: UserRole
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id),
            "email": self.email,
            "role": self.role.value,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TenantQuota:
    """
    Resource quota tracking for a tenant
    
    Attributes:
        tenant_id: Associated tenant
        period_start: Quota period start
        period_end: Quota period end
        transactions_used: Transactions used in period
        api_calls_used: API calls used in period
        storage_used_mb: Storage used in MB
    """
    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    transactions_used: int = 0
    api_calls_used: int = 0
    storage_used_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": str(self.tenant_id),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "transactions_used": self.transactions_used,
            "api_calls_used": self.api_calls_used,
            "storage_used_mb": self.storage_used_mb
        }


# Plan configurations
PLAN_CONFIGS = {
    TenantPlan.FREE: {
        "max_transactions": 10000,
        "max_users": 5,
        "max_api_calls": 1000,
        "max_storage_mb": 100,
        "features": ["basic_fraud_detection"]
    },
    TenantPlan.PROFESSIONAL: {
        "max_transactions": 100000,
        "max_users": 25,
        "max_api_calls": 10000,
        "max_storage_mb": 1000,
        "features": [
            "basic_fraud_detection",
            "advanced_ml",
            "real_time_alerts",
            "custom_rules"
        ]
    },
    TenantPlan.ENTERPRISE: {
        "max_transactions": -1,  # unlimited
        "max_users": -1,
        "max_api_calls": 100000,
        "max_storage_mb": -1,
        "features": ["all"]
    }
}


def create_tenant_from_plan(
    name: str,
    slug: str,
    plan: TenantPlan = TenantPlan.FREE,
    **kwargs
) -> Tenant:
    """
    Create a tenant with plan-based defaults
    
    Args:
        name: Tenant name
        slug: URL-friendly slug
        plan: Subscription plan
        **kwargs: Additional tenant attributes
    
    Returns:
        Tenant instance with plan defaults
    """
    config = PLAN_CONFIGS[plan]
    
    # Handle metadata separately to allow merging
    metadata = kwargs.pop("metadata", {})
    metadata["max_storage_mb"] = config["max_storage_mb"]
    
    # Handle features separately to allow merging
    custom_features = kwargs.pop("features", [])
    features = config["features"].copy()
    features.extend(custom_features)
    
    return Tenant(
        id=uuid4(),
        name=name,
        slug=slug,
        plan=plan,
        max_transactions=config["max_transactions"],
        max_users=config["max_users"],
        max_api_calls=config["max_api_calls"],
        features=features,
        metadata=metadata,
        **kwargs
    )
