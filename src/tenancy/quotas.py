"""
Resource Quota Management

Tracks and enforces resource quotas for tenants.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import UUID
from enum import Enum


class QuotaType(Enum):
    """Types of quotas"""
    TRANSACTIONS = "transactions"
    API_CALLS = "api_calls"
    STORAGE = "storage"
    USERS = "users"


class QuotaPeriod(Enum):
    """Quota reset periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    UNLIMITED = "unlimited"


@dataclass
class QuotaUsage:
    """
    Track usage for a specific quota
    
    Attributes:
        quota_type: Type of quota being tracked
        limit: Maximum allowed usage (-1 for unlimited)
        used: Current usage
        period: Reset period
        period_start: When current period started
        period_end: When current period ends
    """
    quota_type: QuotaType
    limit: int
    used: int = 0
    period: QuotaPeriod = QuotaPeriod.MONTHLY
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate period end if not set"""
        if self.period_end is None:
            self.period_end = self._calculate_period_end()
    
    def _calculate_period_end(self) -> datetime:
        """Calculate when the current period ends"""
        if self.period == QuotaPeriod.UNLIMITED:
            return datetime.max
        elif self.period == QuotaPeriod.HOURLY:
            return self.period_start + timedelta(hours=1)
        elif self.period == QuotaPeriod.DAILY:
            return self.period_start + timedelta(days=1)
        elif self.period == QuotaPeriod.MONTHLY:
            # Add one month (approximate)
            return self.period_start + timedelta(days=30)
        return self.period_start
    
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded"""
        if self.limit == -1:  # Unlimited
            return False
        return self.used >= self.limit
    
    def remaining(self) -> int:
        """Get remaining quota"""
        if self.limit == -1:  # Unlimited
            return -1
        return max(0, self.limit - self.used)
    
    def percentage_used(self) -> float:
        """Get percentage of quota used"""
        if self.limit == -1:  # Unlimited
            return 0.0
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100.0
    
    def needs_reset(self) -> bool:
        """Check if quota period has expired"""
        if self.period == QuotaPeriod.UNLIMITED:
            return False
        return datetime.utcnow() >= self.period_end
    
    def reset(self):
        """Reset quota usage for new period"""
        self.used = 0
        self.period_start = datetime.utcnow()
        self.period_end = self._calculate_period_end()
    
    def increment(self, amount: int = 1) -> bool:
        """
        Increment usage
        
        Args:
            amount: Amount to increment by
            
        Returns:
            True if increment succeeded, False if quota exceeded
        """
        # Check if needs reset
        if self.needs_reset():
            self.reset()
        
        # Check if would exceed
        if self.limit != -1 and (self.used + amount) > self.limit:
            return False
        
        self.used += amount
        return True


class QuotaManager:
    """
    Manage quotas for all tenants
    
    Tracks and enforces resource quotas across all tenants.
    """
    
    def __init__(self):
        """Initialize quota manager"""
        # tenant_id -> quota_type -> QuotaUsage
        self._quotas: Dict[UUID, Dict[QuotaType, QuotaUsage]] = {}
    
    def set_quota(
        self,
        tenant_id: UUID,
        quota_type: QuotaType,
        limit: int,
        period: QuotaPeriod = QuotaPeriod.MONTHLY,
    ):
        """
        Set quota for a tenant
        
        Args:
            tenant_id: Tenant UUID
            quota_type: Type of quota
            limit: Maximum allowed usage (-1 for unlimited)
            period: Reset period
        """
        if tenant_id not in self._quotas:
            self._quotas[tenant_id] = {}
        
        self._quotas[tenant_id][quota_type] = QuotaUsage(
            quota_type=quota_type,
            limit=limit,
            period=period,
        )
    
    def get_quota(self, tenant_id: UUID, quota_type: QuotaType) -> Optional[QuotaUsage]:
        """
        Get quota usage for a tenant
        
        Args:
            tenant_id: Tenant UUID
            quota_type: Type of quota
            
        Returns:
            QuotaUsage object or None if not set
        """
        if tenant_id not in self._quotas:
            return None
        return self._quotas[tenant_id].get(quota_type)
    
    def check_quota(self, tenant_id: UUID, quota_type: QuotaType, amount: int = 1) -> bool:
        """
        Check if quota allows usage
        
        Args:
            tenant_id: Tenant UUID
            quota_type: Type of quota
            amount: Amount to check
            
        Returns:
            True if quota allows usage, False otherwise
        """
        quota = self.get_quota(tenant_id, quota_type)
        if not quota:
            return True  # No quota set, allow
        
        # Check if needs reset
        if quota.needs_reset():
            quota.reset()
        
        # Check if would exceed
        if quota.limit == -1:  # Unlimited
            return True
        
        return (quota.used + amount) <= quota.limit
    
    def use_quota(self, tenant_id: UUID, quota_type: QuotaType, amount: int = 1) -> bool:
        """
        Use quota (check and increment)
        
        Args:
            tenant_id: Tenant UUID
            quota_type: Type of quota
            amount: Amount to use
            
        Returns:
            True if quota was used successfully, False if exceeded
        """
        quota = self.get_quota(tenant_id, quota_type)
        if not quota:
            return True  # No quota set, allow
        
        return quota.increment(amount)
    
    def get_all_quotas(self, tenant_id: UUID) -> Dict[QuotaType, QuotaUsage]:
        """
        Get all quotas for a tenant
        
        Args:
            tenant_id: Tenant UUID
            
        Returns:
            Dictionary of quota type to QuotaUsage
        """
        return self._quotas.get(tenant_id, {})
    
    def reset_quota(self, tenant_id: UUID, quota_type: QuotaType):
        """
        Reset a specific quota
        
        Args:
            tenant_id: Tenant UUID
            quota_type: Type of quota to reset
        """
        quota = self.get_quota(tenant_id, quota_type)
        if quota:
            quota.reset()
    
    def reset_all_quotas(self, tenant_id: UUID):
        """
        Reset all quotas for a tenant
        
        Args:
            tenant_id: Tenant UUID
        """
        if tenant_id in self._quotas:
            for quota in self._quotas[tenant_id].values():
                quota.reset()
    
    def delete_tenant_quotas(self, tenant_id: UUID):
        """
        Delete all quotas for a tenant
        
        Args:
            tenant_id: Tenant UUID
        """
        if tenant_id in self._quotas:
            del self._quotas[tenant_id]


# Global quota manager instance
_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """
    Get the global quota manager instance
    
    Returns:
        Global QuotaManager instance
    """
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager


class QuotaExceededError(Exception):
    """Raised when a quota is exceeded"""
    
    def __init__(self, tenant_id: UUID, quota_type: QuotaType, message: str = None):
        self.tenant_id = tenant_id
        self.quota_type = quota_type
        self.message = message or f"Quota exceeded for {quota_type.value}"
        super().__init__(self.message)
