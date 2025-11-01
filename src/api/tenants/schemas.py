"""
Tenant API Schemas

Pydantic schemas for tenant API requests and responses.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class TenantCreate(BaseModel):
    """Schema for creating a new tenant"""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    slug: str = Field(..., min_length=1, max_length=100, description="URL-friendly identifier")
    plan: str = Field(default="free", description="Subscription plan (free, professional, enterprise)")
    features: List[str] = Field(default_factory=list, description="Enabled feature flags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TenantUpdate(BaseModel):
    """Schema for updating a tenant"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = Field(None, description="Tenant status (active, suspended, inactive, deleted)")
    plan: Optional[str] = Field(None, description="Subscription plan")
    features: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class TenantResponse(BaseModel):
    """Schema for tenant response"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    slug: str
    status: str
    plan: str
    max_transactions: int
    max_users: int
    max_api_calls: int
    features: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class TenantUserCreate(BaseModel):
    """Schema for creating a tenant user"""
    user_id: UUID = Field(..., description="Global user ID from auth system")
    email: EmailStr = Field(..., description="User email address")
    role: str = Field(..., description="User role (tenant_admin, analyst, operator, auditor, api_user)")
    permissions: List[str] = Field(default_factory=list, description="Custom permission overrides")


class TenantUserUpdate(BaseModel):
    """Schema for updating a tenant user"""
    role: Optional[str] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None


class TenantUserResponse(BaseModel):
    """Schema for tenant user response"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    tenant_id: UUID
    user_id: UUID
    email: str
    role: str
    permissions: List[str]
    is_active: bool
    created_at: datetime


class TenantStatsResponse(BaseModel):
    """Schema for tenant statistics"""
    tenant_id: UUID
    total_users: int
    active_users: int
    transactions_used: int
    transactions_limit: int
    api_calls_used: int
    api_calls_limit: int
    storage_used_mb: float
    plan: str
    status: str
    created_at: datetime
    last_activity: Optional[datetime] = None


class TenantQuotaResponse(BaseModel):
    """Schema for tenant quota response"""
    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    transactions_used: int
    transactions_limit: int
    transactions_remaining: int
    api_calls_used: int
    api_calls_limit: int
    api_calls_remaining: int
    storage_used_mb: float
    usage_percentage: float
