"""
Tenant Management Routes

FastAPI routes for tenant CRUD operations and management.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
from uuid import UUID
from datetime import datetime

from .schemas import (
    TenantCreate,
    TenantUpdate,
    TenantResponse,
    TenantUserCreate,
    TenantUserUpdate,
    TenantUserResponse,
    TenantStatsResponse,
    TenantQuotaResponse,
)
from src.tenancy.models import (
    Tenant,
    TenantUser,
    TenantStatus,
    TenantPlan,
    UserRole,
    create_tenant_from_plan,
)
from src.tenancy.rbac import require_permission, get_rbac_manager
from src.tenancy.context import require_tenant_id, require_user_id
from src.tenancy.permissions import Permission


router = APIRouter(prefix="/tenants", tags=["tenants"])

# In-memory storage for demo (replace with database in production)
_tenants: Dict[UUID, Tenant] = {}
_tenant_users: Dict[UUID, Dict[UUID, TenantUser]] = {}


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(tenant_data: TenantCreate) -> TenantResponse:
    """
    Create a new tenant
    
    Creates a new tenant organization with the specified plan and features.
    Only accessible to system administrators.
    """
    # Convert plan string to enum
    try:
        plan = TenantPlan[tenant_data.plan.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan: {tenant_data.plan}. Must be one of: free, professional, enterprise"
        )
    
    # Create tenant from plan
    tenant = create_tenant_from_plan(
        name=tenant_data.name,
        slug=tenant_data.slug,
        plan=plan,
        features=tenant_data.features,
        metadata=tenant_data.metadata,
    )
    
    # Store tenant
    _tenants[tenant.id] = tenant
    _tenant_users[tenant.id] = {}
    
    return TenantResponse(**tenant.to_dict())


@router.get("/{tenant_id}", response_model=TenantResponse)
@require_permission(Permission.TENANT_MANAGE)
async def get_tenant(tenant_id: UUID) -> TenantResponse:
    """
    Get tenant by ID
    
    Retrieves detailed information about a specific tenant.
    Requires TENANT_MANAGE permission.
    """
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}"
        )
    
    return TenantResponse(**tenant.to_dict())


@router.get("", response_model=List[TenantResponse])
async def list_tenants(
    status_filter: str = None,
    plan_filter: str = None,
    skip: int = 0,
    limit: int = 100,
) -> List[TenantResponse]:
    """
    List all tenants
    
    Returns a paginated list of tenants with optional filtering.
    System administrators only.
    """
    tenants = list(_tenants.values())
    
    # Apply filters
    if status_filter:
        try:
            status_enum = TenantStatus[status_filter.upper()]
            tenants = [t for t in tenants if t.status == status_enum]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )
    
    if plan_filter:
        try:
            plan_enum = TenantPlan[plan_filter.upper()]
            tenants = [t for t in tenants if t.plan == plan_enum]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan: {plan_filter}"
            )
    
    # Apply pagination
    tenants = tenants[skip : skip + limit]
    
    return [TenantResponse(**t.to_dict()) for t in tenants]


@router.patch("/{tenant_id}", response_model=TenantResponse)
@require_permission(Permission.TENANT_MANAGE)
async def update_tenant(tenant_id: UUID, update_data: TenantUpdate) -> TenantResponse:
    """
    Update tenant
    
    Updates tenant information such as name, status, plan, or features.
    Requires TENANT_MANAGE permission.
    """
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}"
        )
    
    # Update fields
    if update_data.name is not None:
        tenant.name = update_data.name
    
    if update_data.status is not None:
        try:
            tenant.status = TenantStatus[update_data.status.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {update_data.status}"
            )
    
    if update_data.plan is not None:
        try:
            tenant.plan = TenantPlan[update_data.plan.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan: {update_data.plan}"
            )
    
    if update_data.features is not None:
        tenant.features = update_data.features
    
    if update_data.metadata is not None:
        tenant.metadata.update(update_data.metadata)
    
    tenant.updated_at = datetime.utcnow()
    
    return TenantResponse(**tenant.to_dict())


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission(Permission.TENANT_MANAGE)
async def delete_tenant(tenant_id: UUID):
    """
    Delete tenant
    
    Soft deletes a tenant by setting its status to DELETED.
    Requires TENANT_MANAGE permission.
    """
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}"
        )
    
    # Soft delete
    tenant.status = TenantStatus.DELETED
    tenant.updated_at = datetime.utcnow()


@router.get("/{tenant_id}/stats", response_model=TenantStatsResponse)
@require_permission(Permission.ANALYTICS_VIEW)
async def get_tenant_stats(tenant_id: UUID) -> TenantStatsResponse:
    """
    Get tenant statistics
    
    Returns usage statistics and metrics for the tenant.
    Requires ANALYTICS_VIEW permission.
    """
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}"
        )
    
    users = _tenant_users.get(tenant_id, {})
    active_users = sum(1 for u in users.values() if u.is_active)
    
    return TenantStatsResponse(
        tenant_id=tenant.id,
        total_users=len(users),
        active_users=active_users,
        transactions_used=0,  # Would come from quota tracking
        transactions_limit=tenant.max_transactions,
        api_calls_used=0,  # Would come from quota tracking
        api_calls_limit=tenant.max_api_calls,
        storage_used_mb=0.0,  # Would come from quota tracking
        plan=tenant.plan.value,
        status=tenant.status.value,
        created_at=tenant.created_at,
        last_activity=None,  # Would track last API call
    )


# Tenant User Management

@router.post("/{tenant_id}/users", response_model=TenantUserResponse, status_code=status.HTTP_201_CREATED)
@require_permission(Permission.USERS_CREATE)
async def create_tenant_user(tenant_id: UUID, user_data: TenantUserCreate) -> TenantUserResponse:
    """
    Create a user within a tenant
    
    Adds a new user to the tenant with specified role and permissions.
    Requires USERS_CREATE permission.
    """
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}"
        )
    
    # Validate role
    try:
        role = UserRole[user_data.role.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {user_data.role}"
        )
    
    # Check user limit
    if tenant_id in _tenant_users:
        if len(_tenant_users[tenant_id]) >= tenant.max_users:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User limit reached for tenant ({tenant.max_users} users)"
            )
    
    # Create user
    from uuid import uuid4
    user = TenantUser(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_data.user_id,
        email=user_data.email,
        role=role,
        permissions=user_data.permissions,
    )
    
    # Store user
    if tenant_id not in _tenant_users:
        _tenant_users[tenant_id] = {}
    _tenant_users[tenant_id][user.id] = user
    
    return TenantUserResponse(**user.to_dict())


@router.get("/{tenant_id}/users", response_model=List[TenantUserResponse])
@require_permission(Permission.USERS_VIEW)
async def list_tenant_users(
    tenant_id: UUID,
    active_only: bool = False,
    role_filter: str = None,
) -> List[TenantUserResponse]:
    """
    List users in a tenant
    
    Returns all users associated with the tenant.
    Requires USERS_VIEW permission.
    """
    if tenant_id not in _tenant_users:
        return []
    
    users = list(_tenant_users[tenant_id].values())
    
    # Apply filters
    if active_only:
        users = [u for u in users if u.is_active]
    
    if role_filter:
        try:
            role_enum = UserRole[role_filter.upper()]
            users = [u for u in users if u.role == role_enum]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role_filter}"
            )
    
    return [TenantUserResponse(**u.to_dict()) for u in users]


@router.get("/{tenant_id}/users/{user_id}", response_model=TenantUserResponse)
@require_permission(Permission.USERS_VIEW)
async def get_tenant_user(tenant_id: UUID, user_id: UUID) -> TenantUserResponse:
    """
    Get a specific user in a tenant
    
    Retrieves detailed information about a tenant user.
    Requires USERS_VIEW permission.
    """
    if tenant_id not in _tenant_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found in tenant"
        )
    
    user = _tenant_users[tenant_id].get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}"
        )
    
    return TenantUserResponse(**user.to_dict())


@router.patch("/{tenant_id}/users/{user_id}", response_model=TenantUserResponse)
@require_permission(Permission.USERS_UPDATE)
async def update_tenant_user(
    tenant_id: UUID,
    user_id: UUID,
    update_data: TenantUserUpdate,
) -> TenantUserResponse:
    """
    Update a tenant user
    
    Updates user role, permissions, or active status.
    Requires USERS_UPDATE permission.
    """
    if tenant_id not in _tenant_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found in tenant"
        )
    
    user = _tenant_users[tenant_id].get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}"
        )
    
    # Update fields
    if update_data.role is not None:
        try:
            user.role = UserRole[update_data.role.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {update_data.role}"
            )
    
    if update_data.permissions is not None:
        user.permissions = update_data.permissions
    
    if update_data.is_active is not None:
        user.is_active = update_data.is_active
    
    return TenantUserResponse(**user.to_dict())


@router.delete("/{tenant_id}/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission(Permission.USERS_DELETE)
async def delete_tenant_user(tenant_id: UUID, user_id: UUID):
    """
    Delete a tenant user
    
    Removes a user from the tenant.
    Requires USERS_DELETE permission.
    """
    if tenant_id not in _tenant_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found in tenant"
        )
    
    if user_id not in _tenant_users[tenant_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}"
        )
    
    # Delete user
    del _tenant_users[tenant_id][user_id]
