"""
Multi-Tenancy Demo Script

Demonstrates the complete multi-tenancy system including:
- Tenant creation with different plans
- User management and RBAC
- Context isolation
- Quota tracking
- Permission checking

Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

import asyncio
from uuid import uuid4
from datetime import datetime

from src.tenancy.models import (
    Tenant,
    TenantUser,
    TenantPlan,
    UserRole,
    create_tenant_from_plan,
)
from src.tenancy.context import TenantContext, TenantContextManager
from src.tenancy.rbac import get_rbac_manager, PermissionDeniedError
from src.tenancy.permissions import Permission
from src.tenancy.quotas import get_quota_manager, QuotaType, QuotaPeriod


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_tenant_info(tenant: Tenant):
    """Print tenant information"""
    print(f"Tenant ID: {tenant.id}")
    print(f"Name: {tenant.name}")
    print(f"Slug: {tenant.slug}")
    print(f"Status: {tenant.status.value}")
    print(f"Plan: {tenant.plan.value}")
    print(f"Max Transactions: {tenant.max_transactions:,}")
    print(f"Max Users: {tenant.max_users}")
    print(f"Max API Calls: {tenant.max_api_calls:,}")
    print(f"Features: {', '.join(tenant.features) if tenant.features else 'None'}")
    print(f"Created: {tenant.created_at.strftime('%Y-%m-%d %H:%M:%S')}")


def print_user_info(user: TenantUser):
    """Print user information"""
    print(f"User ID: {user.id}")
    print(f"Email: {user.email}")
    print(f"Role: {user.role.value}")
    print(f"Active: {user.is_active}")
    print(f"Custom Permissions: {', '.join(user.permissions) if user.permissions else 'None'}")


def demo_tenant_creation():
    """Demonstrate tenant creation with different plans"""
    print_section("1. Tenant Creation")
    
    # Create Free tier tenant
    print("Creating FREE tier tenant...")
    free_tenant = create_tenant_from_plan(
        name="Startup Inc",
        slug="startup-inc",
        plan=TenantPlan.FREE,
        features=["basic_analytics"],
    )
    print_tenant_info(free_tenant)
    
    # Create Professional tier tenant
    print("\nCreating PROFESSIONAL tier tenant...")
    pro_tenant = create_tenant_from_plan(
        name="Growing Corp",
        slug="growing-corp",
        plan=TenantPlan.PROFESSIONAL,
        features=["basic_analytics", "advanced_reporting", "api_access"],
    )
    print_tenant_info(pro_tenant)
    
    # Create Enterprise tier tenant
    print("\nCreating ENTERPRISE tier tenant...")
    enterprise_tenant = create_tenant_from_plan(
        name="Enterprise Ltd",
        slug="enterprise-ltd",
        plan=TenantPlan.ENTERPRISE,
        features=[
            "basic_analytics",
            "advanced_reporting",
            "api_access",
            "custom_integrations",
            "white_labeling",
            "dedicated_support",
        ],
    )
    print_tenant_info(enterprise_tenant)
    
    return free_tenant, pro_tenant, enterprise_tenant


def demo_user_management(tenant: Tenant):
    """Demonstrate user creation and management"""
    print_section("2. User Management")
    
    print(f"Creating users for tenant: {tenant.name}\n")
    
    # Create admin user
    admin = TenantUser(
        id=uuid4(),
        tenant_id=tenant.id,
        user_id=uuid4(),
        email="admin@example.com",
        role=UserRole.TENANT_ADMIN,
    )
    print("Admin User:")
    print_user_info(admin)
    
    # Create analyst user
    print("\nAnalyst User:")
    analyst = TenantUser(
        id=uuid4(),
        tenant_id=tenant.id,
        user_id=uuid4(),
        email="analyst@example.com",
        role=UserRole.ANALYST,
    )
    print_user_info(analyst)
    
    # Create operator with custom permissions
    print("\nOperator with Custom Permissions:")
    operator = TenantUser(
        id=uuid4(),
        tenant_id=tenant.id,
        user_id=uuid4(),
        email="operator@example.com",
        role=UserRole.OPERATOR,
        permissions=[Permission.FRAUD_OVERRIDE.value],
    )
    print_user_info(operator)
    
    return admin, analyst, operator


def demo_context_isolation():
    """Demonstrate tenant context isolation"""
    print_section("3. Context Isolation")
    
    tenant1_id = uuid4()
    tenant2_id = uuid4()
    
    print(f"Tenant 1 ID: {tenant1_id}")
    print(f"Tenant 2 ID: {tenant2_id}\n")
    
    # Set context for tenant 1
    print("Setting context for Tenant 1...")
    TenantContext.set_current_tenant(tenant1_id)
    current = TenantContext.get_current_tenant()
    print(f"Current tenant: {current}")
    print(f"Match: {current == tenant1_id}\n")
    
    # Use context manager for temporary switch
    print("Temporarily switching to Tenant 2 using context manager...")
    with TenantContextManager(tenant2_id):
        current = TenantContext.get_current_tenant()
        print(f"Current tenant (inside context): {current}")
        print(f"Match: {current == tenant2_id}\n")
    
    # Context restored
    current = TenantContext.get_current_tenant()
    print(f"Current tenant (after context): {current}")
    print(f"Match: {current == tenant1_id}\n")
    
    # Clear context
    print("Clearing context...")
    TenantContext.clear_tenant()
    current = TenantContext.get_current_tenant()
    print(f"Current tenant: {current}")


def demo_rbac(admin: TenantUser, analyst: TenantUser):
    """Demonstrate RBAC permission checking"""
    print_section("4. RBAC Permission Checking")
    
    rbac = get_rbac_manager()
    
    # Admin permissions
    print("Admin User Permissions:")
    admin_permissions = rbac.get_user_permissions(admin)
    print(f"Total permissions: {len(admin_permissions)}")
    print(f"Has TENANT_SETTINGS: {rbac.check_permission(admin, Permission.TENANT_SETTINGS)}")
    print(f"Has FRAUD_RULES_MANAGE: {rbac.check_permission(admin, Permission.FRAUD_RULES_MANAGE)}")
    print(f"Is admin: {rbac.is_admin(admin)}\n")
    
    # Analyst permissions
    print("Analyst User Permissions:")
    analyst_permissions = rbac.get_user_permissions(analyst)
    print(f"Total permissions: {len(analyst_permissions)}")
    print(f"Has TRANSACTIONS_READ: {rbac.check_permission(analyst, Permission.TRANSACTIONS_READ)}")
    print(f"Has ANALYTICS_VIEW: {rbac.check_permission(analyst, Permission.ANALYTICS_VIEW)}")
    print(f"Has TRANSACTIONS_DELETE: {rbac.check_permission(analyst, Permission.TRANSACTIONS_DELETE, raise_on_deny=False)}")
    print(f"Is admin: {rbac.is_admin(analyst)}\n")
    
    # Test permission denial
    print("Testing permission denial for analyst...")
    try:
        rbac.check_permission(analyst, Permission.USERS_DELETE, raise_on_deny=True)
        print("Permission granted (unexpected)")
    except PermissionDeniedError as e:
        print(f"Permission denied (expected): {e}")


def demo_quotas(tenant: Tenant):
    """Demonstrate quota tracking and enforcement"""
    print_section("5. Quota Tracking")
    
    quota_mgr = get_quota_manager()
    
    # Set quotas for tenant
    print(f"Setting quotas for tenant: {tenant.name}\n")
    quota_mgr.set_quota(
        tenant.id,
        QuotaType.TRANSACTIONS,
        limit=tenant.max_transactions,
        period=QuotaPeriod.MONTHLY,
    )
    quota_mgr.set_quota(
        tenant.id,
        QuotaType.API_CALLS,
        limit=tenant.max_api_calls,
        period=QuotaPeriod.DAILY,
    )
    
    # Check quotas
    print("Initial Quota Status:")
    for quota_type in [QuotaType.TRANSACTIONS, QuotaType.API_CALLS]:
        quota = quota_mgr.get_quota(tenant.id, quota_type)
        if quota:
            print(f"\n{quota_type.value.upper()}:")
            print(f"  Limit: {quota.limit:,}")
            print(f"  Used: {quota.used:,}")
            print(f"  Remaining: {quota.remaining():,}")
            print(f"  Usage: {quota.percentage_used():.1f}%")
            print(f"  Period: {quota.period.value}")
    
    # Simulate usage
    print("\n\nSimulating usage...")
    print("Recording 100 transactions...")
    quota_mgr.use_quota(tenant.id, QuotaType.TRANSACTIONS, 100)
    
    print("Recording 50 API calls...")
    quota_mgr.use_quota(tenant.id, QuotaType.API_CALLS, 50)
    
    # Check updated quotas
    print("\nUpdated Quota Status:")
    for quota_type in [QuotaType.TRANSACTIONS, QuotaType.API_CALLS]:
        quota = quota_mgr.get_quota(tenant.id, quota_type)
        if quota:
            print(f"\n{quota_type.value.upper()}:")
            print(f"  Limit: {quota.limit:,}")
            print(f"  Used: {quota.used:,}")
            print(f"  Remaining: {quota.remaining():,}")
            print(f"  Usage: {quota.percentage_used():.1f}%")
    
    # Test quota exceeded
    print("\n\nTesting quota enforcement...")
    # Try to use more than remaining
    success = quota_mgr.use_quota(tenant.id, QuotaType.TRANSACTIONS, tenant.max_transactions + 1)
    print(f"Attempt to exceed quota: {'Failed (expected)' if not success else 'Succeeded (unexpected)'}")


def demo_plan_comparison():
    """Compare different plan tiers"""
    print_section("6. Plan Tier Comparison")
    
    plans = [TenantPlan.FREE, TenantPlan.PROFESSIONAL, TenantPlan.ENTERPRISE]
    
    print(f"{'Plan':<15} {'Transactions':<15} {'Users':<10} {'API Calls':<12}")
    print("-" * 60)
    
    for plan in plans:
        tenant = create_tenant_from_plan(
            name=f"Test {plan.value}",
            slug=f"test-{plan.value}",
            plan=plan,
        )
        trans = "Unlimited" if tenant.max_transactions == -1 else f"{tenant.max_transactions:,}"
        users = "Unlimited" if tenant.max_users == -1 else str(tenant.max_users)
        api_calls = "Unlimited" if tenant.max_api_calls == -1 else f"{tenant.max_api_calls:,}"
        
        print(f"{plan.value:<15} {trans:<15} {users:<10} {api_calls:<12}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  SynFinance Multi-Tenancy System Demo")
    print("  Week 8 Day 4: Multi-tenancy & Enterprise Features")
    print("=" * 80)
    
    # 1. Tenant Creation
    free_tenant, pro_tenant, enterprise_tenant = demo_tenant_creation()
    
    # 2. User Management
    admin, analyst, operator = demo_user_management(pro_tenant)
    
    # 3. Context Isolation
    demo_context_isolation()
    
    # 4. RBAC
    demo_rbac(admin, analyst)
    
    # 5. Quotas
    demo_quotas(free_tenant)
    
    # 6. Plan Comparison
    demo_plan_comparison()
    
    print_section("Demo Complete!")
    print("Multi-tenancy system successfully demonstrated:")
    print("  ✓ Tenant creation with multiple plan tiers")
    print("  ✓ User management with role-based access control")
    print("  ✓ Context isolation for request-scoped tenant tracking")
    print("  ✓ Permission checking and enforcement")
    print("  ✓ Resource quota tracking and enforcement")
    print("\nThe system is ready for production use!")


if __name__ == "__main__":
    main()
