"""
Tests for RBAC System

Tests role-based access control and permission checking.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

import pytest
from uuid import uuid4

from src.tenancy.models import TenantUser, UserRole
from src.tenancy.permissions import Permission, get_role_permissions
from src.tenancy.rbac import (
    RBACManager,
    PermissionDeniedError,
    get_rbac_manager,
)


class TestRBACManager:
    """Test RBACManager class"""
    
    def setup_method(self):
        """Create RBAC manager for each test"""
        self.rbac = RBACManager()
    
    def test_check_permission_allowed(self):
        """Test permission check when allowed"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="admin@example.com",
            role=UserRole.TENANT_ADMIN
        )
        
        # Admin has all permissions
        result = self.rbac.check_permission(
            user,
            Permission.TRANSACTIONS_READ,
            raise_on_deny=False
        )
        
        assert result is True
    
    def test_check_permission_denied(self):
        """Test permission check when denied"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="analyst@example.com",
            role=UserRole.ANALYST
        )
        
        # Analyst cannot create users
        result = self.rbac.check_permission(
            user,
            Permission.USERS_CREATE,
            raise_on_deny=False
        )
        
        assert result is False
    
    def test_check_permission_raises_on_deny(self):
        """Test permission check raises when denied"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="analyst@example.com",
            role=UserRole.ANALYST
        )
        
        with pytest.raises(PermissionDeniedError, match="lacks permission"):
            self.rbac.check_permission(
                user,
                Permission.USERS_CREATE,
                raise_on_deny=True
            )
    
    def test_check_permission_with_custom_override(self):
        """Test permission with custom override"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="user@example.com",
            role=UserRole.ANALYST,
            permissions=["users.create"]  # Custom override
        )
        
        # Normally analyst can't create users, but has override
        result = self.rbac.check_permission(
            user,
            Permission.USERS_CREATE,
            raise_on_deny=False
        )
        
        assert result is True
    
    def test_check_permission_inactive_user(self):
        """Test permission denied for inactive user"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="inactive@example.com",
            role=UserRole.TENANT_ADMIN,
            is_active=False
        )
        
        result = self.rbac.check_permission(
            user,
            Permission.TRANSACTIONS_READ,
            raise_on_deny=False
        )
        
        assert result is False
    
    def test_check_any_permission_success(self):
        """Test check any permission when one matches"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="operator@example.com",
            role=UserRole.OPERATOR
        )
        
        result = self.rbac.check_any_permission(
            user,
            [Permission.TRANSACTIONS_CREATE, Permission.USERS_CREATE],
            raise_on_deny=False
        )
        
        # Operator can create transactions
        assert result is True
    
    def test_check_any_permission_failure(self):
        """Test check any permission when none match"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="analyst@example.com",
            role=UserRole.ANALYST
        )
        
        with pytest.raises(PermissionDeniedError, match="lacks any of"):
            self.rbac.check_any_permission(
                user,
                [Permission.USERS_CREATE, Permission.USERS_DELETE],
                raise_on_deny=True
            )
    
    def test_check_all_permissions_success(self):
        """Test check all permissions when all match"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="admin@example.com",
            role=UserRole.TENANT_ADMIN
        )
        
        result = self.rbac.check_all_permissions(
            user,
            [Permission.TRANSACTIONS_READ, Permission.FRAUD_DETECT],
            raise_on_deny=False
        )
        
        assert result is True
    
    def test_check_all_permissions_failure(self):
        """Test check all permissions when some missing"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="analyst@example.com",
            role=UserRole.ANALYST
        )
        
        result = self.rbac.check_all_permissions(
            user,
            [Permission.TRANSACTIONS_READ, Permission.USERS_CREATE],
            raise_on_deny=False
        )
        
        # Analyst can read but not create users
        assert result is False
    
    def test_get_user_permissions(self):
        """Test getting all permissions for a user"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="operator@example.com",
            role=UserRole.OPERATOR
        )
        
        permissions = self.rbac.get_user_permissions(user)
        
        assert Permission.TRANSACTIONS_READ in permissions
        assert Permission.FRAUD_DETECT in permissions
        assert Permission.USERS_CREATE not in permissions
    
    def test_get_user_permissions_with_overrides(self):
        """Test getting permissions with custom overrides"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="user@example.com",
            role=UserRole.ANALYST,
            permissions=["custom.permission"]
        )
        
        permissions = self.rbac.get_user_permissions(user)
        
        # Should have role permissions plus custom
        analyst_perms = get_role_permissions(UserRole.ANALYST)
        assert len(permissions) >= len(analyst_perms)
    
    def test_has_role(self):
        """Test role checking"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="operator@example.com",
            role=UserRole.OPERATOR
        )
        
        assert self.rbac.has_role(user, UserRole.OPERATOR)
        assert not self.rbac.has_role(user, UserRole.TENANT_ADMIN)
    
    def test_is_admin(self):
        """Test admin check"""
        admin = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="admin@example.com",
            role=UserRole.TENANT_ADMIN
        )
        
        non_admin = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="user@example.com",
            role=UserRole.ANALYST
        )
        
        assert self.rbac.is_admin(admin)
        assert not self.rbac.is_admin(non_admin)


class TestRBACGlobalManager:
    """Test global RBAC manager"""
    
    def test_get_rbac_manager_returns_instance(self):
        """Test getting global RBAC manager"""
        manager = get_rbac_manager()
        assert isinstance(manager, RBACManager)
    
    def test_get_rbac_manager_returns_same_instance(self):
        """Test manager is singleton"""
        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()
        assert manager1 is manager2


class TestRolePermissions:
    """Test role permission assignments"""
    
    def test_tenant_admin_has_all_permissions(self):
        """Test tenant admin has comprehensive permissions"""
        permissions = get_role_permissions(UserRole.TENANT_ADMIN)
        
        assert Permission.TRANSACTIONS_READ in permissions
        assert Permission.TRANSACTIONS_CREATE in permissions
        assert Permission.USERS_CREATE in permissions
        assert Permission.TENANT_SETTINGS in permissions
        assert len(permissions) > 20  # Should have many permissions
    
    def test_analyst_read_only(self):
        """Test analyst has read-only permissions"""
        permissions = get_role_permissions(UserRole.ANALYST)
        
        assert Permission.TRANSACTIONS_READ in permissions
        assert Permission.ANALYTICS_VIEW in permissions
        assert Permission.TRANSACTIONS_CREATE not in permissions
        assert Permission.USERS_CREATE not in permissions
    
    def test_operator_operational_access(self):
        """Test operator has operational permissions"""
        permissions = get_role_permissions(UserRole.OPERATOR)
        
        assert Permission.TRANSACTIONS_CREATE in permissions
        assert Permission.FRAUD_DETECT in permissions
        assert Permission.USERS_CREATE not in permissions
    
    def test_auditor_audit_access(self):
        """Test auditor has audit permissions"""
        permissions = get_role_permissions(UserRole.AUDITOR)
        
        assert Permission.AUDIT_VIEW in permissions
        assert Permission.AUDIT_EXPORT in permissions
        assert Permission.TRANSACTIONS_CREATE not in permissions
    
    def test_api_user_api_access(self):
        """Test API user has API permissions"""
        permissions = get_role_permissions(UserRole.API_USER)
        
        assert Permission.API_ACCESS in permissions
        assert Permission.TRANSACTIONS_READ in permissions
        assert Permission.USERS_CREATE not in permissions
