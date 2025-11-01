"""
Tests for Tenant Models

Tests tenant dataclasses, plan configurations, and model functionality.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

import pytest
from uuid import uuid4, UUID
from datetime import datetime

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


class TestTenant:
    """Test Tenant dataclass"""
    
    def test_tenant_creation(self):
        """Test creating a tenant"""
        tenant_id = uuid4()
        tenant = Tenant(
            id=tenant_id,
            name="Test Organization",
            slug="test-org"
        )
        
        assert tenant.id == tenant_id
        assert tenant.name == "Test Organization"
        assert tenant.slug == "test-org"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.plan == TenantPlan.FREE
        assert tenant.max_transactions == 10000
        assert tenant.max_users == 5
        assert tenant.max_api_calls == 1000
    
    def test_tenant_is_active(self):
        """Test tenant active status check"""
        active_tenant = Tenant(
            id=uuid4(),
            name="Active",
            slug="active",
            status=TenantStatus.ACTIVE
        )
        assert active_tenant.is_active()
        
        suspended_tenant = Tenant(
            id=uuid4(),
            name="Suspended",
            slug="suspended",
            status=TenantStatus.SUSPENDED
        )
        assert not suspended_tenant.is_active()
    
    def test_tenant_has_feature(self):
        """Test feature flag checking"""
        tenant = Tenant(
            id=uuid4(),
            name="Test",
            slug="test",
            features=["basic_fraud_detection", "real_time_alerts"]
        )
        
        assert tenant.has_feature("basic_fraud_detection")
        assert tenant.has_feature("real_time_alerts")
        assert not tenant.has_feature("advanced_ml")
    
    def test_tenant_has_all_features(self):
        """Test 'all' feature flag"""
        tenant = Tenant(
            id=uuid4(),
            name="Enterprise",
            slug="enterprise",
            features=["all"]
        )
        
        assert tenant.has_feature("any_feature")
        assert tenant.has_feature("another_feature")
    
    def test_tenant_to_dict(self):
        """Test tenant serialization"""
        tenant_id = uuid4()
        tenant = Tenant(
            id=tenant_id,
            name="Test",
            slug="test",
            features=["feature1"]
        )
        
        data = tenant.to_dict()
        
        assert data["id"] == str(tenant_id)
        assert data["name"] == "Test"
        assert data["slug"] == "test"
        assert data["status"] == "active"
        assert data["plan"] == "free"
        assert "features" in data
        assert "created_at" in data


class TestTenantUser:
    """Test TenantUser dataclass"""
    
    def test_tenant_user_creation(self):
        """Test creating a tenant user"""
        user_id = uuid4()
        tenant_id = uuid4()
        user_entity_id = uuid4()
        
        user = TenantUser(
            id=user_id,
            tenant_id=tenant_id,
            user_id=user_entity_id,
            email="test@example.com",
            role=UserRole.ANALYST
        )
        
        assert user.id == user_id
        assert user.tenant_id == tenant_id
        assert user.user_id == user_entity_id
        assert user.email == "test@example.com"
        assert user.role == UserRole.ANALYST
        assert user.is_active
        assert user.permissions == []
    
    def test_tenant_user_with_permissions(self):
        """Test user with custom permissions"""
        user = TenantUser(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="admin@example.com",
            role=UserRole.OPERATOR,
            permissions=["custom.permission", "special.access"]
        )
        
        assert len(user.permissions) == 2
        assert "custom.permission" in user.permissions
    
    def test_tenant_user_to_dict(self):
        """Test user serialization"""
        user_id = uuid4()
        user = TenantUser(
            id=user_id,
            tenant_id=uuid4(),
            user_id=uuid4(),
            email="user@example.com",
            role=UserRole.ANALYST
        )
        
        data = user.to_dict()
        
        assert data["id"] == str(user_id)
        assert data["email"] == "user@example.com"
        assert data["role"] == "analyst"
        assert data["is_active"] is True


class TestTenantQuota:
    """Test TenantQuota dataclass"""
    
    def test_quota_creation(self):
        """Test creating a quota"""
        tenant_id = uuid4()
        start = datetime.utcnow()
        end = datetime.utcnow()
        
        quota = TenantQuota(
            tenant_id=tenant_id,
            period_start=start,
            period_end=end,
            transactions_used=1500,
            api_calls_used=500,
            storage_used_mb=50.5
        )
        
        assert quota.tenant_id == tenant_id
        assert quota.transactions_used == 1500
        assert quota.api_calls_used == 500
        assert quota.storage_used_mb == 50.5
    
    def test_quota_to_dict(self):
        """Test quota serialization"""
        quota = TenantQuota(
            tenant_id=uuid4(),
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow()
        )
        
        data = quota.to_dict()
        
        assert "tenant_id" in data
        assert "transactions_used" in data
        assert "api_calls_used" in data
        assert "storage_used_mb" in data


class TestPlanConfigurations:
    """Test plan configurations"""
    
    def test_free_plan_config(self):
        """Test free plan configuration"""
        config = PLAN_CONFIGS[TenantPlan.FREE]
        
        assert config["max_transactions"] == 10000
        assert config["max_users"] == 5
        assert config["max_api_calls"] == 1000
        assert "basic_fraud_detection" in config["features"]
    
    def test_professional_plan_config(self):
        """Test professional plan configuration"""
        config = PLAN_CONFIGS[TenantPlan.PROFESSIONAL]
        
        assert config["max_transactions"] == 100000
        assert config["max_users"] == 25
        assert config["max_api_calls"] == 10000
        assert "advanced_ml" in config["features"]
        assert "real_time_alerts" in config["features"]
    
    def test_enterprise_plan_config(self):
        """Test enterprise plan configuration"""
        config = PLAN_CONFIGS[TenantPlan.ENTERPRISE]
        
        assert config["max_transactions"] == -1  # unlimited
        assert config["max_users"] == -1
        assert "all" in config["features"]


class TestCreateTenantFromPlan:
    """Test tenant creation from plan"""
    
    def test_create_free_tenant(self):
        """Test creating tenant with free plan"""
        tenant = create_tenant_from_plan(
            name="Free Org",
            slug="free-org",
            plan=TenantPlan.FREE
        )
        
        assert tenant.name == "Free Org"
        assert tenant.slug == "free-org"
        assert tenant.plan == TenantPlan.FREE
        assert tenant.max_transactions == 10000
        assert tenant.max_users == 5
        assert "basic_fraud_detection" in tenant.features
    
    def test_create_professional_tenant(self):
        """Test creating tenant with professional plan"""
        tenant = create_tenant_from_plan(
            name="Pro Org",
            slug="pro-org",
            plan=TenantPlan.PROFESSIONAL
        )
        
        assert tenant.plan == TenantPlan.PROFESSIONAL
        assert tenant.max_transactions == 100000
        assert tenant.max_users == 25
        assert "advanced_ml" in tenant.features
    
    def test_create_enterprise_tenant(self):
        """Test creating tenant with enterprise plan"""
        tenant = create_tenant_from_plan(
            name="Enterprise Org",
            slug="enterprise-org",
            plan=TenantPlan.ENTERPRISE
        )
        
        assert tenant.plan == TenantPlan.ENTERPRISE
        assert tenant.max_transactions == -1
        assert "all" in tenant.features
    
    def test_create_tenant_with_custom_metadata(self):
        """Test creating tenant with custom metadata"""
        tenant = create_tenant_from_plan(
            name="Custom Org",
            slug="custom-org",
            plan=TenantPlan.FREE,
            metadata={"custom_field": "value"}
        )
        
        assert "custom_field" in tenant.metadata
        assert tenant.metadata["custom_field"] == "value"
    
    def test_tenant_id_is_uuid(self):
        """Test that generated tenant ID is valid UUID"""
        tenant = create_tenant_from_plan(
            name="Test",
            slug="test"
        )
        
        assert isinstance(tenant.id, UUID)
