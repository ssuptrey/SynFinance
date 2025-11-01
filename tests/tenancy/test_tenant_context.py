"""
Tests for Tenant Context

Tests tenant context management and isolation.
Week 8 Day 4: Multi-tenancy & Enterprise Features
"""

import pytest
from uuid import uuid4

from src.tenancy.context import (
    TenantContext,
    TenantContextManager,
    TenantContextError,
    get_tenant_id,
    require_tenant_id,
    get_user_id,
)


class TestTenantContext:
    """Test TenantContext class"""
    
    def setup_method(self):
        """Clear context before each test"""
        TenantContext.clear_all()
    
    def test_set_and_get_tenant(self):
        """Test setting and getting tenant ID"""
        tenant_id = uuid4()
        TenantContext.set_current_tenant(tenant_id)
        
        assert TenantContext.get_current_tenant() == tenant_id
    
    def test_get_tenant_when_not_set(self):
        """Test getting tenant when not set"""
        assert TenantContext.get_current_tenant() is None
    
    def test_require_tenant_when_set(self):
        """Test requiring tenant when set"""
        tenant_id = uuid4()
        TenantContext.set_current_tenant(tenant_id)
        
        assert TenantContext.require_tenant() == tenant_id
    
    def test_require_tenant_when_not_set(self):
        """Test requiring tenant when not set raises error"""
        with pytest.raises(TenantContextError, match="No tenant set"):
            TenantContext.require_tenant()
    
    def test_clear_tenant(self):
        """Test clearing tenant"""
        TenantContext.set_current_tenant(uuid4())
        TenantContext.clear_tenant()
        
        assert TenantContext.get_current_tenant() is None
    
    def test_set_and_get_user(self):
        """Test setting and getting user ID"""
        user_id = uuid4()
        TenantContext.set_current_user(user_id)
        
        assert TenantContext.get_current_user() == user_id
    
    def test_require_user_when_set(self):
        """Test requiring user when set"""
        user_id = uuid4()
        TenantContext.set_current_user(user_id)
        
        assert TenantContext.require_user() == user_id
    
    def test_require_user_when_not_set(self):
        """Test requiring user when not set raises error"""
        with pytest.raises(TenantContextError, match="No user set"):
            TenantContext.require_user()
    
    def test_clear_all(self):
        """Test clearing all context"""
        TenantContext.set_current_tenant(uuid4())
        TenantContext.set_current_user(uuid4())
        
        TenantContext.clear_all()
        
        assert TenantContext.get_current_tenant() is None
        assert TenantContext.get_current_user() is None


class TestTenantContextManager:
    """Test TenantContextManager context manager"""
    
    def setup_method(self):
        """Clear context before each test"""
        TenantContext.clear_all()
    
    def test_context_manager_sets_tenant(self):
        """Test context manager sets tenant"""
        tenant_id = uuid4()
        
        with TenantContextManager(tenant_id):
            assert TenantContext.get_current_tenant() == tenant_id
    
    def test_context_manager_restores_previous_tenant(self):
        """Test context manager restores previous tenant"""
        original_tenant = uuid4()
        new_tenant = uuid4()
        
        TenantContext.set_current_tenant(original_tenant)
        
        with TenantContextManager(new_tenant):
            assert TenantContext.get_current_tenant() == new_tenant
        
        assert TenantContext.get_current_tenant() == original_tenant
    
    def test_context_manager_sets_user(self):
        """Test context manager sets user"""
        tenant_id = uuid4()
        user_id = uuid4()
        
        with TenantContextManager(tenant_id, user_id):
            assert TenantContext.get_current_tenant() == tenant_id
            assert TenantContext.get_current_user() == user_id
    
    def test_context_manager_restores_on_exception(self):
        """Test context manager restores context on exception"""
        original_tenant = uuid4()
        new_tenant = uuid4()
        
        TenantContext.set_current_tenant(original_tenant)
        
        try:
            with TenantContextManager(new_tenant):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert TenantContext.get_current_tenant() == original_tenant
    
    def test_nested_context_managers(self):
        """Test nested context managers"""
        tenant1 = uuid4()
        tenant2 = uuid4()
        tenant3 = uuid4()
        
        with TenantContextManager(tenant1):
            assert TenantContext.get_current_tenant() == tenant1
            
            with TenantContextManager(tenant2):
                assert TenantContext.get_current_tenant() == tenant2
                
                with TenantContextManager(tenant3):
                    assert TenantContext.get_current_tenant() == tenant3
                
                assert TenantContext.get_current_tenant() == tenant2
            
            assert TenantContext.get_current_tenant() == tenant1


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def setup_method(self):
        """Clear context before each test"""
        TenantContext.clear_all()
    
    def test_get_tenant_id_function(self):
        """Test get_tenant_id convenience function"""
        tenant_id = uuid4()
        TenantContext.set_current_tenant(tenant_id)
        
        assert get_tenant_id() == tenant_id
    
    def test_require_tenant_id_function(self):
        """Test require_tenant_id convenience function"""
        tenant_id = uuid4()
        TenantContext.set_current_tenant(tenant_id)
        
        assert require_tenant_id() == tenant_id
    
    def test_require_tenant_id_raises_when_not_set(self):
        """Test require_tenant_id raises error when not set"""
        with pytest.raises(TenantContextError):
            require_tenant_id()
    
    def test_get_user_id_function(self):
        """Test get_user_id convenience function"""
        user_id = uuid4()
        TenantContext.set_current_user(user_id)
        
        assert get_user_id() == user_id
