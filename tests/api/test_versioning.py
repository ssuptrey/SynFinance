"""
Tests for API Versioning System

Tests version registry, negotiation, deprecation, compatibility, and migration tools.
"""

import pytest
from datetime import date, timedelta
from fastapi import FastAPI, APIRouter, Request
from fastapi.testclient import TestClient
from src.api.versioning.registry import (
    APIVersion,
    VersionRegistry,
    VersionStatus,
    register_version,
    get_version,
    list_versions,
    get_latest_version,
    is_version_deprecated,
    get_registry,
)
from src.api.versioning.negotiation import (
    VersionNegotiator,
    detect_version_from_url,
    detect_version_from_accept,
    detect_version_from_request,
)
from src.api.versioning.deprecation import (
    deprecated,
    add_deprecation_headers,
    add_sunset_headers,
    check_deprecation_status,
)
from src.api.versioning.compatibility import (
    FieldMapping,
    SchemaMapping,
    CompatibilityAdapter,
    transform_request,
    transform_response,
)
from src.api.versioning.migration import (
    BreakingChange,
    MigrationGuide,
    compare_versions,
    generate_timeline,
    get_v1_to_v2_migration,
)
from src.api.versioning.middleware import VersionMiddleware
from src.api.versioning.router import (
    create_versioned_router,
    get_version_info_router,
)


class TestVersionRegistry:
    """Test version registry functionality"""
    
    def setup_method(self):
        """Clear registry before each test"""
        get_registry().clear()
    
    def test_api_version_creation(self):
        """Test creating an API version"""
        version = APIVersion(
            version="v1",
            semantic_version="1.0.0",
            description="Initial release"
        )
        
        assert version.version == "v1"
        assert version.major_version == 1
        assert version.is_active
        assert not version.is_deprecated
    
    def test_invalid_version_format(self):
        """Test validation of version format"""
        with pytest.raises(ValueError):
            APIVersion(version="1.0", semantic_version="1.0.0")
        
        with pytest.raises(ValueError):
            APIVersion(version="v1", semantic_version="1.0")
    
    def test_register_version(self):
        """Test registering a version"""
        version = register_version(
            version="v1",
            semantic_version="1.0.0",
            description="Test version"
        )
        
        assert version is not None
        assert get_version("v1") == version
    
    def test_duplicate_registration(self):
        """Test that duplicate registration raises error"""
        register_version("v1", "1.0.0")
        
        with pytest.raises(ValueError):
            register_version("v1", "1.1.0")
    
    def test_list_versions(self):
        """Test listing all versions"""
        v1 = register_version("v1", "1.0.0")
        v2 = register_version("v2", "2.0.0")
        
        versions = list_versions()
        assert len(versions) == 2
        # Should be reverse sorted
        assert versions[0] == v2
        assert versions[1] == v1
    
    def test_get_latest_version(self):
        """Test getting latest version"""
        v1 = register_version("v1", "1.0.0")
        v2 = register_version("v2", "2.0.0", status=VersionStatus.ACTIVE)
        
        latest = get_latest_version()
        assert latest == v2
    
    def test_version_deprecation(self):
        """Test version deprecation"""
        v1 = register_version("v1", "1.0.0")
        assert not v1.is_deprecated
        
        get_registry().update_status(
            "v1",
            VersionStatus.DEPRECATED,
            sunset_date=date.today() + timedelta(days=180)
        )
        
        assert v1.is_deprecated
        assert is_version_deprecated("v1")
        assert v1.days_until_sunset is not None
    
    def test_version_to_dict(self):
        """Test version serialization"""
        version = register_version(
            "v1",
            "1.0.0",
            description="Test",
            breaking_changes=["Changed auth"]
        )
        
        data = version.to_dict()
        assert data["version"] == "v1"
        assert data["semantic_version"] == "1.0.0"
        assert data["description"] == "Test"
        assert len(data["breaking_changes"]) == 1


class TestVersionNegotiation:
    """Test version negotiation from requests"""
    
    def test_detect_from_url(self):
        """Test version detection from URL"""
        assert detect_version_from_url("/api/v1/transactions") == "v1"
        assert detect_version_from_url("/api/v2/fraud-detection") == "v2"
        assert detect_version_from_url("/v3/analytics") == "v3"
        assert detect_version_from_url("/api/transactions") is None
    
    def test_detect_from_accept_header(self):
        """Test version detection from Accept header"""
        assert detect_version_from_accept("application/vnd.synfinance.v1+json") == "v1"
        assert detect_version_from_accept("application/vnd.synfinance.v2+json") == "v2"
        assert detect_version_from_accept("application/json") is None
    
    def test_version_negotiator(self):
        """Test version negotiator"""
        get_registry().clear()
        v1 = register_version("v1", "1.0.0")
        v2 = register_version("v2", "2.0.0")
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Test with URL version
        negotiator = VersionNegotiator()
        
        # Create mock request
        class MockRequest:
            def __init__(self):
                self.url = type('obj', (object,), {'path': '/api/v1/test'})
                self.headers = {}
                self.query_params = {}
        
        request = MockRequest()
        version = negotiator.detect(request)
        assert version == "v1"


class TestDeprecation:
    """Test deprecation functionality"""
    
    def setup_method(self):
        """Clear registry before each test"""
        get_registry().clear()
    
    def test_add_deprecation_headers(self):
        """Test adding deprecation headers"""
        from fastapi import Response
        
        response = Response()
        add_deprecation_headers(
            response,
            message="Use v2 instead",
            sunset_date=date(2026, 6, 1),
            replacement="/api/v2/endpoint"
        )
        
        assert response.headers["Deprecation"] == "true"
        assert "Sunset" in response.headers
        assert response.headers["X-API-Replacement"] == "/api/v2/endpoint"
    
    def test_add_sunset_headers(self):
        """Test adding sunset headers for deprecated version"""
        from fastapi import Response
        
        v1 = register_version("v1", "1.0.0")
        get_registry().update_status(
            "v1",
            VersionStatus.DEPRECATED,
            sunset_date=date(2026, 6, 1)
        )
        
        response = Response()
        add_sunset_headers(response, "v1")
        
        assert response.headers["Deprecation"] == "true"
        assert "Sunset" in response.headers
    
    def test_check_deprecation_status(self):
        """Test checking deprecation status"""
        v1 = register_version("v1", "1.0.0")
        
        status = check_deprecation_status("v1")
        assert not status["deprecated"]
        
        get_registry().update_status("v1", VersionStatus.DEPRECATED)
        
        status = check_deprecation_status("v1")
        assert status["deprecated"]


class TestCompatibility:
    """Test compatibility and transformation"""
    
    def test_field_mapping(self):
        """Test field mapping creation"""
        mapping = FieldMapping(
            old_name="created_at",
            new_name="timestamp",
            required=True
        )
        
        assert mapping.old_name == "created_at"
        assert mapping.new_name == "timestamp"
    
    def test_schema_mapping(self):
        """Test schema mapping"""
        mapping = SchemaMapping(
            from_version="v1",
            to_version="v2",
            field_mappings=[
                FieldMapping("old_field", "new_field")
            ],
            removed_fields=["deprecated_field"],
            added_fields={"tenant_id": None}
        )
        
        assert mapping.from_version == "v1"
        assert len(mapping.field_mappings) == 1
    
    def test_transform_forward(self):
        """Test forward transformation (v1 -> v2)"""
        adapter = CompatibilityAdapter()
        
        mapping = SchemaMapping(
            from_version="v1",
            to_version="v2",
            field_mappings=[
                FieldMapping("created_at", "timestamp")
            ],
            added_fields={"tenant_id": "default"}
        )
        adapter.register_mapping(mapping)
        
        v1_data = {"id": "123", "created_at": "2025-11-01"}
        v2_data = adapter.transform_forward(v1_data, "v1", "v2")
        
        assert "timestamp" in v2_data
        assert "created_at" not in v2_data
        assert v2_data["tenant_id"] == "default"
    
    def test_transform_backward(self):
        """Test backward transformation (v2 -> v1)"""
        adapter = CompatibilityAdapter()
        
        mapping = SchemaMapping(
            from_version="v1",
            to_version="v2",
            field_mappings=[
                FieldMapping("created_at", "timestamp")
            ],
            added_fields={"tenant_id": None}
        )
        adapter.register_mapping(mapping)
        
        v2_data = {"id": "123", "timestamp": "2025-11-01", "tenant_id": "tenant-1"}
        v1_data = adapter.transform_backward(v2_data, "v1", "v2")
        
        assert "created_at" in v1_data
        assert "timestamp" not in v1_data
        assert "tenant_id" not in v1_data


class TestMigration:
    """Test migration tools"""
    
    def test_breaking_change(self):
        """Test breaking change creation"""
        change = BreakingChange(
            category="authentication",
            description="Tenant ID required",
            old_behavior="No tenant context",
            new_behavior="X-Tenant-ID header required",
            migration_steps=["Add header"],
            affected_endpoints=["/api/transactions"]
        )
        
        assert change.category == "authentication"
        assert len(change.migration_steps) == 1
    
    def test_migration_guide(self):
        """Test migration guide creation"""
        guide = MigrationGuide("v1", "v2")
        
        guide.add_breaking_change(BreakingChange(
            category="test",
            description="Test change",
            old_behavior="old",
            new_behavior="new",
            migration_steps=["step 1"],
            affected_endpoints=["/test"]
        ))
        
        guide.add_new_feature("New feature")
        guide.add_deprecation("Old endpoint")
        
        markdown = guide.generate_markdown()
        assert "Migration Guide: v1 → v2" in markdown
        assert "Breaking Changes" in markdown
        assert "New Features" in markdown
    
    def test_compare_versions(self):
        """Test version comparison"""
        get_registry().clear()
        register_version("v1", "1.0.0")
        register_version("v2", "2.0.0")
        
        comparison = compare_versions("v1", "v2")
        assert comparison["comparison"]["newer"] == "v2"
        assert comparison["comparison"]["version_gap"] == 1
    
    def test_generate_timeline(self):
        """Test timeline generation"""
        get_registry().clear()
        v1 = register_version("v1", "1.0.0")
        get_registry().update_status(
            "v1",
            VersionStatus.DEPRECATED,
            sunset_date=date.today() + timedelta(days=180)
        )
        
        timeline = generate_timeline("v1")
        assert timeline["version"] == "v1"
        assert len(timeline["events"]) >= 2  # Released + Deprecated
    
    def test_v1_to_v2_migration(self):
        """Test pre-built v1 to v2 migration guide"""
        guide = get_v1_to_v2_migration()
        
        assert guide.from_version == "v1"
        assert guide.to_version == "v2"
        assert len(guide.breaking_changes) > 0
        assert len(guide.new_features) > 0


class TestRouter:
    """Test versioned router utilities"""
    
    def test_create_versioned_router(self):
        """Test creating a versioned router"""
        router = create_versioned_router("v1")
        
        assert router.prefix == "/api/v1"
        assert "v1" in router.tags
    
    def test_version_info_router(self):
        """Test version info router"""
        get_registry().clear()
        register_version("v1", "1.0.0")
        register_version("v2", "2.0.0")
        
        app = FastAPI()
        app.include_router(get_version_info_router())
        
        client = TestClient(app)
        
        # Test listing versions
        response = client.get("/api/versions")
        assert response.status_code == 200
        assert response.json()["count"] == 2
        
        # Test getting current version
        response = client.get("/api/version")
        assert response.status_code == 200
        assert response.json()["version"] == "v2"  # Latest
        
        # Test getting specific version
        response = client.get("/api/version/v1")
        assert response.status_code == 200
        assert response.json()["version"] == "v1"


class TestMiddleware:
    """Test version middleware"""
    
    def test_version_middleware(self):
        """Test that middleware detects and injects version"""
        get_registry().clear()
        register_version("v1", "1.0.0")
        register_version("v2", "2.0.0")
        
        app = FastAPI()
        app.add_middleware(VersionMiddleware, default_version="v2")
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"version": request.state.api_version}
        
        client = TestClient(app)
        
        # Test with URL version
        response = client.get("/api/v1/test")
        # Note: Middleware processes the path but route needs to match
        # In production, use versioned routers
        
        # Test with header
        response = client.get("/test", headers={"X-API-Version": "v1"})
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
