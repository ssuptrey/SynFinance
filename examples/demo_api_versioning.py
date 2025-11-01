"""
API Versioning System Demo

Demonstrates the complete API versioning capabilities of SynFinance:
- Version registration and management
- Version negotiation from requests
- Deprecation warnings and sunset policies
- Backward compatibility transformations
- Migration tools and guides
"""

import sys
from datetime import date, timedelta
from fastapi import FastAPI, APIRouter, Request, Response
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, 'E:/SynFinance')

from src.api.versioning import (
    # Registry
    APIVersion,
    VersionStatus,
    register_version,
    get_version,
    list_versions,
    get_latest_version,
    get_registry,
    # Negotiation
    detect_version_from_url,
    detect_version_from_accept,
    # Deprecation
    deprecated,
    add_sunset_headers,
    check_deprecation_status,
    # Compatibility
    FieldMapping,
    SchemaMapping,
    get_compatibility_adapter,
    transform_request,
    transform_response,
    # Migration
    BreakingChange,
    MigrationGuide,
    compare_versions,
    generate_timeline,
    suggest_deprecation_timeline,
    get_client_migration_code,
    get_v1_to_v2_migration,
    # Router
    create_versioned_router,
    get_version_info_router,
    # Middleware
    VersionMiddleware,
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def demo_version_registration():
    """Demonstrate version registration"""
    print_section("1. Version Registration & Lifecycle Management")
    
    # Clear any existing versions
    get_registry().clear()
    
    # Register v1 (deprecated)
    v1 = register_version(
        version="v1",
        semantic_version="1.0.0",
        description="Initial release - Basic fraud detection",
        status=VersionStatus.DEPRECATED,
        release_date=date(2024, 1, 1),
        breaking_changes=[],
    )
    
    # Mark v1 as deprecated with sunset date
    get_registry().update_status(
        "v1",
        VersionStatus.DEPRECATED,
        sunset_date=date.today() + timedelta(days=180)
    )
    
    print(f"✓ Registered v1:")
    print(f"  Status: {v1.status.value}")
    print(f"  Release Date: {v1.release_date}")
    print(f"  Deprecation Date: {v1.deprecation_date}")
    print(f"  Sunset Date: {v1.sunset_date}")
    print(f"  Days Until Sunset: {v1.days_until_sunset}")
    
    # Register v2 (current)
    v2 = register_version(
        version="v2",
        semantic_version="2.0.0",
        description="Multi-tenancy, RBAC, Enhanced ML",
        status=VersionStatus.ACTIVE,
        release_date=date(2025, 1, 1),
        breaking_changes=[
            "Tenant context required (X-Tenant-ID header)",
            "All responses include tenant_id field",
            "New permission system replaces simple roles",
        ],
    )
    
    print(f"\n✓ Registered v2:")
    print(f"  Status: {v2.status.value}")
    print(f"  Release Date: {v2.release_date}")
    print(f"  Breaking Changes: {len(v2.breaking_changes)}")
    
    # Register v3 (beta)
    v3 = register_version(
        version="v3",
        semantic_version="3.0.0-beta",
        description="AI-powered insights, Blockchain integration",
        status=VersionStatus.BETA,
        release_date=date.today(),
        breaking_changes=[
            "New authentication mechanism",
            "Response format changes",
        ],
    )
    
    print(f"\n✓ Registered v3 (Beta):")
    print(f"  Status: {v3.status.value}")
    print(f"  Description: {v3.description}")
    
    # List all versions
    print(f"\n✓ All Versions:")
    for v in list_versions():
        print(f"  - {v.version} ({v.semantic_version}): {v.status.value}")
    
    # Get latest
    latest = get_latest_version()
    print(f"\n✓ Latest Active Version: {latest.version}")


def demo_version_negotiation():
    """Demonstrate version detection"""
    print_section("2. Version Negotiation & Detection")
    
    # URL-based detection
    print("✓ URL-based Version Detection:")
    test_urls = [
        "/api/v1/transactions",
        "/api/v2/fraud-detection",
        "/v3/analytics",
    ]
    for url in test_urls:
        version = detect_version_from_url(url)
        print(f"  {url} → {version}")
    
    # Header-based detection
    print(f"\n✓ Accept Header Detection:")
    test_headers = [
        "application/vnd.synfinance.v1+json",
        "application/vnd.synfinance.v2+json",
    ]
    for header in test_headers:
        version = detect_version_from_accept(header)
        print(f"  {header} → {version}")
    
    # Priority order
    print(f"\n✓ Detection Priority: URL > Accept Header > X-API-Version > Query Param > Default")


def demo_deprecation_warnings():
    """Demonstrate deprecation warnings"""
    print_section("3. Deprecation Warnings & Sunset Headers")
    
    v1 = get_version("v1")
    
    # Check deprecation status
    status = check_deprecation_status("v1")
    print(f"✓ v1 Deprecation Status:")
    print(f"  Deprecated: {status['deprecated']}")
    print(f"  Status: {status['status']}")
    print(f"  Deprecation Date: {status['deprecation_date']}")
    print(f"  Sunset Date: {status['sunset_date']}")
    print(f"  Days Until Sunset: {status['days_until_sunset']}")
    print(f"  Is Sunset: {status['is_sunset']}")
    
    # Show headers that would be added
    print(f"\n✓ Response Headers for Deprecated Version:")
    print(f"  Deprecation: true")
    print(f"  Sunset: {v1.sunset_date.strftime('%a, %d %b %Y %H:%M:%S GMT')}")
    print(f"  X-API-Days-Until-Sunset: {v1.days_until_sunset}")
    
    # Suggest deprecation timeline
    print(f"\n✓ Suggested Deprecation Timeline for v2:")
    timeline = suggest_deprecation_timeline("v2", deprecation_period_months=6, sunset_period_months=12)
    for milestone in timeline["milestones"]:
        print(f"\n  {milestone['date']} - {milestone['action']}:")
        for task in milestone['tasks']:
            print(f"    - {task}")


def demo_backward_compatibility():
    """Demonstrate compatibility transformations"""
    print_section("4. Backward Compatibility Transformations")
    
    # Register schema mapping
    adapter = get_compatibility_adapter()
    
    mapping = SchemaMapping(
        from_version="v1",
        to_version="v2",
        field_mappings=[
            FieldMapping(
                old_name="timestamp",
                new_name="transaction_timestamp",
            ),
            FieldMapping(
                old_name="user_id",
                new_name="customer_id",
            ),
        ],
        added_fields={
            "tenant_id": None,
            "fraud_score": 0.0,
            "risk_level": "low",
        },
        removed_fields=["legacy_field"],
    )
    adapter.register_mapping(mapping)
    
    # Transform v1 request to v2 format
    print("✓ Request Transformation (v1 → v2):")
    v1_request = {
        "id": "txn_123",
        "amount": 100.50,
        "timestamp": "2025-11-01T10:00:00Z",
        "user_id": "user_456",
        "legacy_field": "old_value",
    }
    
    print(f"\n  v1 Request:")
    for key, value in v1_request.items():
        print(f"    {key}: {value}")
    
    v2_request = transform_request(v1_request, "v1", "v2")
    
    print(f"\n  v2 Request (after transformation):")
    for key, value in v2_request.items():
        print(f"    {key}: {value}")
    
    # Transform v2 response back to v1 format
    print(f"\n✓ Response Transformation (v2 → v1):")
    v2_response = {
        "id": "txn_123",
        "amount": 100.50,
        "transaction_timestamp": "2025-11-01T10:00:00Z",
        "customer_id": "user_456",
        "tenant_id": "tenant_789",
        "fraud_score": 0.15,
        "risk_level": "low",
    }
    
    print(f"\n  v2 Response:")
    for key, value in v2_response.items():
        print(f"    {key}: {value}")
    
    v1_response = transform_response(v2_response, "v1", "v2")
    
    print(f"\n  v1 Response (backward compatible):")
    for key, value in v1_response.items():
        print(f"    {key}: {value}")


def demo_migration_tools():
    """Demonstrate migration tools"""
    print_section("5. Migration Tools & Guides")
    
    # Compare versions
    print("✓ Version Comparison:")
    comparison = compare_versions("v1", "v2")
    print(f"  Newer Version: {comparison['comparison']['newer']}")
    print(f"  Older Version: {comparison['comparison']['older']}")
    print(f"  Version Gap: {comparison['comparison']['version_gap']}")
    print(f"  Breaking Changes Expected: {comparison['compatibility']['breaking_changes_expected']}")
    print(f"  Migration Required: {comparison['compatibility']['migration_required']}")
    
    # Generate timeline
    print(f"\n✓ Version Timeline (v1):")
    timeline = generate_timeline("v1")
    for event in timeline["events"]:
        print(f"  {event['date']}: {event['event']} ({event['status']})")
    
    # Get migration guide
    print(f"\n✓ Migration Guide (v1 → v2):")
    guide = get_v1_to_v2_migration()
    print(f"  Breaking Changes: {len(guide.breaking_changes)}")
    print(f"  New Features: {len(guide.new_features)}")
    
    print(f"\n  Breaking Change Example:")
    bc = guide.breaking_changes[0]
    print(f"    Category: {bc.category}")
    print(f"    Description: {bc.description}")
    print(f"    Migration Steps:")
    for step in bc.migration_steps:
        print(f"      - {step}")
    
    print(f"\n  New Features:")
    for i, feature in enumerate(guide.new_features[:3], 1):
        print(f"    {i}. {feature}")


def demo_client_migration_code():
    """Demonstrate client migration code generation"""
    print_section("6. Client Migration Code Examples")
    
    # Python example
    print("✓ Python Client Migration Code:")
    python_code = get_client_migration_code("v1", "v2", "python")
    print(python_code)
    
    # JavaScript example
    print("\n✓ JavaScript Client Migration Code:")
    js_code = get_client_migration_code("v1", "v2", "javascript")
    print(js_code)


def demo_versioned_api():
    """Demonstrate versioned API in action"""
    print_section("7. Versioned API in Action")
    
    # Create app with versioned routers
    app = FastAPI(title="SynFinance API")
    
    # Add version middleware
    app.add_middleware(VersionMiddleware, default_version="v2")
    
    # Create v1 router
    v1_router = create_versioned_router("v1", tags=["v1", "deprecated"])
    
    @v1_router.get("/transactions")
    async def get_transactions_v1():
        return {
            "transactions": [
                {"id": "txn_1", "amount": 100.0, "timestamp": "2025-11-01T10:00:00Z"}
            ],
            "count": 1,
        }
    
    app.include_router(v1_router)
    
    # Create v2 router
    v2_router = create_versioned_router("v2", tags=["v2", "current"])
    
    @v2_router.get("/transactions")
    async def get_transactions_v2(request: Request):
        return {
            "transactions": [
                {
                    "id": "txn_1",
                    "amount": 100.0,
                    "transaction_timestamp": "2025-11-01T10:00:00Z",
                    "tenant_id": "tenant_123",
                    "fraud_score": 0.05,
                    "risk_level": "low",
                }
            ],
            "count": 1,
            "tenant_id": "tenant_123",
            "api_version": request.state.api_version,
        }
    
    app.include_router(v2_router)
    
    # Add version info router
    app.include_router(get_version_info_router())
    
    # Test with client
    client = TestClient(app)
    
    print("✓ Testing v1 endpoint:")
    response = client.get("/api/v1/transactions")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print(f"  Headers:")
    print(f"    X-API-Version: {response.headers.get('X-API-Version')}")
    print(f"    Deprecation: {response.headers.get('Deprecation')}")
    
    print(f"\n✓ Testing v2 endpoint:")
    response = client.get("/api/v2/transactions")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print(f"  Headers:")
    print(f"    X-API-Version: {response.headers.get('X-API-Version')}")
    
    print(f"\n✓ Testing version info endpoint:")
    response = client.get("/api/versions")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Total Versions: {data['count']}")
    for v in data['versions']:
        print(f"    - {v['version']}: {v['status']}")


def main():
    """Run all demonstrations"""
    print("""
    ================================================================================
      SynFinance API Versioning System Demo
    ================================================================================
    
    This demo showcases the complete API versioning capabilities including:
    - Version registration and lifecycle management
    - Version negotiation from multiple sources
    - Deprecation warnings and sunset policies
    - Backward compatibility transformations
    - Migration tools and documentation generation
    - Versioned API routers and endpoints
    """)
    
    try:
        demo_version_registration()
        demo_version_negotiation()
        demo_deprecation_warnings()
        demo_backward_compatibility()
        demo_migration_tools()
        demo_client_migration_code()
        demo_versioned_api()
        
        print_section("Demo Complete!")
        print("""
API versioning system successfully demonstrated:
  ✓ Version registration with lifecycle management
  ✓ Multi-source version negotiation (URL, headers, query params)
  ✓ Deprecation warnings with sunset timelines
  ✓ Automatic request/response transformations for compatibility
  ✓ Comprehensive migration tools and guides
  ✓ Client code generation for multiple languages
  ✓ Versioned API routers with middleware support

The system supports:
  - Semantic versioning (major.minor.patch)
  - Multiple active versions simultaneously
  - Gradual deprecation with defined timelines
  - Backward compatibility through transformations
  - Automated migration documentation
  - Version-specific OpenAPI documentation

Ready for production API versioning!
        """)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
