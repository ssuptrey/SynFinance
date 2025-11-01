"""
API Versioning System for SynFinance

This module provides comprehensive API versioning capabilities including:
- Version registration and management
- URL-based and header-based version detection
- Backward compatibility layer
- Deprecation warnings and sunset policies
- Migration tools and utilities

Version Format:
- Public API: v1, v2, v3 (major versions)
- Internal: 1.0.0, 1.1.0, 2.0.0 (semantic versioning)

Example Usage:
    from src.api.versioning import APIVersion, register_version
    
    # Register a new API version
    register_version(
        version="v2",
        router=v2_router,
        deprecated=False,
        sunset_date=None
    )
    
    # Detect version from request
    version = APIVersion.from_request(request)
"""

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
    detect_version_from_request,
    detect_version_from_url,
    detect_version_from_header,
    detect_version_from_accept,
)

from src.api.versioning.deprecation import (
    deprecated,
    deprecate_field,
    DeprecationWarning as APIDeprecationWarning,
    add_sunset_headers,
    check_deprecation_status,
)

from src.api.versioning.compatibility import (
    FieldMapping,
    SchemaMapping,
    CompatibilityAdapter,
    RequestTransformer,
    ResponseTransformer,
    transform_request,
    transform_response,
    get_compatibility_adapter,
)

from src.api.versioning.middleware import (
    VersionMiddleware,
    create_version_middleware,
)

from src.api.versioning.router import (
    create_versioned_router,
    mount_versioned_routers,
    create_latest_router_alias,
    get_version_info_router,
)

from src.api.versioning.migration import (
    BreakingChange,
    MigrationGuide,
    compare_versions,
    generate_timeline,
    suggest_deprecation_timeline,
    get_client_migration_code,
    get_v1_to_v2_migration,
)

__all__ = [
    # Registry
    "APIVersion",
    "VersionRegistry",
    "VersionStatus",
    "register_version",
    "get_version",
    "list_versions",
    "get_latest_version",
    "is_version_deprecated",
    "get_registry",
    # Negotiation
    "VersionNegotiator",
    "detect_version_from_request",
    "detect_version_from_url",
    "detect_version_from_header",
    "detect_version_from_accept",
    # Deprecation
    "deprecated",
    "deprecate_field",
    "APIDeprecationWarning",
    "add_sunset_headers",
    "check_deprecation_status",
    # Compatibility
    "FieldMapping",
    "SchemaMapping",
    "CompatibilityAdapter",
    "RequestTransformer",
    "ResponseTransformer",
    "transform_request",
    "transform_response",
    "get_compatibility_adapter",
    # Middleware
    "VersionMiddleware",
    "create_version_middleware",
    # Router
    "create_versioned_router",
    "mount_versioned_routers",
    "create_latest_router_alias",
    "get_version_info_router",
    # Migration
    "BreakingChange",
    "MigrationGuide",
    "compare_versions",
    "generate_timeline",
    "suggest_deprecation_timeline",
    "get_client_migration_code",
    "get_v1_to_v2_migration",
]
