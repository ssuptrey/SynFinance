"""
Versioned API Router Utilities

Helper functions for creating version-specific API routers.
"""

from typing import Optional, List
from fastapi import APIRouter, FastAPI
from src.api.versioning.registry import APIVersion, get_version


def create_versioned_router(
    version: str,
    prefix: str = "",
    tags: Optional[List[str]] = None,
    **kwargs,
) -> APIRouter:
    """
    Create a router for a specific API version.
    
    Args:
        version: Version identifier (e.g., "v1", "v2")
        prefix: Router prefix (default: /api/{version})
        tags: OpenAPI tags
        **kwargs: Additional router arguments
        
    Returns:
        Configured APIRouter
    """
    if not prefix:
        prefix = f"/api/{version}"
    
    # Add version to tags
    version_tags = tags or []
    if version not in version_tags:
        version_tags.append(version)
    
    return APIRouter(
        prefix=prefix,
        tags=version_tags,
        **kwargs
    )


def mount_versioned_routers(
    app: FastAPI,
    routers: dict[str, APIRouter],
) -> None:
    """
    Mount multiple versioned routers to the app.
    
    Args:
        app: FastAPI application
        routers: Dictionary mapping version to router
        
    Example:
        mount_versioned_routers(app, {
            "v1": v1_router,
            "v2": v2_router,
        })
    """
    for version, router in routers.items():
        api_version = get_version(version)
        if api_version:
            # Update router to use APIVersion's router if available
            actual_router = api_version.router or router
            app.include_router(actual_router)
        else:
            app.include_router(router)


def create_latest_router_alias(
    app: FastAPI,
    latest_version: str,
) -> None:
    """
    Create /api/latest/ alias pointing to the latest version.
    
    Args:
        app: FastAPI application
        latest_version: Latest version identifier
    """
    api_version = get_version(latest_version)
    if api_version and api_version.router:
        # Create alias router
        alias_router = APIRouter(prefix="/api/latest", tags=["latest"])
        
        # Copy routes from latest version router
        for route in api_version.router.routes:
            alias_router.routes.append(route)
        
        app.include_router(alias_router)


def get_version_info_router() -> APIRouter:
    """
    Create a router that provides version information endpoints.
    
    Returns:
        APIRouter with version info endpoints
    """
    from src.api.versioning.registry import list_versions, get_latest_version
    
    router = APIRouter(prefix="/api", tags=["versioning"])
    
    @router.get("/versions")
    async def list_api_versions():
        """List all available API versions"""
        versions = list_versions()
        return {
            "versions": [v.to_dict() for v in versions],
            "count": len(versions),
        }
    
    @router.get("/version")
    async def get_current_version():
        """Get current/latest API version"""
        latest = get_latest_version()
        if not latest:
            return {"error": "No versions registered"}
        
        return latest.to_dict()
    
    @router.get("/version/{version}")
    async def get_version_details(version: str):
        """Get details for a specific version"""
        api_version = get_version(version)
        if not api_version:
            return {"error": f"Version {version} not found"}
        
        return api_version.to_dict()
    
    return router
