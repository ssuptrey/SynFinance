"""
API Version Registry

Manages registration and lookup of API versions, including:
- Version metadata (release date, deprecation status, sunset date)
- Router associations for each version
- Version lifecycle tracking
- Breaking change documentation
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List, Any
from enum import Enum
import re


class VersionStatus(Enum):
    """Version lifecycle status"""
    ACTIVE = "active"  # Current supported version
    DEPRECATED = "deprecated"  # Still supported but discouraged
    SUNSET = "sunset"  # No longer supported
    BETA = "beta"  # Preview/testing only


@dataclass
class APIVersion:
    """
    Represents an API version with metadata and lifecycle information.
    
    Attributes:
        version: Version identifier (e.g., "v1", "v2")
        semantic_version: Full semantic version (e.g., "2.1.0")
        status: Current lifecycle status
        release_date: When this version was released
        deprecation_date: When deprecation was announced (if deprecated)
        sunset_date: When support will end (if deprecated)
        router: FastAPI router for this version
        description: Human-readable description
        breaking_changes: List of breaking changes from previous version
        migration_guide: URL or path to migration documentation
    """
    version: str  # e.g., "v1", "v2"
    semantic_version: str  # e.g., "1.0.0", "2.1.0"
    status: VersionStatus = VersionStatus.ACTIVE
    release_date: date = field(default_factory=date.today)
    deprecation_date: Optional[date] = None
    sunset_date: Optional[date] = None
    router: Optional[Any] = None
    description: str = ""
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    
    def __post_init__(self):
        """Validate version format"""
        if not re.match(r'^v\d+$', self.version):
            raise ValueError(f"Version must be in format 'v1', 'v2', etc. Got: {self.version}")
        
        # Allow semantic version with optional pre-release/build metadata
        if not re.match(r'^\d+\.\d+\.\d+(?:-[\w.]+)?(?:\+[\w.]+)?$', self.semantic_version):
            raise ValueError(f"Semantic version must be in format 'X.Y.Z' (with optional -prerelease+build). Got: {self.semantic_version}")
    
    @property
    def major_version(self) -> int:
        """Extract major version number"""
        return int(self.version[1:])
    
    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated"""
        return self.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)
    
    @property
    def is_active(self) -> bool:
        """Check if version is actively supported"""
        return self.status == VersionStatus.ACTIVE
    
    @property
    def is_sunset(self) -> bool:
        """Check if version has reached sunset"""
        if self.status == VersionStatus.SUNSET:
            return True
        if self.sunset_date and date.today() >= self.sunset_date:
            return True
        return False
    
    @property
    def days_until_sunset(self) -> Optional[int]:
        """Calculate days remaining until sunset"""
        if not self.sunset_date:
            return None
        delta = self.sunset_date - date.today()
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "version": self.version,
            "semantic_version": self.semantic_version,
            "status": self.status.value,
            "release_date": self.release_date.isoformat(),
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "description": self.description,
            "breaking_changes": self.breaking_changes,
            "migration_guide": self.migration_guide,
            "is_deprecated": self.is_deprecated,
            "is_sunset": self.is_sunset,
            "days_until_sunset": self.days_until_sunset,
        }


class VersionRegistry:
    """
    Central registry for managing API versions.
    
    Provides version registration, lookup, and lifecycle management.
    Implements singleton pattern for global access.
    """
    
    _instance: Optional['VersionRegistry'] = None
    _versions: Dict[str, APIVersion] = {}
    
    def __new__(cls):
        """Ensure singleton instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._versions = {}
        return cls._instance
    
    def register(self, version: APIVersion) -> None:
        """
        Register a new API version.
        
        Args:
            version: APIVersion instance to register
            
        Raises:
            ValueError: If version already registered
        """
        if version.version in self._versions:
            raise ValueError(f"Version {version.version} already registered")
        
        self._versions[version.version] = version
    
    def get(self, version: str) -> Optional[APIVersion]:
        """
        Get version by identifier.
        
        Args:
            version: Version identifier (e.g., "v1", "v2")
            
        Returns:
            APIVersion instance or None if not found
        """
        return self._versions.get(version)
    
    def list_all(self) -> List[APIVersion]:
        """
        Get all registered versions.
        
        Returns:
            List of all APIVersion instances, sorted by version number
        """
        return sorted(
            self._versions.values(),
            key=lambda v: v.major_version,
            reverse=True
        )
    
    def list_active(self) -> List[APIVersion]:
        """
        Get all active (non-deprecated, non-sunset) versions.
        
        Returns:
            List of active APIVersion instances
        """
        return [v for v in self.list_all() if v.is_active]
    
    def get_latest(self) -> Optional[APIVersion]:
        """
        Get the latest active version.
        
        Returns:
            Latest APIVersion instance or None if no versions registered
        """
        active = self.list_active()
        return active[0] if active else None
    
    def is_deprecated(self, version: str) -> bool:
        """
        Check if a version is deprecated.
        
        Args:
            version: Version identifier
            
        Returns:
            True if deprecated, False otherwise
        """
        v = self.get(version)
        return v.is_deprecated if v else False
    
    def update_status(self, version: str, status: VersionStatus, sunset_date: Optional[date] = None) -> None:
        """
        Update version status.
        
        Args:
            version: Version identifier
            status: New status
            sunset_date: Optional sunset date for deprecated versions
        """
        v = self.get(version)
        if not v:
            raise ValueError(f"Version {version} not found")
        
        v.status = status
        
        if status == VersionStatus.DEPRECATED and not v.deprecation_date:
            v.deprecation_date = date.today()
        
        if sunset_date:
            v.sunset_date = sunset_date
    
    def clear(self) -> None:
        """Clear all registered versions (mainly for testing)"""
        self._versions.clear()


# Global registry instance
_registry = VersionRegistry()


# Convenience functions for common operations

def register_version(
    version: str,
    semantic_version: str,
    router: Optional[Any] = None,
    description: str = "",
    status: VersionStatus = VersionStatus.ACTIVE,
    release_date: Optional[date] = None,
    breaking_changes: Optional[List[str]] = None,
    migration_guide: Optional[str] = None,
) -> APIVersion:
    """
    Register a new API version.
    
    Args:
        version: Version identifier (e.g., "v1", "v2")
        semantic_version: Semantic version (e.g., "1.0.0")
        router: FastAPI router for this version
        description: Human-readable description
        status: Initial status
        release_date: Release date (defaults to today)
        breaking_changes: List of breaking changes
        migration_guide: Path to migration documentation
        
    Returns:
        Registered APIVersion instance
    """
    api_version = APIVersion(
        version=version,
        semantic_version=semantic_version,
        router=router,
        description=description,
        status=status,
        release_date=release_date or date.today(),
        breaking_changes=breaking_changes or [],
        migration_guide=migration_guide,
    )
    
    _registry.register(api_version)
    return api_version


def get_version(version: str) -> Optional[APIVersion]:
    """Get version by identifier"""
    return _registry.get(version)


def list_versions() -> List[APIVersion]:
    """Get all registered versions"""
    return _registry.list_all()


def get_latest_version() -> Optional[APIVersion]:
    """Get the latest active version"""
    return _registry.get_latest()


def is_version_deprecated(version: str) -> bool:
    """Check if version is deprecated"""
    return _registry.is_deprecated(version)


def get_registry() -> VersionRegistry:
    """Get the global version registry"""
    return _registry
