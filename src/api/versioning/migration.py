"""
API Migration Tools

Utilities for managing API version migrations:
- Migration script generation
- Breaking change detection
- Client migration helpers
- Version comparison
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import date, timedelta
from src.api.versioning.registry import APIVersion, get_version, list_versions


@dataclass
class BreakingChange:
    """
    Represents a breaking change between API versions.
    
    Attributes:
        category: Type of change (field_removed, field_renamed, type_changed, etc.)
        description: Human-readable description
        old_behavior: How it worked in old version
        new_behavior: How it works in new version
        migration_steps: Steps to migrate
        affected_endpoints: List of affected endpoints
    """
    category: str
    description: str
    old_behavior: str
    new_behavior: str
    migration_steps: List[str]
    affected_endpoints: List[str]


class MigrationGuide:
    """
    Generates migration guides between API versions.
    """
    
    def __init__(self, from_version: str, to_version: str):
        """
        Initialize migration guide.
        
        Args:
            from_version: Source version
            to_version: Target version
        """
        self.from_version = from_version
        self.to_version = to_version
        self.breaking_changes: List[BreakingChange] = []
        self.new_features: List[str] = []
        self.deprecations: List[str] = []
    
    def add_breaking_change(self, change: BreakingChange) -> None:
        """Add a breaking change to the guide"""
        self.breaking_changes.append(change)
    
    def add_new_feature(self, feature: str) -> None:
        """Add a new feature description"""
        self.new_features.append(feature)
    
    def add_deprecation(self, deprecation: str) -> None:
        """Add a deprecation notice"""
        self.deprecations.append(deprecation)
    
    def generate_markdown(self) -> str:
        """
        Generate migration guide as markdown.
        
        Returns:
            Markdown-formatted migration guide
        """
        lines = [
            f"# Migration Guide: {self.from_version} → {self.to_version}",
            "",
            f"This guide will help you migrate from API {self.from_version} to {self.to_version}.",
            "",
        ]
        
        # Breaking changes
        if self.breaking_changes:
            lines.extend([
                "## Breaking Changes",
                "",
                "⚠️ The following changes require code updates:",
                "",
            ])
            
            for i, change in enumerate(self.breaking_changes, 1):
                lines.extend([
                    f"### {i}. {change.description}",
                    "",
                    f"**Category:** {change.category}",
                    "",
                    f"**Old Behavior ({self.from_version}):**",
                    f"```",
                    change.old_behavior,
                    f"```",
                    "",
                    f"**New Behavior ({self.to_version}):**",
                    f"```",
                    change.new_behavior,
                    f"```",
                    "",
                    "**Migration Steps:**",
                ])
                
                for step in change.migration_steps:
                    lines.append(f"1. {step}")
                
                lines.append("")
                
                if change.affected_endpoints:
                    lines.append("**Affected Endpoints:**")
                    for endpoint in change.affected_endpoints:
                        lines.append(f"- `{endpoint}`")
                    lines.append("")
        
        # New features
        if self.new_features:
            lines.extend([
                "## New Features",
                "",
                "✨ New capabilities in this version:",
                "",
            ])
            
            for feature in self.new_features:
                lines.append(f"- {feature}")
            
            lines.append("")
        
        # Deprecations
        if self.deprecations:
            lines.extend([
                "## Deprecations",
                "",
                "⚠️ The following features are deprecated and will be removed in a future version:",
                "",
            ])
            
            for deprecation in self.deprecations:
                lines.append(f"- {deprecation}")
            
            lines.append("")
        
        # Migration checklist
        lines.extend([
            "## Migration Checklist",
            "",
            "- [ ] Review all breaking changes above",
            "- [ ] Update client code to handle new request/response formats",
            "- [ ] Test all affected endpoints",
            "- [ ] Update API version in client configuration",
            "- [ ] Deploy and monitor for errors",
            "",
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert migration guide to dictionary"""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "breaking_changes": [
                {
                    "category": bc.category,
                    "description": bc.description,
                    "old_behavior": bc.old_behavior,
                    "new_behavior": bc.new_behavior,
                    "migration_steps": bc.migration_steps,
                    "affected_endpoints": bc.affected_endpoints,
                }
                for bc in self.breaking_changes
            ],
            "new_features": self.new_features,
            "deprecations": self.deprecations,
        }


def compare_versions(v1: str, v2: str) -> Dict[str, Any]:
    """
    Compare two API versions.
    
    Args:
        v1: First version
        v2: Second version
        
    Returns:
        Comparison results
    """
    version1 = get_version(v1)
    version2 = get_version(v2)
    
    if not version1 or not version2:
        return {"error": "One or both versions not found"}
    
    return {
        "versions": {
            "v1": version1.to_dict(),
            "v2": version2.to_dict(),
        },
        "comparison": {
            "newer": v2 if version2.major_version > version1.major_version else v1,
            "older": v1 if version2.major_version > version1.major_version else v2,
            "version_gap": abs(version2.major_version - version1.major_version),
        },
        "compatibility": {
            "breaking_changes_expected": version2.major_version != version1.major_version,
            "migration_required": version2.major_version > version1.major_version,
        }
    }


def generate_timeline(version: str) -> Dict[str, Any]:
    """
    Generate deprecation/sunset timeline for a version.
    
    Args:
        version: Version identifier
        
    Returns:
        Timeline information
    """
    api_version = get_version(version)
    if not api_version:
        return {"error": f"Version {version} not found"}
    
    timeline = {
        "version": version,
        "current_status": api_version.status.value,
        "events": []
    }
    
    # Release
    timeline["events"].append({
        "date": api_version.release_date.isoformat(),
        "event": "Released",
        "status": "active",
    })
    
    # Deprecation
    if api_version.deprecation_date:
        timeline["events"].append({
            "date": api_version.deprecation_date.isoformat(),
            "event": "Deprecated",
            "status": "deprecated",
        })
    
    # Sunset
    if api_version.sunset_date:
        timeline["events"].append({
            "date": api_version.sunset_date.isoformat(),
            "event": "Sunset (End of Support)",
            "status": "sunset",
        })
        
        timeline["days_until_sunset"] = api_version.days_until_sunset
    
    return timeline


def suggest_deprecation_timeline(
    version: str,
    deprecation_period_months: int = 6,
    sunset_period_months: int = 12,
) -> Dict[str, Any]:
    """
    Suggest deprecation timeline for a version.
    
    Args:
        version: Version to deprecate
        deprecation_period_months: Months before marking deprecated
        sunset_period_months: Months before sunset
        
    Returns:
        Suggested timeline
    """
    today = date.today()
    deprecation_date = today + timedelta(days=deprecation_period_months * 30)
    sunset_date = today + timedelta(days=sunset_period_months * 30)
    
    return {
        "version": version,
        "suggested_timeline": {
            "deprecation_announcement": today.isoformat(),
            "deprecation_date": deprecation_date.isoformat(),
            "sunset_date": sunset_date.isoformat(),
        },
        "periods": {
            "deprecation_period": f"{deprecation_period_months} months",
            "total_support": f"{sunset_period_months} months",
        },
        "milestones": [
            {
                "date": today.isoformat(),
                "action": "Announce deprecation",
                "tasks": [
                    "Update documentation",
                    "Add deprecation headers",
                    "Notify users via email/blog",
                ]
            },
            {
                "date": deprecation_date.isoformat(),
                "action": "Mark as deprecated",
                "tasks": [
                    "Update version status",
                    "Add sunset headers",
                    "Increase warning visibility",
                ]
            },
            {
                "date": sunset_date.isoformat(),
                "action": "Remove version",
                "tasks": [
                    "Disable endpoints",
                    "Return 410 Gone status",
                    "Redirect to migration guide",
                ]
            }
        ]
    }


def get_client_migration_code(from_version: str, to_version: str, language: str = "python") -> str:
    """
    Generate example client migration code.
    
    Args:
        from_version: Source version
        to_version: Target version
        language: Programming language (python, javascript, etc.)
        
    Returns:
        Example migration code
    """
    if language == "python":
        return f"""# Migration from {from_version} to {to_version}

import requests

# Old ({from_version}) - Basic authentication
response = requests.get(
    "https://api.synfinance.com/api/{from_version}/transactions",
    headers={{
        "Authorization": "Bearer YOUR_TOKEN"
    }}
)

# New ({to_version}) - With tenant context
response = requests.get(
    "https://api.synfinance.com/api/{to_version}/transactions",
    headers={{
        "Authorization": "Bearer YOUR_TOKEN",
        "X-Tenant-ID": "your-tenant-id",  # NEW: Required in {to_version}
        "X-API-Version": "{to_version}",  # Optional: Can also use URL
    }}
)

# Handle new response format
if response.status_code == 200:
    data = response.json()
    # {to_version} includes tenant_id in response
    tenant_id = data.get("tenant_id")
    transactions = data.get("transactions", [])
"""
    
    elif language == "javascript":
        return f"""// Migration from {from_version} to {to_version}

// Old ({from_version})
const response = await fetch(
  'https://api.synfinance.com/api/{from_version}/transactions',
  {{
    headers: {{
      'Authorization': 'Bearer YOUR_TOKEN'
    }}
  }}
);

// New ({to_version}) - With tenant context
const response = await fetch(
  'https://api.synfinance.com/api/{to_version}/transactions',
  {{
    headers: {{
      'Authorization': 'Bearer YOUR_TOKEN',
      'X-Tenant-ID': 'your-tenant-id',  // NEW: Required in {to_version}
      'X-API-Version': '{to_version}'    // Optional: Can also use URL
    }}
  }}
);

// Handle new response format
const data = await response.json();
const tenantId = data.tenant_id;  // NEW in {to_version}
const transactions = data.transactions;
"""
    
    else:
        return f"# Migration code for {language} not yet available"


# Pre-built migration guides

def get_v1_to_v2_migration() -> MigrationGuide:
    """Get pre-built migration guide from v1 to v2"""
    guide = MigrationGuide("v1", "v2")
    
    # Breaking changes
    guide.add_breaking_change(BreakingChange(
        category="authentication",
        description="Tenant context now required",
        old_behavior="GET /api/v1/transactions\nAuthorization: Bearer TOKEN",
        new_behavior="GET /api/v2/transactions\nAuthorization: Bearer TOKEN\nX-Tenant-ID: tenant-123",
        migration_steps=[
            "Obtain tenant ID from onboarding process",
            "Add X-Tenant-ID header to all requests",
            "Update client configuration with tenant ID",
        ],
        affected_endpoints=[
            "/api/v2/transactions",
            "/api/v2/fraud-detection",
            "/api/v2/customers",
            "/api/v2/merchants",
        ]
    ))
    
    guide.add_breaking_change(BreakingChange(
        category="response_format",
        description="All responses now include tenant_id field",
        old_behavior='{"id": "123", "amount": 100.0}',
        new_behavior='{"id": "123", "amount": 100.0, "tenant_id": "tenant-123"}',
        migration_steps=[
            "Update response parsing to handle tenant_id field",
            "Optionally validate tenant_id matches expected value",
        ],
        affected_endpoints=["All endpoints"]
    ))
    
    # New features
    guide.add_new_feature("Multi-tenant support with complete isolation")
    guide.add_new_feature("Role-based access control (RBAC) with 30+ permissions")
    guide.add_new_feature("Resource quotas and usage tracking")
    guide.add_new_feature("GraphQL API for flexible querying")
    guide.add_new_feature("WebSocket support for real-time updates")
    guide.add_new_feature("Ensemble ML models for improved fraud detection")
    
    return guide
