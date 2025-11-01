"""
API Compatibility Layer

Provides backward compatibility through request/response transformations:
- Field mapping between versions
- Data structure conversions
- Automatic adaptation of old requests to new schema
- Response transformation back to old format
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class FieldMapping:
    """
    Maps a field from one version to another.
    
    Attributes:
        old_name: Field name in old version
        new_name: Field name in new version
        transformer: Optional function to transform the value
        default: Default value if field missing
        required: Whether field is required in new version
    """
    old_name: str
    new_name: str
    transformer: Optional[Callable[[Any], Any]] = None
    default: Any = None
    required: bool = False


@dataclass
class SchemaMapping:
    """
    Complete schema mapping between two versions.
    
    Attributes:
        from_version: Source version
        to_version: Target version
        field_mappings: List of field mappings
        removed_fields: Fields removed in new version
        added_fields: Fields added in new version with defaults
        custom_transformer: Optional custom transformation function
    """
    from_version: str
    to_version: str
    field_mappings: List[FieldMapping] = field(default_factory=list)
    removed_fields: List[str] = field(default_factory=list)
    added_fields: Dict[str, Any] = field(default_factory=dict)
    custom_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class CompatibilityAdapter:
    """
    Adapts requests and responses between API versions.
    
    Manages schema mappings and applies transformations automatically.
    """
    
    def __init__(self):
        self.mappings: Dict[tuple[str, str], SchemaMapping] = {}
    
    def register_mapping(self, mapping: SchemaMapping) -> None:
        """
        Register a schema mapping.
        
        Args:
            mapping: SchemaMapping instance
        """
        key = (mapping.from_version, mapping.to_version)
        self.mappings[key] = mapping
    
    def get_mapping(self, from_version: str, to_version: str) -> Optional[SchemaMapping]:
        """
        Get schema mapping between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            SchemaMapping or None if not found
        """
        return self.mappings.get((from_version, to_version))
    
    def transform_forward(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Transform data from old version to new version.
        
        Used for incoming requests to adapt old format to new schema.
        
        Args:
            data: Input data in old format
            from_version: Source version
            to_version: Target version
            
        Returns:
            Transformed data in new format
        """
        mapping = self.get_mapping(from_version, to_version)
        if not mapping:
            # No mapping = no changes needed
            return data
        
        result = deepcopy(data)
        
        # Apply custom transformer first if provided
        if mapping.custom_transformer:
            result = mapping.custom_transformer(result)
        
        # Apply field mappings
        for field_map in mapping.field_mappings:
            if field_map.old_name in result:
                value = result.pop(field_map.old_name)
                
                # Transform value if transformer provided
                if field_map.transformer:
                    value = field_map.transformer(value)
                
                result[field_map.new_name] = value
            elif field_map.required and field_map.default is not None:
                # Add default value for required fields
                result[field_map.new_name] = field_map.default
        
        # Remove old fields that don't exist in new version
        for old_field in mapping.removed_fields:
            result.pop(old_field, None)
        
        # Add new fields with defaults
        for new_field, default_value in mapping.added_fields.items():
            if new_field not in result:
                result[new_field] = default_value
        
        return result
    
    def transform_backward(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Transform data from new version back to old version.
        
        Used for outgoing responses to maintain backward compatibility.
        
        Args:
            data: Input data in new format
            from_version: Target (old) version
            to_version: Source (current) version
            
        Returns:
            Transformed data in old format
        """
        mapping = self.get_mapping(from_version, to_version)
        if not mapping:
            return data
        
        result = deepcopy(data)
        
        # Reverse field mappings
        for field_map in mapping.field_mappings:
            if field_map.new_name in result:
                value = result.pop(field_map.new_name)
                
                # Note: We don't reverse the transformer as that might not be bijective
                # If reverse transformation is needed, create a separate mapping
                
                result[field_map.old_name] = value
        
        # Remove fields that didn't exist in old version
        for new_field in mapping.added_fields:
            result.pop(new_field, None)
        
        return result


class RequestTransformer:
    """
    Transforms incoming requests to match current API version.
    
    Allows old clients to continue using old request formats.
    """
    
    def __init__(self, adapter: CompatibilityAdapter):
        self.adapter = adapter
    
    def transform(
        self,
        data: Dict[str, Any],
        request_version: str,
        current_version: str,
    ) -> Dict[str, Any]:
        """
        Transform request data to current version.
        
        Args:
            data: Request data
            request_version: Version used by client
            current_version: Current API version
            
        Returns:
            Transformed request data
        """
        if request_version == current_version:
            return data
        
        # For now, only handle one-step transformations
        # Future: Chain transformations for multi-version jumps
        return self.adapter.transform_forward(data, request_version, current_version)


class ResponseTransformer:
    """
    Transforms outgoing responses to match requested API version.
    
    Allows serving responses in old format for backward compatibility.
    """
    
    def __init__(self, adapter: CompatibilityAdapter):
        self.adapter = adapter
    
    def transform(
        self,
        data: Dict[str, Any],
        request_version: str,
        current_version: str,
    ) -> Dict[str, Any]:
        """
        Transform response data to requested version.
        
        Args:
            data: Response data in current format
            request_version: Version requested by client
            current_version: Current API version
            
        Returns:
            Transformed response data
        """
        if request_version == current_version:
            return data
        
        return self.adapter.transform_backward(data, request_version, current_version)


# Global adapter instance
_adapter = CompatibilityAdapter()


def get_compatibility_adapter() -> CompatibilityAdapter:
    """Get global compatibility adapter"""
    return _adapter


def transform_request(
    data: Dict[str, Any],
    from_version: str,
    to_version: str,
) -> Dict[str, Any]:
    """
    Convenience function to transform request data.
    
    Args:
        data: Request data
        from_version: Source version
        to_version: Target version
        
    Returns:
        Transformed data
    """
    return _adapter.transform_forward(data, from_version, to_version)


def transform_response(
    data: Dict[str, Any],
    from_version: str,
    to_version: str,
) -> Dict[str, Any]:
    """
    Convenience function to transform response data.
    
    Args:
        data: Response data
        from_version: Target version
        to_version: Source version
        
    Returns:
        Transformed data
    """
    return _adapter.transform_backward(data, from_version, to_version)


# Example mappings for v1 -> v2 migration

def register_default_mappings() -> None:
    """Register default schema mappings for version migrations"""
    adapter = get_compatibility_adapter()
    
    # v1 -> v2: Add tenant context
    v1_to_v2 = SchemaMapping(
        from_version="v1",
        to_version="v2",
        field_mappings=[],
        removed_fields=[],
        added_fields={
            "tenant_id": None,  # Will be filled by middleware
        },
    )
    adapter.register_mapping(v1_to_v2)
    
    # Example: Transaction schema changes
    transaction_v1_to_v2 = SchemaMapping(
        from_version="v1",
        to_version="v2",
        field_mappings=[
            FieldMapping(
                old_name="timestamp",
                new_name="transaction_timestamp",
                required=False,
            ),
        ],
        added_fields={
            "tenant_id": None,
            "fraud_score": 0.0,
            "risk_level": "low",
        },
    )
    adapter.register_mapping(transaction_v1_to_v2)
