"""
Debug Inspector for SynFinance

Tools for inspecting and debugging:
- Transaction inspection
- Feature inspection
- Model decision inspection
- Data flow tracing
- State introspection

Week 7 Day 4: Enhanced Observability
"""

import inspect
import sys
import traceback
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class InspectionResult:
    """Result from an inspection operation"""
    target_type: str  # transaction, feature, model, object
    target_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'target_type': self.target_type,
            'target_name': self.target_name,
            'timestamp': self.timestamp.isoformat(),
            'attributes': self.attributes,
            'methods': self.methods,
            'properties': self.properties,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ObjectInspector:
    """
    Inspector for Python objects
    
    Provides detailed introspection of objects, classes, and instances.
    """
    
    def inspect(self, obj: Any, name: str = "object") -> InspectionResult:
        """
        Inspect a Python object
        
        Args:
            obj: Object to inspect
            name: Name for the object
        
        Returns:
            InspectionResult with detailed information
        """
        result = InspectionResult(
            target_type=type(obj).__name__,
            target_name=name
        )
        
        # Extract attributes
        result.attributes = self._extract_attributes(obj)
        
        # Extract methods
        result.methods = self._extract_methods(obj)
        
        # Extract properties
        result.properties = self._extract_properties(obj)
        
        # Metadata
        result.metadata['class'] = obj.__class__.__name__
        result.metadata['module'] = obj.__class__.__module__
        result.metadata['doc'] = inspect.getdoc(obj)
        result.metadata['file'] = self._get_source_file(obj)
        
        return result
    
    def _extract_attributes(self, obj: Any) -> Dict[str, Any]:
        """Extract object attributes"""
        attributes = {}
        
        for attr_name in dir(obj):
            # Skip magic methods
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                
                # Skip methods
                if callable(attr_value):
                    continue
                
                # Convert to serializable format
                attributes[attr_name] = self._make_serializable(attr_value)
            except Exception:
                # Some attributes may raise exceptions
                continue
        
        return attributes
    
    def _extract_methods(self, obj: Any) -> List[str]:
        """Extract method names"""
        methods = []
        
        for name in dir(obj):
            # Skip magic methods
            if name.startswith('__') and name.endswith('__'):
                continue
            
            try:
                attr = getattr(obj, name)
                if callable(attr):
                    # Get signature
                    try:
                        sig = inspect.signature(attr)
                        methods.append(f"{name}{sig}")
                    except (ValueError, TypeError):
                        methods.append(name)
            except Exception:
                continue
        
        return methods
    
    def _extract_properties(self, obj: Any) -> Dict[str, Any]:
        """Extract property descriptors"""
        properties = {}
        
        # Get class properties
        for name, value in inspect.getmembers(type(obj)):
            if isinstance(value, property):
                try:
                    prop_value = getattr(obj, name)
                    properties[name] = self._make_serializable(prop_value)
                except Exception:
                    properties[name] = "<error accessing property>"
        
        return properties
    
    def _get_source_file(self, obj: Any) -> Optional[str]:
        """Get source file for object"""
        try:
            return inspect.getfile(obj.__class__)
        except (TypeError, AttributeError):
            return None
    
    def _make_serializable(self, value: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
        """Convert value to JSON-serializable format"""
        if current_depth >= max_depth:
            return f"<max depth reached: {type(value).__name__}>"
        
        # Handle None
        if value is None:
            return None
        
        # Handle primitives
        if isinstance(value, (int, float, str, bool)):
            return value
        
        # Handle datetime
        if isinstance(value, datetime):
            return value.isoformat()
        
        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            return [self._make_serializable(v, max_depth, current_depth + 1) for v in value[:100]]
        
        # Handle dicts
        if isinstance(value, dict):
            return {
                k: self._make_serializable(v, max_depth, current_depth + 1)
                for k, v in list(value.items())[:100]
            }
        
        # Handle dataclasses
        if hasattr(value, '__dataclass_fields__'):
            try:
                return asdict(value)
            except Exception:
                pass
        
        # Default: string representation
        try:
            return str(value)[:1000]  # Limit string length
        except Exception:
            return f"<{type(value).__name__}>"


class TransactionInspector:
    """
    Inspector for transaction objects
    
    Specialized inspector for SynFinance transactions.
    """
    
    def __init__(self):
        self.obj_inspector = ObjectInspector()
    
    def inspect_transaction(self, transaction: Any) -> InspectionResult:
        """
        Inspect a transaction object
        
        Args:
            transaction: Transaction object to inspect
        
        Returns:
            InspectionResult with transaction details
        """
        result = self.obj_inspector.inspect(transaction, "transaction")
        result.target_type = "transaction"
        
        # Extract transaction-specific fields
        tx_fields = {}
        for field in ['transaction_id', 'customer_id', 'merchant_id', 'amount', 
                      'timestamp', 'category', 'is_fraud', 'fraud_score']:
            try:
                value = getattr(transaction, field, None)
                if value is not None:
                    tx_fields[field] = value
            except Exception:
                pass
        
        result.metadata['transaction_fields'] = tx_fields
        
        return result
    
    def validate_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Validate transaction object structure
        
        Args:
            transaction: Transaction to validate
        
        Returns:
            Validation results with issues found
        """
        issues = []
        warnings = []
        
        # Required fields
        required_fields = ['transaction_id', 'customer_id', 'amount', 'timestamp']
        for field in required_fields:
            if not hasattr(transaction, field):
                issues.append(f"Missing required field: {field}")
            elif getattr(transaction, field, None) is None:
                issues.append(f"Required field is None: {field}")
        
        # Type validation
        if hasattr(transaction, 'amount'):
            amount = getattr(transaction, 'amount')
            if not isinstance(amount, (int, float)):
                issues.append(f"Amount should be numeric, got {type(amount).__name__}")
            elif amount < 0:
                warnings.append("Amount is negative")
        
        if hasattr(transaction, 'timestamp'):
            timestamp = getattr(transaction, 'timestamp')
            if not isinstance(timestamp, (datetime, str)):
                issues.append(f"Timestamp should be datetime/str, got {type(timestamp).__name__}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }


class FeatureInspector:
    """
    Inspector for ML features
    
    Analyzes feature vectors and feature engineering outputs.
    """
    
    def inspect_features(self, features: Union[Dict, List, Any], name: str = "features") -> InspectionResult:
        """
        Inspect feature vector
        
        Args:
            features: Feature data (dict, list, or dataframe)
            name: Name for the features
        
        Returns:
            InspectionResult with feature analysis
        """
        result = InspectionResult(
            target_type="features",
            target_name=name
        )
        
        # Handle different feature formats
        if isinstance(features, dict):
            result.attributes = features
            result.metadata['feature_count'] = len(features)
            result.metadata['feature_names'] = list(features.keys())
            
            # Analyze feature values
            analysis = self._analyze_feature_dict(features)
            result.metadata['analysis'] = analysis
        
        elif isinstance(features, list):
            result.metadata['feature_count'] = len(features)
            result.metadata['feature_type'] = 'vector'
            
            # Analyze vector
            analysis = self._analyze_feature_vector(features)
            result.metadata['analysis'] = analysis
        
        elif hasattr(features, 'to_dict'):
            # Pandas Series/DataFrame
            feature_dict = features.to_dict()
            result.attributes = feature_dict
            result.metadata['feature_count'] = len(feature_dict)
            result.metadata['analysis'] = self._analyze_feature_dict(feature_dict)
        
        return result
    
    def _analyze_feature_dict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature dictionary"""
        analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'null_features': [],
            'constant_features': []
        }
        
        for name, value in features.items():
            if value is None:
                analysis['null_features'].append(name)
            elif isinstance(value, (int, float)):
                analysis['numeric_features'].append(name)
                if value == 0:
                    analysis['constant_features'].append(name)
            else:
                analysis['categorical_features'].append(name)
        
        return analysis
    
    def _analyze_feature_vector(self, features: List) -> Dict[str, Any]:
        """Analyze feature vector"""
        analysis = {
            'length': len(features),
            'null_count': sum(1 for f in features if f is None),
            'zero_count': sum(1 for f in features if f == 0),
            'numeric_count': sum(1 for f in features if isinstance(f, (int, float)))
        }
        
        # Statistics for numeric features
        numeric_features = [f for f in features if isinstance(f, (int, float))]
        if numeric_features:
            analysis['min'] = min(numeric_features)
            analysis['max'] = max(numeric_features)
            analysis['mean'] = sum(numeric_features) / len(numeric_features)
        
        return analysis


class ModelInspector:
    """
    Inspector for ML models
    
    Analyzes model structure, parameters, and decision-making.
    """
    
    def __init__(self):
        self.obj_inspector = ObjectInspector()
    
    def inspect_model(self, model: Any, name: str = "model") -> InspectionResult:
        """
        Inspect ML model
        
        Args:
            model: Model object to inspect
            name: Name for the model
        
        Returns:
            InspectionResult with model details
        """
        result = self.obj_inspector.inspect(model, name)
        result.target_type = "model"
        
        # Extract model-specific information
        model_info = {}
        
        # Check for common model attributes
        for attr in ['n_features_', 'feature_names_in_', 'classes_', 'coef_', 
                     'feature_importances_', 'n_estimators_']:
            if hasattr(model, attr):
                try:
                    value = getattr(model, attr)
                    model_info[attr] = self.obj_inspector._make_serializable(value)
                except Exception:
                    pass
        
        result.metadata['model_info'] = model_info
        
        return result
    
    def explain_prediction(self, model: Any, features: Any, prediction: Any) -> Dict[str, Any]:
        """
        Explain model prediction
        
        Args:
            model: ML model
            features: Input features
            prediction: Model prediction
        
        Returns:
            Explanation of prediction
        """
        explanation = {
            'prediction': prediction,
            'model_type': type(model).__name__,
            'features': features if isinstance(features, dict) else None
        }
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                explanation['feature_importance'] = dict(zip(feature_names, importances))
        
        # Coefficients (for linear models)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                explanation['coefficients'] = dict(zip(feature_names, coef))
        
        return explanation


class DebugInspector:
    """
    Comprehensive debug inspector
    
    Combines all inspection capabilities for debugging SynFinance operations.
    """
    
    def __init__(self):
        self.obj_inspector = ObjectInspector()
        self.tx_inspector = TransactionInspector()
        self.feature_inspector = FeatureInspector()
        self.model_inspector = ModelInspector()
    
    def inspect(self, obj: Any, name: str = "object", obj_type: Optional[str] = None) -> InspectionResult:
        """
        Inspect any object with appropriate inspector
        
        Args:
            obj: Object to inspect
            name: Name for the object
            obj_type: Type hint (transaction, feature, model, or None for auto-detect)
        
        Returns:
            InspectionResult
        """
        # Auto-detect type if not specified
        if obj_type is None:
            obj_type = self._detect_type(obj)
        
        # Route to appropriate inspector
        if obj_type == "transaction":
            return self.tx_inspector.inspect_transaction(obj)
        elif obj_type == "features":
            return self.feature_inspector.inspect_features(obj, name)
        elif obj_type == "model":
            return self.model_inspector.inspect_model(obj, name)
        else:
            return self.obj_inspector.inspect(obj, name)
    
    def _detect_type(self, obj: Any) -> str:
        """Auto-detect object type"""
        class_name = obj.__class__.__name__.lower()
        
        if 'transaction' in class_name:
            return "transaction"
        elif 'feature' in class_name or isinstance(obj, dict):
            return "features"
        elif any(x in class_name for x in ['model', 'classifier', 'regressor', 'estimator']):
            return "model"
        
        return "object"
    
    def print_report(self, result: InspectionResult) -> None:
        """Print human-readable inspection report"""
        print("=" * 80)
        print(f"Debug Inspection: {result.target_name}")
        print("=" * 80)
        print(f"Type: {result.target_type}")
        print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if result.attributes:
            print("\nAttributes:")
            for key, value in list(result.attributes.items())[:20]:
                print(f"  {key}: {value}")
            if len(result.attributes) > 20:
                print(f"  ... and {len(result.attributes) - 20} more")
        
        if result.methods:
            print(f"\nMethods ({len(result.methods)}):")
            for method in result.methods[:10]:
                print(f"  - {method}")
            if len(result.methods) > 10:
                print(f"  ... and {len(result.methods) - 10} more")
        
        if result.properties:
            print("\nProperties:")
            for key, value in result.properties.items():
                print(f"  {key}: {value}")
        
        if result.metadata:
            print("\nMetadata:")
            for key, value in result.metadata.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in list(value.items())[:10]:
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print("=" * 80)


def inspect_object(obj: Any, name: str = "object", verbose: bool = True) -> InspectionResult:
    """
    Quick inspection utility
    
    Args:
        obj: Object to inspect
        name: Name for the object
        verbose: Print detailed report
    
    Returns:
        InspectionResult
    """
    inspector = DebugInspector()
    result = inspector.inspect(obj, name)
    
    if verbose:
        inspector.print_report(result)
    
    return result
