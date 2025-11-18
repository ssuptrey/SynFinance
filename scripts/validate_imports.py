"""
Import Validation Script
Tests all SynFinance modules to verify imports work correctly.
Run this to validate documentation accuracy.
"""

import sys
from typing import Dict, List, Tuple

def test_import(module_path: str, class_name: str) -> Tuple[bool, str]:
    """Test if an import works."""
    try:
        parts = module_path.rsplit('.', 1)
        if len(parts) == 2:
            module = __import__(parts[0], fromlist=[parts[1]])
            getattr(module, class_name)
        else:
            __import__(module_path)
        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

# Define all imports to test
IMPORTS_TO_TEST = [
    # Config
    ("src.config", "ConfigManager", "Configuration management"),
    ("src.config", "AppConfig", "Application configuration"),
    
    # Generators
    ("src.customer_generator", "CustomerGenerator", "Customer generation"),
    ("src.generators.merchant_generator", "MerchantGenerator", "Merchant generation"),
    ("src.generators.transaction_core", "TransactionGenerator", "Transaction generation"),
    ("src.generators.geographic_generator", "GeographicPatternGenerator", "Geographic patterns"),
    ("src.generators.temporal_generator", "TemporalPatternGenerator", "Temporal patterns"),
    ("src.generators.fraud_patterns", "FraudPatternGenerator", "Fraud patterns"),
    ("src.generators.ml_features", "MLFeatureGenerator", "ML features"),
    
    # Fraud Detection
    ("src.fraud.scoring_engine", "FraudScoringEngine", "Fraud scoring"),
    ("src.fraud.pattern_detector", "PatternDetector", "Pattern detection"),
    ("src.fraud.behavioral_analyzer", "BehavioralAnalyzer", "Behavioral analysis"),
    ("src.fraud.velocity_checker", "VelocityChecker", "Velocity checking"),
    ("src.fraud.decision_engine", "DecisionEngine", "Fraud decisions"),
    
    # Database
    ("src.database.db_manager", "DatabaseManager", "Database management"),
    ("src.database.models", "Customer", "Customer model"),
    ("src.database.models", "Merchant", "Merchant model"),
    ("src.database.models", "Transaction", "Transaction model"),
    
    # Analytics
    ("src.analytics.statistical_analyzer", "StatisticalAnalyzer", "Statistical analysis"),
    ("src.analytics.correlation_analyzer", "CorrelationAnalyzer", "Correlation analysis"),
    ("src.analytics.visualization", "VisualizationFramework", "Visualization"),
    ("src.analytics.data_profiler", "DataProfiler", "Data profiling"),
    ("src.analytics.trend_analyzer", "TrendAnalyzer", "Trend analysis"),
    
    # Performance
    ("src.performance.optimizer", "BatchProcessor", "Batch processing"),
    ("src.performance.query_optimizer", "QueryOptimizer", "Query optimization"),
    ("src.performance.profiler", "Profiler", "Performance profiling"),
    ("src.performance.metrics_collector", "MetricsCollector", "Metrics collection"),
    ("src.performance.load_tester", "LoadTester", "Load testing"),
    ("src.performance.cache_manager", "CacheManager", "Cache management"),
    
    # Reporting
    ("src.reporting.html_generator", "HTMLReportGenerator", "HTML reports"),
    ("src.reporting.excel_generator", "ExcelDashboardGenerator", "Excel reports"),
    ("src.reporting.dataset_comparator", "DatasetComparator", "Dataset comparison"),
    
    # ML
    ("src.ml.base_model", "BaseMLModel", "Base ML model"),
    ("src.ml.model_registry", "ModelRegistry", "Model registry"),
    ("src.ml.model_optimization", "ModelOptimizer", "Model optimization"),
    
    # API (just check if module loads)
    ("src.api.app", None, "FastAPI application"),
    ("src.api.health", None, "Health checks"),
]

def main():
    """Run all import tests."""
    print("=" * 80)
    print("SynFinance Import Validation")
    print("=" * 80)
    print()
    
    results: Dict[str, List[Tuple[str, str, bool, str]]] = {
        "PASSED": [],
        "FAILED": []
    }
    
    for module_path, class_name, description in IMPORTS_TO_TEST:
        if class_name:
            import_str = f"from {module_path} import {class_name}"
        else:
            import_str = f"import {module_path}"
        
        success, message = test_import(module_path, class_name) if class_name else test_import(module_path, "")
        
        if success:
            results["PASSED"].append((import_str, description, success, message))
        else:
            results["FAILED"].append((import_str, description, success, message))
    
    # Print passed imports
    print(f"✓ PASSED ({len(results['PASSED'])} imports)")
    print("-" * 80)
    for import_str, description, _, _ in results["PASSED"]:
        print(f"  ✓ {import_str:<65} # {description}")
    print()
    
    # Print failed imports
    if results["FAILED"]:
        print(f"✗ FAILED ({len(results['FAILED'])} imports)")
        print("-" * 80)
        for import_str, description, _, message in results["FAILED"]:
            print(f"  ✗ {import_str:<65} # {description}")
            print(f"     {message}")
        print()
    
    # Summary
    total = len(results["PASSED"]) + len(results["FAILED"])
    pass_rate = (len(results["PASSED"]) / total * 100) if total > 0 else 0
    
    print("=" * 80)
    print(f"SUMMARY: {len(results['PASSED'])}/{total} passed ({pass_rate:.1f}%)")
    print("=" * 80)
    
    # Exit with error if any failed
    if results["FAILED"]:
        sys.exit(1)
    else:
        print("\n✓ All imports validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
