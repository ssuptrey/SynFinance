"""
SynFinance Analytics Module

Advanced analytics and visualization components for fraud detection analysis.
"""

from .advanced_analytics import (
    CorrelationAnalyzer,
    FeatureImportanceAnalyzer,
    ModelPerformanceAnalyzer,
    StatisticalTestAnalyzer,
    AnalyticsReport,
    CorrelationResult,
    FeatureImportanceResult,
    ModelMetrics,
    StatisticalTestResult,
)
from .visualization import VisualizationFramework
from .dashboard import HTMLDashboardGenerator

__all__ = [
    # Analytics
    "CorrelationAnalyzer",
    "FeatureImportanceAnalyzer",
    "ModelPerformanceAnalyzer",
    "StatisticalTestAnalyzer",
    "AnalyticsReport",
    "CorrelationResult",
    "FeatureImportanceResult",
    "ModelMetrics",
    "StatisticalTestResult",
    # Visualization
    "VisualizationFramework",
    # Dashboard
    "HTMLDashboardGenerator",
]
