"""
SynFinance Quality Assurance Framework

Comprehensive data quality validation, quality gates, and automated QA pipelines.

Week 7 Day 3 Deliverable
"""

from src.quality.data_quality_checker import (
    DataQualityChecker,
    QualityReport,
    QualityViolation,
    ValidationResult,
    ViolationType,
    Severity,
)
from src.quality.quality_gates import (
    QualityGate,
    QualityGateReport,
    GateDefinition,
    GateResult,
    GateType,
    GateStatus,
)
from src.quality.qa_pipeline import (
    QAPipeline,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    # Data Quality Checker
    "DataQualityChecker",
    "QualityReport",
    "QualityViolation",
    "ValidationResult",
    "ViolationType",
    "Severity",
    # Quality Gates
    "QualityGate",
    "QualityGateReport",
    "GateDefinition",
    "GateResult",
    "GateType",
    "GateStatus",
    # QA Pipeline
    "QAPipeline",
    "PipelineConfig",
    "PipelineResult",
]

__version__ = "1.0.0"
