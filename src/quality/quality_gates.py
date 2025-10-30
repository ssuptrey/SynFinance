"""
Quality Gates for SynFinance

Define and enforce quality thresholds for automated pass/fail decisions.

Quality gates block deployments or issue warnings based on:
- Data quality scores
- Specific validation failures
- Business rule violations
- Performance metrics

Week 7 Day 3: Automated Quality Assurance Framework
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime

from src.quality.data_quality_checker import QualityReport, Severity, ViolationType

logger = logging.getLogger(__name__)


class GateType(Enum):
    """Types of quality gates"""
    BLOCKING = "blocking"      # Must pass to proceed
    WARNING = "warning"         # Generates warning but allows proceed
    INFORMATIONAL = "informational"  # Just for tracking


class GateStatus(Enum):
    """Status of a quality gate"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateDefinition:
    """
    Definition of a quality gate
    
    Attributes:
        name: Gate name/identifier
        gate_type: Type of gate (blocking/warning/informational)
        description: Description of what this gate checks
        threshold: Threshold value for the check
        comparator: Comparison function (e.g., >=, <=, ==)
        metric_name: Name of the metric to check
    """
    name: str
    gate_type: GateType
    description: str
    threshold: float
    comparator: str  # '>=', '<=', '==', '!=', '>', '<'
    metric_name: str
    
    def evaluate(self, actual_value: float) -> bool:
        """
        Evaluate if the gate passes
        
        Args:
            actual_value: Actual value of the metric
            
        Returns:
            True if gate passes, False otherwise
        """
        comparators = {
            '>=': lambda a, t: a >= t,
            '<=': lambda a, t: a <= t,
            '==': lambda a, t: a == t,
            '!=': lambda a, t: a != t,
            '>': lambda a, t: a > t,
            '<': lambda a, t: a < t,
        }
        
        if self.comparator not in comparators:
            raise ValueError(f"Invalid comparator: {self.comparator}")
        
        return comparators[self.comparator](actual_value, self.threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.gate_type.value,
            'description': self.description,
            'threshold': self.threshold,
            'comparator': self.comparator,
            'metric': self.metric_name
        }


@dataclass
class GateResult:
    """
    Result of evaluating a quality gate
    
    Attributes:
        gate_name: Name of the gate
        status: Status of the gate
        gate_type: Type of gate
        actual_value: Actual value measured
        threshold: Expected threshold
        message: Result message
        passed: Whether the gate passed
    """
    gate_name: str
    status: GateStatus
    gate_type: GateType
    actual_value: Optional[float]
    threshold: Optional[float]
    message: str
    passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'gate': self.gate_name,
            'status': self.status.value,
            'type': self.gate_type.value,
            'actual': self.actual_value,
            'threshold': self.threshold,
            'message': self.message,
            'passed': self.passed
        }


@dataclass
class QualityGateReport:
    """
    Report of all quality gate evaluations
    
    Attributes:
        timestamp: When gates were evaluated
        total_gates: Total number of gates evaluated
        passed_gates: Number of gates that passed
        failed_gates: Number of gates that failed
        blocking_failures: Number of blocking gates that failed
        gate_results: List of individual gate results
        overall_passed: Whether all blocking gates passed
    """
    timestamp: datetime
    total_gates: int
    passed_gates: int
    failed_gates: int
    blocking_failures: int
    gate_results: List[GateResult] = field(default_factory=list)
    overall_passed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'summary': {
                'total': self.total_gates,
                'passed': self.passed_gates,
                'failed': self.failed_gates,
                'blocking_failures': self.blocking_failures
            },
            'overall_passed': self.overall_passed,
            'gates': [r.to_dict() for r in self.gate_results]
        }


class QualityGate:
    """
    Quality gate manager for enforcing quality standards
    
    Manages a set of quality gates and evaluates them against
    quality reports to make automated pass/fail decisions.
    """
    
    def __init__(self):
        """Initialize quality gate manager"""
        self.gates: List[GateDefinition] = []
        self._setup_default_gates()
    
    def _setup_default_gates(self) -> None:
        """Setup default quality gates"""
        
        # Overall quality score gate (BLOCKING)
        self.add_gate(GateDefinition(
            name="minimum_quality_score",
            gate_type=GateType.BLOCKING,
            description="Minimum overall quality score required",
            threshold=80.0,
            comparator='>=',
            metric_name='quality_score'
        ))
        
        # Critical violations gate (BLOCKING)
        self.add_gate(GateDefinition(
            name="no_critical_violations",
            gate_type=GateType.BLOCKING,
            description="No critical violations allowed",
            threshold=0.0,
            comparator='==',
            metric_name='critical_violations'
        ))
        
        # Error violations gate (WARNING)
        self.add_gate(GateDefinition(
            name="max_error_violations",
            gate_type=GateType.WARNING,
            description="Maximum number of error-level violations",
            threshold=5.0,
            comparator='<=',
            metric_name='error_violations'
        ))
        
        # Missing values gate (WARNING)
        self.add_gate(GateDefinition(
            name="missing_values_threshold",
            gate_type=GateType.WARNING,
            description="Maximum percentage of missing values",
            threshold=1.0,
            comparator='<=',
            metric_name='missing_value_percentage'
        ))
        
        # Fraud rate gate (WARNING)
        self.add_gate(GateDefinition(
            name="fraud_rate_minimum",
            gate_type=GateType.WARNING,
            description="Minimum fraud rate for realistic data",
            threshold=0.5,
            comparator='>=',
            metric_name='fraud_rate_percentage'
        ))
        
        self.add_gate(GateDefinition(
            name="fraud_rate_maximum",
            gate_type=GateType.WARNING,
            description="Maximum fraud rate for realistic data",
            threshold=3.0,
            comparator='<=',
            metric_name='fraud_rate_percentage'
        ))
        
        # Anomaly rate gate (WARNING)
        self.add_gate(GateDefinition(
            name="anomaly_rate_minimum",
            gate_type=GateType.WARNING,
            description="Minimum anomaly rate for realistic data",
            threshold=5.0,
            comparator='>=',
            metric_name='anomaly_rate_percentage'
        ))
        
        self.add_gate(GateDefinition(
            name="anomaly_rate_maximum",
            gate_type=GateType.WARNING,
            description="Maximum anomaly rate for realistic data",
            threshold=20.0,
            comparator='<=',
            metric_name='anomaly_rate_percentage'
        ))
        
        # Passed checks gate (INFORMATIONAL)
        self.add_gate(GateDefinition(
            name="minimum_passed_checks",
            gate_type=GateType.INFORMATIONAL,
            description="Minimum percentage of checks that should pass",
            threshold=80.0,
            comparator='>=',
            metric_name='passed_checks_percentage'
        ))
    
    def add_gate(self, gate: GateDefinition) -> None:
        """
        Add a quality gate
        
        Args:
            gate: Gate definition to add
        """
        self.gates.append(gate)
        logger.debug(f"Added quality gate: {gate.name}")
    
    def remove_gate(self, gate_name: str) -> bool:
        """
        Remove a quality gate by name
        
        Args:
            gate_name: Name of the gate to remove
            
        Returns:
            True if gate was removed, False if not found
        """
        initial_count = len(self.gates)
        self.gates = [g for g in self.gates if g.name != gate_name]
        removed = len(self.gates) < initial_count
        
        if removed:
            logger.debug(f"Removed quality gate: {gate_name}")
        return removed
    
    def get_gate(self, gate_name: str) -> Optional[GateDefinition]:
        """
        Get a gate by name
        
        Args:
            gate_name: Name of the gate
            
        Returns:
            Gate definition or None if not found
        """
        for gate in self.gates:
            if gate.name == gate_name:
                return gate
        return None
    
    def list_gates(self) -> List[Dict[str, Any]]:
        """
        List all gates
        
        Returns:
            List of gate definitions as dictionaries
        """
        return [gate.to_dict() for gate in self.gates]
    
    def evaluate(self, quality_report: QualityReport) -> QualityGateReport:
        """
        Evaluate all quality gates against a quality report
        
        Args:
            quality_report: Quality report to evaluate
            
        Returns:
            Quality gate report with results
        """
        logger.info(f"Evaluating {len(self.gates)} quality gates")
        
        gate_results = []
        passed_count = 0
        failed_count = 0
        blocking_failures = 0
        
        # Extract metrics from quality report
        metrics = self._extract_metrics(quality_report)
        
        # Evaluate each gate
        for gate in self.gates:
            result = self._evaluate_gate(gate, metrics)
            gate_results.append(result)
            
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                if gate.gate_type == GateType.BLOCKING:
                    blocking_failures += 1
        
        # Overall pass/fail decision (all blocking gates must pass)
        overall_passed = blocking_failures == 0
        
        report = QualityGateReport(
            timestamp=datetime.now(),
            total_gates=len(self.gates),
            passed_gates=passed_count,
            failed_gates=failed_count,
            blocking_failures=blocking_failures,
            gate_results=gate_results,
            overall_passed=overall_passed
        )
        
        if overall_passed:
            logger.info("All quality gates passed!")
        else:
            logger.warning(f"Quality gates failed: {blocking_failures} blocking failures")
        
        return report
    
    def _extract_metrics(self, quality_report: QualityReport) -> Dict[str, float]:
        """
        Extract metrics from quality report for gate evaluation
        
        Args:
            quality_report: Quality report
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'quality_score': quality_report.quality_score,
            'critical_violations': quality_report.critical_violations,
            'total_violations': quality_report.total_violations,
            'passed_checks': quality_report.passed_checks,
            'failed_checks': quality_report.failed_checks,
        }
        
        # Calculate derived metrics
        total_records = quality_report.total_records
        if total_records > 0:
            # Missing value percentage
            missing_values = quality_report.summary.get('missing_values', 0)
            total_cells = total_records * quality_report.summary.get('total_fields', 1)
            metrics['missing_value_percentage'] = (missing_values / total_cells) * 100 if total_cells > 0 else 0
            
            # Passed checks percentage
            total_checks = len(quality_report.validation_results)
            metrics['passed_checks_percentage'] = (quality_report.passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Count error violations
        error_violations = sum(
            sum(1 for v in r.violations if v.severity == Severity.ERROR)
            for r in quality_report.validation_results
        )
        metrics['error_violations'] = error_violations
        
        # Extract fraud and anomaly rates from validation results
        for result in quality_report.validation_results:
            if result.check_name == "Business Rule Validation":
                fraud_rate = result.metrics.get('fraud_rate')
                if fraud_rate is not None:
                    metrics['fraud_rate_percentage'] = fraud_rate * 100
                
                anomaly_rate = result.metrics.get('anomaly_rate')
                if anomaly_rate is not None:
                    metrics['anomaly_rate_percentage'] = anomaly_rate * 100
        
        return metrics
    
    def _evaluate_gate(self, gate: GateDefinition, metrics: Dict[str, float]) -> GateResult:
        """
        Evaluate a single gate
        
        Args:
            gate: Gate definition
            metrics: Metrics dictionary
            
        Returns:
            Gate result
        """
        # Check if metric exists
        if gate.metric_name not in metrics:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.SKIPPED,
                gate_type=gate.gate_type,
                actual_value=None,
                threshold=gate.threshold,
                message=f"Metric '{gate.metric_name}' not available",
                passed=True  # Skipped gates don't fail
            )
        
        actual_value = metrics[gate.metric_name]
        passed = gate.evaluate(actual_value)
        
        if passed:
            status = GateStatus.PASSED
            message = f"{gate.description}: {actual_value:.2f} {gate.comparator} {gate.threshold:.2f} ✓"
        else:
            status = GateStatus.FAILED
            message = f"{gate.description}: {actual_value:.2f} NOT {gate.comparator} {gate.threshold:.2f} ✗"
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            gate_type=gate.gate_type,
            actual_value=actual_value,
            threshold=gate.threshold,
            message=message,
            passed=passed
        )
    
    def get_summary(self, gate_report: QualityGateReport) -> str:
        """
        Get a human-readable summary of gate results
        
        Args:
            gate_report: Quality gate report
            
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            "QUALITY GATE EVALUATION SUMMARY",
            "=" * 70,
            f"Timestamp: {gate_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Gates: {gate_report.total_gates}",
            f"Passed: {gate_report.passed_gates}",
            f"Failed: {gate_report.failed_gates}",
            f"Blocking Failures: {gate_report.blocking_failures}",
            f"Overall Result: {'PASSED ✓' if gate_report.overall_passed else 'FAILED ✗'}",
            "",
            "Gate Results:",
            "-" * 70
        ]
        
        # Group by gate type
        for gate_type in GateType:
            type_results = [r for r in gate_report.gate_results if r.gate_type == gate_type]
            if type_results:
                lines.append(f"\n{gate_type.value.upper()} Gates:")
                for result in type_results:
                    status_symbol = "✓" if result.passed else "✗"
                    lines.append(f"  [{status_symbol}] {result.gate_name}: {result.message}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
