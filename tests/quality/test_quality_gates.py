"""
Tests for QualityGate

Week 7 Day 3: Automated Quality Assurance Framework
"""

import pytest
from datetime import datetime

from src.quality.quality_gates import (
    QualityGate,
    GateDefinition,
    GateType,
    GateStatus,
)
from src.quality.data_quality_checker import (
    DataQualityChecker,
    QualityReport,
    ValidationResult,
    ViolationType,
    Severity,
)


@pytest.fixture
def quality_gate():
    """Fixture for QualityGate instance"""
    return QualityGate()


@pytest.fixture
def sample_quality_report():
    """Fixture for a sample quality report"""
    return QualityReport(
        dataset_name='test_dataset',
        timestamp=datetime.now(),
        quality_score=85.0,
        total_records=1000,
        summary={
            'total_records': 1000,
            'total_fields': 8,
            'missing_values': 10,
        },
        validation_results=[
            ValidationResult(
                check_name="Business Rule Validation",
                passed=True,
                violations=[],
                metrics={
                    'fraud_rate': 0.015,  # 1.5%
                    'anomaly_rate': 0.10,  # 10%
                }
            )
        ],
        recommendations=["Fix missing values in amount field"]
    )


@pytest.fixture
def failing_quality_report():
    """Fixture for a quality report that should fail gates"""
    from src.quality.data_quality_checker import QualityViolation, Severity
    
    return QualityReport(
        dataset_name='failing_dataset',
        timestamp=datetime.now(),
        quality_score=65.0,  # Below 80 threshold
        total_records=1000,
        summary={
            'total_records': 1000,
            'total_fields': 8,
            'missing_values': 100,
        },
        validation_results=[
            ValidationResult(
                check_name="Schema Validation",
                passed=False,
                violations=[
                    QualityViolation(
                        violation_type=ViolationType.SCHEMA,
                        severity=Severity.CRITICAL,
                        field='test',
                        message='Test critical violation',
                        actual_value='bad',
                        expected_value='good',
                        count=3,
                        percentage=0.3
                    )
                ],
                metrics={}
            ),
            ValidationResult(
                check_name="Business Rule Validation",
                passed=False,
                violations=[],
                metrics={
                    'fraud_rate': 0.001,  # Too low
                    'anomaly_rate': 0.30,  # Too high
                }
            )
        ],
        recommendations=["Critical issues found"]
    )


class TestGateDefinition:
    """Test suite for GateDefinition"""
    
    def test_gate_definition_creation(self):
        """Test creating a gate definition"""
        gate = GateDefinition(
            name="test_gate",
            gate_type=GateType.BLOCKING,
            description="Test gate",
            threshold=80.0,
            comparator='>=',
            metric_name='quality_score'
        )
        
        assert gate.name == "test_gate"
        assert gate.gate_type == GateType.BLOCKING
        assert gate.threshold == 80.0
        assert gate.comparator == '>='
    
    def test_gate_evaluate_greater_equal(self):
        """Test gate evaluation with >= comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.BLOCKING,
            description="Test",
            threshold=80.0,
            comparator='>=',
            metric_name='score'
        )
        
        assert gate.evaluate(85.0) is True
        assert gate.evaluate(80.0) is True
        assert gate.evaluate(75.0) is False
    
    def test_gate_evaluate_less_equal(self):
        """Test gate evaluation with <= comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.WARNING,
            description="Test",
            threshold=5.0,
            comparator='<=',
            metric_name='errors'
        )
        
        assert gate.evaluate(3.0) is True
        assert gate.evaluate(5.0) is True
        assert gate.evaluate(7.0) is False
    
    def test_gate_evaluate_equal(self):
        """Test gate evaluation with == comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.BLOCKING,
            description="Test",
            threshold=0.0,
            comparator='==',
            metric_name='critical'
        )
        
        assert gate.evaluate(0.0) is True
        assert gate.evaluate(1.0) is False
    
    def test_gate_evaluate_not_equal(self):
        """Test gate evaluation with != comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.WARNING,
            description="Test",
            threshold=0.0,
            comparator='!=',
            metric_name='value'
        )
        
        assert gate.evaluate(1.0) is True
        assert gate.evaluate(0.0) is False
    
    def test_gate_evaluate_greater(self):
        """Test gate evaluation with > comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.WARNING,
            description="Test",
            threshold=5.0,
            comparator='>',
            metric_name='count'
        )
        
        assert gate.evaluate(6.0) is True
        assert gate.evaluate(5.0) is False
        assert gate.evaluate(4.0) is False
    
    def test_gate_evaluate_less(self):
        """Test gate evaluation with < comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.WARNING,
            description="Test",
            threshold=10.0,
            comparator='<',
            metric_name='percentage'
        )
        
        assert gate.evaluate(9.0) is True
        assert gate.evaluate(10.0) is False
        assert gate.evaluate(11.0) is False
    
    def test_gate_evaluate_invalid_comparator(self):
        """Test gate evaluation with invalid comparator"""
        gate = GateDefinition(
            name="test",
            gate_type=GateType.WARNING,
            description="Test",
            threshold=5.0,
            comparator='<>',  # Invalid
            metric_name='value'
        )
        
        with pytest.raises(ValueError, match="Invalid comparator"):
            gate.evaluate(5.0)
    
    def test_gate_to_dict(self):
        """Test gate conversion to dictionary"""
        gate = GateDefinition(
            name="test_gate",
            gate_type=GateType.BLOCKING,
            description="Test gate",
            threshold=80.0,
            comparator='>=',
            metric_name='quality_score'
        )
        
        gate_dict = gate.to_dict()
        
        assert gate_dict['name'] == "test_gate"
        assert gate_dict['type'] == "blocking"
        assert gate_dict['threshold'] == 80.0
        assert gate_dict['comparator'] == '>='
        assert gate_dict['metric'] == 'quality_score'


class TestQualityGate:
    """Test suite for QualityGate"""
    
    def test_initialization(self, quality_gate):
        """Test QualityGate initialization"""
        assert quality_gate is not None
        assert len(quality_gate.gates) > 0  # Should have default gates
    
    def test_default_gates_exist(self, quality_gate):
        """Test that default gates are created"""
        gate_names = [g.name for g in quality_gate.gates]
        
        assert "minimum_quality_score" in gate_names
        assert "no_critical_violations" in gate_names
        assert "max_error_violations" in gate_names
    
    def test_add_gate(self):
        """Test adding a custom gate"""
        qg = QualityGate()
        initial_count = len(qg.gates)
        
        new_gate = GateDefinition(
            name="custom_gate",
            gate_type=GateType.WARNING,
            description="Custom test gate",
            threshold=100.0,
            comparator='<=',
            metric_name='custom_metric'
        )
        
        qg.add_gate(new_gate)
        
        assert len(qg.gates) == initial_count + 1
        assert qg.get_gate("custom_gate") is not None
    
    def test_remove_gate(self, quality_gate):
        """Test removing a gate"""
        # Add a gate to remove
        gate = GateDefinition(
            name="temp_gate",
            gate_type=GateType.WARNING,
            description="Temporary gate",
            threshold=50.0,
            comparator='>=',
            metric_name='temp_metric'
        )
        quality_gate.add_gate(gate)
        
        # Remove it
        result = quality_gate.remove_gate("temp_gate")
        
        assert result is True
        assert quality_gate.get_gate("temp_gate") is None
    
    def test_remove_nonexistent_gate(self, quality_gate):
        """Test removing a gate that doesn't exist"""
        result = quality_gate.remove_gate("nonexistent_gate")
        
        assert result is False
    
    def test_get_gate(self, quality_gate):
        """Test retrieving a gate by name"""
        gate = quality_gate.get_gate("minimum_quality_score")
        
        assert gate is not None
        assert gate.name == "minimum_quality_score"
        assert gate.gate_type == GateType.BLOCKING
    
    def test_get_nonexistent_gate(self, quality_gate):
        """Test retrieving a gate that doesn't exist"""
        gate = quality_gate.get_gate("nonexistent")
        
        assert gate is None
    
    def test_list_gates(self, quality_gate):
        """Test listing all gates"""
        gates_list = quality_gate.list_gates()
        
        assert isinstance(gates_list, list)
        assert len(gates_list) > 0
        assert all(isinstance(g, dict) for g in gates_list)
    
    def test_evaluate_passing_report(self, quality_gate, sample_quality_report):
        """Test evaluating a quality report that should pass"""
        report = quality_gate.evaluate(sample_quality_report)
        
        assert report is not None
        assert report.overall_passed is True
        assert report.blocking_failures == 0
    
    def test_evaluate_failing_report(self, quality_gate, failing_quality_report):
        """Test evaluating a quality report that should fail"""
        report = quality_gate.evaluate(failing_quality_report)
        
        assert report is not None
        assert report.overall_passed is False
        assert report.blocking_failures > 0
    
    def test_gate_report_structure(self, quality_gate, sample_quality_report):
        """Test structure of gate report"""
        report = quality_gate.evaluate(sample_quality_report)
        
        assert hasattr(report, 'timestamp')
        assert hasattr(report, 'total_gates')
        assert hasattr(report, 'passed_gates')
        assert hasattr(report, 'failed_gates')
        assert hasattr(report, 'blocking_failures')
        assert hasattr(report, 'gate_results')
        assert hasattr(report, 'overall_passed')
        
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.total_gates, int)
        assert isinstance(report.overall_passed, bool)
    
    def test_gate_results_count(self, quality_gate, sample_quality_report):
        """Test that gate results match number of gates"""
        report = quality_gate.evaluate(sample_quality_report)
        
        assert len(report.gate_results) == len(quality_gate.gates)
        assert report.total_gates == len(quality_gate.gates)
    
    def test_blocking_gate_failure_blocks_overall(self):
        """Test that blocking gate failure blocks overall pass"""
        qg = QualityGate()
        
        # Create a report that fails blocking gate
        failing_report = QualityReport(
            dataset_name='test',
            timestamp=datetime.now(),
            quality_score=75.0,  # Below 80 threshold - fails blocking gate
            total_records=1000,
            summary={'total_records': 1000, 'total_fields': 8},
            validation_results=[],
            recommendations=[]
        )
        
        report = qg.evaluate(failing_report)
        
        assert report.overall_passed is False
        assert report.blocking_failures > 0
    
    def test_warning_gate_failure_allows_overall_pass(self):
        """Test that warning gate failure doesn't block overall pass"""
        qg = QualityGate()
        
        # Create a report that passes blocking gates but fails warning gates
        report_data = QualityReport(
            dataset_name='test',
            timestamp=datetime.now(),
            quality_score=85.0,  # Passes blocking gate
            total_records=1000,
            summary={'total_records': 1000, 'total_fields': 8, 'missing_values': 100},
            validation_results=[
                ValidationResult(
                    check_name="Business Rule Validation",
                    passed=True,
                    violations=[],
                    metrics={
                        'fraud_rate': 0.0,  # Fails warning gate
                        'anomaly_rate': 0.25,  # Fails warning gate
                    }
                )
            ],
            recommendations=[]
        )
        
        report = qg.evaluate(report_data)
        
        # Should pass overall despite warning failures
        assert report.blocking_failures == 0
        # May have failed warning gates, but overall should pass
        assert report.overall_passed is True
    
    def test_gate_result_status(self, quality_gate, sample_quality_report):
        """Test gate result status values"""
        report = quality_gate.evaluate(sample_quality_report)
        
        for gate_result in report.gate_results:
            assert gate_result.status in [GateStatus.PASSED, GateStatus.FAILED, GateStatus.SKIPPED]
    
    def test_skipped_gate_when_metric_missing(self):
        """Test that gate is skipped when metric is not available"""
        qg = QualityGate()
        
        # Add a gate for a metric that won't exist
        qg.add_gate(GateDefinition(
            name="missing_metric_gate",
            gate_type=GateType.WARNING,
            description="Test missing metric",
            threshold=50.0,
            comparator='>=',
            metric_name='nonexistent_metric'
        ))
        
        report_data = QualityReport(
            dataset_name='test',
            timestamp=datetime.now(),
            quality_score=85.0,
            total_records=1000,
            summary={'total_records': 1000, 'total_fields': 8},
            validation_results=[],
            recommendations=[]
        )
        
        report = qg.evaluate(report_data)
        
        # Find the skipped gate
        skipped_gates = [gr for gr in report.gate_results if gr.gate_name == "missing_metric_gate"]
        
        assert len(skipped_gates) == 1
        assert skipped_gates[0].status == GateStatus.SKIPPED
        assert skipped_gates[0].passed is True  # Skipped gates don't fail
    
    def test_get_summary(self, quality_gate, sample_quality_report):
        """Test generating gate summary"""
        report = quality_gate.evaluate(sample_quality_report)
        summary = quality_gate.get_summary(report)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "QUALITY GATE EVALUATION SUMMARY" in summary
        assert "PASSED" in summary or "FAILED" in summary
    
    def test_gate_report_to_dict(self, quality_gate, sample_quality_report):
        """Test converting gate report to dictionary"""
        report = quality_gate.evaluate(sample_quality_report)
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert 'timestamp' in report_dict
        assert 'summary' in report_dict
        assert 'overall_passed' in report_dict
        assert 'gates' in report_dict
        
        assert isinstance(report_dict['gates'], list)
