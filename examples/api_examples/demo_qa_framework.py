"""
Demo: Week 7 Day 3 - Automated Quality Assurance Framework

Demonstrates the comprehensive QA framework including:
1. Data quality validation
2. Quality gates evaluation
3. Automated QA pipeline

Run: python examples/demo_qa_framework.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.quality import (
    DataQualityChecker,
    QualityGate,
    QAPipeline,
    PipelineConfig,
    GateDefinition,
    GateType
)


def create_sample_data(size=1000, quality='good'):
    """
    Create sample transaction data
    
    Args:
        size: Number of records
        quality: 'good', 'medium', or 'bad'
    """
    np.random.seed(42)
    
    # Base data
    data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(size)],
        'customer_id': [f'CUST{i%100:05d}' for i in range(size)],
        'merchant_id': [f'MERCH{i%50:04d}' for i in range(size)],
        'amount': np.random.lognormal(mean=5, sigma=1.5, size=size),
        'timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 30))
            for _ in range(size)
        ],
        'category': np.random.choice(['food', 'retail', 'gas', 'entertainment'], size=size),
        'payment_mode': np.random.choice(['credit', 'debit', 'upi', 'cash'], size=size),
        'is_fraud': np.random.choice([0, 1], size=size, p=[0.99, 0.01]),
        'is_anomaly': np.random.choice([0, 1], size=size, p=[0.9, 0.1]),
        'fraud_type': ['none'] * size,
    }
    
    # Introduce quality issues based on quality parameter
    if quality == 'medium':
        # Add some missing values
        indices = np.random.choice(size, size // 20, replace=False)
        for idx in indices:
            data['amount'][idx] = np.nan
        
        # Add some future dates
        future_indices = np.random.choice(size, size // 50, replace=False)
        for idx in future_indices:
            data['timestamp'][idx] = datetime.now() + timedelta(days=5)
    
    elif quality == 'bad':
        # Missing required fields
        data.pop('category')
        data.pop('payment_mode')
        
        # Lots of missing values
        indices = np.random.choice(size, size // 5, replace=False)
        for idx in indices:
            data['amount'][idx] = np.nan
        
        # No fraud/anomalies (violates business rules)
        data['is_fraud'] = [0] * size
        data['is_anomaly'] = [0] * size
        
        # Many future dates
        future_indices = np.random.choice(size, size // 10, replace=False)
        for idx in future_indices:
            data['timestamp'][idx] = datetime.now() + timedelta(days=30)
    
    return pd.DataFrame(data)


def demo_data_quality_checker():
    """Demonstrate DataQualityChecker"""
    print("=" * 80)
    print("DEMO 1: Data Quality Checker")
    print("=" * 80)
    
    checker = DataQualityChecker()
    
    # Test with good quality data
    print("\n1. Checking GOOD quality data...")
    good_data = create_sample_data(size=1000, quality='good')
    report = checker.check_quality(good_data)
    
    print(f"   Quality Score: {report.quality_score:.2f}/100")
    print(f"   Total Violations: {report.total_violations}")
    print(f"   Critical Violations: {report.critical_violations}")
    print(f"   Passed Checks: {report.passed_checks}/{report.passed_checks + report.failed_checks}")
    
    # Test with bad quality data
    print("\n2. Checking BAD quality data...")
    bad_data = create_sample_data(size=1000, quality='bad')
    report = checker.check_quality(bad_data)
    
    print(f"   Quality Score: {report.quality_score:.2f}/100")
    print(f"   Total Violations: {report.total_violations}")
    print(f"   Critical Violations: {report.critical_violations}")
    print(f"   Passed Checks: {report.passed_checks}/{report.passed_checks + report.failed_checks}")
    
    if report.recommendations:
        print("\n   Top Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"     {i}. {rec}")


def demo_quality_gates():
    """Demonstrate QualityGate"""
    print("\n" + "=" * 80)
    print("DEMO 2: Quality Gates")
    print("=" * 80)
    
    checker = DataQualityChecker()
    gate = QualityGate()
    
    # Test with good data
    print("\n1. Evaluating gates for GOOD data...")
    good_data = create_sample_data(size=1000, quality='good')
    quality_report = checker.check_quality(good_data)
    gate_report = gate.evaluate(quality_report)
    
    print(f"   Overall Result: {'PASSED ✓' if gate_report.overall_passed else 'FAILED ✗'}")
    print(f"   Total Gates: {gate_report.total_gates}")
    print(f"   Passed: {gate_report.passed_gates}")
    print(f"   Failed: {gate_report.failed_gates}")
    print(f"   Blocking Failures: {gate_report.blocking_failures}")
    
    # Test with bad data
    print("\n2. Evaluating gates for BAD data...")
    bad_data = create_sample_data(size=1000, quality='bad')
    quality_report = checker.check_quality(bad_data)
    gate_report = gate.evaluate(quality_report)
    
    print(f"   Overall Result: {'PASSED ✓' if gate_report.overall_passed else 'FAILED ✗'}")
    print(f"   Total Gates: {gate_report.total_gates}")
    print(f"   Passed: {gate_report.passed_gates}")
    print(f"   Failed: {gate_report.failed_gates}")
    print(f"   Blocking Failures: {gate_report.blocking_failures}")
    
    # Show failed gates
    if gate_report.failed_gates > 0:
        print("\n   Failed Gates:")
        for result in gate_report.gate_results:
            if not result.passed:
                print(f"     ✗ {result.gate_name}: {result.message}")


def demo_qa_pipeline():
    """Demonstrate QAPipeline"""
    print("\n" + "=" * 80)
    print("DEMO 3: Automated QA Pipeline")
    print("=" * 80)
    
    # Configure pipeline
    output_dir = Path("output/qa_demo")
    config = PipelineConfig(
        output_dir=output_dir,
        generate_test_data=False,
        run_quality_checks=True,
        evaluate_gates=True,
        save_reports=True
    )
    
    pipeline = QAPipeline(config)
    
    # Run pipeline on good data
    print("\n1. Running pipeline on GOOD data...")
    good_data = create_sample_data(size=500, quality='good')
    result = pipeline.run(good_data)
    
    print(f"   Success: {result.success}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    print(f"   Output Files: {len(result.output_files)}")
    
    if result.output_files:
        print("\n   Generated Files:")
        for file in result.output_files:
            print(f"     • {file.name}")
    
    # Run batch processing
    print("\n2. Running batch processing...")
    datasets = {
        'good': create_sample_data(size=500, quality='good'),
        'medium': create_sample_data(size=500, quality='medium'),
    }
    
    batch_results = pipeline.run_batch(datasets)
    
    print(f"   Processed {len(batch_results)} datasets")
    for name, result in batch_results.items():
        status = "✓" if result.success else "✗"
        quality_score = result.quality_report.quality_score if result.quality_report else 0
        print(f"     {status} {name}: Quality Score {quality_score:.1f}/100")
    
    # Show batch summary location
    batch_summary = output_dir / "batch_summary.txt"
    if batch_summary.exists():
        print(f"\n   Batch summary saved to: {batch_summary}")


def demo_custom_gates():
    """Demonstrate custom quality gates"""
    print("\n" + "=" * 80)
    print("DEMO 4: Custom Quality Gates")
    print("=" * 80)
    
    gate = QualityGate()
    
    # Add a custom gate
    print("\n1. Adding custom gate...")
    custom_gate = GateDefinition(
        name="max_duplicate_records",
        gate_type=GateType.WARNING,
        description="Maximum percentage of duplicate records",
        threshold=1.0,
        comparator='<=',
        metric_name='duplicate_percentage'
    )
    gate.add_gate(custom_gate)
    
    print(f"   Added gate: {custom_gate.name}")
    print(f"   Type: {custom_gate.gate_type.value}")
    print(f"   Threshold: {custom_gate.threshold}%")
    
    # List all gates
    print("\n2. All gates:")
    gates = gate.list_gates()
    for g in gates[:5]:  # Show first 5
        print(f"   • {g['name']} ({g['type']}): {g['description']}")
    print(f"   ... and {len(gates) - 5} more")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("Week 7 Day 3: Automated Quality Assurance Framework Demo")
    print("=" * 80)
    
    try:
        demo_data_quality_checker()
        demo_quality_gates()
        demo_qa_pipeline()
        demo_custom_gates()
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        print("\nCheck output/qa_demo/ for generated reports")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
