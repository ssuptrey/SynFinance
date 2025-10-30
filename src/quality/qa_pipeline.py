"""
QA Pipeline for SynFinance

Automated quality assurance pipeline that:
1. Generates test datasets
2. Runs quality checks
3. Evaluates quality gates
4. Generates comprehensive reports

Week 7 Day 3: Automated Quality Assurance Framework
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from src.quality.data_quality_checker import DataQualityChecker, QualityReport
from src.quality.quality_gates import QualityGate, QualityGateReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for QA pipeline
    
    Attributes:
        output_dir: Directory for pipeline outputs
        generate_test_data: Whether to generate test data
        test_data_size: Size of test dataset
        run_quality_checks: Whether to run quality checks
        evaluate_gates: Whether to evaluate quality gates
        save_reports: Whether to save reports to disk
    """
    output_dir: Path = Path("output/qa_pipeline")
    generate_test_data: bool = False
    test_data_size: int = 10000
    run_quality_checks: bool = True
    evaluate_gates: bool = True
    save_reports: bool = True


@dataclass
class PipelineResult:
    """
    Result of QA pipeline execution
    
    Attributes:
        timestamp: When pipeline was run
        success: Whether pipeline completed successfully
        quality_report: Quality check results
        gate_report: Quality gate results
        errors: List of errors encountered
        warnings: List of warnings
        output_files: List of generated output files
    """
    timestamp: datetime
    success: bool
    quality_report: Optional[QualityReport] = None
    gate_report: Optional[QualityGateReport] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_files: List[Path] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'quality_passed': self.quality_report.quality_score >= 80 if self.quality_report else False,
            'gates_passed': self.gate_report.overall_passed if self.gate_report else False,
            'errors': self.errors,
            'warnings': self.warnings,
            'output_files': [str(f) for f in self.output_files]
        }


class QAPipeline:
    """
    Automated QA pipeline for comprehensive quality assurance
    
    Orchestrates the entire QA process:
    - Optional test data generation
    - Quality checks via DataQualityChecker
    - Gate evaluation via QualityGate
    - Report generation and storage
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize QA pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.quality_checker = DataQualityChecker()
        self.quality_gate = QualityGate()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"QA Pipeline initialized with output dir: {self.config.output_dir}")
    
    def run(self, data: Optional[pd.DataFrame] = None) -> PipelineResult:
        """
        Run the complete QA pipeline
        
        Args:
            data: DataFrame to validate (if None, will generate test data if configured)
            
        Returns:
            Pipeline result with all reports and status
        """
        logger.info("Starting QA pipeline execution")
        timestamp = datetime.now()
        result = PipelineResult(timestamp=timestamp, success=False)
        
        try:
            # Step 1: Get or generate data
            if data is None and self.config.generate_test_data:
                logger.info(f"Generating test dataset of size {self.config.test_data_size}")
                data = self._generate_test_data()
                if self.config.save_reports:
                    test_data_path = self.config.output_dir / f"test_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
                    data.to_csv(test_data_path, index=False)
                    result.output_files.append(test_data_path)
                    logger.info(f"Test data saved to {test_data_path}")
            
            if data is None or data.empty:
                raise ValueError("No data provided and test data generation disabled")
            
            # Step 2: Run quality checks
            if self.config.run_quality_checks:
                logger.info("Running quality checks")
                quality_report = self.quality_checker.check_quality(data)
                result.quality_report = quality_report
                
                # Save quality report
                if self.config.save_reports:
                    quality_report_path = self.config.output_dir / f"quality_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                    self._save_quality_report(quality_report, quality_report_path)
                    result.output_files.append(quality_report_path)
                
                # Check for critical issues
                if quality_report.critical_violations > 0:
                    result.warnings.append(f"Found {quality_report.critical_violations} critical violations")
                
                if quality_report.quality_score < 80:
                    result.warnings.append(f"Quality score ({quality_report.quality_score:.1f}) below 80")
            
            # Step 3: Evaluate quality gates
            if self.config.evaluate_gates and result.quality_report:
                logger.info("Evaluating quality gates")
                gate_report = self.quality_gate.evaluate(result.quality_report)
                result.gate_report = gate_report
                
                # Save gate report
                if self.config.save_reports:
                    gate_report_path = self.config.output_dir / f"gate_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                    self._save_gate_report(gate_report, gate_report_path)
                    result.output_files.append(gate_report_path)
                
                # Check for gate failures
                if not gate_report.overall_passed:
                    result.errors.append(f"Quality gates failed: {gate_report.blocking_failures} blocking failures")
                
                if gate_report.failed_gates > 0:
                    result.warnings.append(f"{gate_report.failed_gates} gates failed")
            
            # Step 4: Generate summary report
            if self.config.save_reports:
                summary_path = self.config.output_dir / f"pipeline_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
                self._save_summary(result, summary_path)
                result.output_files.append(summary_path)
            
            # Determine overall success
            result.success = (
                len(result.errors) == 0 and
                (not result.gate_report or result.gate_report.overall_passed)
            )
            
            if result.success:
                logger.info("QA pipeline completed successfully")
            else:
                logger.warning(f"QA pipeline completed with issues: {len(result.errors)} errors, {len(result.warnings)} warnings")
            
        except Exception as e:
            logger.error(f"QA pipeline failed: {e}", exc_info=True)
            result.errors.append(str(e))
            result.success = False
        
        return result
    
    def _generate_test_data(self) -> pd.DataFrame:
        """
        Generate synthetic test data
        
        Returns:
            Test dataset
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate simple test data directly
        size = min(self.config.test_data_size, 1000)
        np.random.seed(42)
        
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
            'fraud_type': [
                np.random.choice(['none', 'card_theft', 'account_takeover'])
                if fraud else 'none'
                for fraud in np.random.choice([0, 1], size=size, p=[0.99, 0.01])
            ],
        }
        
        return pd.DataFrame(data)
    
    def _save_quality_report(self, report: QualityReport, path: Path) -> None:
        """
        Save quality report to JSON file
        
        Args:
            report: Quality report
            path: Output file path
        """
        import json
        import numpy as np
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'quality_score': report.quality_score,
            'total_records': report.total_records,
            'passed_checks': report.passed_checks,
            'failed_checks': report.failed_checks,
            'total_violations': report.total_violations,
            'critical_violations': report.critical_violations,
            'summary': report.summary,
            'validation_results': [
                {
                    'check_name': r.check_name,
                    'passed': r.passed,
                    'violations_count': len(r.violations),
                    'metrics': r.metrics,
                    'violations': [
                        {
                            'type': v.violation_type.value,
                            'severity': v.severity.value,
                            'field': v.field,
                            'message': v.message,
                            'count': int(v.count) if isinstance(v.count, (np.integer, np.int64)) else v.count,
                            'percentage': float(v.percentage) if isinstance(v.percentage, (np.floating, np.float64)) else v.percentage
                        }
                        for v in r.violations[:10]  # Limit to first 10
                    ]
                }
                for r in report.validation_results
            ],
            'recommendations': report.recommendations
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Quality report saved to {path}")
    
    def _save_gate_report(self, report: QualityGateReport, path: Path) -> None:
        """
        Save gate report to JSON file
        
        Args:
            report: Gate report
            path: Output file path
        """
        import json
        import numpy as np
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Gate report saved to {path}")
    
    def _save_summary(self, result: PipelineResult, path: Path) -> None:
        """
        Save pipeline summary to text file
        
        Args:
            result: Pipeline result
            path: Output file path
        """
        lines = [
            "=" * 80,
            "QA PIPELINE EXECUTION SUMMARY",
            "=" * 80,
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {'SUCCESS ✓' if result.success else 'FAILED ✗'}",
            "",
        ]
        
        # Quality report summary
        if result.quality_report:
            qr = result.quality_report
            lines.extend([
                "QUALITY CHECK RESULTS",
                "-" * 80,
                f"Quality Score: {qr.quality_score:.2f}/100",
                f"Total Records: {qr.total_records:,}",
                f"Passed Checks: {qr.passed_checks}/{qr.passed_checks + qr.failed_checks}",
                f"Total Violations: {qr.total_violations}",
                f"Critical Violations: {qr.critical_violations}",
                "",
                "Top Issues:",
            ])
            
            for i, rec in enumerate(qr.recommendations[:5], 1):
                lines.append(f"  {i}. {rec}")
            
            lines.append("")
        
        # Gate report summary
        if result.gate_report:
            gr = result.gate_report
            lines.extend([
                "QUALITY GATE RESULTS",
                "-" * 80,
                f"Overall Result: {'PASSED ✓' if gr.overall_passed else 'FAILED ✗'}",
                f"Total Gates: {gr.total_gates}",
                f"Passed: {gr.passed_gates}",
                f"Failed: {gr.failed_gates}",
                f"Blocking Failures: {gr.blocking_failures}",
                "",
                "Gate Details:",
            ])
            
            for gate_result in gr.gate_results:
                symbol = "✓" if gate_result.passed else "✗"
                lines.append(f"  [{symbol}] {gate_result.gate_name}: {gate_result.message}")
            
            lines.append("")
        
        # Errors and warnings
        if result.errors:
            lines.extend([
                "ERRORS",
                "-" * 80,
            ])
            for error in result.errors:
                lines.append(f"  ✗ {error}")
            lines.append("")
        
        if result.warnings:
            lines.extend([
                "WARNINGS",
                "-" * 80,
            ])
            for warning in result.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")
        
        # Output files
        if result.output_files:
            lines.extend([
                "OUTPUT FILES",
                "-" * 80,
            ])
            for output_file in result.output_files:
                lines.append(f"  • {output_file}")
            lines.append("")
        
        lines.append("=" * 80)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Pipeline summary saved to {path}")
    
    def run_batch(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, PipelineResult]:
        """
        Run QA pipeline on multiple datasets
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            
        Returns:
            Dictionary mapping dataset names to pipeline results
        """
        logger.info(f"Running batch QA on {len(datasets)} datasets")
        results = {}
        
        for name, data in datasets.items():
            logger.info(f"Processing dataset: {name}")
            
            # Create subdirectory for this dataset
            dataset_dir = self.config.output_dir / name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Temporarily update config
            original_dir = self.config.output_dir
            self.config.output_dir = dataset_dir
            
            # Run pipeline
            result = self.run(data)
            results[name] = result
            
            # Restore config
            self.config.output_dir = original_dir
        
        # Generate batch summary
        if self.config.save_reports:
            self._save_batch_summary(results, self.config.output_dir / "batch_summary.txt")
        
        return results
    
    def _save_batch_summary(self, results: Dict[str, PipelineResult], path: Path) -> None:
        """
        Save batch processing summary
        
        Args:
            results: Dictionary of pipeline results
            path: Output file path
        """
        lines = [
            "=" * 80,
            "BATCH QA PIPELINE SUMMARY",
            "=" * 80,
            f"Total Datasets: {len(results)}",
            f"Successful: {sum(1 for r in results.values() if r.success)}",
            f"Failed: {sum(1 for r in results.values() if not r.success)}",
            "",
            "Dataset Results:",
            "-" * 80,
        ]
        
        for name, result in results.items():
            status = "✓" if result.success else "✗"
            quality_score = result.quality_report.quality_score if result.quality_report else 0
            gates_passed = result.gate_report.overall_passed if result.gate_report else False
            
            lines.append(f"\n{name}:")
            lines.append(f"  Status: {status}")
            lines.append(f"  Quality Score: {quality_score:.2f}/100")
            lines.append(f"  Gates Passed: {gates_passed}")
            
            if result.errors:
                lines.append(f"  Errors: {len(result.errors)}")
            if result.warnings:
                lines.append(f"  Warnings: {len(result.warnings)}")
        
        lines.append("\n" + "=" * 80)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Batch summary saved to {path}")
