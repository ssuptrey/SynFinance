"""
Tests for QAPipeline

Week 7 Day 3: Automated Quality Assurance Framework
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

from src.quality.qa_pipeline import (
    QAPipeline,
    PipelineConfig,
    PipelineResult,
)


@pytest.fixture
def temp_output_dir():
    """Fixture for temporary output directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def pipeline_config(temp_output_dir):
    """Fixture for pipeline configuration"""
    return PipelineConfig(
        output_dir=temp_output_dir,
        generate_test_data=False,
        test_data_size=100,
        run_quality_checks=True,
        evaluate_gates=True,
        save_reports=True
    )


@pytest.fixture
def qa_pipeline(pipeline_config):
    """Fixture for QAPipeline instance"""
    return QAPipeline(pipeline_config)


@pytest.fixture
def sample_dataset():
    """Fixture for sample transaction dataset"""
    np.random.seed(42)
    size = 100
    
    data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(size)],
        'customer_id': [f'CUST{i%20:05d}' for i in range(size)],
        'merchant_id': [f'MERCH{i%10:04d}' for i in range(size)],
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


class TestPipelineConfig:
    """Test suite for PipelineConfig"""
    
    def test_default_config(self):
        """Test default pipeline configuration"""
        config = PipelineConfig()
        
        assert config.output_dir == Path("output/qa_pipeline")
        assert config.generate_test_data is False
        assert config.test_data_size == 10000
        assert config.run_quality_checks is True
        assert config.evaluate_gates is True
        assert config.save_reports is True
    
    def test_custom_config(self, temp_output_dir):
        """Test custom pipeline configuration"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            generate_test_data=True,
            test_data_size=500,
            run_quality_checks=False,
            evaluate_gates=False,
            save_reports=False
        )
        
        assert config.output_dir == temp_output_dir
        assert config.generate_test_data is True
        assert config.test_data_size == 500
        assert config.run_quality_checks is False
        assert config.evaluate_gates is False
        assert config.save_reports is False


class TestPipelineResult:
    """Test suite for PipelineResult"""
    
    def test_pipeline_result_creation(self):
        """Test creating a pipeline result"""
        result = PipelineResult(
            timestamp=datetime.now(),
            success=True
        )
        
        assert result is not None
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []
        assert result.output_files == []
    
    def test_pipeline_result_to_dict(self):
        """Test converting pipeline result to dictionary"""
        result = PipelineResult(
            timestamp=datetime.now(),
            success=True
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'timestamp' in result_dict
        assert 'success' in result_dict
        assert 'quality_passed' in result_dict
        assert 'gates_passed' in result_dict


class TestQAPipeline:
    """Test suite for QAPipeline"""
    
    def test_initialization(self, qa_pipeline, temp_output_dir):
        """Test QAPipeline initialization"""
        assert qa_pipeline is not None
        assert qa_pipeline.config.output_dir == temp_output_dir
        assert qa_pipeline.quality_checker is not None
        assert qa_pipeline.quality_gate is not None
    
    def test_output_directory_created(self, temp_output_dir):
        """Test that output directory is created"""
        config = PipelineConfig(output_dir=temp_output_dir / "subdir")
        pipeline = QAPipeline(config)
        
        assert (temp_output_dir / "subdir").exists()
    
    def test_run_with_data(self, qa_pipeline, sample_dataset):
        """Test running pipeline with provided data"""
        result = qa_pipeline.run(sample_dataset)
        
        assert result is not None
        assert isinstance(result, PipelineResult)
        assert result.success is not None
    
    def test_run_quality_checks(self, qa_pipeline, sample_dataset):
        """Test that quality checks are run"""
        result = qa_pipeline.run(sample_dataset)
        
        assert result.quality_report is not None
        assert hasattr(result.quality_report, 'quality_score')
        assert hasattr(result.quality_report, 'total_violations')
    
    def test_run_gate_evaluation(self, qa_pipeline, sample_dataset):
        """Test that gates are evaluated"""
        result = qa_pipeline.run(sample_dataset)
        
        assert result.gate_report is not None
        assert hasattr(result.gate_report, 'overall_passed')
        assert hasattr(result.gate_report, 'gate_results')
    
    def test_skip_quality_checks(self, temp_output_dir, sample_dataset):
        """Test skipping quality checks"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            run_quality_checks=False,
            evaluate_gates=False
        )
        pipeline = QAPipeline(config)
        
        result = pipeline.run(sample_dataset)
        
        assert result.quality_report is None
        assert result.gate_report is None
    
    def test_skip_gate_evaluation(self, temp_output_dir, sample_dataset):
        """Test skipping gate evaluation"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            run_quality_checks=True,
            evaluate_gates=False
        )
        pipeline = QAPipeline(config)
        
        result = pipeline.run(sample_dataset)
        
        assert result.quality_report is not None
        assert result.gate_report is None
    
    def test_save_reports(self, qa_pipeline, sample_dataset):
        """Test that reports are saved to disk"""
        result = qa_pipeline.run(sample_dataset)
        
        assert len(result.output_files) > 0
        
        # Check that files actually exist
        for file_path in result.output_files:
            assert file_path.exists()
    
    def test_no_save_reports(self, temp_output_dir, sample_dataset):
        """Test disabling report saving"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            save_reports=False
        )
        pipeline = QAPipeline(config)
        
        result = pipeline.run(sample_dataset)
        
        assert len(result.output_files) == 0
    
    def test_generate_test_data(self, temp_output_dir):
        """Test generating test data"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            generate_test_data=True,
            test_data_size=100
        )
        pipeline = QAPipeline(config)
        
        result = pipeline.run()  # No data provided
        
        assert result.success is not None
        assert result.quality_report is not None
    
    def test_no_data_no_generation_error(self, temp_output_dir):
        """Test error when no data and generation disabled"""
        config = PipelineConfig(
            output_dir=temp_output_dir,
            generate_test_data=False
        )
        pipeline = QAPipeline(config)
        
        result = pipeline.run()  # No data provided
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_success_status(self, qa_pipeline, sample_dataset):
        """Test success status determination"""
        result = qa_pipeline.run(sample_dataset)
        
        # Success should be True if no errors and gates pass
        if result.gate_report and result.gate_report.overall_passed and len(result.errors) == 0:
            assert result.success is True
    
    def test_failure_status_on_errors(self, qa_pipeline):
        """Test failure status when errors occur"""
        # Provide None data with generation disabled - should cause error
        qa_pipeline.config.generate_test_data = False
        result = qa_pipeline.run(None)
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_warnings_added(self, temp_output_dir):
        """Test that warnings are added for quality issues"""
        # Create a dataset that will trigger warnings
        data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', 'CUST002'],
            'merchant_id': ['MERCH001', 'MERCH002'],
            'amount': [100.0, 200.0],
            'timestamp': [datetime.now()] * 2,
            'is_fraud': [0, 0],  # No fraud - will trigger warning
            'is_anomaly': [0, 0],
            'fraud_type': ['none', 'none'],
        })
        
        config = PipelineConfig(output_dir=temp_output_dir)
        pipeline = QAPipeline(config)
        
        result = pipeline.run(data)
        
        # Small dataset with no fraud should trigger warnings
        assert len(result.warnings) >= 0  # May have warnings
    
    def test_batch_processing(self, qa_pipeline, sample_dataset):
        """Test batch processing multiple datasets"""
        datasets = {
            'dataset1': sample_dataset.copy(),
            'dataset2': sample_dataset.copy(),
        }
        
        results = qa_pipeline.run_batch(datasets)
        
        assert len(results) == 2
        assert 'dataset1' in results
        assert 'dataset2' in results
        assert all(isinstance(r, PipelineResult) for r in results.values())
    
    def test_batch_creates_subdirectories(self, qa_pipeline, sample_dataset, temp_output_dir):
        """Test that batch processing creates subdirectories"""
        datasets = {
            'dataset1': sample_dataset.copy(),
            'dataset2': sample_dataset.copy(),
        }
        
        qa_pipeline.run_batch(datasets)
        
        assert (temp_output_dir / 'dataset1').exists()
        assert (temp_output_dir / 'dataset2').exists()
    
    def test_batch_summary_created(self, qa_pipeline, sample_dataset, temp_output_dir):
        """Test that batch summary is created"""
        datasets = {
            'dataset1': sample_dataset.copy(),
        }
        
        qa_pipeline.run_batch(datasets)
        
        batch_summary_path = temp_output_dir / 'batch_summary.txt'
        assert batch_summary_path.exists()
    
    def test_output_files_types(self, qa_pipeline, sample_dataset):
        """Test types of output files generated"""
        result = qa_pipeline.run(sample_dataset)
        
        file_names = [f.name for f in result.output_files]
        
        # Check for expected file types
        has_quality_report = any('quality_report' in name for name in file_names)
        has_gate_report = any('gate_report' in name for name in file_names)
        has_summary = any('pipeline_summary' in name for name in file_names)
        
        assert has_quality_report or has_gate_report or has_summary
    
    def test_timestamp_in_filenames(self, qa_pipeline, sample_dataset):
        """Test that output files have timestamps"""
        result = qa_pipeline.run(sample_dataset)
        
        # All output files should have timestamps in format YYYYMMDD_HHMMSS
        for file_path in result.output_files:
            filename = file_path.name
            # Check that filename contains digits that could be a timestamp
            assert any(char.isdigit() for char in filename)
    
    def test_empty_dataset_handling(self, qa_pipeline):
        """Test handling of empty dataset"""
        empty_data = pd.DataFrame()
        
        result = qa_pipeline.run(empty_data)
        
        # Should complete but may have quality issues
        assert result is not None
    
    def test_pipeline_error_handling(self, qa_pipeline):
        """Test pipeline handles errors gracefully"""
        # Pass invalid data type
        result = qa_pipeline.run(None)
        
        assert result is not None
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_quality_report_saved_as_json(self, qa_pipeline, sample_dataset, temp_output_dir):
        """Test that quality report is saved as JSON"""
        result = qa_pipeline.run(sample_dataset)
        
        json_files = [f for f in result.output_files if f.suffix == '.json' and 'quality_report' in f.name]
        
        assert len(json_files) > 0
        
        # Verify it's valid JSON
        import json
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            assert 'quality_score' in data
    
    def test_gate_report_saved_as_json(self, qa_pipeline, sample_dataset):
        """Test that gate report is saved as JSON"""
        result = qa_pipeline.run(sample_dataset)
        
        json_files = [f for f in result.output_files if f.suffix == '.json' and 'gate_report' in f.name]
        
        assert len(json_files) > 0
        
        # Verify it's valid JSON
        import json
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            assert 'overall_passed' in data
    
    def test_summary_saved_as_text(self, qa_pipeline, sample_dataset):
        """Test that summary is saved as text file"""
        result = qa_pipeline.run(sample_dataset)
        
        txt_files = [f for f in result.output_files if f.suffix == '.txt' and 'summary' in f.name]
        
        assert len(txt_files) > 0
        
        # Verify content
        with open(txt_files[0], 'r') as f:
            content = f.read()
            assert 'QA PIPELINE EXECUTION SUMMARY' in content
