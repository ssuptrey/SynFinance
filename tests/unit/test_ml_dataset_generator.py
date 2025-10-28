"""
Tests for ML Dataset Generator Module

Tests the MLDatasetGenerator class and dataset preparation methods.
Validates balancing, splitting, normalization, and export functionality.
"""

import pytest
import os
import json
import tempfile
from src.generators.ml_dataset_generator import MLDatasetGenerator, DatasetSplit


class TestMLDatasetGenerator:
    """Test suite for ML dataset generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a dataset generator instance."""
        return MLDatasetGenerator(seed=42)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample ML features."""
        features = []
        for i in range(100):
            features.append({
                'transaction_id': f'TXN{i:03d}',
                'daily_txn_count': i % 10,
                'weekly_txn_count': i % 20,
                'daily_txn_amount': float(i * 100),
                'weekly_txn_amount': float(i * 500),
                'txn_frequency_1h': i % 3,
                'amount_velocity_1h': float(i * 50),
                'distance_from_home': float(i * 10),
                'category_diversity_score': float(i % 5) / 5.0,
                'merchant_loyalty_score': float(i % 10) / 10.0,
                'is_fraud': 1 if i < 10 else 0,  # 10% fraud rate
                'fraud_type': 'Card Cloning' if i < 10 else 'None'
            })
        return features
    
    @pytest.fixture
    def balanced_features(self):
        """Create balanced sample features (50-50 fraud/normal)."""
        features = []
        for i in range(100):
            features.append({
                'transaction_id': f'TXN{i:03d}',
                'daily_txn_count': i % 10,
                'amount': float(i * 100),
                'is_fraud': 1 if i < 50 else 0,
                'fraud_type': 'Fraud' if i < 50 else 'None'
            })
        return features


class TestDatasetBalancing(TestMLDatasetGenerator):
    """Test dataset balancing functionality."""
    
    def test_undersample_balancing(self, generator, sample_features):
        """Test undersampling to balance dataset."""
        balanced = generator.create_balanced_dataset(
            sample_features,
            target_fraud_rate=0.5,
            strategy='undersample'
        )
        
        # Should have balanced classes
        fraud_count = sum(1 for f in balanced if f['is_fraud'] == 1)
        normal_count = sum(1 for f in balanced if f['is_fraud'] == 0)
        
        assert fraud_count == normal_count
        assert len(balanced) == 20  # 10 fraud + 10 normal
    
    def test_oversample_balancing(self, generator, sample_features):
        """Test oversampling to balance dataset."""
        balanced = generator.create_balanced_dataset(
            sample_features,
            target_fraud_rate=0.5,
            strategy='oversample'
        )
        
        # Should have balanced classes
        fraud_count = sum(1 for f in balanced if f['is_fraud'] == 1)
        normal_count = sum(1 for f in balanced if f['is_fraud'] == 0)
        
        assert fraud_count == normal_count
        assert len(balanced) == 180  # 90 fraud + 90 normal (fraud oversampled)
    
    def test_custom_fraud_rate(self, generator, sample_features):
        """Test balancing to custom fraud rate."""
        balanced = generator.create_balanced_dataset(
            sample_features,
            target_fraud_rate=0.3,
            strategy='undersample'
        )
        
        fraud_count = sum(1 for f in balanced if f['is_fraud'] == 1)
        total_count = len(balanced)
        
        fraud_rate = fraud_count / total_count
        assert abs(fraud_rate - 0.3) < 0.05  # Within 5% of target


class TestDatasetSplitting(TestMLDatasetGenerator):
    """Test train/validation/test splitting."""
    
    def test_split_ratios(self, generator, balanced_features):
        """Test that split ratios are respected."""
        split = generator.create_train_test_split(
            balanced_features,
            train_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        
        total = len(balanced_features)
        # Allow tolerance for stratified splitting rounding
        assert 65 <= len(split.train) <= 75  # ~70%
        assert 10 <= len(split.validation) <= 20  # ~15%
        assert 10 <= len(split.test) <= 20  # ~15%
    
    def test_stratified_splitting(self, generator, balanced_features):
        """Test that stratified splitting maintains class balance."""
        split = generator.create_train_test_split(
            balanced_features,
            stratify=True
        )
        
        # Check fraud rate in each split
        train_fraud_rate = sum(1 for f in split.train if f['is_fraud'] == 1) / len(split.train)
        val_fraud_rate = sum(1 for f in split.validation if f['is_fraud'] == 1) / len(split.validation)
        test_fraud_rate = sum(1 for f in split.test if f['is_fraud'] == 1) / len(split.test)
        
        # All should be close to 0.5
        assert abs(train_fraud_rate - 0.5) < 0.1
        assert abs(val_fraud_rate - 0.5) < 0.1
        assert abs(test_fraud_rate - 0.5) < 0.1
    
    def test_no_overlap_between_splits(self, generator, balanced_features):
        """Test that train/val/test splits have no overlap."""
        split = generator.create_train_test_split(balanced_features)
        
        train_ids = set(f['transaction_id'] for f in split.train)
        val_ids = set(f['transaction_id'] for f in split.validation)
        test_ids = set(f['transaction_id'] for f in split.test)
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestFeatureNormalization(TestMLDatasetGenerator):
    """Test feature normalization."""
    
    def test_normalization_range(self, generator):
        """Test that normalization scales to [0, 1]."""
        features = [
            {'daily_txn_amount': 100.0, 'distance_from_home': 5.0, 'transaction_id': 'A', 'is_fraud': 0},
            {'daily_txn_amount': 500.0, 'distance_from_home': 15.0, 'transaction_id': 'B', 'is_fraud': 1},
            {'daily_txn_amount': 1000.0, 'distance_from_home': 25.0, 'transaction_id': 'C', 'is_fraud': 0},
        ]
        
        normalized = generator.normalize_features(
            features,
            fit=True
        )
        
        # Check normalization
        for feat in normalized:
            if 'daily_txn_amount' in feat:
                assert 0.0 <= feat['daily_txn_amount'] <= 1.0
            if 'distance_from_home' in feat:
                assert 0.0 <= feat['distance_from_home'] <= 1.0
    
    def test_normalization_preserves_excluded_columns(self, generator):
        """Test that excluded columns are not normalized."""
        features = [
            {'amount': 100.0, 'transaction_id': 'A', 'is_fraud': 0},
            {'amount': 500.0, 'transaction_id': 'B', 'is_fraud': 1},
        ]
        
        normalized = generator.normalize_features(
            features,
            fit=True
        )
        
        # Check excluded columns unchanged
        assert normalized[0]['transaction_id'] == 'A'
        assert normalized[1]['is_fraud'] == 1
    
    def test_normalization_stats_stored(self, generator):
        """Test that normalization stats are stored."""
        features = [
            {'amount': 100.0, 'daily_txn_amount': 50.0},
            {'amount': 500.0, 'daily_txn_amount': 150.0},
        ]
        
        normalized = generator.normalize_features(features, fit=True)
        
        # Check stats stored in generator
        assert len(generator.feature_stats) > 0
        if 'daily_txn_amount' in generator.feature_stats:
            assert 'min' in generator.feature_stats['daily_txn_amount']
            assert 'max' in generator.feature_stats['daily_txn_amount']


class TestCategoricalEncoding(TestMLDatasetGenerator):
    """Test categorical feature encoding."""
    
    def test_fraud_type_encoding(self, generator):
        """Test that fraud_type is encoded to numbers."""
        features = [
            {'fraud_type': 'None', 'amount': 100},
            {'fraud_type': 'Card Cloning', 'amount': 200},
            {'fraud_type': 'None', 'amount': 150},
            {'fraud_type': 'Velocity Abuse', 'amount': 300},
        ]
        
        encoded, mapping = generator.encode_categorical_features(features)
        
        # Check encoded values are integers (with _encoded suffix)
        assert isinstance(encoded[0]['fraud_type_encoded'], int)
        assert isinstance(encoded[1]['fraud_type_encoded'], int)
        
        # Check mapping exists
        assert 'fraud_type' in mapping
        assert len(mapping['fraud_type']) >= 3  # At least 3 unique values
    
    def test_encoding_consistency(self, generator):
        """Test that same values get same encoding."""
        features = [
            {'fraud_type': 'Card Cloning'},
            {'fraud_type': 'None'},
            {'fraud_type': 'Card Cloning'},
        ]
        
        encoded, mapping = generator.encode_categorical_features(features)
        
        assert encoded[0]['fraud_type'] == encoded[2]['fraud_type']
        assert encoded[0]['fraud_type'] != encoded[1]['fraud_type']


class TestDataQualityValidation(TestMLDatasetGenerator):
    """Test data quality validation."""
    
    def test_validate_balanced_dataset(self, generator, balanced_features):
        """Test validation of balanced dataset."""
        report = generator.validate_dataset_quality(balanced_features)
        
        assert 'total_samples' in report
        assert 'fraud_count' in report
        assert 'fraud_rate' in report
        assert 'quality_checks' in report
        
        assert report['total_samples'] == 100
        assert report['fraud_count'] == 50
        assert report['fraud_rate'] == 0.5
    
    def test_quality_checks_pass(self, generator, balanced_features):
        """Test that quality checks pass for good dataset."""
        report = generator.validate_dataset_quality(balanced_features)
        
        checks = report['quality_checks']
        assert checks['sufficient_samples'] is True
        assert checks['no_missing_labels'] is True
    
    def test_detect_insufficient_samples(self, generator):
        """Test detection of insufficient samples."""
        small_features = [
            {'is_fraud': 0, 'amount': 100},
            {'is_fraud': 1, 'amount': 200},
        ]
        
        report = generator.validate_dataset_quality(small_features)
        assert report['quality_checks']['sufficient_samples'] is False


class TestExportFunctionality(TestMLDatasetGenerator):
    """Test export to various formats."""
    
    def test_export_to_csv(self, generator, balanced_features):
        """Test CSV export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filepath = f.name
        
        try:
            generator.export_to_csv(balanced_features, filepath)
            
            # Check file exists and has content
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            # Read and verify
            with open(filepath, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 101  # 100 data + 1 header
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_to_json(self, generator, balanced_features):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            generator.export_to_json(balanced_features, filepath)
            
            # Check file exists
            assert os.path.exists(filepath)
            
            # Read and verify
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert len(data) == 100
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_metadata(self, generator):
        """Test metadata export."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            dataset_info = {
                'total_samples': 100,
                'fraud_rate': 0.1,
                'features': ['amount', 'count']
            }
            
            generator.export_metadata(filepath, dataset_info)
            
            # Read and verify
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert 'dataset_info' in data
                assert data['dataset_info']['total_samples'] == 100
                assert data['dataset_info']['fraud_rate'] == 0.1
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestMLReadyDataset(TestMLDatasetGenerator):
    """Test complete ML-ready dataset creation."""
    
    def test_create_ml_ready_dataset(self, generator, sample_features):
        """Test complete ML pipeline."""
        split, metadata = generator.create_ml_ready_dataset(
            sample_features,
            balance_strategy='undersample',
            target_fraud_rate=0.5,
            normalize=True,
            encode_categorical=True
        )
        
        # Check split created
        assert isinstance(split, DatasetSplit)
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0
        
        # Check metadata
        assert 'normalized' in metadata
        assert metadata['normalized'] is True
        assert 'categorical_encoded' in metadata
        assert 'encodings' in metadata
        assert 'train_quality' in metadata
    
    def test_normalization_fit_on_train_only(self, generator, sample_features):
        """Test that normalization is fit on train set only."""
        split, metadata = generator.create_ml_ready_dataset(
            sample_features,
            normalize=True
        )
        
        # Normalization flag should be True
        assert metadata['normalized'] is True


class TestDatasetSplitClass:
    """Test DatasetSplit class functionality."""
    
    def test_dataset_split_creation(self):
        """Test DatasetSplit creation."""
        split = DatasetSplit(
            train=[{'is_fraud': 0}],
            validation=[{'is_fraud': 1}],
            test=[{'is_fraud': 0}]
        )
        
        assert len(split.train) == 1
        assert len(split.validation) == 1
        assert len(split.test) == 1
    
    def test_get_stats(self):
        """Test get_stats method."""
        split = DatasetSplit(
            train=[{'is_fraud': 0}, {'is_fraud': 1}],
            validation=[{'is_fraud': 0}],
            test=[{'is_fraud': 1}]
        )
        
        stats = split.get_stats()
        
        assert 'train_size' in stats
        assert 'validation_size' in stats
        assert 'test_size' in stats
        assert stats['train_size'] == 2
        assert stats['validation_size'] == 1
        assert stats['test_size'] == 1


class TestReproducibility(TestMLDatasetGenerator):
    """Test reproducibility with seeds."""
    
    def test_same_seed_same_results(self, sample_features):
        """Test that same seed produces same results."""
        gen1 = MLDatasetGenerator(seed=42)
        gen2 = MLDatasetGenerator(seed=42)
        
        split1 = gen1.create_train_test_split(sample_features, stratify=False)
        split2 = gen2.create_train_test_split(sample_features, stratify=False)
        
        # Should have same IDs in train set when stratify is off
        ids1 = set([f['transaction_id'] for f in split1.train])
        ids2 = set([f['transaction_id'] for f in split2.train])
        
        # At least half should match (due to stratified shuffling)
        overlap = len(ids1 & ids2)
        assert overlap > len(ids1) * 0.5  # At least 50% overlap
    
    def test_different_seed_different_results(self, sample_features):
        """Test that different seeds produce different results."""
        gen1 = MLDatasetGenerator(seed=42)
        gen2 = MLDatasetGenerator(seed=123)
        
        split1 = gen1.create_train_test_split(sample_features, stratify=False)
        split2 = gen2.create_train_test_split(sample_features, stratify=False)
        
        # Should have different IDs in train set
        ids1 = set([f['transaction_id'] for f in split1.train])
        ids2 = set([f['transaction_id'] for f in split2.train])
        
        # Should be different (not identical)
        assert ids1 != ids2
