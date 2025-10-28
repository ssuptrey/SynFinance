"""
ML Dataset Generator for Fraud Detection

This module provides functionality to generate ML-ready datasets from transaction data,
including stratified sampling, train/test splitting, feature normalization, and export
to various formats (Parquet, NumPy, CSV).

Features:
- Stratified sampling for class balance
- Train/validation/test splitting (70/15/15)
- Feature normalization and scaling
- Label encoding for categorical features
- Export to multiple formats
- Data quality validation

Author: SynFinance Development Team
Version: 0.5.0
Date: October 26, 2025
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict, Counter
import json
import csv


@dataclass
class DatasetSplit:
    """Container for train/validation/test split."""
    train: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    
    def get_stats(self) -> Dict:
        """Get statistics about the split."""
        return {
            'train_size': len(self.train),
            'validation_size': len(self.validation),
            'test_size': len(self.test),
            'total_size': len(self.train) + len(self.validation) + len(self.test),
            'train_fraud_rate': self._fraud_rate(self.train),
            'validation_fraud_rate': self._fraud_rate(self.validation),
            'test_fraud_rate': self._fraud_rate(self.test)
        }
    
    def _fraud_rate(self, data: List[Dict]) -> float:
        """Calculate fraud rate in dataset."""
        if not data:
            return 0.0
        fraud_count = sum(1 for item in data if item.get('is_fraud', 0) == 1)
        return fraud_count / len(data)


class MLDatasetGenerator:
    """
    Generate ML-ready datasets for fraud detection.
    
    Features:
    - Stratified sampling for class balance
    - Train/validation/test splitting
    - Feature normalization
    - Label encoding
    - Multiple export formats
    - Data quality validation
    
    Example:
        generator = MLDatasetGenerator(seed=42)
        split = generator.create_train_test_split(features, test_size=0.3)
        generator.export_to_parquet(split.train, 'train.parquet')
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize ML dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.feature_stats = {}  # For normalization
    
    def create_balanced_dataset(
        self,
        features: List[Dict],
        target_fraud_rate: float = 0.5,
        strategy: str = 'undersample'
    ) -> List[Dict]:
        """
        Create balanced dataset for training.
        
        Args:
            features: List of feature dictionaries
            target_fraud_rate: Desired fraud rate (0.0-1.0)
            strategy: 'undersample' or 'oversample'
            
        Returns:
            Balanced list of features
        """
        # Separate fraud and non-fraud
        fraud_samples = [f for f in features if f.get('is_fraud', 0) == 1]
        normal_samples = [f for f in features if f.get('is_fraud', 0) == 0]
        
        fraud_count = len(fraud_samples)
        normal_count = len(normal_samples)
        
        if fraud_count == 0:
            return normal_samples
        
        if strategy == 'undersample':
            # Reduce majority class
            target_normal = int(fraud_count * (1 - target_fraud_rate) / target_fraud_rate)
            if target_normal < normal_count:
                normal_samples = random.sample(normal_samples, target_normal)
        
        elif strategy == 'oversample':
            # Increase minority class
            target_fraud = int(normal_count * target_fraud_rate / (1 - target_fraud_rate))
            if target_fraud > fraud_count:
                # Oversample with replacement
                additional = target_fraud - fraud_count
                fraud_samples.extend(random.choices(fraud_samples, k=additional))
        
        # Combine and shuffle
        balanced = fraud_samples + normal_samples
        random.shuffle(balanced)
        
        return balanced
    
    def create_train_test_split(
        self,
        features: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        stratify: bool = True
    ) -> DatasetSplit:
        """
        Split data into train/validation/test sets.
        
        Args:
            features: List of feature dictionaries
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            stratify: Maintain class balance in splits
            
        Returns:
            DatasetSplit with train/validation/test splits
        """
        if stratify:
            # Stratified split to maintain fraud rate
            fraud_samples = [f for f in features if f.get('is_fraud', 0) == 1]
            normal_samples = [f for f in features if f.get('is_fraud', 0) == 0]
            
            # Shuffle each class
            random.shuffle(fraud_samples)
            random.shuffle(normal_samples)
            
            # Split fraud samples
            fraud_train_size = int(len(fraud_samples) * train_ratio)
            fraud_val_size = int(len(fraud_samples) * val_ratio)
            
            fraud_train = fraud_samples[:fraud_train_size]
            fraud_val = fraud_samples[fraud_train_size:fraud_train_size + fraud_val_size]
            fraud_test = fraud_samples[fraud_train_size + fraud_val_size:]
            
            # Split normal samples
            normal_train_size = int(len(normal_samples) * train_ratio)
            normal_val_size = int(len(normal_samples) * val_ratio)
            
            normal_train = normal_samples[:normal_train_size]
            normal_val = normal_samples[normal_train_size:normal_train_size + normal_val_size]
            normal_test = normal_samples[normal_train_size + normal_val_size:]
            
            # Combine and shuffle
            train = fraud_train + normal_train
            val = fraud_val + normal_val
            test = fraud_test + normal_test
            
        else:
            # Simple random split
            random.shuffle(features)
            train_size = int(len(features) * train_ratio)
            val_size = int(len(features) * val_ratio)
            
            train = features[:train_size]
            val = features[train_size:train_size + val_size]
            test = features[train_size + val_size:]
        
        # Shuffle final splits
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return DatasetSplit(train=train, validation=val, test=test)
    
    def normalize_features(
        self,
        features: List[Dict],
        fit: bool = True,
        exclude_columns: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Normalize numerical features using min-max scaling.
        
        Args:
            features: List of feature dictionaries
            fit: If True, fit scaler on this data; if False, use existing stats
            exclude_columns: Columns to skip normalization
            
        Returns:
            List of feature dictionaries with normalized values
        """
        if exclude_columns is None:
            exclude_columns = ['transaction_id', 'is_fraud', 'fraud_type',
                              'is_unusual_hour', 'is_weekend', 'is_holiday',
                              'new_merchant_flag', 'temporal_cluster_flag']
        
        numerical_cols = [
            'daily_txn_count', 'weekly_txn_count', 'daily_txn_amount', 'weekly_txn_amount',
            'avg_daily_amount', 'avg_weekly_amount', 'txn_frequency_1h', 'txn_frequency_6h',
            'txn_frequency_24h', 'amount_velocity_1h', 'amount_velocity_6h', 'amount_velocity_24h',
            'distance_from_home', 'avg_distance_last_10', 'distance_variance', 'unique_cities_7d',
            'travel_velocity_kmh', 'hour_of_day', 'day_of_week', 'category_diversity_score',
            'merchant_loyalty_score', 'avg_merchant_reputation', 'refund_rate_30d',
            'declined_rate_7d', 'shared_merchant_count', 'shared_location_count',
            'customer_proximity_score'
        ]
        
        # Filter out excluded columns
        cols_to_normalize = [col for col in numerical_cols if col not in exclude_columns]
        
        if fit:
            # Calculate min/max for each column
            self.feature_stats = {}
            for col in cols_to_normalize:
                values = [f.get(col, 0) for f in features if col in f]
                if values:
                    self.feature_stats[col] = {
                        'min': min(values),
                        'max': max(values)
                    }
        
        # Normalize
        normalized = []
        for feature in features:
            norm_feature = feature.copy()
            for col in cols_to_normalize:
                if col in feature and col in self.feature_stats:
                    val = feature[col]
                    min_val = self.feature_stats[col]['min']
                    max_val = self.feature_stats[col]['max']
                    
                    # Min-max normalization
                    if max_val > min_val:
                        norm_feature[col] = (val - min_val) / (max_val - min_val)
                    else:
                        norm_feature[col] = 0
            
            normalized.append(norm_feature)
        
        return normalized
    
    def encode_categorical_features(
        self,
        features: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Encode categorical features using label encoding.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Tuple of (encoded features, encoding mapping)
        """
        categorical_cols = ['fraud_type']
        
        # Build encoding mappings
        encodings = {}
        for col in categorical_cols:
            unique_values = list(set(f.get(col, 'None') for f in features))
            encodings[col] = {val: idx for idx, val in enumerate(sorted(unique_values))}
        
        # Apply encodings
        encoded = []
        for feature in features:
            enc_feature = feature.copy()
            for col in categorical_cols:
                if col in feature:
                    val = feature[col]
                    enc_feature[f"{col}_encoded"] = encodings[col].get(val, 0)
            encoded.append(enc_feature)
        
        return encoded, encodings
    
    def validate_dataset_quality(self, features: List[Dict]) -> Dict:
        """
        Validate dataset quality.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Quality validation report
        """
        report = {
            'total_samples': len(features),
            'fraud_count': sum(1 for f in features if f.get('is_fraud', 0) == 1),
            'normal_count': sum(1 for f in features if f.get('is_fraud', 0) == 0),
            'fraud_rate': 0,
            'missing_values': {},
            'feature_ranges': {},
            'correlations': {},
            'quality_checks': {}
        }
        
        if features:
            report['fraud_rate'] = report['fraud_count'] / len(features)
            
            # Check for missing values
            all_keys = set()
            for f in features:
                all_keys.update(f.keys())
            
            for key in all_keys:
                missing = sum(1 for f in features if key not in f or f[key] is None)
                if missing > 0:
                    report['missing_values'][key] = missing
            
            # Feature ranges
            numerical_features = ['daily_txn_amount', 'travel_velocity_kmh', 'distance_from_home']
            for feat in numerical_features:
                values = [f.get(feat, 0) for f in features if feat in f]
                if values:
                    report['feature_ranges'][feat] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values)
                    }
            
            # Quality checks
            report['quality_checks']['class_balance'] = (
                0.01 <= report['fraud_rate'] <= 0.99  # Not too imbalanced
            )
            report['quality_checks']['sufficient_samples'] = len(features) >= 100
            report['quality_checks']['no_missing_labels'] = 'is_fraud' not in report['missing_values']
        
        return report
    
    def export_to_csv(self, features: List[Dict], filepath: str):
        """Export features to CSV file."""
        if not features:
            return
        
        # Get all column names
        columns = list(features[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(features)
    
    def export_to_json(self, features: List[Dict], filepath: str):
        """Export features to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2)
    
    def export_to_parquet(self, features: List[Dict], filepath: str):
        """
        Export features to Parquet format using pandas and pyarrow.
        Falls back to JSON if pandas/pyarrow not available.
        """
        try:
            import pandas as pd
            df = pd.DataFrame(features)
            df.to_parquet(filepath, engine='pyarrow', compression='snappy')
            print(f"✓ Exported to Parquet: {filepath}")
        except ImportError:
            print("⚠ pandas/pyarrow not available. Falling back to JSON.")
            json_path = filepath.replace('.parquet', '.json')
            self.export_to_json(features, json_path)
            print(f"✓ Exported to JSON: {json_path}")
    
    def export_to_numpy(self, features: List[Dict], filepath_prefix: str):
        """
        Export features to NumPy arrays (X and y).
        
        Args:
            features: List of feature dictionaries
            filepath_prefix: Prefix for output files (e.g., 'train' -> 'train_X.npy', 'train_y.npy')
        """
        import numpy as np
        
        # Define excluded columns
        exclude_cols = {'transaction_id', 'is_fraud', 'fraud_type', 'fraud_type_encoded'}
        
        if not features:
            return
        
        # Get feature columns
        all_cols = set(features[0].keys())
        feature_cols = sorted(all_cols - exclude_cols)
        
        # Extract X and y
        X = np.array([[sample.get(col, 0) for col in feature_cols] for sample in features])
        y = np.array([sample.get('is_fraud', 0) for sample in features])
        
        # Save arrays
        np.save(f'{filepath_prefix}_X.npy', X)
        np.save(f'{filepath_prefix}_y.npy', y)
        
        # Save feature names
        with open(f'{filepath_prefix}_features.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        print(f"✓ Exported to NumPy: {filepath_prefix}_X.npy, {filepath_prefix}_y.npy")
        print(f"  - Shape: X={X.shape}, y={y.shape}")
        return X, y, feature_cols
    
    def export_metadata(self, filepath: str, dataset_info: Dict):
        """Export dataset metadata."""
        metadata = {
            'generator': 'MLDatasetGenerator',
            'version': '0.5.0',
            'seed': self.seed,
            'dataset_info': dataset_info,
            'feature_stats': self.feature_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def create_ml_ready_dataset(
        self,
        features: List[Dict],
        balance_strategy: str = 'undersample',
        target_fraud_rate: float = 0.5,
        normalize: bool = True,
        encode_categorical: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[DatasetSplit, Dict]:
        """
        Create complete ML-ready dataset with all preprocessing.
        
        Args:
            features: Raw feature dictionaries
            balance_strategy: 'undersample', 'oversample', or 'none'
            target_fraud_rate: Target fraud rate for balancing
            normalize: Whether to normalize features
            encode_categorical: Whether to encode categorical features
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (DatasetSplit, metadata_dict)
        """
        print(f"Creating ML-ready dataset from {len(features)} samples...")
        
        # Step 1: Balance dataset
        if balance_strategy != 'none':
            features = self.create_balanced_dataset(
                features,
                target_fraud_rate=target_fraud_rate,
                strategy=balance_strategy
            )
            print(f"Balanced to {len(features)} samples (fraud rate: {target_fraud_rate})")
        
        # Step 2: Encode categorical features
        encodings = {}
        if encode_categorical:
            features, encodings = self.encode_categorical_features(features)
            print("Encoded categorical features")
        
        # Step 3: Split data
        split = self.create_train_test_split(
            features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify=True
        )
        print(f"Split: {len(split.train)} train, {len(split.validation)} val, {len(split.test)} test")
        
        # Step 4: Normalize features (fit on train only)
        if normalize:
            split.train = self.normalize_features(split.train, fit=True)
            split.validation = self.normalize_features(split.validation, fit=False)
            split.test = self.normalize_features(split.test, fit=False)
            print("Normalized features")
        
        # Step 5: Validate quality
        train_quality = self.validate_dataset_quality(split.train)
        
        # Compile metadata
        metadata = {
            'total_samples': len(features),
            'balance_strategy': balance_strategy,
            'target_fraud_rate': target_fraud_rate,
            'normalized': normalize,
            'categorical_encoded': encode_categorical,
            'encodings': encodings,
            'split_stats': split.get_stats(),
            'train_quality': train_quality,
            'feature_count': 32
        }
        
        print("Dataset creation complete!")
        return split, metadata


def generate_ml_dataset_from_transactions(
    transactions: List[Dict],
    customers: List[Dict],
    output_dir: str = 'output/ml_dataset',
    balance_strategy: str = 'undersample',
    target_fraud_rate: float = 0.5
) -> Dict:
    """
    High-level function to generate complete ML dataset from transactions.
    
    Args:
        transactions: List of transaction dictionaries
        customers: List of customer dictionaries
        output_dir: Output directory for files
        balance_strategy: Balancing strategy
        target_fraud_rate: Target fraud rate
        
    Returns:
        Dictionary with file paths and metadata
    """
    from .ml_features import MLFeatureEngineer
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Engineer features
    print("Engineering features...")
    engineer = MLFeatureEngineer()
    customer_map = {c['Customer_ID']: c for c in customers}
    history_map = defaultdict(list)
    
    features = []
    for txn in transactions:
        customer_id = txn.get('Customer_ID')
        customer = customer_map.get(customer_id, {})
        history = history_map[customer_id]
        
        # Engineer features
        ml_features = engineer.engineer_features(txn, customer, history)
        features.append(ml_features.to_dict())
        
        # Update history
        history_map[customer_id].append(txn)
    
    print(f"Engineered {len(features)} feature sets")
    
    # Step 2: Create ML-ready dataset
    generator = MLDatasetGenerator(seed=42)
    split, metadata = generator.create_ml_ready_dataset(
        features,
        balance_strategy=balance_strategy,
        target_fraud_rate=target_fraud_rate
    )
    
    # Step 3: Export datasets
    print("Exporting datasets...")
    generator.export_to_csv(split.train, f"{output_dir}/train.csv")
    generator.export_to_csv(split.validation, f"{output_dir}/validation.csv")
    generator.export_to_csv(split.test, f"{output_dir}/test.csv")
    generator.export_metadata(f"{output_dir}/metadata.json", metadata)
    
    print(f"Datasets exported to {output_dir}/")
    
    return {
        'output_dir': output_dir,
        'files': {
            'train': f"{output_dir}/train.csv",
            'validation': f"{output_dir}/validation.csv",
            'test': f"{output_dir}/test.csv",
            'metadata': f"{output_dir}/metadata.json"
        },
        'metadata': metadata
    }
