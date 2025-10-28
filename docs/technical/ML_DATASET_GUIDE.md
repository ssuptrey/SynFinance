# ML Dataset Preparation Guide

**Version**: 0.5.0  
**Last Updated**: October 26, 2025

## Overview

This guide explains how to prepare ML-ready datasets from SynFinance transaction data for fraud detection model training. The dataset preparation pipeline handles balancing, splitting, normalization, and export to multiple formats.

## Quick Start

```python
from src.generators.ml_dataset_generator import MLDatasetGenerator

# Initialize generator
dataset_gen = MLDatasetGenerator(seed=42)

# Create ML-ready dataset
split, metadata = dataset_gen.create_ml_ready_dataset(
    features_list,
    balance_strategy='undersample',
    target_fraud_rate=0.5,
    normalize=True,
    encode_categorical=True
)

# Access train/validation/test sets
print(f"Train: {len(split.train)} samples")
print(f"Validation: {len(split.validation)} samples")
print(f"Test: {len(split.test)} samples")
```

## Dataset Preparation Pipeline

### Step 1: Class Balancing

Fraud detection datasets are typically imbalanced (1-10% fraud). Balance the dataset for better model training.

#### Undersampling (Recommended)
Reduces majority class to match minority class.

```python
balanced = dataset_gen.create_balanced_dataset(
    features_list,
    target_fraud_rate=0.5,
    strategy='undersample'
)
```

**Pros:**
- Faster training
- Prevents majority class bias
- Good for small datasets

**Cons:**
- Loses normal transaction data
- May miss rare normal patterns

#### Oversampling
Duplicates minority class to match majority class.

```python
balanced = dataset_gen.create_balanced_dataset(
    features_list,
    target_fraud_rate=0.5,
    strategy='oversample'
)
```

**Pros:**
- Keeps all data
- Good for large datasets

**Cons:**
- Slower training
- Risk of overfitting
- Duplicates fraud examples

#### Custom Fraud Rate
Set any target fraud rate (not just 50-50).

```python
balanced = dataset_gen.create_balanced_dataset(
    features_list,
    target_fraud_rate=0.3,  # 30% fraud
    strategy='undersample'
)
```

### Step 2: Train/Validation/Test Split

Split dataset into three sets with stratification to maintain class balance.

```python
split = dataset_gen.create_train_test_split(
    balanced_features,
    train_ratio=0.70,   # 70% training
    val_ratio=0.15,     # 15% validation
    stratify=True       # Maintain fraud rate
)

# Access splits
train_data = split.train
val_data = split.validation
test_data = split.test

# Get statistics
stats = split.get_stats()
print(f"Train fraud rate: {stats['train_fraud_rate']:.1%}")
print(f"Val fraud rate: {stats['validation_fraud_rate']:.1%}")
print(f"Test fraud rate: {stats['test_fraud_rate']:.1%}")
```

**Split Ratios:**
- **Train (70%)**: Model training
- **Validation (15%)**: Hyperparameter tuning
- **Test (15%)**: Final evaluation

**Stratification**: Ensures each split has same fraud rate as original dataset.

### Step 3: Feature Normalization

Normalize continuous features to [0, 1] range using min-max scaling.

```python
# Normalize (fit on train, apply to all)
split.train = dataset_gen.normalize_features(split.train, fit=True)
split.validation = dataset_gen.normalize_features(split.validation, fit=False)
split.test = dataset_gen.normalize_features(split.test, fit=False)
```

**Important:** Always fit normalization on training data only to prevent data leakage.

**Excluded Columns:**
- `transaction_id`
- `is_fraud`
- `fraud_type`
- Binary flags (is_unusual_hour, is_weekend, etc.)

**Normalization Formula:**
```
normalized = (value - min) / (max - min)
```

### Step 4: Categorical Encoding

Encode categorical features (fraud_type) to integers.

```python
encoded, encodings = dataset_gen.encode_categorical_features(features_list)

# Access encoding mappings
print(encodings['fraud_type'])
# {'None': 0, 'Card Cloning': 1, 'Velocity Abuse': 2, ...}
```

**Output:** Creates `fraud_type_encoded` column with integer values.

### Step 5: Data Quality Validation

Validate dataset quality before training.

```python
quality_report = dataset_gen.validate_dataset_quality(split.train)

print(f"Total samples: {quality_report['total_samples']}")
print(f"Fraud rate: {quality_report['fraud_rate']:.2%}")
print(f"Missing values: {quality_report['missing_values']}")

# Check quality flags
checks = quality_report['quality_checks']
assert checks['class_balance'] == True
assert checks['sufficient_samples'] == True
assert checks['no_missing_labels'] == True
```

**Quality Checks:**
- Class balance (fraud rate 1-99%)
- Sufficient samples (≥100)
- No missing labels

## Complete Pipeline Example

```python
from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.ml_dataset_generator import MLDatasetGenerator

# 1. Generate transaction data
generator = DataGenerator(num_customers=100, num_days=30)
customers = generator.generate_customers()
transactions = generator.generate_transactions(num_transactions=5000)

# 2. Inject fraud patterns
fraud_gen = FraudPatternGenerator(seed=42)
transactions = fraud_gen.inject_fraud_patterns(
    transactions, customers, fraud_rate=0.1
)

# 3. Engineer features
feature_engineer = MLFeatureEngineer()
features_list = []

# Build history and engineer features
# ... (see ML_FEATURES.md for details)

# 4. Create ML-ready dataset
dataset_gen = MLDatasetGenerator(seed=42)

split, metadata = dataset_gen.create_ml_ready_dataset(
    features_list,
    balance_strategy='undersample',
    target_fraud_rate=0.5,
    normalize=True,
    encode_categorical=True
)

# 5. Validate quality
train_quality = dataset_gen.validate_dataset_quality(split.train)
print(f"Dataset ready: {train_quality['total_samples']} train samples")
```

## Export Formats

### CSV Export

```python
dataset_gen.export_to_csv(split.train, 'output/train.csv')
dataset_gen.export_to_csv(split.validation, 'output/validation.csv')
dataset_gen.export_to_csv(split.test, 'output/test.csv')
```

**Use Case:** Easy viewing in Excel, compatible with pandas

### JSON Export

```python
dataset_gen.export_to_json(split.train, 'output/train.json')
dataset_gen.export_to_json(split.test, 'output/test.json')
```

**Use Case:** Web APIs, JavaScript frameworks

### Parquet Export (Efficient Storage)

```python
# Requires: pip install pyarrow pandas
dataset_gen.export_to_parquet(split.train, 'output/train.parquet')
dataset_gen.export_to_parquet(split.test, 'output/test.parquet')
```

**Use Case:** Large datasets, data warehousing, Spark/Dask

**Benefits:**
- 10-100x smaller than CSV
- Faster read/write
- Column-based storage
- Preserves data types

### NumPy Export (Direct ML Use)

```python
# Export train set
X_train, y_train, feature_names = dataset_gen.export_to_numpy(
    split.train,
    'output/train'
)
# Creates: train_X.npy, train_y.npy, train_features.json

# Export test set
X_test, y_test, _ = dataset_gen.export_to_numpy(
    split.test,
    'output/test'
)

# Load and use
import numpy as np
X = np.load('output/train_X.npy')
y = np.load('output/train_y.npy')

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
```

**Use Case:** Direct scikit-learn, TensorFlow, PyTorch input

### Metadata Export

```python
dataset_gen.export_metadata('output/metadata.json', metadata)
```

**Metadata Contents:**
- Total samples
- Balance strategy
- Fraud rate
- Normalization status
- Categorical encodings
- Split statistics
- Quality metrics
- Feature count

## Data Quality Analysis

### Correlation Analysis

Check for highly correlated features.

```bash
python scripts/validate_data_quality.py output/train.csv \
    --correlation-threshold 0.8
```

**Output:**
- Correlation heatmap
- List of high correlation pairs
- Recommendation to remove redundant features

### Missing Value Detection

```python
quality_report = dataset_gen.validate_dataset_quality(features_list)
missing = quality_report['missing_values']

if missing:
    print("Missing values found:")
    for feature, count in missing.items():
        print(f"  {feature}: {count}")
else:
    print("No missing values")
```

### Outlier Detection

```bash
python scripts/validate_data_quality.py output/train.csv \
    --iqr-multiplier 1.5
```

**Method:** IQR (Interquartile Range)
- Lower bound: Q1 - 1.5 × IQR
- Upper bound: Q3 + 1.5 × IQR

**Output:**
- Features with outliers
- Outlier counts and percentages
- Recommended actions

### Variance Analysis

Low-variance features (near-constant values) provide little information.

```bash
python scripts/analyze_variance.py output/train.csv
```

**Threshold:** Variance < 0.01

**Action:** Consider removing low-variance features.

## Best Practices

### 1. Always Use Stratified Splitting

```python
# ✓ Good - maintains fraud rate
split = dataset_gen.create_train_test_split(
    features, stratify=True
)

# ✗ Bad - may create imbalanced splits
split = dataset_gen.create_train_test_split(
    features, stratify=False
)
```

### 2. Fit Normalization on Train Only

```python
# ✓ Good - prevents data leakage
split.train = dataset_gen.normalize_features(split.train, fit=True)
split.test = dataset_gen.normalize_features(split.test, fit=False)

# ✗ Bad - data leakage
split.train = dataset_gen.normalize_features(split.train, fit=True)
split.test = dataset_gen.normalize_features(split.test, fit=True)  # Wrong!
```

### 3. Validate Dataset Quality

```python
# Always validate before training
quality = dataset_gen.validate_dataset_quality(split.train)
assert quality['quality_checks']['class_balance']
assert quality['quality_checks']['sufficient_samples']
assert quality['quality_checks']['no_missing_labels']
```

### 4. Use Reproducible Seeds

```python
# Set seed for reproducibility
dataset_gen = MLDatasetGenerator(seed=42)

# Results will be identical across runs
split1 = dataset_gen.create_train_test_split(features)
split2 = dataset_gen.create_train_test_split(features)
# split1.train == split2.train (same samples)
```

### 5. Monitor Class Balance

```python
stats = split.get_stats()
for set_name in ['train', 'validation', 'test']:
    fraud_rate = stats[f'{set_name}_fraud_rate']
    assert 0.4 <= fraud_rate <= 0.6, f"{set_name} fraud rate {fraud_rate:.1%} out of range"
```

## Common Issues and Solutions

### Issue 1: Imbalanced Test Set

**Problem:** Test set has different fraud rate than train/validation.

**Solution:** Use stratified splitting.
```python
split = dataset_gen.create_train_test_split(features, stratify=True)
```

### Issue 2: Data Leakage

**Problem:** Normalization fit on entire dataset instead of train only.

**Solution:** Always fit on train, apply to validation/test.
```python
split.train = dataset_gen.normalize_features(split.train, fit=True)
split.validation = dataset_gen.normalize_features(split.validation, fit=False)
split.test = dataset_gen.normalize_features(split.test, fit=False)
```

### Issue 3: Insufficient Training Data

**Problem:** After undersampling, too few samples remain.

**Solution:** Use oversampling or generate more data.
```python
# Option 1: Oversample instead
balanced = dataset_gen.create_balanced_dataset(
    features, strategy='oversample'
)

# Option 2: Generate more transactions
transactions = generator.generate_transactions(num_transactions=10000)
```

### Issue 4: High Correlation Between Features

**Problem:** Features with r > 0.9 correlation.

**Solution:** Remove one of the correlated features.
```python
# Run correlation analysis
python scripts/validate_data_quality.py train.csv

# Remove redundant feature
for sample in features:
    del sample['redundant_feature_name']
```

## Performance Tips

### 1. Batch Processing

```python
# Process in batches for large datasets
batch_size = 1000
for i in range(0, len(features), batch_size):
    batch = features[i:i+batch_size]
    # Process batch
```

### 2. Use Parquet for Large Datasets

```python
# CSV: ~500 MB
dataset_gen.export_to_csv(data, 'large.csv')

# Parquet: ~50 MB (10x smaller)
dataset_gen.export_to_parquet(data, 'large.parquet')
```

### 3. Cache Normalized Data

```python
# Normalize once, save results
split.train = dataset_gen.normalize_features(split.train, fit=True)
dataset_gen.export_to_parquet(split.train, 'train_normalized.parquet')

# Load later
import pandas as pd
train = pd.read_parquet('train_normalized.parquet').to_dict('records')
```

## API Reference

### MLDatasetGenerator

```python
class MLDatasetGenerator:
    """Generate ML-ready datasets for fraud detection."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        
    def create_balanced_dataset(
        self,
        features: List[Dict],
        target_fraud_rate: float = 0.5,
        strategy: str = 'undersample'
    ) -> List[Dict]:
        """Balance dataset."""
        
    def create_train_test_split(
        self,
        features: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        stratify: bool = True
    ) -> DatasetSplit:
        """Split into train/validation/test."""
        
    def normalize_features(
        self,
        features: List[Dict],
        fit: bool = True,
        exclude_columns: Optional[List[str]] = None
    ) -> List[Dict]:
        """Normalize features to [0, 1]."""
        
    def encode_categorical_features(
        self,
        features: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Encode categorical features."""
        
    def validate_dataset_quality(
        self,
        features: List[Dict]
    ) -> Dict:
        """Validate dataset quality."""
        
    def export_to_csv(self, features: List[Dict], filepath: str):
        """Export to CSV."""
        
    def export_to_json(self, features: List[Dict], filepath: str):
        """Export to JSON."""
        
    def export_to_parquet(self, features: List[Dict], filepath: str):
        """Export to Parquet."""
        
    def export_to_numpy(
        self,
        features: List[Dict],
        filepath_prefix: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Export to NumPy arrays."""
        
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
        """Complete ML-ready dataset pipeline."""
```

### DatasetSplit

```python
@dataclass
class DatasetSplit:
    """Container for train/validation/test split."""
    
    train: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    
    def get_stats(self) -> Dict:
        """Get split statistics."""
```

## Example Scripts

### Generate Training Dataset

```bash
python examples/train_fraud_detector.py \
    --num-transactions 5000 \
    --fraud-rate 0.1 \
    --output-dir output/ml_training
```

### Validate Dataset Quality

```bash
python scripts/validate_data_quality.py \
    output/train.csv \
    --output-dir output/quality_validation
```

### Analyze Correlations

```bash
python scripts/analyze_correlations.py output/train.csv
```

## Additional Resources

- [ML Features Guide](ML_FEATURES.md)
- [Fraud Detection Tutorial](../../examples/fraud_detection_tutorial.ipynb)
- [Integration Guide](../guides/INTEGRATION_GUIDE.md)
- [API Documentation](../INDEX.md)

## Version History

- **0.5.0** (Oct 26, 2025): Complete dataset preparation pipeline
- **0.4.0**: Initial ML dataset support
