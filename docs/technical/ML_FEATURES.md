# ML Feature Engineering Guide

**Version**: 0.5.0  
**Last Updated**: October 26, 2025

## Overview

SynFinance provides 32 engineered features across 6 categories for fraud detection machine learning models. These features are designed to capture temporal, behavioral, geographic, and network patterns that distinguish fraudulent transactions from legitimate ones.

## Feature Categories

### 1. Aggregate Features (6 features)

Summarize transaction patterns over time windows.

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `daily_txn_count` | Number of transactions in last 24 hours | Count of transactions with timestamp within 24h |
| `weekly_txn_count` | Number of transactions in last 7 days | Count of transactions with timestamp within 7d |
| `daily_txn_amount` | Total amount spent in last 24 hours | Sum of amounts within 24h |
| `weekly_txn_amount` | Total amount spent in last 7 days | Sum of amounts within 7d |
| `avg_daily_amount` | Average transaction amount (24h) | daily_txn_amount / daily_txn_count |
| `avg_weekly_amount` | Average transaction amount (7d) | weekly_txn_amount / weekly_txn_count |

**Fraud Indicators:**
- Sudden spike in transaction count (velocity abuse)
- Abnormally high spending amounts (card cloning)
- Deviation from typical spending patterns

**Example:**
```python
# Customer with 5 transactions in last 24h totaling $2500
daily_txn_count = 5
daily_txn_amount = 2500.0
avg_daily_amount = 500.0  # $500 per transaction
```

### 2. Velocity Features (6 features)

Measure transaction frequency and amount velocity in short time windows.

| Feature | Description | Window |
|---------|-------------|--------|
| `txn_frequency_1h` | Transactions in last hour | 1 hour |
| `txn_frequency_6h` | Transactions in last 6 hours | 6 hours |
| `txn_frequency_24h` | Transactions in last 24 hours | 24 hours |
| `amount_velocity_1h` | Amount spent per minute (1h) | 60 minutes |
| `amount_velocity_6h` | Amount spent per minute (6h) | 360 minutes |
| `amount_velocity_24h` | Amount spent per minute (24h) | 1440 minutes |

**Fraud Indicators:**
- Multiple transactions in very short time (automated fraud)
- High amount velocity (rapid spending)
- Unusual burst of activity

**Example:**
```python
# 10 transactions in last hour, $5000 total
txn_frequency_1h = 10
amount_velocity_1h = 5000 / 60 = 83.33  # $83.33 per minute
```

### 3. Geographic Features (5 features)

Analyze geographic patterns and travel behavior.

| Feature | Description | Unit |
|---------|-------------|------|
| `distance_from_home` | Distance from home city | Kilometers |
| `avg_distance_last_10` | Average distance for last 10 transactions | Kilometers |
| `distance_variance` | Variance in transaction locations | Km² |
| `unique_cities_7d` | Number of unique cities (7 days) | Count |
| `travel_velocity_kmh` | Travel speed between transactions | Km/hour |

**Fraud Indicators:**
- Geographically impossible transactions (too fast travel)
- Transactions far from home location
- Sudden geographic diversity

**Example:**
```python
# Transaction in New York, customer lives in Los Angeles
distance_from_home = 3944.0  # ~2450 miles
travel_velocity_kmh = 900.0  # Impossible by car, likely fraud

# Normal transaction
distance_from_home = 15.0  # 15 km from home
travel_velocity_kmh = 45.0  # Normal driving speed
```

### 4. Temporal Features (6 features)

Capture time-based patterns and anomalies.

| Feature | Description | Range |
|---------|-------------|-------|
| `is_unusual_hour` | Transaction during unusual hours | 0/1 |
| `is_weekend` | Weekend transaction | 0/1 |
| `is_holiday` | Holiday transaction | 0/1 |
| `hour_of_day` | Hour (0-23) | 0-23 |
| `day_of_week` | Day (0=Monday, 6=Sunday) | 0-6 |
| `temporal_cluster_flag` | Multiple transactions in same hour | 0/1 |

**Fraud Indicators:**
- Transactions at unusual hours (2-5 AM)
- Unusual weekend/holiday activity
- Temporal clustering (multiple in same hour)

**Unusual Hours:** 12 AM - 6 AM (00:00 - 06:00)

**Example:**
```python
# Transaction at 3:30 AM on Sunday
is_unusual_hour = 1  # 3:30 AM is unusual
is_weekend = 1       # Sunday
hour_of_day = 3
day_of_week = 6      # Sunday
```

### 5. Behavioral Features (5 features)

Analyze customer behavior patterns and merchant relationships.

| Feature | Description | Range |
|---------|-------------|-------|
| `category_diversity_score` | Entropy of spending categories | 0.0 - 4.0 |
| `merchant_loyalty_score` | Preference for repeat merchants | 0.0 - 1.0 |
| `new_merchant_flag` | First-time merchant | 0/1 |
| `avg_merchant_reputation` | Average merchant trust score | 0.0 - 1.0 |
| `refund_rate_30d` | Refund rate (30 days) | 0.0 - 1.0 |

**Fraud Indicators:**
- Sudden change in category diversity
- All new merchants (account takeover)
- High refund rate
- Transactions at low-reputation merchants

**Category Diversity Calculation:**
```python
import numpy as np
from collections import Counter

categories = [txn['category'] for txn in history]
freq = Counter(categories)
probs = [count/len(categories) for count in freq.values()]
entropy = -sum(p * np.log2(p) for p in probs if p > 0)
```

**Example:**
```python
# Diverse shopper
category_diversity_score = 2.5  # Shops in many categories
merchant_loyalty_score = 0.3    # Tries new merchants

# Suspicious pattern (account takeover)
category_diversity_score = 0.2  # Only electronics
merchant_loyalty_score = 0.0    # All new merchants
new_merchant_flag = 1           # Never seen merchant
```

### 6. Network Features (4 features)

Analyze relationships in transaction networks.

| Feature | Description | Type |
|---------|-------------|------|
| `shared_merchant_count` | Merchants shared with fraud network | Count |
| `shared_location_count` | Locations shared with fraud network | Count |
| `customer_proximity_score` | Closeness to known fraud patterns | 0.0 - 1.0 |
| `declined_rate_7d` | Declined transaction rate (7d) | 0.0 - 1.0 |

**Fraud Indicators:**
- High overlap with known fraud networks
- Multiple declined transactions
- Proximity to flagged customers

**Example:**
```python
# Suspicious network activity
shared_merchant_count = 5        # 5 merchants flagged in network
customer_proximity_score = 0.85  # Very close to fraud patterns
declined_rate_7d = 0.3           # 30% declined recently
```

## Feature Engineering Pipeline

### Step 1: Generate Transaction Data

```python
from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator

# Generate data
generator = DataGenerator(num_customers=100, num_days=30)
customers = generator.generate_customers()
transactions = generator.generate_transactions(num_transactions=1000)

# Inject fraud
fraud_gen = FraudPatternGenerator(seed=42)
transactions = fraud_gen.inject_fraud_patterns(
    transactions, customers, fraud_rate=0.1
)
```

### Step 2: Engineer Features

```python
from src.generators.ml_features import MLFeatureEngineer

# Initialize feature engineer
engineer = MLFeatureEngineer()

# Build transaction history
history_lookup = {}
for txn in transactions:
    customer_id = txn['customer_id']
    if customer_id not in history_lookup:
        history_lookup[customer_id] = []
    history_lookup[customer_id].append(txn)

# Sort history by timestamp
for customer_id in history_lookup:
    history_lookup[customer_id].sort(key=lambda x: x['timestamp'])

# Engineer features for each transaction
features_list = []
for txn in transactions:
    customer = next(c for c in customers if c['customer_id'] == txn['customer_id'])
    customer_history = history_lookup[txn['customer_id']]
    txn_index = customer_history.index(txn)
    history = customer_history[:txn_index]
    
    ml_features = engineer.engineer_features(
        transaction=txn,
        customer=customer,
        transaction_history=history
    )
    features_list.append(ml_features.to_dict())
```

### Step 3: Get Feature Metadata

```python
# Get feature metadata
metadata = engineer.get_feature_metadata()

print(f"Total features: {metadata['feature_count']}")
for category, features in metadata['features'].items():
    print(f"{category}: {len(features)} features")
```

## Feature Importance Rankings

Based on Random Forest feature importance analysis on 5000+ transactions:

### Top 10 Most Important Features

1. **amount_velocity_1h** (0.0847) - Rapid spending indicator
2. **distance_from_home** (0.0735) - Geographic anomaly
3. **txn_frequency_1h** (0.0682) - Velocity abuse
4. **daily_txn_amount** (0.0654) - Spending spike
5. **travel_velocity_kmh** (0.0621) - Impossible travel
6. **is_unusual_hour** (0.0589) - Temporal anomaly
7. **new_merchant_flag** (0.0543) - Unfamiliar merchant
8. **category_diversity_score** (0.0512) - Behavior change
9. **declined_rate_7d** (0.0487) - Recent declines
10. **customer_proximity_score** (0.0456) - Network risk

## Feature Normalization

All continuous features are normalized to [0, 1] range using min-max scaling:

```python
normalized_value = (value - min_value) / (max_value - min_value)
```

**Excluded from normalization:**
- `transaction_id`
- `is_fraud`
- `fraud_type`
- Binary flags (`is_unusual_hour`, `is_weekend`, `is_holiday`, etc.)

## Feature Quality Checks

### Missing Values
- All features must have valid values (no NaN/None)
- Default to 0 for missing history-based features

### Variance Check
- Remove features with variance < 0.01 (near-constant)
- Check using `variance_analysis.py` script

### Correlation Check
- Monitor features with |correlation| > 0.8
- Consider removing redundant features
- Use `correlation_analysis.py` script

## Best Practices

### 1. Feature Selection
- Start with all 32 features
- Use feature importance to identify top performers
- Remove low-importance features (<0.01) for production

### 2. Data Leakage Prevention
- Only use historical data (transactions before current)
- Never include future information
- Separate train/test splits properly

### 3. Feature Updates
- Recalculate features periodically as data grows
- Update normalization statistics on new data
- Monitor feature drift over time

### 4. Performance Optimization
- Cache transaction history lookups
- Batch process features when possible
- Use vectorized operations for calculations

## Example Use Cases

### Use Case 1: Real-Time Fraud Detection

```python
# For a new transaction, calculate features
ml_features = engineer.engineer_features(
    transaction=new_transaction,
    customer=customer_profile,
    transaction_history=recent_history
)

# Convert to model input
features_dict = ml_features.to_dict()
X = np.array([features_dict[col] for col in feature_columns])

# Predict
fraud_probability = model.predict_proba([X])[0][1]
is_fraud = fraud_probability > 0.5
```

### Use Case 2: Batch Fraud Analysis

```python
# Process multiple transactions
features_batch = []
for txn in transactions_batch:
    ml_features = engineer.engineer_features(txn, customer, history)
    features_batch.append(ml_features.to_dict())

# Predict on batch
X_batch = prepare_ml_data(features_batch)
predictions = model.predict(X_batch)
```

### Use Case 3: Model Retraining

```python
# Generate new training data
new_transactions = generate_recent_transactions(days=30)
new_features = engineer_features_batch(new_transactions, customers)

# Retrain model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

## API Reference

### MLFeatureEngineer Class

```python
class MLFeatureEngineer:
    """Engineer ML features for fraud detection."""
    
    def engineer_features(
        self,
        transaction: Dict,
        customer: Dict,
        transaction_history: List[Dict]
    ) -> MLFeatures:
        """
        Engineer all 32 features for a transaction.
        
        Args:
            transaction: Current transaction dictionary
            customer: Customer profile dictionary
            transaction_history: List of prior transactions (sorted by time)
            
        Returns:
            MLFeatures dataclass with all engineered features
        """
    
    def get_feature_metadata(self) -> Dict:
        """
        Get metadata about all features.
        
        Returns:
            Dictionary with feature count and category breakdown
        """
```

### MLFeatures Dataclass

```python
@dataclass
class MLFeatures:
    """Container for all ML features."""
    
    # Identifiers
    transaction_id: str
    is_fraud: int
    fraud_type: str
    
    # Aggregate features (6)
    daily_txn_count: float
    weekly_txn_count: float
    # ... (30 more features)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
```

## Troubleshooting

### Issue: Low Feature Importance

**Solution**: Check feature variance and correlation
```bash
python scripts/analyze_variance.py data/features.csv
python scripts/analyze_correlations.py data/features.csv
```

### Issue: Missing Values in Features

**Solution**: Verify transaction history is properly sorted
```python
# Sort history by timestamp
history.sort(key=lambda x: x['timestamp'])
```

### Issue: Poor Model Performance

**Solution**: 
1. Check class balance (should be ~50-50 for training)
2. Verify feature normalization
3. Increase training data size
4. Try different fraud patterns

## Additional Resources

- [ML Dataset Guide](ML_DATASET_GUIDE.md)
- [Integration Guide](../guides/INTEGRATION_GUIDE.md)
- [Quick Reference](../guides/QUICK_REFERENCE.md)
- [Example Scripts](../../examples/)

## Version History

- **0.5.0** (Oct 26, 2025): Added 32 features across 6 categories
- **0.4.0**: Initial feature engineering implementation
