# SynFinance Integration Guide

**Enterprise API Reference and Integration Patterns**

**Complete Field Reference:** [FIELD_REFERENCE.md](../technical/FIELD_REFERENCE.md)

---

## Overview

This guide covers integrating SynFinance into your data pipelines, testing frameworks, and production systems. SynFinance generates realistic synthetic financial transaction data with **45 comprehensive fields** spanning transaction details, customer demographics, merchant information, device metadata, and risk indicators.

---

## Core APIs

### 1. Customer Generation API

```python
from src.customer_generator import CustomerGenerator

# Initialize generator
gen = CustomerGenerator(seed=42)  # Optional seed for reproducibility

# Generate single customer
customer = gen.generate_customer()

# Generate multiple customers
customers = [gen.generate_customer() for _ in range(100)]

# Generate specific segment
from src.customer_profile import CustomerSegment
gen_segment = CustomerGenerator(seed=42)
# (Use generate_customer() - segment assigned by distribution)
```

**CustomerProfile Fields:**
- `customer_id`: Unique ID (CUST0000001)
- `age`, `gender`, `city`, `state`, `region`
- `income_bracket`, `occupation`, `monthly_income`
- `segment`: CustomerSegment enum
- `preferred_categories`: List of category strings
- `preferred_payment_modes`: List of payment methods

### 2. Transaction Generation API

```python
from src.generators.transaction_core import TransactionGenerator
from datetime import datetime, timedelta

# Initialize generator
txn_gen = TransactionGenerator(seed=42)

# Generate single transaction
transaction = txn_gen.generate_transaction(customer, datetime.now())

# Generate multiple transactions for a customer
transactions = txn_gen.generate_customer_transactions(
    customer=customer,
    count=50,
    days=30
)

# Returns list of transaction dictionaries
```

**Transaction Dictionary (45 fields):**

**See [FIELD_REFERENCE.md](../technical/FIELD_REFERENCE.md) for complete specifications.**

**Field Categories:**
- **Core Transaction** (8 fields): Transaction_ID, Date, Time, Day_of_Week, Hour, Amount, Category, Merchant_ID
- **Customer Demographics** (7 fields): Customer_ID, Age, Gender, City, State, Region, Customer_Segment
- **Merchant Details** (5 fields): Merchant_Name, Merchant_Category, Merchant_City, Merchant_Rating, Merchant_Years_Operating
- **Location Context** (3 fields): City_Tier, Location_Type, Distance_From_Home_km
- **Device & Channel** (6 fields): Device_Type, Device_OS, Browser, App_Version, Channel, IP_Address
- **Payment Details** (3 fields): Payment_Mode, Card_Type, Bank_Name
- **Temporal Context** (4 fields): Is_Weekend, Is_Holiday, Festival_Name, Time_Of_Day
- **Risk & Fraud** (5 fields): Is_Fraud, Fraud_Type, Distance_From_Last_Txn_km, Time_Since_Last_Txn_hours, Unusual_Spending
- **Customer Profile** (2 fields): Digital_Savviness, Income_Bracket
- **Derived Metrics** (2 fields): Transaction_Velocity, Customer_Lifetime_Days

### 3. Full Dataset Generation API

```python
from src.data_generator import generate_realistic_dataset

# Generate complete dataset
df = generate_realistic_dataset(
    num_customers=1000,
    transactions_per_customer=50,
    start_date="2025-01-01",
    days=90,
    seed=42
)

# Returns pandas DataFrame with all transactions
print(df.shape)  # (50000, 45)
```

**Quality Validation Workflow:**

```python
from scripts.analyze_variance import VarianceAnalyzer

# Generate dataset
df = generate_realistic_dataset(
    num_customers=1000,
    transactions_per_customer=50,
    days=30,
    seed=42
)

# Save and validate
df.to_csv('output/dataset.csv', index=False)

# Run variance analysis
analyzer = VarianceAnalyzer('output/dataset.csv', output_dir='output/quality')
results = analyzer.run_analysis()

# Check quality
print(f"Quality: {results['summary']['quality_metrics']['pass_rate']:.1%}")
print(f"Fields: {results['summary']['quality_metrics']['pass_count']}/{results['summary']['quality_metrics']['total_fields']} PASS")
```

---

## Integration Patterns

### Pattern 1: Daily Data Pipeline

```python
from datetime import datetime, timedelta
import pandas as pd

def generate_daily_transactions(date, num_transactions=10000):
    """Generate transactions for a specific date"""
    num_customers = num_transactions // 10
    
    df = generate_realistic_dataset(
        num_customers=num_customers,
        transactions_per_customer=10,
        start_date=date.strftime("%Y-%m-%d"),
        days=1,
        seed=int(date.timestamp())
    )
    
    return df

# Example: Generate last 7 days
for i in range(7):
    date = datetime.now() - timedelta(days=i)
    df = generate_daily_transactions(date)
    df.to_csv(f"output/transactions_{date.strftime('%Y%m%d')}.csv")
```

### Pattern 2: ML Training Data

```python
def generate_ml_training_data(fraud_ratio=0.02):
    """Generate balanced dataset for ML training"""
    
    # Generate large dataset
    df = generate_realistic_dataset(
        num_customers=5000,
        transactions_per_customer=100,
        days=180,
        seed=42
    )
    
    # Split by fraud flag
    fraud = df[df['Is_Fraud'] == 1]
    normal = df[df['Is_Fraud'] == 0]
    
    # Balance classes
    fraud_count = len(fraud)
    normal_sample = normal.sample(n=int(fraud_count * (1-fraud_ratio) / fraud_ratio))
    
    balanced = pd.concat([fraud, normal_sample]).sample(frac=1)
    return balanced

# Example usage
train_data = generate_ml_training_data()
print(f"Fraud rate: {train_data['Is_Fraud'].mean():.2%}")
```

### Pattern 3: Data Quality Validation Pipeline

```python
from scripts.analyze_variance import VarianceAnalyzer

def generate_and_validate(num_customers, min_quality=0.80):
    """Generate dataset and validate quality"""
    
    # Generate data
    df = generate_realistic_dataset(
        num_customers=num_customers,
        transactions_per_customer=50,
        days=30
    )
    
    # Save for analysis
    df.to_csv('output/dataset.csv', index=False)
    
    # Run variance analysis
    analyzer = VarianceAnalyzer('output/dataset.csv')
    results = analyzer.run_analysis()
    
    quality_rate = results['summary']['quality_metrics']['pass_rate']
    
    if quality_rate < min_quality:
        raise ValueError(f"Quality {quality_rate:.1%} below threshold {min_quality:.1%}")
    
    print(f"Dataset quality: {quality_rate:.1%} - PASS")
    return df

# Example: Generate validated data
df = generate_and_validate(1000, min_quality=0.80)
```

### Pattern 4: Streaming Data Simulation

```python
import time

def stream_transactions(customers, interval_seconds=1):
    """Simulate real-time transaction stream"""
    txn_gen = TransactionGenerator()
    
    while True:
        # Select random customer
        customer = random.choice(customers)
        
        # Generate transaction
        txn = txn_gen.generate_transaction(customer, datetime.now())
        
        # Yield transaction
        yield txn
        
        # Wait for interval
        time.sleep(interval_seconds)

# Example: Stream to Kafka
from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
gen = CustomerGenerator()
customers = [gen.generate_customer() for _ in range(1000)]

for txn in stream_transactions(customers, interval_seconds=0.1):
    producer.send('transactions', json.dumps(txn).encode())
```

### Pattern 5: Fraud Detection Training Data

**NEW in v0.4.0:** Inject realistic fraud patterns for ML model training.

```python
from src.generators.fraud_patterns import inject_fraud_into_dataset, FraudPatternGenerator
from src.customer_generator import CustomerGenerator
from src.data_generator import generate_realistic_dataset

# Generate base dataset
df = generate_realistic_dataset(
    num_customers=1000,
    transactions_per_customer=50,
    days=90,
    seed=42
)

# Convert to list of dicts for fraud injection
transactions = df.to_dict('records')
gen = CustomerGenerator(seed=42)
customers = [gen.generate_customer() for _ in range(1000)]

# Inject fraud patterns at 2% rate
modified_transactions, fraud_stats = inject_fraud_into_dataset(
    transactions=transactions,
    customers=customers,
    fraud_rate=0.02,  # 2% fraud rate
    seed=42
)

# Convert back to DataFrame
fraud_df = pd.DataFrame(modified_transactions)

# View statistics
print(f"Total Transactions: {fraud_stats['total_transactions']}")
print(f"Total Fraud: {fraud_stats['total_fraud']}")
print(f"Actual Fraud Rate: {fraud_stats['fraud_rate']:.2%}")
print(f"\nFraud Distribution:")
for fraud_type, count in fraud_stats['fraud_by_type'].items():
    percentage = fraud_stats['fraud_type_distribution'][fraud_type]
    print(f"  {fraud_type}: {count} ({percentage:.1%})")

# Save dataset with fraud labels
fraud_df.to_csv('output/fraud_training_data.csv', index=False)
```

**Fraud Fields Added (5 fields):**
- `Fraud_Type`: Name of fraud pattern (e.g., "Card Cloning")
- `Fraud_Confidence`: Confidence score 0.0-1.0
- `Fraud_Reason`: Detailed explanation
- `Fraud_Severity`: "low", "medium", "high", or "critical"
- `Fraud_Evidence`: JSON string with supporting evidence

**10 Fraud Pattern Types:**
1. **Card Cloning** - Impossible travel detection (>800 km/h)
2. **Account Takeover** - 3-10x spending spikes
3. **Merchant Collusion** - Round amounts near thresholds
4. **Velocity Abuse** - 5+ transactions/hour
5. **Amount Manipulation** - Structuring detection
6. **Refund Fraud** - >3x normal refund rate
7. **Stolen Card** - Inactivity spike detection
8. **Synthetic Identity** - Limited history patterns
9. **First Party Fraud** - Bust-out detection
10. **Friendly Fraud** - Chargeback abuse

**Per-Transaction Fraud Control:**
```python
# For more control, use FraudPatternGenerator directly
fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)

customer_history = {}
for customer in customers:
    customer_history[customer.customer_id] = []

modified_transactions = []
for txn in transactions:
    customer = next((c for c in customers if c.customer_id == txn['Customer_ID']), None)
    history = customer_history.get(txn['Customer_ID'], [])
    
    # Apply fraud with custom logic
    modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer, history)
    
    # Add fraud labels
    if fraud_indicator:
        modified_txn['Fraud_Type'] = fraud_indicator.fraud_type.value
        modified_txn['Fraud_Confidence'] = fraud_indicator.confidence
        modified_txn['Fraud_Reason'] = fraud_indicator.reason
        modified_txn['Fraud_Severity'] = fraud_indicator.severity
        modified_txn['Fraud_Evidence'] = json.dumps(fraud_indicator.evidence)
    
    customer_history[txn['Customer_ID']].append(modified_txn)
    modified_transactions.append(modified_txn)

# Get final statistics
stats = fraud_gen.get_fraud_statistics()
```

**ML Training Example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load fraud dataset
df = pd.read_csv('output/fraud_training_data.csv')

# Prepare features (exclude fraud label fields)
X = df.drop(['Fraud_Type', 'Fraud_Confidence', 'Fraud_Reason', 
             'Fraud_Severity', 'Fraud_Evidence', 'Is_Fraud'], axis=1)
y = df['Is_Fraud']

# Split data (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

**See [FRAUD_PATTERNS.md](../technical/FRAUD_PATTERNS.md) for complete fraud pattern documentation.**

---

## Pattern 6: ML Feature Engineering and Model Training

**NEW in v0.5.0:** Complete ML workflow for fraud detection model training.

### Step 1: Generate Transactions with Fraud

```python
from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator

# Generate base transactions
generator = DataGenerator(num_customers=100, num_days=30)
customers = generator.generate_customers()
transactions = generator.generate_transactions(num_transactions=5000)

# Inject fraud patterns (10% fraud rate)
fraud_gen = FraudPatternGenerator(seed=42)
transactions = fraud_gen.inject_fraud_patterns(
    transactions, customers, fraud_rate=0.1
)

# Check fraud statistics
stats = fraud_gen.get_fraud_statistics()
print(f"Total fraud: {stats['total_fraud']} ({stats['fraud_rate']:.1%})")
```

### Step 2: Engineer ML Features

```python
from src.generators.ml_features import MLFeatureEngineer

# Initialize feature engineer
engineer = MLFeatureEngineer()

# Build transaction history for feature engineering
transaction_history = {}
for txn in sorted(transactions, key=lambda x: x.timestamp):
    customer_id = txn.customer_id
    if customer_id not in transaction_history:
        transaction_history[customer_id] = []
    transaction_history[customer_id].append(txn)

# Engineer 32 ML features for each transaction
features_list = []
for txn in transactions:
    customer_id = txn.customer_id
    history = transaction_history.get(customer_id, [])
    
    # Get features for this transaction
    features = engineer.engineer_features([txn], {customer_id: history})
    features_list.extend(features)

print(f"Engineered {len(features_list)} samples with 32 features each")
```

### Step 3: Create ML-Ready Dataset

```python
from src.generators.ml_dataset_generator import MLDatasetGenerator

# Initialize dataset generator
dataset_gen = MLDatasetGenerator(seed=42)

# Create balanced, normalized, split dataset
split, metadata = dataset_gen.create_ml_ready_dataset(
    features_list,
    balance_strategy='undersample',  # or 'oversample'
    target_fraud_rate=0.5,  # 50-50 balance
    normalize=True,
    encode_categorical=True,
    train_ratio=0.70,
    val_ratio=0.15
)

print(f"Train: {len(split.train)} samples")
print(f"Validation: {len(split.validation)} samples")
print(f"Test: {len(split.test)} samples")

# Validate dataset quality
quality = dataset_gen.validate_dataset_quality(split.train)
print(f"Fraud rate: {quality['fraud_rate']:.1%}")
print(f"Quality checks: {quality['quality_checks']}")
```

### Step 4: Export for ML Libraries

```python
# Export to CSV
dataset_gen.export_to_csv(split.train, 'output/train.csv')
dataset_gen.export_to_csv(split.validation, 'output/validation.csv')
dataset_gen.export_to_csv(split.test, 'output/test.csv')

# Export to Parquet (10x smaller, faster)
dataset_gen.export_to_parquet(split.train, 'output/train.parquet')

# Export to NumPy arrays (direct sklearn input)
X_train, y_train, feature_names = dataset_gen.export_to_numpy(
    split.train, 'output/train'
)
X_test, y_test, _ = dataset_gen.export_to_numpy(
    split.test, 'output/test'
)

print(f"NumPy arrays: X_train={X_train.shape}, y_train={y_train.shape}")
```

### Step 5: Train ML Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Feature importance
feature_importance = sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
print("\nTop 10 Features:")
for name, importance in feature_importance[:10]:
    print(f"  {name}: {importance:.4f}")
```

### Step 6: Using XGBoost (Optional)

```python
try:
    import xgboost as xgb
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.3f}")
    
except ImportError:
    print("XGBoost not installed. Install: pip install xgboost")
```

### Complete End-to-End Script

```python
# examples/train_fraud_detector.py
python examples/train_fraud_detector.py \
    --num-transactions 5000 \
    --fraud-rate 0.1 \
    --output-dir output/ml_training
```

### Interactive Jupyter Notebook

```bash
jupyter notebook examples/fraud_detection_tutorial.ipynb
```

**ML Features:**
- 32 engineered features across 6 categories (aggregate, velocity, geographic, temporal, behavioral, network)
- Balanced dataset preparation (undersample/oversample)
- Feature normalization (min-max scaling)
- Multiple export formats (CSV, JSON, Parquet, NumPy)
- Data quality validation (correlation, outliers, distributions)

**See Documentation:**
- [ML_FEATURES.md](../technical/ML_FEATURES.md) - Complete 32-feature reference
- [ML_DATASET_GUIDE.md](../technical/ML_DATASET_GUIDE.md) - Dataset preparation guide
- [fraud_detection_tutorial.ipynb](../../examples/fraud_detection_tutorial.ipynb) - Interactive tutorial

---

## Advanced Configuration

### Custom City Distribution

```python
from src.generators.geographic_generator import GeographicPatternGenerator

geo_gen = GeographicPatternGenerator(seed=42)

# Override proximity groups
geo_gen.PROXIMITY_GROUPS["NewCity"] = ["NearbyCity1", "NearbyCity2"]

# Override cost-of-living
geo_gen.COST_OF_LIVING_MULTIPLIERS[1] = 1.5  # 50% higher for Tier 1
```

### Custom Merchant Pool

```python
from src.generators.merchant_generator import MerchantGenerator

merchant_gen = MerchantGenerator(seed=42)

# Pre-create merchant pool for city
merchants = merchant_gen.get_or_create_merchant_pool("Mumbai")
print(f"Mumbai has {len(merchants)} merchants")
```

### Custom Temporal Patterns

```python
from src.generators.temporal_generator import TemporalPatternGenerator

temporal_gen = TemporalPatternGenerator(seed=42)

# Define custom festivals
custom_festivals = {
    "Custom_Festival": {
        "month": [3, 4],
        "spending_multiplier": 2.0
    }
}

multiplier, breakdown = temporal_gen.get_festival_multiplier(
    datetime(2025, 3, 15),
    customer,
    custom_festivals
)
```

---

## Export Formats

### CSV Export
```python
df.to_csv("output/transactions.csv", index=False)
```

### Excel Export (Multiple Sheets)
```python
with pd.ExcelWriter("output/transactions.xlsx") as writer:
    df.to_excel(writer, sheet_name="Transactions", index=False)
    df[df['Is_Fraud']==1].to_excel(writer, sheet_name="Fraud", index=False)
```

### JSON Export
```python
df.to_json("output/transactions.json", orient="records", lines=True)
```

### Parquet Export (Big Data)
```python
df.to_parquet("output/transactions.parquet", compression="snappy")
```

---

## Testing Integration

```python
import pytest
from src.data_generator import generate_realistic_dataset

def test_data_pipeline():
    """Test your data pipeline with synthetic data"""
    # Generate test data
    df = generate_realistic_dataset(
        num_customers=100,
        transactions_per_customer=10,
        days=7,
        seed=42
    )
    
    # Test your pipeline
    result = your_pipeline_function(df)
    
    # Assertions
    assert len(result) > 0
    assert result['Amount'].mean() > 0
```

---

## Performance Tips

1. **Batch Generation:** Use `generate_realistic_dataset()` instead of loops
2. **Seed Control:** Use seeds for reproducible test data
3. **Memory Management:** Generate in chunks for large datasets (1M+ rows)
4. **Parallel Generation:** Use multiprocessing for multiple customer segments

```python
from multiprocessing import Pool

def generate_segment_data(segment_size):
    return generate_realistic_dataset(segment_size, 50, days=30)

# Parallel generation
with Pool(4) as p:
    results = p.map(generate_segment_data, [250, 250, 250, 250])

df = pd.concat(results)
```

---

## Pattern 7: Anomaly Detection and Labeling

**NEW in v0.6.0 - Week 5 Complete (Days 1-7)**

Generate datasets with labeled anomalies, perform statistical analysis, and engineer ML features for anomaly detection. Anomalies are **unusual but legitimate** behaviors (distinct from fraud).

### Use Case: Anomaly Detection ML Pipeline

```python
from src.data_generator import generate_realistic_dataset
from src.generators.fraud_patterns import FraudPatternGenerator, apply_fraud_labels
from src.generators.anomaly_patterns import AnomalyPatternGenerator, apply_anomaly_labels
import pandas as pd

# Step 1: Generate base transactions
df = generate_realistic_dataset(
    num_customers=500,
    transactions_per_customer=100,
    days=90,
    seed=42
)

# Step 2: Convert to list of dictionaries
transactions = df.to_dict('records')

# Step 3: Inject fraud patterns (2%)
fraud_gen = FraudPatternGenerator(seed=42)
customers = df.groupby('Customer_ID').first().to_dict('records')  # Get unique customers
transactions = fraud_gen.inject_fraud_patterns(transactions, customers, fraud_rate=0.02)
transactions = apply_fraud_labels(transactions)  # Label non-fraudulent

# Step 4: Inject anomaly patterns (5%)
anomaly_gen = AnomalyPatternGenerator(seed=42)
transactions = anomaly_gen.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.05)
transactions = apply_anomaly_labels(transactions)  # Label non-anomalous

# Step 5: Convert back to DataFrame
df_final = pd.DataFrame(transactions)

# Analyze results
print(f"Total Transactions: {len(df_final)}")
print(f"Fraudulent: {df_final['Is_Fraud'].sum()} ({df_final['Is_Fraud'].mean():.1%})")
print(f"Anomalous: {(df_final['Anomaly_Type'] != 'None').sum()} ({(df_final['Anomaly_Type'] != 'None').mean():.1%})")
print(f"Both Fraud+Anomaly: {((df_final['Is_Fraud'] == 1) & (df_final['Anomaly_Type'] != 'None')).sum()}")

# Export for ML training
df_final.to_csv('output/anomaly_training_data.csv', index=False)
```

### Anomaly Fields Added

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Anomaly_Type` | string | Type of anomaly | "BEHAVIORAL", "GEOGRAPHIC", "TEMPORAL", "AMOUNT", "None" |
| `Anomaly_Confidence` | float | Detection confidence (0.0-1.0) | 0.75 |
| `Anomaly_Reason` | string | Human-readable explanation | "Unusual purchase: Jewelry (5% of history)" |
| `Anomaly_Severity` | float | Severity score (0.0-1.0) | 0.65 |
| `Anomaly_Evidence` | JSON string | Structured evidence | `{"unusual_category": "Jewelry", "multiplier": 4.2}` |

### Anomaly Pattern Types

**1. Behavioral Anomalies** (out-of-character purchases):
- Shopping in rare categories (<10% of history)
- Amount spikes (3-5x normal, not fraud level)
- Payment method changes

**2. Geographic Anomalies** (unusual locations):
- Impossible travel (>800 km/h)
- Very fast travel (800-2000 km/h, possible flight)
- Unusual locations (never visited cities)

**3. Temporal Anomalies** (unusual hours):
- Late night (0-5 AM)
- Early morning (6-8 AM)
- Uncommon hours (<10% of history)

**4. Amount Anomalies** (unusual amounts):
- Spending spikes (3-5x normal)
- Micro-transactions (Rs. 10-50)
- Round amounts (Rs. 1000, 2000, etc.)

### Severity Scoring

| Severity Range | Interpretation | Action |
|----------------|----------------|--------|
| 0.0 - 0.3 | Low severity | Monitor |
| 0.3 - 0.6 | Medium severity | Review |
| 0.6 - 0.8 | High severity | Alert |
| 0.8 - 1.0 | Critical severity | Block/Investigate |

### Use Case: Filter by Severity

```python
# High severity anomalies only
high_severity = df_final[df_final['Anomaly_Severity'] >= 0.7]
print(f"High severity anomalies: {len(high_severity)}")

# Parse evidence JSON
import json
for idx, row in high_severity.head().iterrows():
    if row['Anomaly_Evidence']:
        evidence = json.loads(row['Anomaly_Evidence'])
        print(f"{row['Anomaly_Type']}: {evidence}")
```

### Integration with ML Models

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Prepare features
le_type = LabelEncoder()
df_final['Anomaly_Type_Encoded'] = le_type.fit_transform(df_final['Anomaly_Type'])

features = df_final[[
    'Amount', 'Hour', 'Distance_From_Last_Txn_km',
    'Time_Since_Last_Txn_hours', 'Anomaly_Severity',
    'Anomaly_Confidence', 'Anomaly_Type_Encoded'
]]

# Train unsupervised model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(features)

# Evaluate (anomalies = -1, normal = 1)
df_final['Predicted_Anomaly'] = (predictions == -1).astype(int)
df_final['Actual_Anomaly'] = (df_final['Anomaly_Type'] != 'None').astype(int)

from sklearn.metrics import classification_report
print(classification_report(
    df_final['Actual_Anomaly'],
    df_final['Predicted_Anomaly'],
    target_names=['Normal', 'Anomaly']
))
```

### CLI Tool: Generate Anomaly Dataset

```bash
# Generate 10K transactions with 2% fraud and 5% anomalies
python examples/generate_anomaly_dataset.py \
    --num-transactions 10000 \
    --num-customers 200 \
    --num-days 90 \
    --fraud-rate 0.02 \
    --anomaly-rate 0.05 \
    --output-dir output/anomaly_dataset \
    --seed 42
```

**Outputs:**
- `anomaly_dataset.csv`: Complete dataset (50 fields = 45 base + 5 anomaly)
- `dataset_summary.json`: Statistics (fraud rate, anomaly rate, overlap, severity metrics)
- `dataset_summary.txt`: Human-readable report

### Best Practices

**1. Rate Selection:**
- Development: 5-8% anomaly rate (easier debugging)
- Production: 3-5% anomaly rate (realistic)
- ML Training: 5-10% anomaly rate (sufficient positive samples)

**2. Combining Fraud and Anomalies:**
- Apply fraud patterns **first** (fraud_rate=0.02)
- Apply anomaly patterns **second** (anomaly_rate=0.05)
- Some transactions will be both fraud AND anomaly (10-20% overlap expected)

**3. Severity Thresholds:**
- Use severity ≥0.7 for alerting
- Use severity ≥0.5 for investigation queue
- Use severity <0.5 for monitoring only

**4. Evidence Parsing:**
```python
import json

def safe_parse_evidence(evidence_str):
    """Safely parse evidence JSON"""
    try:
        return json.loads(evidence_str) if evidence_str else {}
    except json.JSONDecodeError:
        return {}

df_final['Evidence_Dict'] = df_final['Anomaly_Evidence'].apply(safe_parse_evidence)
```

### Anomaly Analysis (Days 3-4)

**NEW in v0.6.0:** Statistical analysis of anomaly patterns

```python
from src.generators.anomaly_analysis import (
    AnomalyFraudCorrelationAnalyzer,
    SeverityDistributionAnalyzer,
    TemporalClusteringAnalyzer,
    GeographicHeatmapAnalyzer
)

transactions = df_final.to_dict('records')

# 1. Anomaly-Fraud Correlation Analysis
correlation_analyzer = AnomalyFraudCorrelationAnalyzer()
correlation_result = correlation_analyzer.analyze_correlation(transactions)

print(f"Phi Coefficient: {correlation_result.phi_coefficient:.4f}")
print(f"Chi-Square: {correlation_result.chi_square_stat:.4f}")
print(f"P-Value: {correlation_result.p_value:.6f}")
print(f"Significant: {correlation_result.is_significant}")

# 2. Severity Distribution Analysis
severity_analyzer = SeverityDistributionAnalyzer()
severity_result = severity_analyzer.analyze_severity_distribution(transactions)

print(f"Mean Severity: {severity_result.mean_severity:.4f}")
print(f"High Severity Outliers: {len(severity_result.high_severity_outliers)}")

# 3. Temporal Clustering Analysis
temporal_analyzer = TemporalClusteringAnalyzer()
temporal_result = temporal_analyzer.analyze_temporal_clustering(transactions)

print(f"Clusters: {temporal_result.cluster_count}")
print(f"Bursts Detected: {len(temporal_result.burst_periods)}")

# 4. Geographic Heatmap Analysis
geographic_analyzer = GeographicHeatmapAnalyzer()
geographic_result = geographic_analyzer.analyze_geographic_distribution(transactions)

print(f"Cities with Anomalies: {len(geographic_result.city_anomaly_counts)}")
print(f"Distance-Severity Correlation: {geographic_result.distance_severity_correlation:.4f}")
```

### ML Feature Engineering (Days 5-6)

**NEW in v0.6.0:** 27 ML features for anomaly detection

```python
from src.generators.anomaly_ml_features import (
    AnomalyMLFeatureGenerator,
    IsolationForestAnomalyDetector
)

# Initialize generators
feature_gen = AnomalyMLFeatureGenerator()
detector = IsolationForestAnomalyDetector(contamination=0.10, random_state=42)

# Build customer histories
customer_histories = {}
for txn in transactions:
    cid = txn['Customer_ID']
    if cid not in customer_histories:
        customer_histories[cid] = []
    customer_histories[cid].append(txn)

# Train Isolation Forest
isolation_scores = detector.fit_predict(transactions)

# Generate 27 ML features per transaction
features_list = feature_gen.generate_features_batch(
    transactions=transactions,
    customer_histories=customer_histories,
    isolation_scores=isolation_scores
)

# Convert to DataFrame
import pandas as pd
features_df = pd.DataFrame([vars(f) for f in features_list])

print(f"Generated {len(features_df)} feature vectors with {features_df.shape[1]} features")
print("\nFeature Categories:")
print("  - Frequency Features: 5 (hourly, daily, weekly counts, trend, time since last)")
print("  - Severity Aggregates: 5 (mean, max, std, high-severity rate, current)")
print("  - Type Distribution: 5 (4 type rates + Shannon entropy diversity)")
print("  - Persistence Metrics: 3 (consecutive count, streak length, days since first)")
print("  - Cross-Pattern: 2 (fraud-anomaly overlap, Jaccard correlation)")
print("  - Evidence Features: 4 (impossible travel, unusual category, hour, spike)")
print("  - Unsupervised: 3 (Isolation Forest score, probability, prediction)")

# Export for ML training
features_df.to_csv('output/anomaly_ml_features.csv', index=False)
```

### Comprehensive Analysis Pipeline (Day 7)

**Complete end-to-end anomaly analysis:**

```python
# Run comprehensive analysis
from examples.analyze_anomaly_patterns import (
    generate_dataset_with_anomalies,
    analyze_anomaly_fraud_correlation,
    analyze_severity_distribution,
    analyze_temporal_clustering,
    analyze_geographic_patterns,
    generate_ml_features,
    save_results
)

# Generate dataset
df, customer_histories = generate_dataset_with_anomalies(
    num_customers=100,
    transactions_per_customer=50
)

# Run all analyses
correlation_result = analyze_anomaly_fraud_correlation(df)
severity_result = analyze_severity_distribution(df)
temporal_result = analyze_temporal_clustering(df)
geographic_result = analyze_geographic_patterns(df)

# Generate ML features
features_df = generate_ml_features(df, customer_histories)

# Save results
save_results(df, features_df, output_dir='output')
```

**Outputs:**
- `anomaly_analysis_dataset.csv`: Complete dataset with anomalies
- `anomaly_ml_features_complete.csv`: 27 ML features per transaction
- `anomaly_analysis_summary.json`: Comprehensive statistics

**Documentation:**
- Complete anomaly patterns documentation: [ANOMALY_PATTERNS.md](../technical/ANOMALY_PATTERNS.md)
- Days 1-2: [WEEK5_DAY1-2_COMPLETE.md](../progress/week5/WEEK5_DAY1-2_COMPLETE.md)
- Days 3-4: [WEEK5_DAY3-4_COMPLETE.md](../progress/week5/WEEK5_DAY3-4_COMPLETE.md)
- Days 5-6: [WEEK5_DAY5-6_COMPLETE.md](../progress/week5/WEEK5_DAY5-6_COMPLETE.md)

---

## Pattern 8: Combined ML Features for Production Models

**NEW in v0.7.0:** Unified feature engineering combining fraud and anomaly signals

### Use Case: Production-Ready ML Pipeline

Generate comprehensive ML features by combining fraud-based features (32), anomaly-based features (26), and interaction features (10) for a total of **68 ML features** per transaction.

```python
from src.customer_generator import CustomerGenerator
from src.generators.transaction_core import TransactionGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.anomaly_ml_features import AnomalyMLFeatureGenerator
from src.generators.combined_ml_features import CombinedMLFeatureGenerator
from dataclasses import asdict

# Step 1: Generate base data
customer_gen = CustomerGenerator(seed=42)
txn_gen = TransactionGenerator(seed=42)
customers = customer_gen.generate_customers(count=100)
df = txn_gen.generate_dataset(customers=customers, transactions_per_customer=50)
transactions = df.to_dict('records')

# Step 2: Inject fraud and anomaly patterns
fraud_gen = FraudPatternGenerator(fraud_rate=0.10, seed=42)
anomaly_gen = AnomalyPatternGenerator(seed=42)

# Create customer map
from types import SimpleNamespace
customer_map = {}
for c in customers:
    customer_ns = SimpleNamespace(
        Customer_ID=c.customer_id,
        Age=c.age,
        Gender=c.gender,
        City=c.city,
        city=c.city,
        State=c.state,
        Region=c.region,
        Occupation=c.occupation.value,
        Income_Bracket=c.income_bracket.value,
        Segment=c.segment.value,
        Digital_Savviness=c.digital_savviness.value,
        digital_savviness=c.digital_savviness,
        Risk_Profile=c.risk_profile.value,
    )
    customer_map[c.customer_id] = customer_ns

# Apply fraud patterns
fraud_transactions = []
customer_history_map = {}

for txn in transactions:
    customer_id = txn.get('Customer_ID')
    customer_dict = customer_map.get(customer_id)
    history = customer_history_map.get(customer_id, [])
    
    modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer_dict, history)
    
    if fraud_indicator:
        modified_txn['Is_Fraud'] = 1
        modified_txn['Fraud_Type'] = fraud_indicator.fraud_type.value
    else:
        modified_txn['Is_Fraud'] = 0
        modified_txn['Fraud_Type'] = None
    
    fraud_transactions.append(modified_txn)
    
    if customer_id not in customer_history_map:
        customer_history_map[customer_id] = []
    customer_history_map[customer_id].append(modified_txn)

# Apply anomaly patterns
final_transactions = anomaly_gen.inject_anomaly_patterns(
    fraud_transactions,
    customers,
    anomaly_rate=0.15
)

# Step 3: Generate all ML features
fraud_feature_gen = MLFeatureEngineer()
anomaly_feature_gen = AnomalyMLFeatureGenerator()
combined_feature_gen = CombinedMLFeatureGenerator()

fraud_features_list = []
anomaly_features_list = []

for i, transaction in enumerate(final_transactions):
    customer_id = transaction.get('Customer_ID')
    customer_ns = customer_map.get(customer_id)
    customer_dict = vars(customer_ns) if customer_ns else {}
    history = [t for t in final_transactions[:i] if t.get('Customer_ID') == customer_id]
    
    # Generate fraud features (32)
    fraud_features = fraud_feature_gen.engineer_features(transaction, customer_dict, history)
    fraud_features_list.append(fraud_features)
    
    # Generate anomaly features (26)
    anomaly_features = anomaly_feature_gen.generate_features(transaction, history)
    anomaly_features_list.append(anomaly_features)

# Convert to dicts
fraud_dicts = [f.to_dict() for f in fraud_features_list]
anomaly_dicts = [asdict(f) for f in anomaly_features_list]

# Generate combined features with interactions (68 total)
combined_features = combined_feature_gen.generate_batch_features(
    final_transactions,
    fraud_dicts,
    anomaly_dicts
)

print(f"Generated {len(combined_features)} transactions with 68 ML features each")
```

### Combined Features Breakdown

**68 Total ML Features:**

1. **Fraud Features (32)**:
   - Aggregate: daily_txn_count, weekly_txn_count, daily_txn_amount, weekly_txn_amount, avg_daily_amount, avg_weekly_amount
   - Velocity: txn_frequency_1h, txn_frequency_6h, txn_frequency_24h, amount_velocity_1h, amount_velocity_6h, amount_velocity_24h
   - Geographic: distance_from_home, avg_distance_last_10, distance_variance, unique_cities_7d, travel_velocity_kmh
   - Temporal: is_unusual_hour, is_weekend, is_holiday, hour_of_day, day_of_week
   - Behavioral: category_diversity_score, merchant_loyalty_score, avg_merchant_reputation, new_merchant_flag, refund_rate_30d, declined_rate_7d
   - Network: shared_merchant_count, shared_location_count, customer_proximity_score, temporal_cluster_flag

2. **Anomaly Features (26)**:
   - Frequency: hourly_anomaly_count, daily_anomaly_count, weekly_anomaly_count, anomaly_frequency_trend, time_since_last_anomaly_hours
   - Severity: mean_severity_last_10, max_severity_last_10, severity_std_last_10, high_severity_rate_last_10, current_severity
   - Type Distribution: behavioral_anomaly_rate, geographic_anomaly_rate, temporal_anomaly_rate, amount_anomaly_rate, anomaly_type_diversity
   - Persistence: consecutive_anomaly_count, anomaly_streak_length, days_since_first_anomaly
   - Cross-Pattern: is_fraud_and_anomaly, fraud_anomaly_correlation_score
   - Evidence: has_impossible_travel, has_unusual_category, has_unusual_hour, has_spending_spike
   - Unsupervised: isolation_forest_score, anomaly_probability

3. **Interaction Features (10)**:
   - **high_risk_combination**: Binary flag for fraud + anomaly + high velocity
   - **risk_amplification_score**: Non-linear combination of fraud and anomaly signals
   - **compound_severity_score**: Product of fraud severity and anomaly severity
   - **behavioral_consistency_score**: Agreement between fraud and anomaly behavioral signals
   - **pattern_alignment_score**: How well fraud and anomaly patterns align across dimensions
   - **conflict_indicator**: Binary flag for conflicting fraud/anomaly signals
   - **velocity_severity_product**: Travel velocity × anomaly severity
   - **geographic_risk_score**: Distance from home × geographic anomaly rate
   - **weighted_risk_score**: Weighted combination of all risk signals
   - **ensemble_fraud_probability**: Ensemble prediction from all available signals

### Export Combined Features

```python
# Export to CSV
features_dicts = combined_feature_gen.export_to_dict_list(combined_features)

import pandas as pd
df_features = pd.DataFrame(features_dicts)
df_features.to_csv('output/combined_ml_features.csv', index=False)

print(f"Exported {len(df_features)} rows × {len(df_features.columns)} columns")
```

### Feature Statistics

```python
# Calculate statistics for all features
stats = combined_feature_gen.get_feature_statistics(combined_features)

# Display statistics for key features
for feature_name in ['risk_amplification_score', 'ensemble_fraud_probability', 'weighted_risk_score']:
    if feature_name in stats:
        s = stats[feature_name]
        print(f"{feature_name}:")
        print(f"  Mean: {s['mean']:.4f}, Std: {s['std']:.4f}")
        print(f"  Min: {s['min']:.4f}, Max: {s['max']:.4f}")
```

### Identify High-Risk Transactions

```python
# Find high-risk transactions using ensemble probability
high_risk = [
    f for f in combined_features
    if f.ensemble_fraud_probability > 0.7
]

print(f"High-risk transactions: {len(high_risk)}")

for txn in high_risk[:5]:
    print(f"Transaction {txn.transaction_id}:")
    print(f"  Is Fraud: {txn.is_fraud}")
    print(f"  Anomaly Type: {txn.anomaly_type}")
    print(f"  Ensemble Probability: {txn.ensemble_fraud_probability:.3f}")
    print(f"  Risk Amplification: {txn.risk_amplification_score:.3f}")
    print(f"  Conflict Indicator: {txn.conflict_indicator}")
```

### CLI Tool: Generate Combined Features

```bash
python examples/generate_combined_features.py
```

**Outputs:**
- `combined_ml_features.csv`: All 68 features for all transactions
- `feature_statistics.json`: Statistics (mean, std, min, max, median) for each feature
- `sample_features.json`: Sample transactions with metadata

### Use Cases for Combined Features

1. **Production Fraud Detection Models**:
   - Use all 68 features for maximum detection accuracy
   - Ensemble probability provides ready-to-use risk score
   - Interaction features capture non-linear patterns

2. **Feature Importance Analysis**:
   - Compare fraud vs anomaly vs interaction feature importance
   - Identify which feature categories contribute most to predictions

3. **Model Comparison**:
   - Train models with different feature subsets
   - Compare fraud-only vs combined feature performance

4. **Alert Prioritization**:
   - Use weighted_risk_score for alert ranking
   - conflict_indicator helps identify false positives

### Integration with ML Models

```python
# Prepare data for sklearn
X = [f.get_feature_values() for f in combined_features]  # 68 features
y = [f.is_fraud for f in combined_features]  # Labels

# Split and train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Model accuracy: {score:.2%}")

# Feature importance
feature_names = combined_features[0].get_feature_names()
importances = clf.feature_importances_

# Top 10 features
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
for name, importance in top_features:
    print(f"{name}: {importance:.4f}")
```

### Best Practices

1. **Feature Selection**:
   - Start with all 68 features
   - Use feature importance to identify top contributors
   - Consider removing low-importance features for production efficiency

2. **Interaction Features**:
   - Interaction features often have high predictive power
   - Monitor conflict_indicator for model debugging
   - Use ensemble_fraud_probability as baseline comparison

3. **Performance**:
   - Batch processing is more efficient than single transactions
   - Cache customer history for faster feature generation
   - Pre-compute statistics for large datasets

4. **Model Training**:
   - Balance classes if fraud rate is low
   - Use cross-validation for robust evaluation
   - Track fraud-only vs combined feature performance

---

### Pattern 9: Advanced Analytics & Visualization

**Use Case:** Generate comprehensive analytics dashboards with feature importance, model performance metrics, and interactive visualizations.

```python
from src.analytics import (
    CorrelationAnalyzer,
    FeatureImportanceAnalyzer,
    ModelPerformanceAnalyzer,
    StatisticalTestAnalyzer,
    VisualizationFramework,
    HTMLDashboardGenerator,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Assume we have: transactions, combined_features, customers

# Prepare ML data
X = np.array([f.get_feature_values() for f in combined_features])
y = np.array([f.is_fraud for f in combined_features])
feature_names = combined_features[0].get_feature_names()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Get predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# ============================================================================
# 1. Correlation Analysis
# ============================================================================
corr_analyzer = CorrelationAnalyzer(threshold=0.7)
corr_result = corr_analyzer.analyze(X_train, feature_names, method='pearson')

print(f"High correlations found: {len(corr_result.get_highly_correlated_pairs())}")

# Get top correlated pairs
for f1, f2, corr in corr_result.get_highly_correlated_pairs()[:5]:
    print(f"  {f1} ↔ {f2}: {corr:.3f}")

# ============================================================================
# 2. Feature Importance Analysis
# ============================================================================
importance_analyzer = FeatureImportanceAnalyzer(n_repeats=5)

# Multiple methods
importance_results = importance_analyzer.analyze_all(
    clf, X_train, y_train, X_test, y_test, feature_names
)

for result in importance_results:
    print(f"\n{result.method.upper()} - Top 5 features:")
    for feat, imp in result.get_top_features(5):
        print(f"  {feat}: {imp:.4f}")

# Tree-based importance only
tree_importance = importance_analyzer.tree_based_importance(clf, feature_names)
top_features = tree_importance.get_top_features(10)

# Permutation importance (most reliable)
perm_importance = importance_analyzer.permutation_importance(
    clf, X_test, y_test, feature_names, scoring='f1'
)

# ============================================================================
# 3. Model Performance Metrics
# ============================================================================
perf_analyzer = ModelPerformanceAnalyzer()
metrics = perf_analyzer.analyze(y_test, y_pred, y_pred_proba)

print(f"Accuracy:  {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall:    {metrics.recall:.4f}")
print(f"F1-Score:  {metrics.f1:.4f}")
print(f"ROC-AUC:   {metrics.roc_auc:.4f}")

# Compare multiple models
models_dict = {
    'RandomForest': (y_pred, y_pred_proba),
    # 'XGBoost': (y_pred_xgb, y_pred_proba_xgb),  # if available
}
comparison = perf_analyzer.compare_models(models_dict, y_test)

# ============================================================================
# 4. Statistical Significance Tests
# ============================================================================
stat_analyzer = StatisticalTestAnalyzer(alpha=0.05)

# Test if fraud has different feature values
ensemble_probs = np.array([f.ensemble_fraud_probability for f in combined_features])
test_result = stat_analyzer.test_fraud_vs_normal(ensemble_probs, y)

print(f"T-test p-value: {test_result.p_value:.2e}")
print(f"Significant: {test_result.is_significant}")

# Chi-square test for categorical relationships
# ANOVA for comparing groups

# ============================================================================
# 5. Generate Visualizations
# ============================================================================
viz = VisualizationFramework()
charts = {}

# Feature distribution
fig = viz.plot_distribution(
    ensemble_probs,
    title="Ensemble Fraud Probability Distribution",
    kde=True,
    show_stats=True
)
charts['ensemble_dist'] = fig

# Correlation heatmap (top 20 features)
top_20_idx = np.argsort(tree_importance.importances)[::-1][:20]
fig = viz.plot_correlation_heatmap(
    corr_result.correlation_matrix[np.ix_(top_20_idx, top_20_idx)],
    [feature_names[i] for i in top_20_idx],
    title="Top 20 Features Correlation"
)
charts['correlation_heatmap'] = fig

# Feature importance chart
fig = viz.plot_feature_importance(
    feature_names,
    tree_importance.importances,
    title="Feature Importance (Tree-based)",
    top_n=15
)
charts['tree_based_importance'] = fig

# Confusion matrix
fig = viz.plot_confusion_matrix(
    metrics.confusion_matrix,
    labels=['Normal', 'Fraud'],
    normalize=False
)
charts['confusion_matrix'] = fig

# ROC curve
fig = viz.plot_roc_curve(
    metrics.fpr,
    metrics.tpr,
    metrics.roc_auc,
    title="ROC Curve"
)
charts['roc_curve'] = fig

# Precision-Recall curve
fig = viz.plot_precision_recall_curve(
    metrics.precision_curve,
    metrics.recall_curve,
    metrics.average_precision,
    title="Precision-Recall Curve"
)
charts['pr_curve'] = fig

# Box plot comparison (fraud vs normal)
fraud_probs = ensemble_probs[y == 1]
normal_probs = ensemble_probs[y == 0]
fig = viz.plot_boxplot(
    {'Fraud': fraud_probs, 'Normal': normal_probs},
    title="Ensemble Probability by Class",
    ylabel="Probability"
)
charts['boxplot'] = fig

# ============================================================================
# 6. Generate HTML Dashboard
# ============================================================================
dashboard_gen = HTMLDashboardGenerator()

# Prepare dataset info
dataset_info = {
    'total_transactions': len(transactions),
    'fraud_transactions': sum(1 for t in transactions if t.get('Is_Fraud') == 1),
    'fraud_rate': sum(1 for t in transactions if t.get('Is_Fraud') == 1) / len(transactions),
    'total_features': len(feature_names),
}

# Prepare anomaly stats (if available)
anomaly_count = sum(1 for t in transactions if t.get('Anomaly_Type', 'None') != 'None')
anomaly_stats = {
    'total_anomalies': anomaly_count,
    'anomaly_rate': anomaly_count / len(transactions),
    'avg_severity': np.mean([
        t.get('Anomaly_Severity', 0) for t in transactions
        if t.get('Anomaly_Type', 'None') != 'None'
    ]) if anomaly_count > 0 else 0,
    'high_severity_count': sum(
        1 for t in transactions
        if t.get('Anomaly_Severity', 0) > 0.7
    ),
}

# Generate comprehensive dashboard
dashboard_gen.generate_dashboard(
    output_path="output/analytics/fraud_detection_dashboard.html",
    title="Fraud Detection Analytics Dashboard",
    subtitle=f"Analysis of {len(transactions)} Transactions",
    dataset_info=dataset_info,
    model_metrics=metrics,
    importance_results=importance_results,
    correlation_results=corr_result,
    anomaly_stats=anomaly_stats,
    charts=charts,
)

print("✅ Dashboard saved to: output/analytics/fraud_detection_dashboard.html")
```

**Analytics Components:**

**1. CorrelationAnalyzer**
- Methods: Pearson, Spearman, Kendall
- Identifies highly correlated feature pairs
- Cross-correlation between feature groups
- Configurable threshold (default 0.7)

**2. FeatureImportanceAnalyzer**
- Permutation importance (model-agnostic)
- Tree-based importance (fast)
- Mutual information (statistical)
- Supports error bars and ranking

**3. ModelPerformanceAnalyzer**
- Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix generation
- ROC and Precision-Recall curves
- Multi-model comparison

**4. StatisticalTestAnalyzer**
- Chi-square test (independence)
- T-test (group comparisons)
- ANOVA (multi-group)
- Significance flagging (configurable α)

**5. VisualizationFramework**
- 14+ chart types:
  - Distributions (histogram, KDE, box, violin)
  - Heatmaps (correlation, confusion matrix)
  - Feature importance (bar, waterfall)
  - Performance curves (ROC, PR)
  - Time series, geographic maps
  - Model comparison charts
- Interactive plots with Plotly (optional)
- Customizable styles and colors

**6. HTMLDashboardGenerator**
- Multi-section dashboard
- Embedded charts (base64)
- Summary statistics grids
- Responsive design
- Print-friendly
- 300+ KB standalone HTML file

**Best Practices:**

1. **Feature Importance**: Use multiple methods and compare
2. **Correlation**: Remove highly correlated features (>0.9) before modeling
3. **Model Evaluation**: Always use held-out test set
4. **Statistical Tests**: Check assumptions before applying
5. **Visualization**: Use appropriate chart types for data types
6. **Dashboard**: Update regularly with new data

**CLI Tool:**
```bash
# Generate analytics dashboard from existing dataset
python examples/demo_analytics_dashboard.py
```

**Output:**
- Interactive HTML dashboard (300-500 KB)
- Multiple visualization charts
- Feature importance rankings
- Model performance metrics
- Statistical test results

---

### Pattern 10: Model Optimization & Registry

**Use Case:** Optimize fraud detection models with hyperparameter tuning, ensemble methods, feature selection, and maintain a versioned model registry for production deployment.

```python
from src.ml.model_optimization import (
    HyperparameterOptimizer,
    EnsembleModelBuilder,
    FeatureSelector
)
from src.ml.model_registry import (
    ModelRegistry,
    ModelComparison,
    ModelMetadata
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# Assume we have: X, y, feature_names from combined features

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================================
# 1. Hyperparameter Optimization
# ============================================================================
optimizer = HyperparameterOptimizer(
    scoring='f1',      # Optimize for F1 score
    cv=5,              # 5-fold cross-validation
    n_jobs=-1,         # Use all CPU cores
    random_state=42
)

# Grid Search for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_result = optimizer.grid_search(
    model=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    X_train=X_train,
    y_train=y_train
)

print(f"Best RF F1 Score: {rf_result.best_score:.4f}")
print(f"Best Parameters: {rf_result.best_params}")
best_rf = rf_result.best_estimator

# Random Search for Gradient Boosting
gb_param_distributions = {
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 0.9, 1.0]
}

gb_result = optimizer.random_search(
    model=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_distributions,
    X_train=X_train,
    y_train=y_train,
    n_iter=30  # Test 30 random combinations
)

print(f"Best GB F1 Score: {gb_result.best_score:.4f}")
best_gb = gb_result.best_estimator

# ============================================================================
# 2. Ensemble Model Building
# ============================================================================
ensemble_builder = EnsembleModelBuilder()

# Prepare base models
base_models = [
    ('rf', best_rf),
    ('gb', best_gb),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# Soft Voting Ensemble
voting_result = ensemble_builder.create_voting_ensemble(
    base_models=base_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    voting='soft'  # or 'hard'
)

print(f"Voting Ensemble F1: {voting_result.ensemble_score:.4f}")
print(f"Improvement: {voting_result.improvement:+.4f}")

# Stacking Ensemble
stacking_result = ensemble_builder.create_stacking_ensemble(
    base_models=base_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    meta_learner=LogisticRegression(random_state=42)
)

print(f"Stacking Ensemble F1: {stacking_result.ensemble_score:.4f}")

# Bagging Ensemble
bagging_result = ensemble_builder.create_bagging_ensemble(
    base_model=RandomForestClassifier(max_depth=20, random_state=42),
    n_estimators=100,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# ============================================================================
# 3. Feature Selection
# ============================================================================
selector = FeatureSelector()

# Recursive Feature Elimination (RFE)
rfe_result = selector.rfe_selection(
    model=best_rf,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    n_features_to_select=20,
    step=1
)

print(f"RFE selected {len(rfe_result.selected_features)} features")
print(f"Top 5: {rfe_result.selected_features[:5]}")

# LASSO Feature Selection
lasso_result = selector.lasso_selection(
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    alpha=0.001
)

print(f"LASSO selected {len(lasso_result.selected_features)} features")

# Correlation-Based Selection (remove highly correlated features)
corr_result = selector.correlation_selection(
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    threshold=0.9
)

print(f"Correlation filtering retained {len(corr_result.selected_features)} features")

# Combined selection (intersection of methods)
combined_features = (
    set(rfe_result.selected_features) & 
    set(lasso_result.selected_features) & 
    set(corr_result.selected_features)
)
print(f"Combined selection: {len(combined_features)} features")

# ============================================================================
# 4. Model Comparison with Business Recommendations
# ============================================================================
comparison = ModelComparison()

# Add models to comparison
models_to_compare = {
    'random_forest': best_rf,
    'gradient_boosting': best_gb,
    'voting_ensemble': voting_result.ensemble_model,
    'stacking_ensemble': stacking_result.ensemble_model,
}

for name, model in models_to_compare.items():
    metadata = ModelMetadata(
        model_id=f"{name}_v1",
        model_name=name.replace('_', ' ').title(),
        model_type=name,
        version='1.0',
        created_at=datetime.now(),
        hyperparameters={},
        feature_names=feature_names,
        training_samples=len(X_train),
        training_duration_seconds=0.0,
        metrics={},
        tags=['optimized', 'production'],
        description=f"Optimized {name} for fraud detection",
        author="SynFinance ML Team"
    )
    
    comparison.add_model(
        model_name=name,
        model=model,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata
    )

# Compare models with business priorities
result = comparison.compare(
    primary_metric='f1',
    business_priority='balanced',  # or 'recall_focused', 'precision_focused'
    min_recall=0.70,  # Minimum recall requirement
    max_fpr=0.10      # Maximum false positive rate
)

print(f"\nBest Model: {result.best_model}")
print(f"Best F1 Score: {result.best_metrics['f1']:.4f}")
print(f"Precision: {result.best_metrics['precision']:.4f}")
print(f"Recall: {result.best_metrics['recall']:.4f}")

print("\nBusiness Recommendations:")
for i, rec in enumerate(result.recommendations, 1):
    print(f"  {i}. {rec}")

# Export comparison report
comparison.export_comparison_report('output/model_comparison_report.txt')

# ============================================================================
# 5. Model Registry (Persistence & Versioning)
# ============================================================================
registry = ModelRegistry(base_dir='models')

# Register best model
best_model_name = result.best_model
best_model = models_to_compare[best_model_name]

model_metadata = ModelMetadata(
    model_id=f"{best_model_name}_production",
    model_name=best_model_name.replace('_', ' ').title(),
    model_type=best_model_name,
    version='1.0.0',
    created_at=datetime.now(),
    hyperparameters=rf_result.best_params if best_model_name == 'random_forest' else {},
    feature_names=feature_names,
    training_samples=len(X_train),
    training_duration_seconds=0.0,
    metrics=result.best_metrics,
    tags=['production', 'fraud_detection', 'v1'],
    description="Production fraud detection model optimized for Indian market",
    author="SynFinance ML Team"
)

# Register model
model_path = registry.register_model(
    model=best_model,
    model_name=f"{best_model_name}_prod",
    metadata=model_metadata,
    overwrite=True
)

print(f"\nModel registered: {model_path}")

# List all registered models
all_models = registry.list_models()
print(f"\nRegistered models: {len(all_models)}")
for model_name in all_models:
    metadata = registry.get_metadata(model_name)
    print(f"  - {model_name} (v{metadata.version}, F1={metadata.metrics.get('f1', 0):.4f})")

# Load model for inference
loaded_model, loaded_metadata = registry.load_model(f"{best_model_name}_prod")
predictions = loaded_model.predict(X_test[:10])
print(f"\nLoaded model predictions (first 10): {predictions}")

# Export registry report
registry.export_registry_report('output/model_registry_report.txt')

# ============================================================================
# 6. Production Deployment Workflow
# ============================================================================

# Filter models by tag
production_models = registry.list_models(tag='production')
print(f"\nProduction models: {production_models}")

# Get latest version
fraud_models = [m for m in all_models if 'fraud' in m.lower()]
if fraud_models:
    latest_fraud_model, metadata = registry.load_model(fraud_models[0])
    print(f"\nLatest fraud model: {metadata.model_name} v{metadata.version}")
    print(f"Training date: {metadata.created_at.strftime('%Y-%m-%d')}")
    print(f"F1 Score: {metadata.metrics.get('f1', 0):.4f}")
```

**Complete Example:**
```bash
python examples/optimize_fraud_models.py
```

**Output:**
- Optimized models with best hyperparameters
- Ensemble models (voting, stacking, bagging)
- Feature selection results (RFE, LASSO, correlation)
- Model comparison with business recommendations
- Registered models in versioned registry
- Exportable comparison and registry reports

**Production Benefits:**
- **Hyperparameter Optimization:** Systematic tuning for optimal performance
- **Ensemble Methods:** Combine multiple models for better predictions
- **Feature Selection:** Reduce dimensionality and improve interpretability
- **Model Registry:** Version control and reproducibility
- **Business Focus:** Recommendations tailored to fraud detection priorities
- **Deployment Ready:** Load registered models for inference

---

## API Reference Summary

| Function | Purpose | Returns |
|----------|---------|---------|
| `CustomerGenerator.generate_customer()` | Create single customer | CustomerProfile |
| `TransactionGenerator.generate_transaction()` | Create single transaction | dict (45 fields) |
| `TransactionGenerator.generate_customer_transactions()` | Create multiple for customer | List[dict] |
| `generate_realistic_dataset()` | Full dataset generation | DataFrame |
| `AnomalyPatternGenerator.detect_anomaly_patterns()` | Detect anomalies | dict (5 anomaly fields) |
| `AnomalyMLFeatureGenerator.generate_features()` | Generate ML features | AnomalyMLFeatures (27 features) |
| `IsolationForestAnomalyDetector.fit_predict()` | Unsupervised detection | List[float] (anomaly scores) |
| `CombinedMLFeatureGenerator.generate_batch_features()` | Generate combined features | List[CombinedMLFeatures] (68 features) |
| `CorrelationAnalyzer.analyze()` | Correlation analysis | CorrelationResult |
| `FeatureImportanceAnalyzer.analyze_all()` | Multi-method importance | List[FeatureImportanceResult] |
| `ModelPerformanceAnalyzer.analyze()` | Model performance metrics | ModelMetrics |
| `StatisticalTestAnalyzer.test_fraud_vs_normal()` | Statistical significance | StatisticalTestResult |
| `VisualizationFramework.plot_*()` | Generate charts | matplotlib.Figure |
| `HTMLDashboardGenerator.generate_dashboard()` | Create HTML dashboard | None (saves file) |
| `HyperparameterOptimizer.grid_search()` | Grid search optimization | OptimizationResult |
| `HyperparameterOptimizer.random_search()` | Random search optimization | OptimizationResult |
| `EnsembleModelBuilder.create_voting_ensemble()` | Build voting ensemble | EnsembleResult |
| `EnsembleModelBuilder.create_stacking_ensemble()` | Build stacking ensemble | EnsembleResult |
| `EnsembleModelBuilder.create_bagging_ensemble()` | Build bagging ensemble | EnsembleResult |
| `FeatureSelector.rfe_selection()` | RFE feature selection | FeatureSelectionResult |
| `FeatureSelector.lasso_selection()` | LASSO feature selection | FeatureSelectionResult |
| `FeatureSelector.correlation_selection()` | Correlation-based selection | FeatureSelectionResult |
| `ModelRegistry.register_model()` | Save model with metadata | str (path) |
| `ModelRegistry.load_model()` | Load registered model | Tuple[model, metadata] |
| `ModelRegistry.list_models()` | List all models | List[str] |
| `ModelComparison.add_model()` | Add model to comparison | None |
| `ModelComparison.compare()` | Compare all models | ModelComparisonResult |
| `VarianceAnalyzer.run_analysis()` | Quality validation | dict (results) |

---

## Quality Assurance

### Variance Analysis

SynFinance includes built-in quality validation through variance analysis:

```python
from scripts.analyze_variance import VarianceAnalyzer

# Analyze dataset quality
analyzer = VarianceAnalyzer('output/dataset.csv', output_dir='output/quality')
results = analyzer.run_analysis()

# Access results
print(f"Pass Rate: {results['summary']['quality_metrics']['pass_rate']:.1%}")
print(f"High Variance Fields: {results['summary']['quality_metrics']['high_variance_count']}")

# Individual field results
for field in results['field_results']:
    print(f"{field['field_name']}: {field['status']} (Entropy={field['entropy']:.2f})")
```

### Quality Thresholds

- **Entropy Threshold**: 1.5 (minimum information content)
- **CV Threshold**: 0.15 (minimum coefficient of variation for numeric fields)
- **Target Pass Rate**: 80%+ of fields meeting thresholds

### Automated Testing

```python
import pytest
from scripts.analyze_variance import VarianceAnalyzer

def test_dataset_quality():
    """Ensure generated data meets quality standards"""
    df = generate_realistic_dataset(1000, 50, days=30)
    df.to_csv('test_dataset.csv', index=False)
    
    analyzer = VarianceAnalyzer('test_dataset.csv')
    results = analyzer.run_analysis()
    
    assert results['summary']['quality_metrics']['pass_rate'] >= 0.80
```

---

## Support

- **Field Reference:** [FIELD_REFERENCE.md](../technical/FIELD_REFERENCE.md)
- **Examples:** See `examples/` folder
- **Tests:** `tests/` for integration examples (111 tests, 100% passing)
- **Issues:** GitHub Issues for bugs

---

**Enterprise-ready integration for production systems.**
