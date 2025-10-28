# SynFinance Quick Reference

**Common Operations & Code Snippets**

**Complete Field Reference:** [FIELD_REFERENCE.md](../technical/FIELD_REFERENCE.md) (45 fields)

---

## Customer Generation

### Create Single Customer
```python
from src.customer_generator import CustomerGenerator
gen = CustomerGenerator(seed=42)
customer = gen.generate_customer()
```

### Create 100 Customers
```python
customers = [gen.generate_customer() for _ in range(100)]
```

### Access Customer Fields
```python
customer.customer_id      # "CUST0000001"
customer.segment          # CustomerSegment.YOUNG_PROFESSIONAL
customer.city             # "Mumbai"
customer.income_bracket   # IncomeBracket.UPPER_MIDDLE
customer.monthly_income   # 120000.0
```

---

## Transaction Generation

### Single Transaction
```python
from src.generators.transaction_core import TransactionGenerator
from datetime import datetime

txn_gen = TransactionGenerator(seed=42)
txn = txn_gen.generate_transaction(customer, datetime.now())
```

### Multiple Transactions
```python
transactions = txn_gen.generate_customer_transactions(
    customer=customer,
    count=50,
    days=30
)
```

### Access Transaction Fields
```python
txn["Transaction_ID"]       # "TXN_20251021_000001"
txn["Amount"]               # 1850.50
txn["Category"]             # "Food & Dining"
txn["City"]                 # "Mumbai"
txn["City_Tier"]            # 1
txn["Is_Fraud"]             # 0
# 45 total fields - see FIELD_REFERENCE.md
```

---

## Full Dataset Generation

### Generate Dataset
```python
from src.data_generator import generate_realistic_dataset

df = generate_realistic_dataset(
    num_customers=1000,
    transactions_per_customer=50,
    start_date="2025-01-01",
    days=90,
    seed=42
)

print(df.shape)  # (50000, 45) - 45 fields per transaction
```

### Save to CSV
```python
df.to_csv("output/transactions.csv", index=False)
```

### Save to Excel
```python
df.to_excel("output/transactions.xlsx", index=False)
```

---

## Fraud Pattern Injection (NEW in v0.4.0)

### Quick Fraud Injection
```python
from src.generators.fraud_patterns import inject_fraud_into_dataset

# Generate base dataset
df = generate_realistic_dataset(1000, 50, days=90, seed=42)
transactions = df.to_dict('records')

# Create customers
gen = CustomerGenerator(seed=42)
customers = [gen.generate_customer() for _ in range(1000)]

# Inject fraud at 2% rate
modified_txns, stats = inject_fraud_into_dataset(
    transactions, customers, fraud_rate=0.02, seed=42
)

# View results
fraud_df = pd.DataFrame(modified_txns)
print(f"Total Fraud: {stats['total_fraud']}")
print(f"Fraud Rate: {stats['fraud_rate']:.2%}")
```

### Fraud Pattern Generator
```python
from src.generators.fraud_patterns import FraudPatternGenerator

# Initialize with 2% fraud rate
fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)

# Apply fraud to single transaction
customer_history = []  # List of previous transactions
modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(
    transaction, customer, customer_history
)

# Check if fraud was applied
if fraud_indicator:
    print(f"Fraud Type: {fraud_indicator.fraud_type.value}")
    print(f"Confidence: {fraud_indicator.confidence}")
    print(f"Severity: {fraud_indicator.severity}")
    print(f"Reason: {fraud_indicator.reason}")
```

### Fraud Statistics
```python
# Get fraud statistics
stats = fraud_gen.get_fraud_statistics()

print(f"Total Transactions: {stats['total_transactions']}")
print(f"Total Fraud: {stats['total_fraud']}")
print(f"Fraud Rate: {stats['fraud_rate']:.2%}")

# Fraud distribution by type
for fraud_type, count in stats['fraud_by_type'].items():
    percentage = stats['fraud_type_distribution'][fraud_type]
    print(f"{fraud_type}: {count} ({percentage:.1%})")
```

### 10 Fraud Pattern Types
```python
from src.generators.fraud_patterns import FraudType

# Available fraud patterns:
FraudType.CARD_CLONING           # Impossible travel
FraudType.ACCOUNT_TAKEOVER       # Behavioral changes
FraudType.MERCHANT_COLLUSION     # Round amounts
FraudType.VELOCITY_ABUSE         # High frequency
FraudType.AMOUNT_MANIPULATION    # Structuring
FraudType.REFUND_FRAUD          # Excessive refunds
FraudType.STOLEN_CARD           # Inactivity spike
FraudType.SYNTHETIC_IDENTITY    # Limited history
FraudType.FIRST_PARTY_FRAUD     # Bust-out
FraudType.FRIENDLY_FRAUD        # Chargebacks
```

### Fraud Fields (5 new fields)
```python
# Every transaction has these fraud fields after injection:
txn["Fraud_Type"]        # "Card Cloning" or "None"
txn["Fraud_Confidence"]  # 0.0-1.0 score
txn["Fraud_Reason"]      # "Impossible travel: Mumbai to Delhi in 30 min"
txn["Fraud_Severity"]    # "low", "medium", "high", "critical"
txn["Fraud_Evidence"]    # JSON: {"distance_km": 1400, "speed_kmh": 2800}
```

### Configure Fraud Rates
```python
# Low fraud (0.5%)
fraud_gen = FraudPatternGenerator(fraud_rate=0.005)

# Medium fraud (2%)
fraud_gen = FraudPatternGenerator(fraud_rate=0.02)

# High fraud (5%)
fraud_gen = FraudPatternGenerator(fraud_rate=0.05)

# Runtime adjustment
fraud_gen.set_fraud_rate(0.03)  # Change to 3%
```

**See [FRAUD_PATTERNS.md](../technical/FRAUD_PATTERNS.md) for detailed fraud pattern documentation.**

---

## Anomaly Pattern Injection (NEW in v0.6.0)

### Quick Anomaly Injection
```python
from src.generators.anomaly_patterns import AnomalyPatternGenerator, apply_anomaly_labels

# Generate base dataset with fraud
df = generate_realistic_dataset(1000, 50, days=90, seed=42)
transactions = df.to_dict('records')

# Create customers
gen = CustomerGenerator(seed=42)
customers = [gen.generate_customer() for _ in range(1000)]

# Inject fraud first (2%)
fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)
transactions = fraud_gen.inject_fraud_patterns(transactions, customers)

# Inject anomalies second (5%)
anomaly_gen = AnomalyPatternGenerator(seed=42)
transactions = anomaly_gen.inject_anomaly_patterns(
    transactions, customers, anomaly_rate=0.05
)

# Label non-anomalous transactions
transactions = apply_anomaly_labels(transactions)

# View results
anomaly_df = pd.DataFrame(transactions)
print(f"Anomalies: {(anomaly_df['Anomaly_Type'] != 'None').sum()}")
print(f"Anomaly Rate: {(anomaly_df['Anomaly_Type'] != 'None').mean():.2%}")
```

### Anomaly Statistics
```python
# Get anomaly statistics
stats = anomaly_gen.get_statistics()

print(f"Total Transactions: {stats['total_transactions']}")
print(f"Anomaly Count: {stats['anomaly_count']}")
print(f"Anomaly Rate: {stats['anomaly_rate']:.2%}")

# Anomaly distribution by type
for anomaly_type, count in stats['anomalies_by_type'].items():
    print(f"{anomaly_type}: {count}")
```

### 4 Anomaly Pattern Types
```python
from src.generators.anomaly_patterns import AnomalyType

# Available anomaly patterns:
AnomalyType.BEHAVIORAL    # Rare categories, amount spikes, payment changes
AnomalyType.GEOGRAPHIC    # Impossible travel, unusual locations
AnomalyType.TEMPORAL      # Unusual hours (late night, early morning)
AnomalyType.AMOUNT        # Spending spikes, micro-transactions, round amounts
AnomalyType.NONE          # No anomaly detected
```

### Anomaly Fields (5 new fields)
```python
# Every transaction has these anomaly fields after injection:
txn["Anomaly_Type"]        # "BEHAVIORAL", "GEOGRAPHIC", "TEMPORAL", "AMOUNT", "None"
txn["Anomaly_Confidence"]  # 0.0-1.0 score
txn["Anomaly_Reason"]      # "Unusual purchase: Jewelry (5% of history)"
txn["Anomaly_Severity"]    # 0.0-1.0 score
txn["Anomaly_Evidence"]    # JSON: {"unusual_category": "Jewelry", "multiplier": 4.2}
```

### Filter Anomalies by Severity
```python
# High severity anomalies only (≥0.7)
high_severity = anomaly_df[anomaly_df['Anomaly_Severity'] >= 0.7]
print(f"High severity: {len(high_severity)}")

# Medium severity (0.3-0.7)
medium_severity = anomaly_df[
    (anomaly_df['Anomaly_Severity'] >= 0.3) & 
    (anomaly_df['Anomaly_Severity'] < 0.7)
]
```

### Parse Anomaly Evidence
```python
import json

def safe_parse_evidence(evidence_str):
    """Safely parse evidence JSON"""
    try:
        return json.loads(evidence_str) if evidence_str else {}
    except json.JSONDecodeError:
        return {}

anomaly_df['Evidence_Dict'] = anomaly_df['Anomaly_Evidence'].apply(safe_parse_evidence)

# Access evidence fields
for idx, row in anomaly_df.head().iterrows():
    if row['Anomaly_Type'] != 'None':
        evidence = row['Evidence_Dict']
        print(f"{row['Anomaly_Type']}: {evidence}")
```

### CLI: Generate Anomaly Dataset
```bash
# Generate complete dataset with fraud + anomalies
python examples/generate_anomaly_dataset.py \
    --num-transactions 10000 \
    --num-customers 200 \
    --num-days 90 \
    --fraud-rate 0.02 \
    --anomaly-rate 0.05 \
    --output-dir output/anomaly_dataset \
    --seed 42

# Outputs:
#   - anomaly_dataset.csv (50 fields)
#   - dataset_summary.json (statistics)
#   - dataset_summary.txt (report)
```

### Anomaly Rate Presets
```python
# Development (high visibility)
anomaly_gen = AnomalyPatternGenerator(seed=42)
transactions = anomaly_gen.inject_anomaly_patterns(txns, custs, anomaly_rate=0.08)

# Production (realistic)
anomaly_gen = AnomalyPatternGenerator(seed=42)
transactions = anomaly_gen.inject_anomaly_patterns(txns, custs, anomaly_rate=0.05)

# ML Training (balanced)
anomaly_gen = AnomalyPatternGenerator(seed=42)
transactions = anomaly_gen.inject_anomaly_patterns(txns, custs, anomaly_rate=0.10)
```

### Anomaly Analysis (Days 3-4)

```python
from src.generators.anomaly_analysis import (
    AnomalyFraudCorrelationAnalyzer,
    SeverityDistributionAnalyzer,
    TemporalClusteringAnalyzer,
    GeographicHeatmapAnalyzer
)

transactions = df.to_dict('records')

# Correlation analysis
correlation_analyzer = AnomalyFraudCorrelationAnalyzer()
result = correlation_analyzer.analyze_correlation(transactions)
print(f"Phi Coefficient: {result.phi_coefficient:.4f}")

# Severity distribution
severity_analyzer = SeverityDistributionAnalyzer()
result = severity_analyzer.analyze_severity_distribution(transactions)
print(f"Mean Severity: {result.mean_severity:.4f}")

# Temporal clustering
temporal_analyzer = TemporalClusteringAnalyzer()
result = temporal_analyzer.analyze_temporal_clustering(transactions)
print(f"Clusters: {result.cluster_count}")

# Geographic heatmap
geographic_analyzer = GeographicHeatmapAnalyzer()
result = geographic_analyzer.analyze_geographic_distribution(transactions)
print(f"Cities: {len(result.city_anomaly_counts)}")
```

### Anomaly ML Features (Days 5-6)

```python
from src.generators.anomaly_ml_features import (
    AnomalyMLFeatureGenerator,
    IsolationForestAnomalyDetector
)

# Generate 27 ML features
feature_gen = AnomalyMLFeatureGenerator()
detector = IsolationForestAnomalyDetector(contamination=0.10)

# Build histories
customer_histories = {}
for txn in transactions:
    cid = txn['Customer_ID']
    if cid not in customer_histories:
        customer_histories[cid] = []
    customer_histories[cid].append(txn)

# Train Isolation Forest
isolation_scores = detector.fit_predict(transactions)

# Generate features
features_list = feature_gen.generate_features_batch(
    transactions=transactions,
    customer_histories=customer_histories,
    isolation_scores=isolation_scores
)

# Convert to DataFrame
features_df = pd.DataFrame([vars(f) for f in features_list])
print(f"Generated {len(features_df)} feature vectors with {features_df.shape[1]} features")
```

### CLI: Comprehensive Anomaly Analysis

```bash
# Run complete anomaly analysis pipeline
python examples/analyze_anomaly_patterns.py

# Generates:
# - output/anomaly_analysis_dataset.csv (full dataset)
# - output/anomaly_ml_features_complete.csv (27 ML features)
# - output/anomaly_analysis_summary.json (statistics)
```

**See [ANOMALY_PATTERNS.md](../technical/ANOMALY_PATTERNS.md) for detailed anomaly pattern documentation.**

---

## Combined ML Features (NEW in v0.7.0)

### Generate All 68 Features

```python
from src.generators.combined_ml_features import CombinedMLFeatureGenerator
from dataclasses import asdict

# Assuming you have fraud_features_list and anomaly_features_list
combined_gen = CombinedMLFeatureGenerator()

# Convert to dicts
fraud_dicts = [f.to_dict() for f in fraud_features_list]
anomaly_dicts = [asdict(f) for f in anomaly_features_list]

# Generate combined features (68 total)
combined_features = combined_gen.generate_batch_features(
    transactions,
    fraud_dicts,
    anomaly_dicts
)

print(f"Generated {len(combined_features)} transactions")
print(f"Features per transaction: 68")
```

### Feature Breakdown

- **Fraud Features**: 32 (velocity, geographic, behavioral, network)
- **Anomaly Features**: 26 (frequency, severity, persistence, evidence)
- **Interaction Features**: 10 (risk amplification, pattern alignment, ensemble)

### Key Interaction Features

```python
for f in combined_features:
    print(f"Transaction {f.transaction_id}:")
    print(f"  Ensemble Probability: {f.ensemble_fraud_probability:.3f}")
    print(f"  Risk Amplification: {f.risk_amplification_score:.3f}")
    print(f"  High Risk Combo: {f.high_risk_combination}")
    print(f"  Conflict Indicator: {f.conflict_indicator}")
```

### Export Combined Features

```python
# Export to list of dicts
features_dicts = combined_gen.export_to_dict_list(combined_features)

# Save to CSV
import pandas as pd
df = pd.DataFrame(features_dicts)
df.to_csv('output/combined_ml_features.csv', index=False)
```

### Feature Statistics

```python
# Calculate statistics
stats = combined_gen.get_feature_statistics(combined_features)

# Display key stats
for feature_name in ['ensemble_fraud_probability', 'weighted_risk_score']:
    s = stats[feature_name]
    print(f"{feature_name}: mean={s['mean']:.3f}, std={s['std']:.3f}")
```

### CLI Tool

```bash
python examples/generate_combined_features.py

# Outputs:
# - output/combined_features/combined_ml_features.csv (68 features)
# - output/combined_features/feature_statistics.json (statistics)
# - output/combined_features/sample_features.json (samples)
```

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
# 359 tests, 100% passing
```

### Run Specific Test File
```bash
pytest tests/generators/test_anomaly_patterns.py -v  # 25 tests
pytest tests/generators/test_anomaly_analysis.py -v  # 21 tests
pytest tests/generators/test_anomaly_ml_features.py -v  # 20 tests
pytest tests/generators/test_combined_ml_features.py -v  # 26 tests
```

### Run Variance Tests
```bash
pytest tests/test_col_variance.py -v
# Tests column variance and data quality
```

### Run Single Test
```bash
pytest tests/test_customer_generation.py::test_segment_distribution -v
```

### Quick Test Run
```bash
pytest tests/ -q
# Quick summary output
```

---

## Data Quality & Variance Analysis

### Run Variance Analysis
```bash
# Analyze dataset quality
python scripts/analyze_variance.py

# With custom input
python scripts/analyze_variance.py --input output/dataset.csv --output output/quality
```

### Variance Analysis in Code
```python
from scripts.analyze_variance import VarianceAnalyzer

# Analyze generated dataset
df = generate_realistic_dataset(1000, 50, days=30)
df.to_csv('output/dataset.csv', index=False)

# Run analysis
analyzer = VarianceAnalyzer('output/dataset.csv', output_dir='output/quality')
results = analyzer.run_analysis()

# Check quality
print(f"Pass Rate: {results['summary']['quality_metrics']['pass_rate']:.1%}")
print(f"Passing: {results['summary']['quality_metrics']['pass_count']} fields")
print(f"Failing: {results['summary']['quality_metrics']['fail_count']} fields")
```

### Quality Workflow
```python
def generate_validated_dataset(num_customers, min_quality=0.80):
    """Generate dataset with quality validation"""
    
    # Generate data
    df = generate_realistic_dataset(num_customers, 50, days=30)
    df.to_csv('output/dataset.csv', index=False)
    
    # Run variance analysis
    analyzer = VarianceAnalyzer('output/dataset.csv')
    results = analyzer.run_analysis()
    
    quality = results['summary']['quality_metrics']['pass_rate']
    
    if quality < min_quality:
        raise ValueError(f"Quality {quality:.1%} below {min_quality:.1%}")
    
    print(f"Quality: {quality:.1%} - PASS")
    return df

# Example
df = generate_validated_dataset(1000)
```

---

## Geographic Operations

### Select Transaction City
```python
from src.generators.geographic_generator import GeographicPatternGenerator

geo_gen = GeographicPatternGenerator(seed=42)
city, location_type = geo_gen.select_transaction_city(customer)
# Returns: ("Mumbai", "home") or ("Pune", "nearby")
```

### Apply Cost-of-Living Adjustment
```python
adjusted_amount = geo_gen.apply_cost_of_living_adjustment(1000, "Mumbai")
# Result: 1300 (30% higher for Tier 1 city)
```

### Get City Tier
```python
tier = geo_gen.get_city_tier("Mumbai")  # 1 (Metro)
tier = geo_gen.get_city_tier("Indore")  # 2 (Major)
tier = geo_gen.get_city_tier("Patna")   # 3 (Smaller)
```

---

## Temporal Operations

### Select Transaction Hour
```python
from src.generators.temporal_generator import TemporalPatternGenerator
from datetime import datetime

temporal_gen = TemporalPatternGenerator(seed=42)
hour = temporal_gen.select_transaction_hour(customer, datetime.now())
# Returns: 0-23 based on occupation and day type
```

### Get Temporal Multiplier
```python
multiplier, breakdown = temporal_gen.get_combined_temporal_multiplier(
    customer,
    datetime(2025, 10, 1),  # Salary day
    festivals
)
# multiplier: 2.0 (salary day boost)
```

---

## Merchant Operations

### Select Merchant
```python
from src.generators.merchant_generator import MerchantGenerator

merchant_gen = MerchantGenerator(seed=42)
merchant = merchant_gen.select_merchant(
    customer=customer,
    category="Food & Dining",
    city="Mumbai"
)
```

### Get Merchant Stats
```python
stats = merchant_gen.get_merchant_stats("Mumbai")
# Returns: total_merchants, chain_count, local_count, avg_reputation
```

---

## Customer Segments

```python
from src.customer_profile import CustomerSegment

# Available segments
CustomerSegment.YOUNG_PROFESSIONAL       # 20%
CustomerSegment.FAMILY_ORIENTED          # 25%
CustomerSegment.BUDGET_CONSCIOUS         # 20%
CustomerSegment.TECH_SAVVY_MILLENNIAL    # 15%
CustomerSegment.AFFLUENT_SHOPPER         # 8%
CustomerSegment.SENIOR_CONSERVATIVE      # 7%
CustomerSegment.STUDENT                  # 5%
```

---

## Transaction Categories

```python
from src.config import TRANSACTION_CATEGORIES

# Available categories
TRANSACTION_CATEGORIES = [
    "Groceries",
    "Food & Dining",
    "Shopping",
    "Entertainment",
    "Travel",
    "Healthcare",
    "Education",
    "Utilities",
    "Transportation",
    "Electronics",
    "Fuel",
    "Insurance"
]
```

---

## Payment Modes

```python
from src.config import PAYMENT_MODES

PAYMENT_MODES = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Digital Wallet",
    "Net Banking",
    "Cash",
    "BNPL"
]
```

---

## Indian Cities (20 cities across 3 tiers)

**Tier 1 (Metros):**
Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad

**Tier 2 (Major):**
Jaipur, Lucknow, Surat, Chandigarh, Indore, Kochi, Nagpur

**Tier 3 (Smaller):**
Bhopal, Visakhapatnam, Patna, Vadodara, Coimbatore

---

## Filtering & Analysis

### Filter by Fraud
```python
fraud_txns = df[df['Is_Fraud'] == 1]
normal_txns = df[df['Is_Fraud'] == 0]
```

### Filter by City Tier
```python
metro_txns = df[df['City_Tier'] == 1]
```

### Filter by Category
```python
food_txns = df[df['Category'] == 'Food & Dining']
```

### Group by Customer
```python
customer_stats = df.groupby('Customer_ID').agg({
    'Amount': ['sum', 'mean', 'count'],
    'Is_Fraud': 'sum'
})
```

---

## Common Patterns

### Generate Test Data for ML
```python
# Balanced fraud dataset
df = generate_realistic_dataset(5000, 100, days=180, seed=42)
fraud = df[df['Is_Fraud'] == 1]
normal = df[df['Is_Fraud'] == 0].sample(n=len(fraud)*49)
balanced = pd.concat([fraud, normal]).sample(frac=1)
```

### Generate Monthly Report
```python
df['Month'] = pd.to_datetime(df['Date']).dt.month
monthly = df.groupby('Month')['Amount'].agg(['sum', 'mean', 'count'])
```

### Generate City Analysis
```python
city_stats = df.groupby(['City', 'City_Tier']).agg({
    'Amount': 'sum',
    'Transaction_ID': 'count'
}).sort_values('Amount', ascending=False)
```

---

## Performance Benchmarks

- **Transaction Generation:** 17,200+ txn/sec
- **Customer Generation:** <1ms per profile
- **Dataset (100K txns):** ~6 seconds
- **Tests (111 tests):** ~7 seconds (100% passing)
- **Variance Analysis:** ~2 seconds for 50K transactions

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run Streamlit app
streamlit run src/app.py
```

---

## ML Fraud Detection Commands

**NEW in v0.5.0:** Machine learning workflow for fraud detection.

### Train Fraud Detection Model

```bash
# Basic training (5000 transactions, 10% fraud)
python examples/train_fraud_detector.py

# Custom configuration
python examples/train_fraud_detector.py \
    --num-transactions 10000 \
    --fraud-rate 0.15 \
    --output-dir output/my_training \
    --seed 123

# Outputs: confusion_matrices.png, roc_curves.png, feature_importance.png, evaluation_results.json
```

### Interactive Jupyter Tutorial

```bash
# Launch Jupyter notebook
jupyter notebook examples/fraud_detection_tutorial.ipynb

# Run all cells for complete ML workflow:
# 1. Data generation
# 2. Fraud injection
# 3. Feature engineering (32 features)
# 4. Dataset preparation
# 5. Model training (Random Forest + XGBoost)
# 6. Evaluation and visualization
```

### Validate Data Quality

```bash
# Analyze dataset quality
python scripts/validate_data_quality.py output/train.csv

# Custom thresholds
python scripts/validate_data_quality.py output/train.csv \
    --output-dir output/quality_reports \
    --correlation-threshold 0.8 \
    --iqr-multiplier 1.5

# Outputs: quality_validation_report.json, correlation_heatmap.png, feature_distributions.png
```

### Generate ML Training Dataset

```python
from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.ml_dataset_generator import MLDatasetGenerator

# 1. Generate transactions with fraud
generator = DataGenerator(num_customers=100, num_days=30)
customers = generator.generate_customers()
transactions = generator.generate_transactions(num_transactions=5000)

fraud_gen = FraudPatternGenerator(seed=42)
transactions = fraud_gen.inject_fraud_patterns(transactions, customers, fraud_rate=0.1)

# 2. Engineer 32 ML features
engineer = MLFeatureEngineer()
# Build transaction history...
features = engineer.engineer_features(transactions, transaction_history)

# 3. Create balanced ML dataset
dataset_gen = MLDatasetGenerator(seed=42)
split, metadata = dataset_gen.create_ml_ready_dataset(
    features, balance_strategy='undersample', target_fraud_rate=0.5
)

# 4. Export to multiple formats
dataset_gen.export_to_csv(split.train, 'output/train.csv')
dataset_gen.export_to_parquet(split.train, 'output/train.parquet')
X_train, y_train, feature_names = dataset_gen.export_to_numpy(split.train, 'output/train')
```

### Feature Engineering

```python
from src.generators.ml_features import MLFeatureEngineer

engineer = MLFeatureEngineer()
features = engineer.engineer_features(transactions, transaction_history)

# 32 features across 6 categories:
# - Aggregate (6): daily_txn_count, weekly_txn_amount, etc.
# - Velocity (6): txn_frequency_1h, amount_velocity_6h, etc.
# - Geographic (5): distance_from_home, travel_velocity_kmh, etc.
# - Temporal (6): is_unusual_hour, is_weekend, temporal_cluster_flag, etc.
# - Behavioral (5): category_diversity_score, merchant_loyalty_score, etc.
# - Network (4): shared_merchant_count, customer_proximity_score, etc.

# Get feature metadata
metadata = engineer.get_feature_metadata()
```

### Export Formats

```python
from src.generators.ml_dataset_generator import MLDatasetGenerator

dataset_gen = MLDatasetGenerator()

# CSV (pandas-compatible)
dataset_gen.export_to_csv(features, 'output/data.csv')

# JSON (structured data)
dataset_gen.export_to_json(features, 'output/data.json')

# Parquet (10x smaller, efficient)
dataset_gen.export_to_parquet(features, 'output/data.parquet')

# NumPy arrays (direct sklearn input)
X, y, feature_names = dataset_gen.export_to_numpy(features, 'output/data')
# Creates: data_X.npy, data_y.npy, data_features.json
```

### Quick ML Model Training

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load NumPy arrays
X_train = np.load('output/train_X.npy')
y_train = np.load('output/train_y.npy')
X_test = np.load('output/test_X.npy')
y_test = np.load('output/test_y.npy')

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Advanced Analytics & Dashboards (NEW in Week 6)

### Quick Analytics Dashboard

```python
# Generate complete analytics dashboard
python examples/demo_analytics_dashboard.py
# Output: fraud_detection_dashboard.html (360+ KB interactive dashboard)
```

### Correlation Analysis

```python
from src.analytics import CorrelationAnalyzer
import numpy as np

# Prepare feature matrix
X = np.array([f.get_feature_values() for f in combined_features])
feature_names = combined_features[0].get_feature_names()

# Analyze correlations
analyzer = CorrelationAnalyzer(threshold=0.7)
result = analyzer.analyze(X, feature_names, method='pearson')

# Get highly correlated pairs
for f1, f2, corr in result.get_highly_correlated_pairs()[:10]:
    print(f"{f1} ↔ {f2}: {corr:.3f}")
```

### Feature Importance

```python
from src.analytics import FeatureImportanceAnalyzer
from sklearn.ensemble import RandomForestClassifier

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Analyze importance (3 methods)
analyzer = FeatureImportanceAnalyzer(n_repeats=5)
results = analyzer.analyze_all(clf, X_train, y_train, X_test, y_test, feature_names)

# Top features per method
for result in results:
    print(f"\n{result.method.upper()}:")
    for feat, imp in result.get_top_features(5):
        print(f"  {feat}: {imp:.4f}")
```

### Model Performance Metrics

```python
from src.analytics import ModelPerformanceAnalyzer

# Get predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Analyze performance
analyzer = ModelPerformanceAnalyzer()
metrics = analyzer.analyze(y_test, y_pred, y_pred_proba)

print(f"Accuracy:  {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall:    {metrics.recall:.4f}")
print(f"F1-Score:  {metrics.f1:.4f}")
print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
```

### Generate Visualizations

```python
from src.analytics import VisualizationFramework

viz = VisualizationFramework()

# Distribution plot
fig = viz.plot_distribution(ensemble_probs, title="Fraud Probability", kde=True)
fig.savefig('output/distribution.png', dpi=150)

# Correlation heatmap
fig = viz.plot_correlation_heatmap(
    result.correlation_matrix, feature_names, title="Feature Correlations"
)
fig.savefig('output/heatmap.png', dpi=150)

# Feature importance chart
fig = viz.plot_feature_importance(
    feature_names, importances, title="Top Features", top_n=15
)
fig.savefig('output/importance.png', dpi=150)

# ROC curve
fig = viz.plot_roc_curve(metrics.fpr, metrics.tpr, metrics.roc_auc)
fig.savefig('output/roc.png', dpi=150)

# Confusion matrix
fig = viz.plot_confusion_matrix(
    metrics.confusion_matrix, labels=['Normal', 'Fraud']
)
fig.savefig('output/confusion.png', dpi=150)
```

### HTML Dashboard Generation

```python
from src.analytics import HTMLDashboardGenerator

dashboard = HTMLDashboardGenerator()

# Prepare data
dataset_info = {
    'total_transactions': len(transactions),
    'fraud_transactions': sum(1 for t in transactions if t.get('Is_Fraud') == 1),
    'fraud_rate': 0.008,
    'total_features': 68,
}

# Generate dashboard
dashboard.generate_dashboard(
    output_path="output/dashboard.html",
    title="Fraud Detection Analytics",
    subtitle=f"{len(transactions)} Transactions Analyzed",
    dataset_info=dataset_info,
    model_metrics=metrics,
    importance_results=results,
    correlation_results=result,
    charts={'roc': roc_fig, 'heatmap': heatmap_fig},
)

print("✅ Dashboard saved: output/dashboard.html")
```

### Statistical Tests

```python
from src.analytics import StatisticalTestAnalyzer

analyzer = StatisticalTestAnalyzer(alpha=0.05)

# Test fraud vs normal distributions
result = analyzer.test_fraud_vs_normal(ensemble_probs, y_true)
print(f"T-test p-value: {result.p_value:.2e}")
print(f"Significant: {result.is_significant}")
```

---

## Model Optimization & Registry (NEW in Week 6 Day 3)

### Complete Optimization Pipeline

```python
# Run full optimization pipeline
python examples/optimize_fraud_models.py
# Output: Optimized models, ensembles, feature selection, model registry
```

### Hyperparameter Optimization

```python
from src.ml.model_optimization import HyperparameterOptimizer
from sklearn.ensemble import RandomForestClassifier

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    scoring='f1',      # or 'roc_auc', 'precision', 'recall'
    cv=5,              # Cross-validation folds
    n_jobs=-1,         # Use all cores
    random_state=42
)

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

result = optimizer.grid_search(
    model=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train
)

print(f"Best F1: {result.best_score:.4f}")
print(f"Best params: {result.best_params}")
best_model = result.best_estimator

# Random Search (faster for large search spaces)
param_distributions = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [5, 10, 15, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

result = optimizer.random_search(
    model=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    X_train=X_train,
    y_train=y_train,
    n_iter=30  # Test 30 random combinations
)
```

### Ensemble Models

```python
from src.ml.model_optimization import EnsembleModelBuilder
from sklearn.linear_model import LogisticRegression

builder = EnsembleModelBuilder()

# Voting Ensemble (combine predictions)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

result = builder.create_voting_ensemble(
    base_models=base_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    voting='soft'  # or 'hard'
)

print(f"Ensemble F1: {result.ensemble_score:.4f}")
print(f"Improvement: {result.improvement:+.4f}")

# Stacking Ensemble (meta-learner)
result = builder.create_stacking_ensemble(
    base_models=base_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    meta_learner=LogisticRegression(random_state=42)
)

# Bagging Ensemble
result = builder.create_bagging_ensemble(
    base_model=DecisionTreeClassifier(max_depth=20, random_state=42),
    n_estimators=100,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

### Feature Selection

```python
from src.ml.model_optimization import FeatureSelector

selector = FeatureSelector()

# RFE (Recursive Feature Elimination)
result = selector.rfe_selection(
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    n_features_to_select=20,
    step=1
)

print(f"Selected {len(result.selected_features)} features")
print(f"Top 5: {result.selected_features[:5]}")

# LASSO Feature Selection
result = selector.lasso_selection(
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    alpha=0.001
)

# Correlation-Based Selection
result = selector.correlation_selection(
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    threshold=0.9  # Remove features with >0.9 correlation
)

# Combined selection (intersection of methods)
rfe_set = set(rfe_result.selected_features)
lasso_set = set(lasso_result.selected_features)
corr_set = set(corr_result.selected_features)
combined = list(rfe_set & lasso_set & corr_set)
```

### Model Comparison

```python
from src.ml.model_registry import ModelComparison, ModelMetadata
from datetime import datetime

comparison = ModelComparison()

# Add models to compare
models = {
    'random_forest': best_rf_model,
    'gradient_boosting': best_gb_model,
    'voting_ensemble': voting_model,
}

for name, model in models.items():
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
        tags=['production'],
        description=f"{name} for fraud detection",
        author="ML Team"
    )
    
    comparison.add_model(
        model_name=name,
        model=model,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata
    )

# Compare with business priorities
result = comparison.compare(
    primary_metric='f1',
    business_priority='balanced',  # or 'recall_focused', 'precision_focused'
    min_recall=0.70,  # Minimum 70% recall
    max_fpr=0.10      # Max 10% false positive rate
)

print(f"Best model: {result.best_model}")
print(f"F1: {result.best_metrics['f1']:.4f}")
print(f"Precision: {result.best_metrics['precision']:.4f}")
print(f"Recall: {result.best_metrics['recall']:.4f}")

# Business recommendations
for i, rec in enumerate(result.recommendations, 1):
    print(f"{i}. {rec}")

# Export report
comparison.export_comparison_report('output/model_comparison.txt')
```

### Model Registry (Versioning & Persistence)

```python
from src.ml.model_registry import ModelRegistry, ModelMetadata
from datetime import datetime

registry = ModelRegistry(base_dir='models')

# Register model
metadata = ModelMetadata(
    model_id="fraud_detector_v1",
    model_name="Random Forest Fraud Detector",
    model_type="random_forest",
    version="1.0.0",
    created_at=datetime.now(),
    hyperparameters={'n_estimators': 200, 'max_depth': 20},
    feature_names=feature_names,
    training_samples=len(X_train),
    training_duration_seconds=45.2,
    metrics={'f1': 0.8234, 'precision': 0.7891, 'recall': 0.8623},
    tags=['production', 'fraud_detection', 'v1'],
    description="Production model optimized for Indian market",
    author="SynFinance ML Team"
)

model_path = registry.register_model(
    model=best_model,
    model_name="fraud_detector_prod",
    metadata=metadata,
    overwrite=True
)

print(f"Model saved: {model_path}")

# List all models
all_models = registry.list_models()
for name in all_models:
    meta = registry.get_metadata(name)
    print(f"- {name} (v{meta.version}, F1={meta.metrics.get('f1', 0):.4f})")

# Filter by tag
prod_models = registry.list_models(tag='production')

# Load model for inference
loaded_model, loaded_metadata = registry.load_model("fraud_detector_prod")
predictions = loaded_model.predict(X_test[:10])

# Export registry report
registry.export_registry_report('output/registry_report.txt')

# Delete old model
registry.delete_model("old_model_name")
```

---

## Troubleshooting

**Import Errors:**
```bash
pip install -r requirements.txt
```

**Test Failures:**
```bash
# Ensure virtual environment is activated
pytest tests/ -v --tb=short
```

**Slow Generation:**
```python
# Use batch generation instead of loops
df = generate_realistic_dataset(...)  # Fast
```

---

**For detailed documentation, see [INDEX.md](../INDEX.md)**
