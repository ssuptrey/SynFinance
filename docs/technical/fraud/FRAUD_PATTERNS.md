# Fraud Pattern Library

**Version:** 0.5.0  
**Week 4, Days 3-4: Advanced Fraud Patterns & Combination System**  
**Author:** SynFinance Team  
**Date:** October 26, 2025

## Overview

The SynFinance Fraud Pattern Library provides a comprehensive system for injecting realistic fraud patterns into synthetic transaction data. This enables machine learning teams to train and test fraud detection models with labeled, diverse, and realistic fraud scenarios.

**Key Features:**
- **15 sophisticated fraud pattern types** (10 base + 5 advanced)
- **Fraud combination system** (chained, coordinated, progressive)
- **Network analysis module** (fraud ring detection, temporal clustering)
- **Cross-pattern statistics** (co-occurrence matrix, isolation tracking)
- Configurable fraud injection rates (0.5-2%)
- Confidence scoring system (0.0-1.0)
- Severity classification (low/medium/high/critical)
- Detailed evidence tracking (JSON serialized)
- History-aware fraud application
- Real-time statistics tracking

## Architecture

### Core Components

```python
from src.generators.fraud_patterns import (
    FraudType,              # Enum of 15 fraud types
    FraudIndicator,         # Fraud metadata container
    FraudPattern,           # Base class for patterns
    FraudPatternGenerator,  # Orchestration system
    FraudCombinationGenerator,  # Combination system (NEW)
    apply_fraud_labels,     # Add fraud fields to transactions
    inject_fraud_into_dataset  # Batch processing utility
)
from src.generators.fraud_network import (
    FraudNetworkAnalyzer,   # Network analysis (NEW)
    FraudRing,              # Fraud ring structure (NEW)
    TemporalCluster,        # Temporal clustering (NEW)
)
```

### FraudPattern Base Class

All fraud patterns inherit from the `FraudPattern` base class:

```python
class FraudPattern:
    def should_apply(self, customer, transaction, customer_history) -> bool:
        """Determine if this pattern is applicable to the transaction"""
        
    def apply_pattern(self, transaction, customer, history) -> Tuple[Dict, FraudIndicator]:
        """Apply fraud characteristics and return modified transaction + indicator"""
        
    def calculate_confidence(self, evidence: Dict) -> float:
        """Calculate confidence score (0.0-1.0) based on evidence"""
```

### FraudIndicator Structure

Each fraud instance is described by a `FraudIndicator`:

```python
@dataclass
class FraudIndicator:
    fraud_type: FraudType          # Type of fraud pattern
    confidence: float              # 0.0-1.0 confidence score
    reason: str                    # Human-readable explanation
    evidence: Dict[str, Any]       # Supporting evidence
    severity: str                  # low/medium/high/critical
```

## Fraud Pattern Types

### 1. Card Cloning

**Description:** Detects impossible travel scenarios where a card is used in geographically distant locations within an impossibly short timeframe.

**Characteristics:**
- Travel speed >800 km/h (impossible for legitimate travel)
- Round amounts (Rs.9,999, Rs.19,999, Rs.49,999)
- Cash withdrawals in unusual locations
- Distance/time violation detection

**Detection Logic:**
```python
# Calculate travel speed
distance = calculate_distance(last_city, current_city)
time_diff_hours = calculate_time_difference(last_txn, current_txn)
speed_kmh = distance / time_diff_hours

# Impossible if >800 km/h
if speed_kmh > 800:
    fraud_detected = True
```

**Confidence Calculation:**
- Speed >2000 km/h: +0.4 confidence
- Speed >800 km/h: +0.3 confidence
- Round amount: +0.2 confidence
- Cash withdrawal: +0.1 confidence

**Example:**
```
Transaction 1: Mumbai, 10:00 AM, Rs.5,000
Transaction 2: Delhi, 10:30 AM, Rs.9,999 (impossible - 1,400 km in 30 min = 2,800 km/h)
Confidence: 0.95, Severity: critical
```

**Evidence Fields:**
- `distance_km`: Distance between cities
- `time_diff_hours`: Time between transactions
- `travel_speed_kmh`: Calculated travel speed
- `impossible_travel`: Boolean flag
- `amount_pattern`: "round" or "normal"

---

### 2. Account Takeover

**Description:** Detects sudden behavioral changes indicating an account has been compromised by an unauthorized party.

**Characteristics:**
- 3-10x spending spike compared to baseline
- Unusual transaction category
- Unusual transaction time (2-5 AM)
- Location change
- High-value items (Electronics, Jewelry)

**Detection Logic:**
```python
# Calculate baseline spending
baseline = mean(customer_history[-20:]['Amount'])

# Detect spike
multiplier = transaction_amount / baseline

# Account takeover if 3x+ spike
if multiplier >= 3.0:
    fraud_detected = True
```

**Confidence Calculation:**
- 10x multiplier: +0.3 confidence
- 5x multiplier: +0.2 confidence
- Unusual hour (2-5 AM): +0.15 confidence
- Unusual category: +0.15 confidence
- Location change: +0.1 confidence

**Example:**
```
Baseline spending: Rs.3,000 per transaction
Suspicious transaction: Rs.30,000 Electronics at 2:30 AM in different city
Confidence: 0.85, Severity: high
```

**Evidence Fields:**
- `baseline_amount`: Customer's typical spending
- `amount_multiplier`: Spike multiplier (e.g., 10x)
- `unusual_hour`: Boolean flag
- `unusual_category`: Boolean flag
- `location_change`: Boolean flag

---

### 3. Merchant Collusion

**Description:** Detects suspicious merchant patterns indicating collusion between merchant and fraudster.

**Characteristics:**
- Round amounts just below reporting thresholds (Rs.49,999, Rs.99,999)
- New merchants (<2 years operating)
- Low-rated merchants (<3.0 rating)
- Repeated high-value transactions at same merchant

**Detection Logic:**
```python
suspicious_amounts = [9999, 19999, 49999, 99999, 199999]

# Check if amount is suspiciously round
for threshold in [10000, 20000, 50000, 100000, 200000]:
    margin = threshold - transaction_amount
    if 0 < margin < 1000:  # Just below threshold
        just_below_threshold = True
```

**Confidence Calculation:**
- Just below threshold (<Rs.100 margin): +0.3 confidence
- Structuring pattern (multiple similar amounts): +0.3 confidence
- New merchant (<1 year): +0.2 confidence
- Low rating (<2.0): +0.2 confidence

**Example:**
```
Transaction: Rs.49,999 at "New Electronics Store" (6 months old, 1.8 rating)
Previous transaction: Rs.49,950 at same merchant last week
Confidence: 0.90, Severity: high
```

**Evidence Fields:**
- `round_amount`: Boolean flag
- `just_below_threshold`: Boolean flag
- `margin_below_threshold`: Margin amount (e.g., Rs.1)
- `new_merchant`: Boolean flag
- `low_reputation`: Boolean flag
- `merchant_age_years`: Merchant age
- `merchant_rating`: Merchant rating

---

### 4. Velocity Abuse

**Description:** Detects abnormally high transaction frequency indicating card testing or automated fraud.

**Characteristics:**
- 5+ transactions within 1 hour
- Small test amounts (Rs.100-500)
- Multiple merchants
- Multiple payment modes

**Detection Logic:**
```python
# Count transactions in last hour
current_time = transaction_timestamp
one_hour_ago = current_time - timedelta(hours=1)

recent_txns = [txn for txn in history 
               if one_hour_ago <= txn.timestamp <= current_time]

if len(recent_txns) >= 5:
    velocity_abuse_detected = True
```

**Confidence Calculation:**
- 10+ transactions/hour: +0.4 confidence
- 7+ transactions/hour: +0.3 confidence
- 5+ transactions/hour: +0.2 confidence
- Small amounts (<Rs.500): +0.15 confidence
- Multiple merchants: +0.1 confidence

**Example:**
```
12:00 PM - 12:15 PM: 10 transactions
Amounts: Rs.100, Rs.150, Rs.200, Rs.100, Rs.250, Rs.150, Rs.300, Rs.100, Rs.200, Rs.150
Different merchants: 8 different stores
Confidence: 0.92, Severity: high
```

**Evidence Fields:**
- `transactions_in_hour`: Count (e.g., 10)
- `transaction_frequency`: Per-hour rate
- `small_amounts`: Boolean flag
- `multiple_merchants`: Count of unique merchants

---

### 5. Amount Manipulation (Structuring)

**Description:** Detects attempts to avoid reporting requirements by keeping transactions just below regulatory thresholds.

**Characteristics:**
- Amounts just below Rs.10K, Rs.20K, Rs.50K, Rs.100K, Rs.200K
- Margin <Rs.1,000 from threshold
- Multiple similar amounts within short period
- Pattern of threshold avoidance

**Detection Logic:**
```python
thresholds = [10000, 20000, 50000, 100000, 200000]

for threshold in thresholds:
    margin = threshold - amount
    if 0 < margin < 1000:  # Within Rs.1000 of threshold
        # Check for structuring pattern
        similar_amounts = count_similar_in_history(amount, margin=500)
        if similar_amounts >= 2:
            structuring_detected = True
```

**Confidence Calculation:**
- Margin <Rs.100: +0.3 confidence
- Margin <Rs.500: +0.2 confidence
- Structuring pattern (3+ similar): +0.3 confidence
- Short time period (<7 days): +0.2 confidence

**Example:**
```
Day 1: Rs.49,999
Day 3: Rs.49,950
Day 5: Rs.49,980
Pattern: Consistently avoiding Rs.50,000 threshold
Confidence: 0.88, Severity: high
```

**Evidence Fields:**
- `just_below_threshold`: Boolean flag
- `threshold_value`: Threshold being avoided (e.g., Rs.50,000)
- `margin_below_threshold`: Margin amount
- `similar_transaction_count`: Count of similar amounts
- `structuring_pattern`: Boolean flag

---

### 6. Refund Fraud

**Description:** Detects abuse of refund policies through excessive return rates.

**Characteristics:**
- Refund rate >6% (normal is ~2%)
- 3x+ higher than normal refund rate
- Online purchases (easier to return)
- High-value categories (Electronics, Fashion)

**Detection Logic:**
```python
# Calculate historical refund rate
total_transactions = len(customer_history)
refund_transactions = sum(1 for txn in customer_history 
                         if txn.get('Refunded', False))

refund_rate = refund_transactions / total_transactions
normal_rate = 0.02  # 2% baseline

if refund_rate > normal_rate * 3:  # 3x normal
    refund_fraud_detected = True
```

**Confidence Calculation:**
- 5x normal rate: +0.4 confidence
- 3x normal rate: +0.3 confidence
- Online channel: +0.15 confidence
- High-value category: +0.15 confidence

**Example:**
```
Customer history: 50 transactions, 10 refunds (20% rate)
Normal rate: 2%
Multiplier: 10x normal
Recent: Rs.25,000 Electronics (Online)
Confidence: 0.85, Severity: medium
```

**Evidence Fields:**
- `refund_rate`: Customer's refund rate (e.g., 0.20)
- `normal_refund_rate`: Baseline rate (e.g., 0.02)
- `refund_multiplier`: Rate multiplier (e.g., 10x)
- `online_purchase`: Boolean flag
- `high_value_category`: Boolean flag

---

### 7. Stolen Card

**Description:** Detects use of stolen cards after period of inactivity.

**Characteristics:**
- 3+ days of card inactivity
- Sudden high-value transaction
- Cash equivalent purchases (Gift cards, Jewelry)
- Different city/location
- 5-10x spending spike

**Detection Logic:**
```python
# Calculate inactivity period
last_transaction_date = max(history, key=lambda x: x['Date'])
current_date = transaction['Date']
inactivity_days = (current_date - last_transaction_date).days

# Stolen card indicators
if inactivity_days >= 3:
    last_amount = history[-1]['Amount']
    spike_multiplier = current_amount / last_amount
    
    if spike_multiplier >= 5.0:
        stolen_card_detected = True
```

**Confidence Calculation:**
- 7+ days inactive: +0.3 confidence
- 5+ days inactive: +0.2 confidence
- 10x amount spike: +0.2 confidence
- Cash equivalent: +0.2 confidence
- Location change: +0.1 confidence

**Example:**
```
Last transaction: October 10, Rs.2,000 (Mumbai)
Card inactive: 7 days
Current transaction: October 17, Rs.40,000 Jewelry (Delhi)
Confidence: 0.90, Severity: high
```

**Evidence Fields:**
- `inactivity_days`: Days inactive (e.g., 7)
- `sudden_large_purchase`: Boolean flag
- `amount_spike_multiplier`: Spike multiplier
- `cash_equivalent`: Boolean flag
- `location_change`: Boolean flag

---

### 8. Synthetic Identity

**Description:** Detects fraudulent identities created by combining real and fake information.

**Characteristics:**
- Limited transaction history (<15 transactions)
- Consistent upward spending growth
- New credit lines being established
- Low merchant loyalty (trying different merchants)
- Gradual trust building pattern

**Detection Logic:**
```python
# Check history length
if len(customer_history) < 15:
    # Check for growth pattern
    amounts = [txn['Amount'] for txn in customer_history]
    
    # Calculate growth rate
    growth_rates = []
    for i in range(1, len(amounts)):
        growth = (amounts[i] - amounts[i-1]) / amounts[i-1]
        growth_rates.append(growth)
    
    avg_growth = mean(growth_rates)
    
    # Synthetic if consistent 15%+ growth
    if avg_growth > 0.15 and std(growth_rates) < 0.1:
        synthetic_identity_detected = True
```

**Confidence Calculation:**
- <5 transactions: +0.3 confidence
- <10 transactions: +0.2 confidence
- Consistent growth (15%+ per txn): +0.25 confidence
- Low variance in growth: +0.15 confidence
- New customer (<30 days): +0.1 confidence

**Example:**
```
Transaction history (5 transactions):
1. Rs.1,000
2. Rs.1,200 (+20%)
3. Rs.1,440 (+20%)
4. Rs.1,728 (+20%)
5. Rs.2,074 (+20%)
Pattern: Suspiciously consistent 20% growth
Confidence: 0.78, Severity: medium
```

**Evidence Fields:**
- `limited_history`: Boolean flag
- `transaction_count`: Number of transactions
- `consistent_growth`: Boolean flag
- `average_growth_rate`: Growth rate (e.g., 0.20)
- `growth_variance`: Variance in growth

---

### 9. First Party Fraud (Bust-Out)

**Description:** Detects legitimate customers who deliberately max out credit and disappear.

**Characteristics:**
- Established history (20+ transactions)
- Sudden 5-15x spending spike
- Large purchases after building trust
- High-value categories
- Multiple channels used

**Detection Logic:**
```python
# Check for established history
if len(customer_history) >= 20:
    # Calculate baseline
    baseline = mean([txn['Amount'] for txn in customer_history[:-5]])
    
    # Check for bust-out spike
    recent_amount = transaction['Amount']
    multiplier = recent_amount / baseline
    
    if multiplier >= 5.0:
        bust_out_detected = True
```

**Confidence Calculation:**
- 10x multiplier: +0.3 confidence
- 5x multiplier: +0.2 confidence
- Established history (30+ txn): +0.1 confidence
- High-value purchase: +0.2 confidence
- Multiple channels: +0.1 confidence

**Example:**
```
Customer history: 25 transactions, average Rs.4,000
Established pattern: Groceries, Shopping, Food & Dining
Bust-out transaction: Rs.60,000 Electronics + Rs.40,000 Jewelry (same day)
Confidence: 0.82, Severity: high
```

**Evidence Fields:**
- `established_history`: Boolean flag
- `baseline_amount`: Historical average
- `sudden_large_purchase`: Boolean flag
- `amount_multiplier`: Spike multiplier
- `bust_out_pattern`: Boolean flag

---

### 10. Friendly Fraud (Chargeback Abuse)

**Description:** Detects customers who make legitimate purchases but falsely claim fraud to get refunds.

**Characteristics:**
- Elevated dispute rate (>3% vs 1% normal)
- Online purchases (harder to prove delivery)
- High-value items
- Chargeback-prone categories (Electronics, Fashion)
- Pattern of disputes

**Detection Logic:**
```python
# Calculate dispute rate
total_transactions = len(customer_history)
disputed_transactions = sum(1 for txn in customer_history 
                           if txn.get('Disputed', False))

dispute_rate = disputed_transactions / total_transactions
normal_rate = 0.01  # 1% baseline

if dispute_rate > normal_rate * 3:  # 3x normal
    friendly_fraud_detected = True
```

**Confidence Calculation:**
- 5x normal dispute rate: +0.4 confidence
- 3x normal dispute rate: +0.3 confidence
- Online channel: +0.15 confidence
- Chargeback-prone category: +0.15 confidence

**Example:**
```
Customer history: 40 transactions, 6 disputes (15% rate)
Normal rate: 1%
Multiplier: 15x normal
Recent: Rs.35,000 Electronics (Online)
Confidence: 0.88, Severity: medium
```

**Evidence Fields:**
- `dispute_rate`: Customer's dispute rate (e.g., 0.15)
- `normal_dispute_rate`: Baseline rate (e.g., 0.01)
- `dispute_multiplier`: Rate multiplier (e.g., 15x)
- `online_purchase`: Boolean flag
- `chargeback_prone_category`: Boolean flag

---

## Usage Guide

### Basic Usage

```python
from src.generators.fraud_patterns import FraudPatternGenerator
from src.customer_generator import CustomerGenerator

# Initialize fraud generator
fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)

# Generate customers
customer_gen = CustomerGenerator(seed=42)
customer = customer_gen.generate_customer()

# Create transaction
transaction = {
    'Transaction_ID': 'TXN_001',
    'Customer_ID': customer.customer_id,
    'Amount': 5000.0,
    'Category': 'Electronics',
    # ... other fields
}

# Maybe apply fraud
customer_history = []  # List of previous transactions
modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(
    transaction, customer, customer_history
)

# Check if fraud was applied
if fraud_indicator:
    print(f"Fraud Type: {fraud_indicator.fraud_type.value}")
    print(f"Confidence: {fraud_indicator.confidence}")
    print(f"Reason: {fraud_indicator.reason}")
    print(f"Severity: {fraud_indicator.severity}")
```

### Batch Processing

```python
from src.generators.fraud_patterns import inject_fraud_into_dataset
from src.customer_generator import CustomerGenerator
from src.data_generator import TransactionGenerator

# Generate dataset
customer_gen = CustomerGenerator(seed=42)
customers = [customer_gen.generate_customer() for _ in range(1000)]

txn_gen = TransactionGenerator(customers, seed=42)
transactions = txn_gen.generate_transactions(num_transactions=10000)

# Inject fraud at 2% rate
modified_transactions, stats = inject_fraud_into_dataset(
    transactions=transactions,
    customers=customers,
    fraud_rate=0.02,
    seed=42
)

# View statistics
print(f"Total Transactions: {stats['total_transactions']}")
print(f"Total Fraud: {stats['total_fraud']}")
print(f"Actual Fraud Rate: {stats['fraud_rate']:.2%}")
print(f"Fraud by Type: {stats['fraud_by_type']}")
```

### Fraud Field Schema

After fraud injection, each transaction has 5 additional fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Fraud_Type` | string | Fraud pattern name | "Card Cloning" |
| `Fraud_Confidence` | float | Confidence score (0.0-1.0) | 0.85 |
| `Fraud_Reason` | string | Detailed explanation | "Impossible travel: Mumbai to Delhi in 30 minutes" |
| `Fraud_Severity` | string | Severity level | "high" |
| `Fraud_Evidence` | JSON string | Evidence dictionary | `{"distance_km": 1400, "speed_kmh": 2800}` |

### Configuring Fraud Rates

```python
# Low fraud rate (0.5%) - Realistic production scenario
fraud_gen = FraudPatternGenerator(fraud_rate=0.005)

# Medium fraud rate (2%) - Balanced ML training
fraud_gen = FraudPatternGenerator(fraud_rate=0.02)

# High fraud rate (5%) - Fraud-focused analysis
fraud_gen = FraudPatternGenerator(fraud_rate=0.05)

# Runtime adjustment
fraud_gen.set_fraud_rate(0.03)  # Change to 3%
```

### Statistics Tracking

```python
# Generate transactions with fraud
for txn in transactions:
    modified_txn, fraud_info = fraud_gen.maybe_apply_fraud(txn, customer, history)
    history.append(modified_txn)

# Get statistics
stats = fraud_gen.get_fraud_statistics()

print(f"Total Transactions: {stats['total_transactions']}")
print(f"Total Fraud: {stats['total_fraud']}")
print(f"Target Rate: {stats['target_fraud_rate']:.1%}")
print(f"Actual Rate: {stats['fraud_rate']:.1%}")

# Fraud distribution by type
for fraud_type, count in stats['fraud_by_type'].items():
    percentage = stats['fraud_type_distribution'][fraud_type]
    print(f"{fraud_type}: {count} ({percentage:.1%})")

# Reset statistics
fraud_gen.reset_statistics()
```

## ML Training Integration

### Feature Engineering

The fraud fields can be used for various ML tasks:

**Binary Classification:**
```python
# Target variable
y = df['Is_Fraud']  # 0 or 1

# Features
X = df.drop(['Fraud_Type', 'Fraud_Confidence', 'Fraud_Reason', 
             'Fraud_Severity', 'Fraud_Evidence', 'Is_Fraud'], axis=1)
```

**Multi-class Classification:**
```python
# Target variable (fraud type)
y = df['Fraud_Type']  # 11 classes (10 fraud types + "None")

# Features
X = df.drop(['Fraud_Type', 'Fraud_Confidence', 'Fraud_Reason', 
             'Fraud_Severity', 'Fraud_Evidence'], axis=1)
```

**Confidence Score Prediction:**
```python
# Target variable (confidence score)
y = df['Fraud_Confidence']  # 0.0-1.0

# Only on fraud transactions
fraud_df = df[df['Is_Fraud'] == 1]
```

### Stratified Sampling

```python
from sklearn.model_selection import train_test_split

# Ensure fraud is represented in train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Option 1: SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Option 2: Under-sampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Option 3: Use fraud_rate parameter to generate balanced dataset
fraud_gen = FraudPatternGenerator(fraud_rate=0.50)  # 50% fraud rate
```

## Performance Characteristics

### Fraud Rate Accuracy

The system maintains accurate fraud rates within ±0.5% of target:

| Target Rate | Actual Rate (10K txns) | Variance |
|-------------|------------------------|----------|
| 0.5% | 0.45% - 0.55% | ±10% |
| 1.0% | 0.9% - 1.1% | ±10% |
| 2.0% | 1.8% - 2.2% | ±10% |
| 5.0% | 4.7% - 5.3% | ±6% |

### Pattern Distribution

Fraud patterns are distributed based on applicability:

- **High Frequency** (30-40%): Velocity Abuse, Amount Manipulation
- **Medium Frequency** (15-25%): Card Cloning, Account Takeover, Merchant Collusion
- **Low Frequency** (5-15%): Refund Fraud, Stolen Card, Synthetic Identity, First Party, Friendly Fraud

**Note:** Distribution varies based on customer segments and transaction patterns.

### Processing Performance

- **Single Transaction:** ~0.5ms overhead
- **Batch Processing (10K txns):** ~5 seconds
- **Memory Overhead:** ~50KB per 1,000 transactions
- **History Tracking:** O(n) time complexity

## Best Practices

### 1. Use Realistic Fraud Rates

```python
# Realistic production scenario (0.5-2%)
fraud_gen = FraudPatternGenerator(fraud_rate=0.01)  # 1%

# Avoid unrealistic rates
fraud_gen = FraudPatternGenerator(fraud_rate=0.50)  # Too high!
```

### 2. Maintain Customer History

```python
# Build accurate history for pattern detection
customer_history = {}
for customer in customers:
    customer_history[customer.customer_id] = []

for txn in transactions:
    history = customer_history[txn['Customer_ID']]
    modified_txn, fraud_info = fraud_gen.maybe_apply_fraud(txn, customer, history)
    customer_history[txn['Customer_ID']].append(modified_txn)
```

### 3. Use Consistent Seeds

```python
# Reproducible fraud injection
fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)

# Same results every time
for _ in range(5):
    modified_txns, stats = inject_fraud_into_dataset(
        transactions, customers, fraud_rate=0.02, seed=42
    )
    # stats will be identical
```

### 4. Validate Fraud Distribution

```python
stats = fraud_gen.get_fraud_statistics()

# Check if all patterns are represented
for fraud_type in FraudType:
    count = stats['fraud_by_type'].get(fraud_type.value, 0)
    if count == 0:
        print(f"Warning: No {fraud_type.value} detected")
```

### 5. Monitor Confidence Scores

```python
# Analyze confidence distribution
confidences = [txn['Fraud_Confidence'] for txn in modified_txns 
               if txn['Is_Fraud'] == 1]

print(f"Mean Confidence: {mean(confidences):.2f}")
print(f"Min Confidence: {min(confidences):.2f}")
print(f"Max Confidence: {max(confidences):.2f}")

# Ensure diverse confidence scores
assert min(confidences) > 0.0
assert max(confidences) <= 1.0
```

## Testing

Comprehensive tests are available in `tests/test_fraud_patterns.py`:

```bash
# Run all fraud pattern tests
pytest tests/test_fraud_patterns.py -v

# Run specific pattern test
pytest tests/test_fraud_patterns.py::TestCardCloningPattern -v

# Run with coverage
pytest tests/test_fraud_patterns.py --cov=src.generators.fraud_patterns
```

**Test Coverage:**
- 26 comprehensive tests
- All 10 fraud patterns validated
- Injection system tested
- Fraud labeling verified
- Statistics tracking validated

## Troubleshooting

### Issue: No Fraud Detected

**Cause:** Fraud rate too low or patterns not applicable

**Solution:**
```python
# Increase fraud rate temporarily
fraud_gen = FraudPatternGenerator(fraud_rate=0.10)  # 10% for testing

# Or use batch processing with guaranteed fraud
modified_txns, stats = inject_fraud_into_dataset(
    transactions, customers, fraud_rate=0.05, seed=42
)
print(f"Fraud Count: {stats['total_fraud']}")
```

### Issue: Unbalanced Pattern Distribution

**Cause:** Customer profiles don't match pattern requirements

**Solution:**
```python
# Check pattern applicability
for fraud_type, pattern in fraud_gen.patterns.items():
    applicable = pattern.should_apply(customer, transaction, history)
    print(f"{fraud_type.value}: {applicable}")

# Ensure diverse customer segments
customers = [customer_gen.generate_customer() for _ in range(1000)]
segments = [c.segment.value for c in customers]
print(f"Segment diversity: {len(set(segments))}")
```

### Issue: Confidence Scores Too High/Low

**Cause:** Evidence weights need adjustment

**Solution:**
```python
# Check evidence for pattern
if fraud_indicator:
    print(f"Evidence: {fraud_indicator.evidence}")
    print(f"Confidence: {fraud_indicator.confidence}")
    
# Confidence calculation is pattern-specific
# Review calculate_confidence() method in pattern class
```

---

## Advanced Fraud Patterns (Week 4 Days 3-4)

### 11. Transaction Replay

**Description:** Detects duplicate transaction attacks where fraudsters replay previously captured transaction details to bypass security systems.

**Characteristics:**
- Similar transactions to same merchant within 2-hour window
- Exact or near-exact amount matches
- Device type changes between replays
- Location consistency checks

**Detection Logic:**
```python
# Count similar transactions in recent history
similar_count = 0
for hist_txn in recent_history:
    if (same_merchant and similar_amount and within_2_hours):
        similar_count += 1

# Device change increases suspicion
if current_device != hist_device:
    confidence += 0.2
```

**Confidence Calculation:**
- 3+ similar transactions: +0.4 confidence
- Exact amount match: +0.3 confidence
- Device changed: +0.2 confidence
- 2-hour window violation: +0.1 confidence

**Example:**
```
Transaction 1: MRCH-001, Rs.1,000, 12:00 PM, Mobile
Transaction 2: MRCH-001, Rs.1,000, 12:30 PM, Web (replay detected)
Confidence: 0.85, Severity: medium
```

---

### 12. Card Testing

**Description:** Detects small test transactions fraudsters make to validate stolen card details before attempting larger fraudulent purchases.

**Characteristics:**
- Small amounts (<Rs.100) relative to customer average
- Multiple small transactions in rapid succession
- Often precede large fraudulent transactions
- Online channel preference

**Detection Logic:**
```python
# Compare to customer average
customer_avg = calculate_average(customer_history)
amount_ratio = current_amount / customer_avg

# Small relative amount indicates testing
if amount_ratio < 0.05:  # Less than 5% of average
    confidence = 0.3 + (1 - amount_ratio) * 0.7
```

**Confidence Calculation:**
- Amount <2% of customer avg: +0.5 confidence
- Multiple small transactions: +0.2 confidence
- Online channel: +0.1 confidence
- Rapid succession (<15 min): +0.2 confidence

**Example:**
```
Customer average: Rs.2,500
Test transaction: Rs.50 (2% of average)
Confidence: 0.75, Severity: low
```

---

### 13. Mule Account

**Description:** Detects money laundering patterns where accounts are used to quickly move illicit funds through the financial system.

**Characteristics:**
- High turnover ratio (90%+ of funds transferred out)
- Rapid fund movement (many transfers in short time)
- Round transfer amounts
- Young account age with high activity

**Detection Logic:**
```python
# Calculate turnover ratio
total_in = sum(transfer_in amounts)
total_out = sum(transfer_out amounts)
turnover_ratio = total_out / total_in if total_in > 0 else 0

# High turnover indicates mule activity
if turnover_ratio > 0.9:
    confidence = 0.5 + (turnover_ratio - 0.9) * 5  # Up to 1.0
```

**Confidence Calculation:**
- Turnover ratio >95%: +0.6 confidence
- 8+ transfers in 24 hours: +0.2 confidence
- Round amounts (multiples of 5000): +0.1 confidence
- Account age <30 days: +0.1 confidence

**Example:**
```
Incoming: Rs.50,000
Outgoing: Rs.48,000 (96% turnover)
8 transfers in 6 hours
Confidence: 0.95, Severity: high
```

---

### 14. Shipping Fraud

**Description:** Detects address manipulation and shipping-related fraud where goods are diverted to unauthorized locations.

**Characteristics:**
- Shipping address different from customer home city
- High-value electronics or luxury items
- Rush/overnight shipping requests
- Late-night or weekend orders

**Detection Logic:**
```python
# Check address change
address_changed = (shipping_city != customer_home_city)

# High value items are higher risk
high_value = (amount > 10000 and category in ['Electronics', 'Jewelry'])

# Rush shipping indicator
rush_shipping = (hour > 20 or hour < 6 or is_weekend)

confidence = 0.25 + (0.25 if address_changed) + 
             (0.3 if high_value) + (0.2 if rush_shipping)
```

**Confidence Calculation:**
- Address change: +0.25 confidence
- High value item (>Rs.10,000): +0.30 confidence
- Rush shipping: +0.20 confidence
- Non-home location: +0.25 confidence

**Example:**
```
Customer city: Mumbai
Shipping to: Delhi
Item: Electronics, Rs.25,000
Time: 11:00 PM
Confidence: 0.90, Severity: high
```

---

### 15. Loyalty Program Abuse

**Description:** Detects systematic exploitation of loyalty/rewards programs through threshold optimization and category manipulation.

**Characteristics:**
- Transaction amounts just below reward thresholds (Rs.999, Rs.1,999, Rs.4,999)
- High concentration in loyalty-earning categories
- Frequent threshold-optimized transactions
- Pattern of just-below-limit spending

**Detection Logic:**
```python
# Check proximity to known thresholds
thresholds = [999, 1999, 4999, 9999, 49999]
min_distance = min(abs(amount - t) for t in thresholds)

# Within 5% of threshold indicates optimization
if min_distance / nearest_threshold < 0.05:
    points_optimization = True
    confidence = 0.3 + (1 - min_distance/50) * 0.5
```

**Confidence Calculation:**
- Within 1% of threshold: +0.5 confidence
- Multiple optimized transactions: +0.2 confidence
- High loyalty category ratio: +0.2 confidence
- Frequent pattern (5+ in month): +0.1 confidence

**Example:**
```
Threshold: Rs.2,000 (for bonus points)
Transaction: Rs.1,999 (99.95% of threshold)
Category: Shopping (high loyalty rewards)
History: 4 similar transactions this month
Confidence: 0.80, Severity: low
```

---

## Fraud Combination System

**NEW in 0.5.0:** Advanced fraud scenarios often involve multiple patterns working together. The `FraudCombinationGenerator` provides three combination modes:

### Chained Fraud

Sequential patterns where one fraud enables another:

```python
# Example: Account takeover → Velocity abuse
combiner = FraudCombinationGenerator(seed=42)
modified_txn, indicator = combiner.apply_chained(
    transaction,
    [AccountTakeoverPattern(), VelocityAbusePattern()],
    customer,
    history
)

# Chained fraud gets 10% confidence boost
# Evidence includes: chain_sequence, chain_length
```

### Coordinated Fraud

Multi-actor fraud rings with shared resources:

```python
# Example: Merchant collusion ring
modified_txn, indicator = combiner.apply_coordinated(
    transaction,
    [MerchantCollusionPattern(), CardCloningPattern()],
    customer,
    history,
    coordination_metadata={'shared_merchants': ['M1', 'M2'], 'fraud_ring_id': 'RING-001'}
)

# Coordinated fraud has elevated severity
# Evidence includes: coordination_actors, shared_merchants
```

### Progressive Fraud

Escalating sophistication over time:

```python
# Example: Increasing sophistication (0.0-1.0 scale)
modified_txn, indicator = combiner.apply_progressive(
    transaction,
    [CardTestingPattern(), TransactionReplayPattern(), AccountTakeoverPattern()],
    customer,
    history,
    sophistication_level=0.7  # Advanced stage
)

# Sophistication scales confidence: base * (0.7 + 0.3 * level)
# Evidence includes: sophistication_level, progression_stage
```

---

## Fraud Network Analysis

**NEW in 0.5.0:** Detect fraud rings and coordinated attacks using the `FraudNetworkAnalyzer`:

### Merchant Networks

```python
analyzer = FraudNetworkAnalyzer(seed=42)
rings = analyzer.analyze_merchant_networks(
    transactions,
    customer_map,
    min_customers=3,
    min_transactions=5
)

# Detects: Multiple customers using same suspicious merchant
# Confidence: 0.3 + (customer_count * 0.1)
```

### Location Networks

```python
rings = analyzer.analyze_location_networks(
    transactions,
    customer_map,
    min_customers=4,
    suspicious_locations=['High_Crime_Zone', 'Border_Area']
)

# Detects: Fraud clusters in suspicious geographic areas
# Confidence: 0.4 + (0.3 if suspicious_area else 0.1)
```

### Temporal Clustering

```python
clusters = analyzer.detect_temporal_clusters(
    transactions,
    time_window_minutes=30,
    min_transactions=5
)

# Detects: Coordinated attacks within time windows
# Suspicious if: 3+ customers, same merchant, <30 min
```

### Network Visualization

```python
graph = analyzer.generate_network_graph()

# Returns: {'nodes': [...], 'edges': [...], 'metadata': {...}}
# Nodes: Fraud rings, customers, clusters
# Edges: Membership, participation relationships
```

---

## Cross-Pattern Statistics

**NEW in 0.5.0:** Track pattern interactions and ensure isolation:

### Co-Occurrence Matrix

```python
generator = FraudPatternGenerator(fraud_rate=0.02, seed=42)

# After processing transactions
matrix = generator.get_pattern_co_occurrence_matrix()

# Shows which patterns appear together
# Example: {'Card Cloning': {'Account Takeover': 5, ...}, ...}
```

### Isolation Statistics

```python
isolation_stats = generator.get_pattern_isolation_stats()

# For each pattern:
# - total_occurrences
# - isolated_occurrences (appeared alone)
# - combined_occurrences (appeared with others)
# - isolation_rate (target: >95%)
```

### Cross-Pattern Analysis

```python
cross_stats = generator.get_cross_pattern_statistics()

# Returns:
# - co_occurrence_matrix
# - isolation_stats
# - overall_isolation_rate
# - most_common_combinations (top 10)
# - patterns_meeting_isolation_target (≥95%)
```

---

## Future Enhancements

**Planned for Future Releases:**
- Cross-channel fraud patterns (mobile → online → POS)
- Time-series anomaly detection
- Behavioral biometrics integration
- Adaptive fraud rates based on customer risk profile
- Real-time fraud detection APIs
- Graph neural network integration

## References

- **Implementation:** 
  - `src/generators/fraud_patterns.py` (2,619 lines) - Core patterns & combination system
  - `src/generators/fraud_network.py` (403 lines) - Network analysis
- **Tests:** 
  - `tests/test_fraud_patterns.py` (26 tests) - Base patterns
  - `tests/test_fraud_combinations.py` (13 tests) - Combination system
  - `tests/test_advanced_fraud_patterns.py` (29 tests) - Advanced patterns
  - `tests/test_fraud_network.py` (22 tests) - Network analysis
  - `tests/test_cross_pattern_stats.py` (10 tests) - Cross-pattern statistics
  - **Total: 209 passing tests**
- **Integration:** See `INTEGRATION_GUIDE.md` for usage examples
- **Architecture:** See `ARCHITECTURE.md` for system design

---

**Version History:**
- **0.5.0** (Oct 26, 2025): Advanced fraud patterns & combination system
  - 5 new advanced fraud patterns (11-15)
  - Fraud combination system (chained, coordinated, progressive)
  - Network analysis module (fraud rings, temporal clustering)
  - Cross-pattern statistics tracking
  - 74 new tests (total 209 passing)
  - Comprehensive documentation updates
- **0.4.0** (Oct 21, 2025): Initial fraud pattern library release
  - 10 fraud pattern types
  - Configurable injection system
  - Comprehensive documentation

**Contributors:** SynFinance Team  
**License:** MIT  
**Contact:** See CONTRIBUTING.md for guidelines
