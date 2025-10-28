# SynFinance Examples

Example scripts demonstrating the usage of the SynFinance synthetic transaction data generator.

## Available Examples

### 1. `demo_merchant_ecosystem.py`
**Purpose:** Demonstrates the merchant ecosystem generation capabilities.

**Features:**
- Generate customers with merchant loyalty patterns
- Create transactions with merchant-specific behaviors
- Analyze merchant distribution (chain vs local)
- Validate loyalty rates by category

**Usage:**
```bash
python examples/demo_merchant_ecosystem.py
```

**Output:**
- Merchant distribution statistics
- Loyalty behavior analysis
- Chain vs local merchant comparison
- Category-wise merchant breakdown

---

### 2. `demo_geographic_patterns.py`
**Purpose:** Demonstrates geographic consistency and patterns.

**Features:**
- Generate transactions across 20 Indian cities
- Test cost-of-living adjustments
- Validate 80/15/5 distribution (home/nearby/distant)
- Analyze city-tier patterns (Tier 1/2/3)

**Usage:**
```bash
python examples/demo_geographic_patterns.py
```

**Output:**
- City distribution statistics
- Cost-of-living multiplier effects
- Proximity group analysis
- Merchant density by city tier

---

### 3. `run_customer_test.py`
**Purpose:** Quick customer generation test and validation.

**Features:**
- Generate sample customer profiles
- Validate customer data integrity
- Output customer statistics
- Export customer data to JSON

**Usage:**
```bash
python examples/run_customer_test.py
```

**Output:**
- Customer validation statistics
- Segment distribution
- Income bracket distribution
- Occupation distribution
- JSON export: `output/customer_validation_stats.json`

---

## Quick Start

### Prerequisites

```bash
# Navigate to project root
cd e:\SynFinance

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run All Examples

```bash
# Merchant ecosystem demo
python examples/demo_merchant_ecosystem.py

# Geographic patterns demo
python examples/demo_geographic_patterns.py

# Customer test
python examples/run_customer_test.py
```

---

## Example Output

### Merchant Ecosystem Demo

```
=== Merchant Ecosystem Demo ===

Generated 1000 transactions across 100 customers

Merchant Distribution:
  Chain Merchants: 48.3%
  Local Merchants: 51.7%

Loyalty Rates by Category:
  Groceries: 87.5%
  Restaurants: 72.3%
  Fashion: 65.1%
  Electronics: 62.8%

Average Reputation Scores:
  Chain: 0.82
  Local: 0.68
```

### Geographic Patterns Demo

```
=== Geographic Patterns Demo ===

Generated 1000 transactions across 20 cities

City Distribution:
  Home City: 79.2%
  Nearby Cities: 15.4%
  Distant Cities: 5.4%

Cost-of-Living Adjustments:
  Tier 1 (Mumbai): +30% (avg ₹1,690)
  Tier 2 (Indore): baseline (avg ₹1,300)
  Tier 3 (Patna): -20% (avg ₹1,040)

Merchant Density:
  Tier 1: 98% availability
  Tier 2: 82% availability
  Tier 3: 61% availability
```

### Customer Test

```
=== Customer Validation Stats ===

Total Customers: 100

Segment Distribution:
  Students: 15%
  Young Professionals: 22%
  Established Professionals: 18%
  Business Owners: 12%
  Homemakers: 16%
  Retirees: 10%
  Freelancers: 7%

Income Bracket Distribution:
  LOW: 25%
  LOWER_MIDDLE: 28%
  MIDDLE: 22%
  UPPER_MIDDLE: 15%
  HIGH: 10%

Output saved to: output/customer_validation_stats.json
```

---

## Creating Custom Examples

### Basic Template

```python
"""
Example: Custom Transaction Generation
"""
from src.customer_generator import CustomerGenerator
from src.generators.transaction_core import TransactionGenerator
from datetime import datetime

def main():
    # Initialize generators
    customer_gen = CustomerGenerator(seed=42)
    transaction_gen = TransactionGenerator(seed=42)
    
    # Generate customers
    customers = customer_gen.generate_customers(100)
    print(f"Generated {len(customers)} customers")
    
    # Generate transactions
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for customer in customers:
        txns = transaction_gen.generate_transactions(
            customer=customer,
            num_transactions=50,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Customer {customer.customer_id}: {len(txns)} transactions")

if __name__ == "__main__":
    main()
```

---

## Integration Examples

### Example 1: Daily Transaction Pipeline

```python
from src.data_generator import generate_dataset

# Generate daily transaction data
df = generate_dataset(
    num_customers=1000,
    transactions_per_customer=30,
    start_date="2024-01-01",
    end_date="2024-01-31",
    seed=42
)

# Export to CSV
df.to_csv("transactions_jan_2024.csv", index=False)
print(f"Generated {len(df)} transactions")
```

### Example 2: ML Training Data

```python
from src.data_generator import generate_dataset

# Generate labeled training data
df = generate_dataset(
    num_customers=10000,
    transactions_per_customer=100,
    start_date="2024-01-01",
    end_date="2024-12-31",
    seed=42
)

# Add fraud labels (synthetic)
import numpy as np
df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])

# Split and export
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train.to_csv("train_data.csv", index=False)
test.to_csv("test_data.csv", index=False)
```

### Example 3: Streaming Generation

```python
from src.generators.transaction_core import TransactionGenerator
from src.customer_generator import CustomerGenerator
from datetime import datetime

# Initialize generators
customer_gen = CustomerGenerator(seed=42)
transaction_gen = TransactionGenerator(seed=42)

# Generate customers once
customers = customer_gen.generate_customers(1000)

# Stream transactions
for customer in customers:
    for txn in transaction_gen.generate_transactions_streaming(
        customer=customer,
        num_transactions=100,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31)
    ):
        # Process transaction immediately
        process_transaction(txn)
```

---

## Performance Tips

### 1. Batch Generation
```python
# Generate large batches for better performance
df = generate_dataset(
    num_customers=10000,  # Larger batches
    transactions_per_customer=100,
    seed=42
)
```

### 2. Use Streaming for Memory Efficiency
```python
# Stream transactions for large datasets
for txn in transaction_gen.generate_transactions_streaming(...):
    # Process one at a time
    write_to_database(txn)
```

### 3. Set Seeds for Reproducibility
```python
# Use same seed for reproducible results
customer_gen = CustomerGenerator(seed=42)
transaction_gen = TransactionGenerator(seed=42)
```

### 4. Filter Early
```python
# Filter at generation time, not after
df = df[df['amount'] > 1000]  # ❌ Slow
# Better: Generate only high-value transactions
```

---

## Export Formats

### CSV Export
```python
df.to_csv("transactions.csv", index=False)
```

### JSON Export
```python
df.to_json("transactions.json", orient="records", lines=True)
```

### Excel Export
```python
df.to_excel("transactions.xlsx", index=False, engine='xlsxwriter')
```

### Parquet Export (Recommended for Large Datasets)
```python
df.to_parquet("transactions.parquet", index=False)
```

---

## Common Use Cases

### Use Case 1: Fraud Detection Training Data
```python
# Generate diverse transaction patterns
df = generate_dataset(
    num_customers=50000,
    transactions_per_customer=200,
    start_date="2023-01-01",
    end_date="2024-12-31"
)

# Add synthetic fraud labels
df['is_fraud'] = synthetic_fraud_labels(df)
```

### Use Case 2: Customer Segmentation Analysis
```python
# Generate customer profiles with rich attributes
customers = customer_gen.generate_customers(10000)

# Analyze segments
segment_analysis = customers.groupby('customer_segment').agg({
    'income_bracket': 'value_counts',
    'digital_savviness': 'mean',
    'preferred_categories': 'first'
})
```

### Use Case 3: Time Series Forecasting
```python
# Generate time series data with temporal patterns
df = generate_dataset(
    num_customers=1000,
    transactions_per_customer=365,  # Daily transactions
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Aggregate by date
daily_totals = df.groupby('transaction_date')['amount'].sum()
```

---

## Troubleshooting

### Error: Module not found
```bash
# Ensure you're in the project root
cd e:\SynFinance

# Set PYTHONPATH
set PYTHONPATH=e:\SynFinance\src
```

### Error: No customers generated
```bash
# Check seed value
customer_gen = CustomerGenerator(seed=42)  # Use fixed seed

# Verify customer count
customers = customer_gen.generate_customers(100)
print(f"Generated {len(customers)} customers")
```

### Performance Issues
```bash
# Reduce batch size
df = generate_dataset(
    num_customers=1000,  # Smaller batch
    transactions_per_customer=50
)

# Use streaming for large datasets
for txn in transaction_gen.generate_transactions_streaming(...):
    process_transaction(txn)
```

---

## Documentation

- **Quick Start:** See [docs/guides/QUICKSTART.md](../docs/guides/QUICKSTART.md)
- **Integration Guide:** See [docs/guides/INTEGRATION_GUIDE.md](../docs/guides/INTEGRATION_GUIDE.md)
- **Quick Reference:** See [docs/guides/QUICK_REFERENCE.md](../docs/guides/QUICK_REFERENCE.md)
- **Architecture:** See [docs/technical/ARCHITECTURE.md](../docs/technical/ARCHITECTURE.md)

---

## Contributing

Found a bug or want to add a new example? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
