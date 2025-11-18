# Getting Started with SynFinance

Welcome to SynFinance! This guide will help you generate realistic synthetic financial data in minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [5-Minute Quickstart](#5-minute-quickstart)
- [10-Minute Tutorial](#10-minute-tutorial)
- [Common Use Cases](#common-use-cases)
- [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, ensure you have:

### Required
- **Python 3.10 or higher** ([Download Python](https://www.python.org/downloads/))
- **PostgreSQL 12 or higher** ([Download PostgreSQL](https://www.postgresql.org/download/))
  - Required for persistent data storage and advanced analytics
  - Tested with PostgreSQL 12, 13, 14, 15, and 16

### Recommended
- **4GB+ RAM** for typical workloads
- **8GB+ RAM** for large datasets (1M+ transactions)
- **SSD storage** for better performance

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| PostgreSQL | 12 | 15+ |
| RAM | 4GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Storage | 5GB | 20GB+ SSD |

---

## Installation

### Option 1: Install from PyPI (Coming Soon)

```bash
pip install synfinance
```

### Option 2: Install from Source (Current Method)

1. **Clone the repository:**

```bash
git clone https://github.com/ssuptrey/SynFinance.git
cd SynFinance
```

2. **Create and activate a virtual environment:**

**Windows (cmd.exe):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up the database:**

```bash
# Create PostgreSQL database
createdb synfinance

# Run migrations
python -m alembic upgrade head
```

5. **Verify installation:**

```bash
python -c "import src; print('SynFinance installed successfully!')"
```

---

## 5-Minute Quickstart

Generate your first synthetic dataset in 5 minutes:

### Step 1: Run the Basic Example

```bash
python examples/week1/hello_world.py
```

### Step 2: Generate Sample Data

```python
from src.generators.customer_generator import CustomerGenerator
from src.core.config import load_config

# Load configuration
config = load_config()

# Create generator
generator = CustomerGenerator(config)

# Generate 1,000 customers
customers = generator.generate_batch(1000)

print(f"Generated {len(customers)} customers")
print(f"Sample customer: {customers[0]}")
```

### Step 3: Save to Database

```python
from src.database.manager import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager(config)

# Save customers
db_manager.bulk_insert('customers', customers)

print("Data saved to database!")
```

### Step 4: View Your Data

```bash
# Connect to PostgreSQL
psql -d synfinance

# Query your data
SELECT customer_id, name, email, city, country 
FROM customers 
LIMIT 5;
```

**Congratulations!** You've generated your first synthetic dataset.

---

## 10-Minute Tutorial

Create a more realistic dataset with transactions and fraud patterns:

### Step 1: Configure Your Dataset

Create a file `my_config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: synfinance
  user: postgres
  password: your_password

generators:
  customers:
    count: 10000
    demographics:
      age_distribution:
        young: 0.3    # 18-35
        middle: 0.5   # 36-55
        senior: 0.2   # 56+
      
  merchants:
    count: 500
    categories:
      - retail
      - dining
      - travel
      - entertainment
      - utilities
      
  transactions:
    count: 100000
    date_range:
      start: "2024-01-01"
      end: "2024-12-31"
    fraud_rate: 0.02  # 2% fraud
```

### Step 2: Run the Complete Pipeline

```python
from src.generators import CustomerGenerator, MerchantGenerator, TransactionGenerator
from src.core.config import load_config
from src.database.manager import DatabaseManager

# Load your custom config
config = load_config('my_config.yaml')
db = DatabaseManager(config)

# Generate customers
print("Generating customers...")
customer_gen = CustomerGenerator(config)
customers = customer_gen.generate_batch(10000)
db.bulk_insert('customers', customers)

# Generate merchants
print("Generating merchants...")
merchant_gen = MerchantGenerator(config)
merchants = merchant_gen.generate_batch(500)
db.bulk_insert('merchants', merchants)

# Generate transactions
print("Generating transactions...")
transaction_gen = TransactionGenerator(config, customers, merchants)
transactions = transaction_gen.generate_batch(100000)
db.bulk_insert('transactions', transactions)

print("✓ Dataset generation complete!")
```

### Step 3: Run Fraud Detection

```python
from src.fraud.detector import FraudDetector

detector = FraudDetector(config)
results = detector.analyze_transactions(transactions)

print(f"Total transactions: {results['total']}")
print(f"Fraudulent: {results['fraudulent']} ({results['fraud_rate']:.2%})")
print(f"High risk: {results['high_risk']}")
```

### Step 4: Generate Reports

```python
from src.reporting.generator import ReportGenerator

report_gen = ReportGenerator(config)

# Generate executive summary
report_gen.create_executive_report(
    output_file='reports/executive_summary.html',
    date_range=('2024-01-01', '2024-12-31')
)

# Generate fraud analysis
report_gen.create_fraud_report(
    output_file='reports/fraud_analysis.pdf'
)

print("Reports generated in ./reports/")
```

**Well done!** You now have a complete synthetic financial dataset with fraud detection and reporting.

---

## Common Use Cases

### Use Case 1: API Testing

Generate test data for your financial API:

```python
from src.generators import TransactionGenerator
from datetime import datetime, timedelta

# Generate transactions for last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

transactions = TransactionGenerator(config).generate_date_range(
    start_date=start_date,
    end_date=end_date,
    daily_volume=5000
)

# Export as JSON for API tests
import json
with open('test_data.json', 'w') as f:
    json.dump(transactions, f, default=str)
```

### Use Case 2: Fraud Detection Training

Create labeled datasets for ML model training:

```python
from src.fraud.pattern_library import FraudPatternLibrary

# Generate balanced dataset
fraud_lib = FraudPatternLibrary(config)

# 80% legitimate, 20% fraudulent
legitimate = transaction_gen.generate_batch(80000, fraud=False)
fraudulent = fraud_lib.generate_fraud_samples(
    count=20000,
    patterns=['card_testing', 'account_takeover', 'synthetic_identity']
)

# Combine and shuffle
all_transactions = legitimate + fraudulent
random.shuffle(all_transactions)
```

### Use Case 3: Performance Testing

Generate high-volume data for load testing:

```python
from src.performance import BatchProcessor

processor = BatchProcessor(config)

# Generate 10M transactions in batches
for batch_num in range(100):
    transactions = transaction_gen.generate_batch(100000)
    processor.process_batch(transactions)
    print(f"Batch {batch_num + 1}/100 complete")

print("10M transactions generated!")
```

### Use Case 4: Data Privacy Compliance

Generate synthetic data that preserves statistical properties without exposing PII:

```python
from src.generators import AnonymizedGenerator

# Load real data schema
real_schema = db.get_schema('production.customers')

# Generate synthetic data matching schema
anon_gen = AnonymizedGenerator(config, schema=real_schema)
synthetic_customers = anon_gen.generate_batch(
    count=50000,
    preserve_distributions=True
)

# Safe to share - no real customer data
db.bulk_insert('synthetic.customers', synthetic_customers)
```

### Use Case 5: Geographic Analysis

Generate data with realistic geographic patterns:

```python
from src.generators import GeographicGenerator

geo_gen = GeographicGenerator(config)

# Generate transactions with geo patterns
transactions = geo_gen.generate_with_patterns(
    count=50000,
    patterns={
        'local_preference': 0.7,     # 70% transactions near home
        'travel_periods': True,       # Include travel patterns
        'regional_merchants': True,   # Region-specific merchants
        'timezone_aware': True        # Respect time zones
    }
)
```

---

## Next Steps

Now that you're up and running, explore these resources:

### Documentation
- **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- **[User Guide](USER_GUIDE.md)** - Complete feature documentation
- **[API Reference](../api/API_REFERENCE.md)** - Full API documentation
- **[FAQ](FAQ.md)** - Frequently asked questions

### Examples
- **[Examples Directory](../../examples/)** - Runnable demo scripts
- **[Interactive Launcher](../../examples/run_demo.py)** - Browse all demos
- **[Week 10 Analytics](../../examples/week10_analytics/)** - Advanced analytics examples

### Configuration
- **[Configuration Guide](USER_GUIDE.md#configuration)** - Customize your setup
- **[Sample Configs](../../config/)** - Production-ready templates

### Support
- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share use cases
- **Contributing Guide** - Help improve SynFinance

---

## Quick Reference

### Essential Commands

```bash
# Activate virtual environment
.venv\Scripts\activate          # Windows cmd
.venv\Scripts\Activate.ps1      # Windows PowerShell
source .venv/bin/activate       # Linux/macOS

# Run examples
python examples/week1/hello_world.py
python examples/run_demo.py --list

# Database operations
createdb synfinance             # Create database
python -m alembic upgrade head  # Run migrations
psql -d synfinance              # Connect to database

# Testing
pytest tests/                   # Run all tests
pytest tests/week1/             # Run specific test suite
pytest -v --tb=short            # Verbose output

# Generate data
python examples/data_generation/generate_dataset.py
```

### Configuration Files

```
config/
├── default.yaml          # Default settings
├── development.yaml      # Development overrides
├── production.yaml       # Production settings
├── test.yaml            # Test configuration
└── schema.json          # Config schema
```

### Project Structure

```
SynFinance/
├── src/                 # Core library code
│   ├── generators/      # Data generators
│   ├── fraud/          # Fraud detection
│   ├── api/            # REST/GraphQL APIs
│   └── database/       # Database management
├── examples/           # Demo scripts
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── config/             # Configuration files
```

---

## Troubleshooting Quick Fixes

### Import Errors

```bash
# Ensure you're in the virtual environment
which python  # Should show .venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
pg_isready

# Verify credentials
psql -d synfinance -U postgres

# Reset database
dropdb synfinance
createdb synfinance
python -m alembic upgrade head
```

### Performance Issues

```python
# Use batch processing for large datasets
from src.performance import BatchProcessor

processor = BatchProcessor(config, batch_size=10000)
processor.process_large_dataset(customers, batch_size=10000)
```

For more troubleshooting help, see the [FAQ](FAQ.md).

---

**Ready to dive deeper?** Check out the [User Guide](USER_GUIDE.md) for comprehensive documentation of all features.
