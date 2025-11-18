# Frequently Asked Questions (FAQ)

Common questions and answers about SynFinance.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Data Generation](#data-generation)
- [Fraud Detection](#fraud-detection)
- [Performance & Scalability](#performance--scalability)
- [Database & Storage](#database--storage)
- [APIs & Integration](#apis--integration)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## General Questions

### What is SynFinance?

SynFinance is a comprehensive synthetic financial data generation platform that creates realistic transaction data, customer profiles, and merchant records for testing, development, and research purposes.

### Why use synthetic data instead of production data?

- **Privacy Compliance:** Avoid exposing sensitive customer information (GDPR, CCPA)
- **Data Availability:** Generate unlimited test data on demand
- **Controlled Testing:** Create specific scenarios and edge cases
- **Cost Effective:** No need to anonymize production data
- **Flexibility:** Customize data to match your exact requirements

### Is SynFinance free to use?

Yes, SynFinance is open-source and free to use under the MIT License. Enterprise support and hosted solutions may be available in the future.

### What programming languages does SynFinance support?

SynFinance is written in Python and provides:
- Native Python SDK
- REST API (language-agnostic)
- GraphQL API (language-agnostic)
- WebSocket API (language-agnostic)

You can use it from any language that supports HTTP requests.

### Can I use SynFinance in production?

Yes! SynFinance is designed for production use. It includes:
- Robust error handling
- Performance optimization
- Comprehensive testing
- Production-grade database support
- API rate limiting and authentication

---

## Installation & Setup

### What are the minimum system requirements?

- Python 3.10+
- PostgreSQL 12+
- 4GB RAM
- 5GB disk space
- 2 CPU cores

For best performance, we recommend 8GB+ RAM and 4+ CPU cores.

### Do I need PostgreSQL? Can I use another database?

PostgreSQL is the primary supported database due to its advanced features. Other databases may work but are not officially supported yet. Future versions may support:
- MySQL/MariaDB
- SQLite (development only)
- MongoDB (NoSQL variant)

### How do I install PostgreSQL?

See our [Installation Guide](INSTALLATION.md#postgresql-installation) for detailed instructions for:
- Windows
- macOS (via Homebrew)
- Linux (apt, yum)

### The virtual environment activation fails. What should I do?

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows cmd:**
```cmd
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### How do I verify the installation?

```bash
# Check Python version
python --version  # Should be 3.10+

# Check PostgreSQL
psql --version

# Check SynFinance installation
python -c "import src; print('SynFinance OK')"

# Run tests
pytest tests/ -v
```

---

## Data Generation

### How do I generate my first dataset?

```python
from src.generators import CustomerGenerator, TransactionGenerator
from src.core.config import load_config

config = load_config()

# Generate customers
customer_gen = CustomerGenerator(config)
customers = customer_gen.generate_batch(1000)

# Generate transactions
txn_gen = TransactionGenerator(config, customers, merchants)
transactions = txn_gen.generate_batch(10000)
```

See the [5-Minute Quickstart](GETTING_STARTED.md#5-minute-quickstart) for details.

### How realistic is the generated data?

SynFinance generates highly realistic data including:
- **Demographics:** Age, income, location distributions match real-world data
- **Temporal Patterns:** Time-of-day, day-of-week, seasonal variations
- **Geographic Patterns:** Location-based merchant preferences, travel patterns
- **Behavioral Consistency:** Customers have consistent spending habits over time
- **Fraud Patterns:** Based on real fraud detection research and industry patterns

### Can I customize the data generation?

Yes! SynFinance is highly customizable:

```python
# Custom demographics
customers = generator.generate_batch(
    count=1000,
    demographics={
        'age_range': (25, 45),
        'income_bracket': 'high',
        'cities': ['New York', 'San Francisco']
    }
)

# Custom transaction patterns
transactions = generator.generate_with_patterns(
    count=10000,
    patterns={
        'temporal': True,
        'seasonal': True,
        'fraud_rate': 0.03  # 3% fraud
    }
)
```

### How do I generate specific date ranges?

```python
transactions = generator.generate_date_range(
    start_date='2024-01-01',
    end_date='2024-12-31',
    daily_volume=1000
)
```

### Can I generate correlated data (e.g., family members)?

Yes:

```python
customer_groups = generator.generate_groups(
    group_count=100,
    members_per_group=(2, 5),
    group_type='family'
)
```

### How do I ensure reproducibility?

Set a random seed in your configuration:

```yaml
generators:
  random_seed: 42
```

Or in code:

```python
import random
random.seed(42)

customers = generator.generate_batch(1000)  # Reproducible
```

---

## Fraud Detection

### What fraud patterns does SynFinance support?

40+ fraud patterns including:
- Card testing
- Account takeover
- Synthetic identity fraud
- First-party fraud
- Chargeback fraud
- Velocity abuse
- Money laundering patterns
- Merchant collusion
- Transaction laundering
- And many more...

See the full list: `fraud_lib.list_patterns()`

### How accurate is the fraud detection?

The ML models achieve:
- **Precision:** 85-95% (few false positives)
- **Recall:** 80-90% (catches most fraud)
- **F1 Score:** 85-92%
- **AUC-ROC:** 0.90-0.95

Performance varies based on your training data and configuration.

### Can I train custom fraud detection models?

Yes:

```python
from src.fraud.ml_models import FraudMLModel

model = FraudMLModel(config, model_type='xgboost')
model.train(features, labels)
model.save('models/my_custom_model.pkl')
```

### How do I adjust the fraud detection threshold?

```python
# Higher threshold = fewer false positives, more false negatives
detector = FraudDetector(config)
predictions = detector.predict(transactions, threshold=0.85)

# Or in config
fraud:
  detection:
    threshold: 0.85
```

### What's the difference between rule-based and ML fraud detection?

- **Rule-based:** Fast, explainable, no training needed. Good for known patterns.
- **ML-based:** Learns patterns from data, adapts to new fraud types, higher accuracy.

SynFinance supports both. Use ensemble methods for best results.

---

## Performance & Scalability

### How much data can SynFinance generate?

SynFinance can generate:
- **Millions of transactions** per run
- **Hundreds of thousands of customers**
- **Tens of thousands of merchants**

Performance depends on your hardware. On a typical development machine (8GB RAM, 4 cores):
- 100K transactions: ~10 seconds
- 1M transactions: ~2 minutes
- 10M transactions: ~20 minutes (with batch processing)

### How do I generate large datasets efficiently?

Use batch processing:

```python
from src.performance import BatchProcessor

processor = BatchProcessor(config, batch_size=10000)

for batch in processor.process_in_batches(large_dataset):
    db.bulk_insert('transactions', batch)
```

### Can I use parallel processing?

Yes:

```python
from src.performance import AsyncProcessor
import asyncio

async_processor = AsyncProcessor(config)
results = asyncio.run(async_processor.generate_all(
    customers=10000,
    merchants=1000,
    transactions=100000,
    workers=4
))
```

### The generation is slow. How can I speed it up?

1. **Use batch processing:** Process in chunks
2. **Increase batch size:** Larger batches = fewer DB round trips
3. **Use async/parallel:** Utilize multiple cores
4. **Optimize database:** Add indexes, tune PostgreSQL
5. **Reduce validation:** Disable in production
6. **Use faster hardware:** SSD, more RAM, more cores

```python
# Optimize config
generators:
  default_batch_size: 10000  # Larger batches
  max_batch_size: 100000

performance:
  parallel_workers: 4  # Use multiple cores
  use_async: true
  cache_enabled: true
```

### How much memory does SynFinance use?

Memory usage depends on batch size:
- 1K transactions: ~50MB
- 10K transactions: ~200MB
- 100K transactions: ~1GB
- 1M transactions: ~5-8GB

Use batch processing to keep memory usage low:

```python
processor = BatchProcessor(config, batch_size=10000)
# Processes 1M transactions in 10K chunks = ~200MB peak memory
```

---

## Database & Storage

### How do I connect to PostgreSQL?

Configure in `config/local.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: synfinance
  user: postgres
  password: your_password
```

Or use environment variables:

```bash
export SYNFINANCE_DB_HOST=localhost
export SYNFINANCE_DB_PASSWORD=your_password
```

### Can I use an existing database?

Yes, point SynFinance to your database in the config. Run migrations to create tables:

```bash
python -m alembic upgrade head
```

### How do I export data to CSV/JSON?

```python
from src.integration import DataExporter

exporter = DataExporter(config)
exporter.to_csv(transactions, 'output/transactions.csv')
exporter.to_json(transactions, 'output/transactions.json')
exporter.to_parquet(transactions, 'output/transactions.parquet')
```

### How do I import existing data?

```python
from src.integration import DataImporter

importer = DataImporter(config)
data = importer.from_csv('input/transactions.csv')
db.bulk_insert('transactions', data)
```

### The database is getting too large. What should I do?

1. **Partition tables:** Split by date range
2. **Archive old data:** Move to separate tables/databases
3. **Delete test data:** Remove old test runs
4. **Use data retention policies:** Auto-delete after N days

```sql
-- Delete transactions older than 1 year
DELETE FROM transactions
WHERE timestamp < NOW() - INTERVAL '1 year';

-- Vacuum to reclaim space
VACUUM FULL transactions;
```

---

## APIs & Integration

### How do I start the REST API?

```bash
# Development
python -m src.api.rest.main

# Production (with uvicorn)
uvicorn src.api.rest.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### How do I authenticate API requests?

Configure authentication in your config:

```yaml
api:
  auth:
    enabled: true
    type: api_key  # or "basic", "oauth"
    api_keys:
      - "your-secret-key-here"
```

Then in requests:

```python
import requests

headers = {'Authorization': 'Bearer your-secret-key-here'}
response = requests.post(url, json=data, headers=headers)
```

### Can I use SynFinance from JavaScript/TypeScript?

Yes, use the REST or GraphQL API:

```javascript
// REST API
const response = await fetch('http://localhost:8000/api/v1/generate/customers', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({ count: 1000 })
});

const data = await response.json();
```

```javascript
// GraphQL API
const query = `
    query {
        customers(limit: 10) {
            customerId
            name
            email
        }
    }
`;

const response = await fetch('http://localhost:8000/graphql', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
});
```

### How do I use the WebSocket API?

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transactions');

ws.onopen = () => {
    ws.send(JSON.stringify({
        action: 'subscribe',
        filters: { fraud_score_min: 0.75 }
    }));
};

ws.onmessage = (event) => {
    const transaction = JSON.parse(event.data);
    console.log('New transaction:', transaction);
};
```

### Are there rate limits on the API?

Yes, default limits are:
- Generate endpoints: 10 req/min
- Query endpoints: 100 req/min
- Fraud detection: 50 req/min

Configure in `config.yaml`:

```yaml
api:
  rate_limits:
    generate: 10
    query: 100
    fraud: 50
```

---

## Troubleshooting

### I get "ModuleNotFoundError: No module named 'src'"

Ensure:
1. You're in the repository root: `cd SynFinance`
2. Virtual environment is activated: `.venv\Scripts\activate`
3. Dependencies are installed: `pip install -r requirements.txt`

### PostgreSQL connection fails

Check:
1. PostgreSQL is running: `pg_isready`
2. Database exists: `psql -l | grep synfinance`
3. Credentials are correct in config
4. Firewall allows connections

```bash
# Test connection
psql -h localhost -U postgres -d synfinance
```

### Alembic migration errors

```bash
# Reset migrations
python -m alembic downgrade base
python -m alembic upgrade head

# Or recreate database
dropdb synfinance
createdb synfinance
python -m alembic upgrade head
```

### Out of memory errors

Reduce batch size:

```python
# Instead of
transactions = generator.generate_batch(1000000)

# Use batch processing
processor = BatchProcessor(config, batch_size=10000)
for batch in processor.process_in_batches(range(100)):
    batch_transactions = generator.generate_batch(10000)
    db.bulk_insert('transactions', batch_transactions)
```

### Tests are failing

```bash
# Ensure test database exists
createdb synfinance_test

# Run with verbose output
pytest -v --tb=short

# Run specific test
pytest tests/generators/test_customer_generator.py -v

# Clear cache and retry
pytest --cache-clear
```

### API returns 500 Internal Server Error

Check logs:

```bash
# View logs
tail -f logs/synfinance.log

# Or run API with debug mode
python -m src.api.rest.main --log-level DEBUG
```

---

## Advanced Topics

### Can I create custom fraud patterns?

Yes:

```python
from src.fraud.pattern_library import FraudPattern

class MyCustomPattern(FraudPattern):
    def __init__(self, config):
        super().__init__(config)
        self.name = "custom_pattern"
    
    def generate(self, count, customers, merchants):
        # Your custom logic here
        return transactions

# Register
fraud_lib.register_pattern(MyCustomPattern)

# Use
fraud = fraud_lib.generate_pattern('custom_pattern', count=100)
```

### How do I profile performance?

```python
from src.performance import Profiler

profiler = Profiler(config)

with profiler.profile_memory():
    transactions = generator.generate_batch(100000)

report = profiler.get_memory_report()
print(f"Peak memory: {report['peak_mb']}MB")
```

### Can I run SynFinance in Docker?

Coming soon! Docker support is planned for a future release.

### How do I contribute to SynFinance?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

### Where can I get help?

- **Documentation:** Read the [User Guide](USER_GUIDE.md)
- **GitHub Issues:** [Report bugs or request features](https://github.com/ssuptrey/SynFinance/issues)
- **Discussions:** [Ask questions](https://github.com/ssuptrey/SynFinance/discussions)
- **Email:** support@synfinance.io (for commercial support)

### Is there commercial support available?

Community support is free via GitHub Issues and Discussions. Commercial support options may be available in the future for:
- Priority bug fixes
- Custom feature development
- Training and consulting
- Hosted solutions

Contact support@synfinance.io for inquiries.

---

## Still have questions?

If your question isn't answered here:

1. **Search the docs:** Check the [User Guide](USER_GUIDE.md) and [API Reference](../api/API_REFERENCE.md)
2. **Search GitHub Issues:** Someone may have already asked
3. **Ask in Discussions:** Post your question
4. **File an issue:** If you think it's a bug

**Happy data generating!**
