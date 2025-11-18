# SynFinance User Guide

Complete guide to using SynFinance for synthetic financial data generation.

## Table of Contents

### Part 1: Core Concepts
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Data Model](#data-model)

### Part 2: Usage
- [Configuration](#configuration)
- [Data Generators](#data-generators)
- [Fraud Detection](#fraud-detection)
- [Database Management](#database-management)
- [APIs](#apis)
- [Analytics & Reporting](#analytics--reporting)

### Part 3: Advanced Topics
- [Performance Optimization](#performance-optimization)
- [Custom Patterns](#custom-patterns)
- [Integration](#integration)
- [Best Practices](#best-practices)

---

## Part 1: Core Concepts

### Overview

SynFinance is a comprehensive synthetic financial data generation platform designed for:

- **Testing & QA:** Generate realistic test data for financial applications
- **Training & Development:** Create datasets for ML model training
- **Performance Testing:** Generate high-volume data for load testing
- **Privacy Compliance:** Replace production data with synthetic equivalents
- **Research & Analysis:** Study fraud patterns and financial behaviors

**Key Capabilities:**
- Generate millions of transactions with realistic patterns
- Simulate fraud with 40+ fraud pattern types
- Real-time API serving (REST, GraphQL, WebSocket)
- Advanced analytics and reporting
- Production-grade performance and scalability

### Architecture

```
SynFinance Architecture
├── Data Layer
│   ├── PostgreSQL Database
│   ├── Connection Pooling
│   └── Migration Management (Alembic)
│
├── Generator Layer
│   ├── Customer Generator (demographics, behaviors)
│   ├── Merchant Generator (categories, locations)
│   ├── Transaction Generator (patterns, timing)
│   └── Geographic Generator (locations, travel)
│
├── Pattern Library
│   ├── Temporal Patterns (hourly, seasonal)
│   ├── Geographic Patterns (regional, travel)
│   ├── Behavioral Patterns (spending habits)
│   └── Fraud Patterns (40+ types)
│
├── Fraud Detection
│   ├── Rule-Based Engine
│   ├── ML Models (XGBoost, LightGBM)
│   ├── Ensemble Methods
│   └── Real-time Scoring
│
├── API Layer
│   ├── REST API (FastAPI)
│   ├── GraphQL API
│   ├── WebSocket (real-time)
│   └── Python SDK
│
└── Analytics Layer
    ├── Statistical Analysis
    ├── Visualization
    ├── Report Generation
    └── Performance Profiling
```

### Key Features

#### 1. Realistic Data Generation

- **Demographic Accuracy:** Age, income, location distributions match real-world data
- **Temporal Patterns:** Time-of-day, day-of-week, seasonal variations
- **Geographic Realism:** Location-aware merchants, travel patterns, timezone handling
- **Behavioral Consistency:** Customers have consistent spending patterns over time

#### 2. Fraud Pattern Library

40+ fraud patterns including:
- Card testing
- Account takeover
- Synthetic identity fraud
- Chargeback fraud
- Money laundering patterns
- First-party fraud
- Merchant collusion

#### 3. Performance & Scalability

- **Batch Processing:** Generate millions of records efficiently
- **Parallel Processing:** Multi-core and async support
- **Memory Optimization:** Streaming and chunked processing
- **Database Optimization:** Connection pooling, bulk inserts

#### 4. APIs & Integration

- **REST API:** Standard HTTP endpoints
- **GraphQL:** Flexible query language
- **WebSocket:** Real-time streaming
- **Python SDK:** Direct library access

### Data Model

#### Core Entities

**1. Customer**
```python
{
    "customer_id": "CUST_00001",
    "name": "John Smith",
    "email": "john.smith@example.com",
    "phone": "+1-555-0123",
    "date_of_birth": "1985-03-15",
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "country": "USA"
    },
    "demographics": {
        "age": 38,
        "income_bracket": "middle",
        "credit_score": 720,
        "risk_profile": "low"
    },
    "created_at": "2024-01-01T00:00:00Z"
}
```

**2. Merchant**
```python
{
    "merchant_id": "MERCH_00001",
    "name": "Acme Coffee Shop",
    "category": "dining",
    "mcc": "5814",  # Merchant Category Code
    "location": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "city": "New York",
        "state": "NY",
        "country": "USA"
    },
    "risk_score": 0.15,
    "created_at": "2024-01-01T00:00:00Z"
}
```

**3. Transaction**
```python
{
    "transaction_id": "TXN_00001",
    "customer_id": "CUST_00001",
    "merchant_id": "MERCH_00001",
    "amount": 45.50,
    "currency": "USD",
    "timestamp": "2024-06-15T14:30:00Z",
    "status": "approved",
    "payment_method": "credit_card",
    "card_type": "visa",
    "location": {
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "fraud_score": 0.02,
    "is_fraud": false,
    "fraud_type": null
}
```

---

## Part 2: Usage

### Configuration

#### Configuration File Structure

SynFinance uses YAML configuration files located in `config/`:

```yaml
# config/default.yaml

database:
  host: localhost
  port: 5432
  name: synfinance
  user: postgres
  password: changeme
  pool_size: 10
  max_overflow: 20
  echo: false  # SQL logging

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/synfinance.log
  console: true

generators:
  default_batch_size: 1000
  max_batch_size: 100000
  random_seed: 42  # For reproducibility
  
  customers:
    count: 10000
    demographics:
      age_distribution:
        young: 0.30      # 18-35 years: 30%
        middle: 0.50     # 36-55 years: 50%
        senior: 0.20     # 56+ years: 20%
      income_distribution:
        low: 0.25
        middle: 0.50
        high: 0.25
    
  merchants:
    count: 500
    categories:
      - retail
      - dining
      - travel
      - entertainment
      - utilities
      - healthcare
      - transportation
    
  transactions:
    count: 100000
    date_range:
      start: "2024-01-01"
      end: "2024-12-31"
    fraud_rate: 0.02  # 2% fraud
    daily_pattern: true
    seasonal_pattern: true

fraud:
  detection:
    threshold: 0.75  # Fraud score threshold
    model_path: models/fraud_detector.pkl
  patterns:
    enabled:
      - card_testing
      - account_takeover
      - synthetic_identity
      - velocity_abuse
    
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins:
    - http://localhost:3000
    - https://app.example.com

performance:
  batch_size: 10000
  parallel_workers: 4
  use_async: true
  cache_enabled: true
  cache_ttl: 3600  # seconds
```

#### Loading Configuration

```python
from src.core.config import load_config

# Load default config
config = load_config()

# Load specific config file
config = load_config('config/production.yaml')

# Load with environment override
config = load_config(env='production')  # Loads config/production.yaml

# Access configuration
db_host = config['database']['host']
batch_size = config['generators']['default_batch_size']
```

#### Environment-Specific Configs

```
config/
├── default.yaml       # Base configuration
├── development.yaml   # Development overrides
├── production.yaml    # Production settings
├── test.yaml         # Test configuration
└── local.yaml        # Local overrides (gitignored)
```

### Data Generators

#### Customer Generator

Generate realistic customer profiles:

```python
from src.generators.customer_generator import CustomerGenerator
from src.core.config import load_config

config = load_config()
generator = CustomerGenerator(config)

# Generate single customer
customer = generator.generate_single()

# Generate batch
customers = generator.generate_batch(1000)

# Generate with specific demographics
customers = generator.generate_batch(
    count=1000,
    demographics={
        'age_range': (25, 45),
        'income_bracket': 'high',
        'cities': ['New York', 'Los Angeles', 'Chicago']
    }
)

# Save to database
from src.database.manager import DatabaseManager
db = DatabaseManager(config)
db.bulk_insert('customers', customers)
```

**Advanced Options:**

```python
# Generate with custom patterns
customers = generator.generate_with_patterns(
    count=1000,
    patterns={
        'high_spenders': 0.1,      # 10% high spenders
        'frequent_travelers': 0.05,  # 5% frequent travelers
        'online_preference': 0.6     # 60% prefer online shopping
    }
)

# Generate correlated groups (families, businesses)
customer_groups = generator.generate_groups(
    group_count=100,
    members_per_group=(2, 5),  # 2-5 members per group
    group_type='family'
)
```

#### Merchant Generator

Generate merchant profiles:

```python
from src.generators.merchant_generator import MerchantGenerator

generator = MerchantGenerator(config)

# Generate merchants
merchants = generator.generate_batch(500)

# Generate by category
restaurants = generator.generate_by_category(
    category='dining',
    count=100
)

# Generate with geographic constraints
nyc_merchants = generator.generate_in_region(
    region='New York',
    count=50,
    categories=['dining', 'retail', 'entertainment']
)

# Save to database
db.bulk_insert('merchants', merchants)
```

#### Transaction Generator

Generate transactions with realistic patterns:

```python
from src.generators.transaction_generator import TransactionGenerator

# Initialize with customers and merchants
generator = TransactionGenerator(config, customers, merchants)

# Generate basic transactions
transactions = generator.generate_batch(10000)

# Generate for date range
transactions = generator.generate_date_range(
    start_date='2024-01-01',
    end_date='2024-12-31',
    daily_volume=1000
)

# Generate with patterns
transactions = generator.generate_with_patterns(
    count=10000,
    patterns={
        'temporal': True,        # Time-of-day patterns
        'geographic': True,      # Location patterns
        'seasonal': True,        # Seasonal variations
        'fraud_rate': 0.02      # 2% fraud
    }
)

# Generate specific scenarios
# Weekend shopping spree
weekend_txns = generator.generate_scenario(
    scenario='weekend_shopping',
    customer_subset=high_spenders,
    date='2024-06-15'  # Saturday
)

# Holiday season
holiday_txns = generator.generate_scenario(
    scenario='holiday_season',
    date_range=('2024-12-01', '2024-12-31'),
    volume_multiplier=2.5
)
```

#### Geographic Generator

Add realistic geographic patterns:

```python
from src.generators.geographic_generator import GeographicGenerator

geo_gen = GeographicGenerator(config)

# Generate with home location preference
transactions = geo_gen.generate_with_home_preference(
    customers=customers,
    merchants=merchants,
    count=10000,
    home_transaction_rate=0.7  # 70% near home
)

# Generate travel patterns
travel_txns = geo_gen.generate_travel_patterns(
    customers=frequent_travelers,
    duration_days=(3, 14),
    count=1000
)

# Generate timezone-aware transactions
txns = geo_gen.generate_timezone_aware(
    customers=customers,
    merchants=merchants,
    count=10000,
    respect_business_hours=True
)
```

### Fraud Detection

#### Fraud Pattern Library

Generate fraudulent transactions:

```python
from src.fraud.pattern_library import FraudPatternLibrary

fraud_lib = FraudPatternLibrary(config)

# List available patterns
patterns = fraud_lib.list_patterns()
# Returns: ['card_testing', 'account_takeover', 'synthetic_identity', ...]

# Generate specific fraud type
card_testing_fraud = fraud_lib.generate_pattern(
    pattern='card_testing',
    count=100,
    customers=customers,
    merchants=merchants
)

# Generate mixed fraud
mixed_fraud = fraud_lib.generate_mixed(
    count=1000,
    pattern_distribution={
        'card_testing': 0.3,
        'account_takeover': 0.2,
        'synthetic_identity': 0.2,
        'velocity_abuse': 0.3
    }
)

# Generate realistic fraud scenario
fraud_scenario = fraud_lib.generate_scenario(
    scenario='organized_fraud_ring',
    affected_customers=50,
    transactions_per_customer=(10, 50),
    time_window_hours=48
)
```

#### Fraud Detector

Detect fraud in transactions:

```python
from src.fraud.detector import FraudDetector

detector = FraudDetector(config)

# Score single transaction
transaction = {...}
score = detector.score_transaction(transaction)
print(f"Fraud score: {score:.3f}")

# Batch scoring
scores = detector.score_batch(transactions)

# Get predictions
predictions = detector.predict(transactions, threshold=0.75)
# Returns: [True, False, False, True, ...]

# Detailed analysis
analysis = detector.analyze_transactions(transactions)
print(f"Total: {analysis['total']}")
print(f"Fraudulent: {analysis['fraudulent']}")
print(f"Fraud rate: {analysis['fraud_rate']:.2%}")
print(f"Average score: {analysis['avg_score']:.3f}")
```

#### ML Models

Train and use ML fraud detection models:

```python
from src.fraud.ml_models import FraudMLModel

# Initialize model
model = FraudMLModel(config, model_type='xgboost')

# Prepare training data
from src.fraud.feature_engineering import FeatureEngineer
engineer = FeatureEngineer(config)

features = engineer.create_features(transactions)
labels = [txn['is_fraud'] for txn in transactions]

# Train model
model.train(features, labels)

# Evaluate
metrics = model.evaluate(test_features, test_labels)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")

# Save model
model.save('models/fraud_detector.pkl')

# Load and predict
loaded_model = FraudMLModel.load('models/fraud_detector.pkl')
predictions = loaded_model.predict(new_transactions)
```

### Database Management

#### Database Manager

Manage database operations:

```python
from src.database.manager import DatabaseManager

db = DatabaseManager(config)

# Connection management
db.connect()
db.disconnect()

# Use context manager (recommended)
with DatabaseManager(config) as db:
    # Operations here
    pass

# Bulk insert
db.bulk_insert('customers', customers)
db.bulk_insert('merchants', merchants)
db.bulk_insert('transactions', transactions)

# Query data
results = db.query("""
    SELECT customer_id, COUNT(*) as txn_count, SUM(amount) as total
    FROM transactions
    WHERE timestamp >= '2024-01-01'
    GROUP BY customer_id
    ORDER BY total DESC
    LIMIT 10
""")

# Execute with parameters
db.execute("""
    UPDATE transactions
    SET fraud_score = :score
    WHERE transaction_id = :txn_id
""", {'score': 0.95, 'txn_id': 'TXN_00001'})

# Batch operations with transaction
with db.transaction():
    db.bulk_insert('customers', batch1)
    db.bulk_insert('transactions', batch2)
    # Commits automatically if no errors
```

#### Schema Management

```python
from src.database.schema import SchemaManager

schema = SchemaManager(config)

# Create tables
schema.create_all_tables()

# Drop tables
schema.drop_all_tables()

# Reset database
schema.reset_database()

# Check schema
table_info = schema.get_table_info('transactions')
indexes = schema.list_indexes('transactions')

# Create index
schema.create_index(
    table='transactions',
    columns=['customer_id', 'timestamp'],
    name='idx_customer_timestamp'
)
```

#### Migrations

```bash
# Create new migration
python -m alembic revision -m "add_fraud_score_index"

# Apply migrations
python -m alembic upgrade head

# Rollback migration
python -m alembic downgrade -1

# View history
python -m alembic history

# Current version
python -m alembic current
```

### APIs

#### REST API

Start the REST API server:

```bash
# Start server
python -m src.api.rest.main

# Or with uvicorn directly
uvicorn src.api.rest.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints:**

```python
# Generate data
POST /api/v1/generate/customers
{
    "count": 1000,
    "demographics": {...}
}

POST /api/v1/generate/transactions
{
    "count": 10000,
    "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
    }
}

# Query data
GET /api/v1/customers?limit=100&offset=0
GET /api/v1/transactions?customer_id=CUST_00001
GET /api/v1/merchants?category=dining

# Fraud detection
POST /api/v1/fraud/score
{
    "transactions": [...]
}

# Analytics
GET /api/v1/analytics/summary?start_date=2024-01-01&end_date=2024-12-31
GET /api/v1/analytics/fraud-stats
```

**Python Client:**

```python
import requests

BASE_URL = 'http://localhost:8000/api/v1'

# Generate customers
response = requests.post(f'{BASE_URL}/generate/customers', json={
    'count': 1000
})
customers = response.json()

# Get transactions
response = requests.get(f'{BASE_URL}/transactions', params={
    'customer_id': 'CUST_00001',
    'limit': 100
})
transactions = response.json()

# Score fraud
response = requests.post(f'{BASE_URL}/fraud/score', json={
    'transactions': transactions
})
scores = response.json()
```

#### GraphQL API

```python
# Start GraphQL server
python -m src.api.graphql.main
```

**Example Queries:**

```graphql
# Query customers
query {
  customers(limit: 10) {
    customerId
    name
    email
    city
    transactions {
      transactionId
      amount
      timestamp
      fraudScore
    }
  }
}

# Generate data
mutation {
  generateCustomers(count: 1000) {
    success
    count
    customers {
      customerId
      name
    }
  }
}

# Complex query
query {
  analytics {
    totalCustomers
    totalTransactions
    fraudRate
    topMerchants(limit: 5) {
      name
      transactionCount
      totalAmount
    }
  }
}
```

**Python Client:**

```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(url='http://localhost:8000/graphql')
client = Client(transport=transport)

query = gql('''
    query {
        customers(limit: 10) {
            customerId
            name
            email
        }
    }
''')

result = client.execute(query)
print(result)
```

#### WebSocket API

Real-time transaction streaming:

```python
# Start WebSocket server
python -m src.api.websocket.main
```

**Python Client:**

```python
import asyncio
import websockets
import json

async def stream_transactions():
    uri = "ws://localhost:8000/ws/transactions"
    async with websockets.connect(uri) as websocket:
        # Subscribe to transaction stream
        await websocket.send(json.dumps({
            'action': 'subscribe',
            'filters': {
                'fraud_score_min': 0.75
            }
        }))
        
        # Receive transactions
        while True:
            message = await websocket.recv()
            transaction = json.loads(message)
            print(f"Received: {transaction['transaction_id']}, "
                  f"Score: {transaction['fraud_score']}")

asyncio.run(stream_transactions())
```

### Analytics & Reporting

#### Statistical Analysis

```python
from src.analytics.statistical import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(config)

# Load data
transactions = db.query("SELECT * FROM transactions LIMIT 100000")

# Basic statistics
stats = analyzer.describe(transactions, columns=['amount', 'fraud_score'])
print(stats)
# Output: count, mean, std, min, 25%, 50%, 75%, max

# Correlation analysis
correlations = analyzer.correlation_matrix(transactions, 
                                          columns=['amount', 'fraud_score', 'merchant_risk'])

# Time series analysis
daily_stats = analyzer.time_series_analysis(
    transactions,
    date_column='timestamp',
    value_column='amount',
    aggregation='sum',
    frequency='D'  # Daily
)

# Fraud pattern analysis
fraud_patterns = analyzer.analyze_fraud_patterns(transactions)
print(f"Peak fraud hours: {fraud_patterns['peak_hours']}")
print(f"High-risk categories: {fraud_patterns['risk_categories']}")
```

#### Visualization

```python
from src.analytics.visualization import Visualizer

viz = Visualizer(config)

# Transaction volume over time
viz.plot_time_series(
    transactions,
    date_column='timestamp',
    value_column='amount',
    output='charts/transaction_volume.png'
)

# Fraud distribution
viz.plot_fraud_distribution(
    transactions,
    output='charts/fraud_dist.png'
)

# Geographic heatmap
viz.plot_geographic_heatmap(
    transactions,
    output='charts/geo_heatmap.html'
)

# Correlation matrix
viz.plot_correlation_matrix(
    correlations,
    output='charts/correlation.png'
)
```

#### Report Generation

```python
from src.reporting.generator import ReportGenerator

report_gen = ReportGenerator(config)

# Executive summary
report_gen.create_executive_report(
    output_file='reports/executive_summary.html',
    date_range=('2024-01-01', '2024-12-31'),
    include_charts=True
)

# Fraud analysis report
report_gen.create_fraud_report(
    output_file='reports/fraud_analysis.pdf',
    transactions=transactions,
    detailed=True
)

# Custom report
report_gen.create_custom_report(
    template='templates/custom_report.html',
    output_file='reports/custom.pdf',
    data={
        'transactions': transactions,
        'customers': customers,
        'merchants': merchants
    }
)
```

---

## Part 3: Advanced Topics

### Performance Optimization

#### Batch Processing

```python
from src.performance.optimizer import BatchProcessor

processor = BatchProcessor(config, batch_size=10000)

# Process large dataset in batches
total_processed = 0
for batch in processor.process_in_batches(large_dataset):
    db.bulk_insert('transactions', batch)
    total_processed += len(batch)
    print(f"Processed {total_processed} records")

# Parallel batch processing
processor.process_parallel(
    data=large_dataset,
    function=process_batch,
    workers=4
)
```

#### Async Processing

```python
from src.performance.optimizer import AsyncProcessor
import asyncio

async def generate_and_save():
    processor = AsyncProcessor(config)
    
    # Generate multiple datasets concurrently
    results = await asyncio.gather(
        processor.generate_customers_async(10000),
        processor.generate_merchants_async(1000),
        processor.generate_transactions_async(100000)
    )
    
    customers, merchants, transactions = results
    return customers, merchants, transactions

# Run async
customers, merchants, transactions = asyncio.run(generate_and_save())
```

#### Query Optimization

```python
from src.performance.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer(config)

# Analyze query performance
analysis = optimizer.analyze_query("""
    SELECT customer_id, SUM(amount)
    FROM transactions
    WHERE timestamp >= '2024-01-01'
    GROUP BY customer_id
""")

print(f"Execution time: {analysis['execution_time_ms']}ms")
print(f"Rows scanned: {analysis['rows_scanned']}")
print(f"Suggested indexes: {analysis['index_recommendations']}")

# Apply recommendations
optimizer.apply_recommendations(analysis)
```

#### Memory Profiling

```python
from src.performance.profiler import Profiler

profiler = Profiler(config)

# Profile function
@profiler.profile_memory
def generate_large_dataset():
    generator = TransactionGenerator(config)
    return generator.generate_batch(1000000)

# Profile with context manager
with profiler.profile_memory():
    transactions = generate_large_dataset()

# Get report
report = profiler.get_memory_report()
print(f"Peak memory: {report['peak_mb']}MB")
print(f"Final memory: {report['final_mb']}MB")
```

### Custom Patterns

#### Create Custom Fraud Pattern

```python
from src.fraud.pattern_library import FraudPattern

class CustomFraudPattern(FraudPattern):
    def __init__(self, config):
        super().__init__(config)
        self.name = "custom_pattern"
        self.description = "My custom fraud pattern"
    
    def generate(self, count, customers, merchants):
        transactions = []
        
        for i in range(count):
            # Custom logic here
            txn = {
                'transaction_id': f'TXN_{i:06d}',
                'customer_id': random.choice(customers)['customer_id'],
                'merchant_id': random.choice(merchants)['merchant_id'],
                'amount': random.uniform(1000, 5000),  # High amounts
                'timestamp': self.generate_timestamp(),
                'is_fraud': True,
                'fraud_type': self.name
            }
            transactions.append(txn)
        
        return transactions

# Register and use
fraud_lib = FraudPatternLibrary(config)
fraud_lib.register_pattern(CustomFraudPattern)

# Generate
custom_fraud = fraud_lib.generate_pattern('custom_pattern', count=100)
```

#### Create Custom Generator

```python
from src.generators.base import BaseGenerator

class CustomDataGenerator(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_single(self):
        # Generate single record
        return {
            'id': self.generate_id(),
            'data': self.generate_custom_data()
        }
    
    def generate_batch(self, count):
        return [self.generate_single() for _ in range(count)]
    
    def generate_custom_data(self):
        # Your custom logic
        pass

# Use custom generator
generator = CustomDataGenerator(config)
data = generator.generate_batch(1000)
```

### Integration

#### Export Data

```python
from src.integration.exporter import DataExporter

exporter = DataExporter(config)

# Export to CSV
exporter.to_csv(
    data=transactions,
    output_file='data/transactions.csv',
    columns=['transaction_id', 'customer_id', 'amount', 'timestamp']
)

# Export to JSON
exporter.to_json(
    data=transactions,
    output_file='data/transactions.json',
    pretty=True
)

# Export to Parquet
exporter.to_parquet(
    data=transactions,
    output_file='data/transactions.parquet',
    compression='snappy'
)

# Export to database
exporter.to_database(
    data=transactions,
    table='transactions_export',
    connection_string='postgresql://user:pass@localhost/external_db'
)
```

#### Import Data

```python
from src.integration.importer import DataImporter

importer = DataImporter(config)

# Import from CSV
data = importer.from_csv('data/external_transactions.csv')

# Import with schema validation
data = importer.from_csv(
    file='data/external_transactions.csv',
    schema={
        'transaction_id': str,
        'amount': float,
        'timestamp': 'datetime'
    },
    validate=True
)

# Import from multiple sources
all_data = importer.import_multiple([
    ('csv', 'data/file1.csv'),
    ('json', 'data/file2.json'),
    ('parquet', 'data/file3.parquet')
])
```

### Best Practices

#### 1. Configuration Management

```python
# Use environment-specific configs
if environment == 'production':
    config = load_config('config/production.yaml')
else:
    config = load_config('config/development.yaml')

# Use environment variables for secrets
import os
config['database']['password'] = os.getenv('DB_PASSWORD')

# Validate configuration
from src.core.config import validate_config
validate_config(config)  # Raises error if invalid
```

#### 2. Error Handling

```python
from src.core.exceptions import (
    GeneratorError,
    DatabaseError,
    ValidationError
)

try:
    transactions = generator.generate_batch(count)
except GeneratorError as e:
    logger.error(f"Generation failed: {e}")
    # Handle gracefully
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Retry or fail
except ValidationError as e:
    logger.error(f"Invalid data: {e}")
    # Fix and retry
```

#### 3. Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/synfinance.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout code
logger.info("Starting data generation")
logger.debug(f"Config: {config}")
logger.warning("High fraud rate detected")
logger.error("Database connection failed")
```

#### 4. Testing Generated Data

```python
# Validate generated data
def validate_transactions(transactions):
    assert len(transactions) > 0, "No transactions generated"
    
    for txn in transactions:
        assert 'transaction_id' in txn
        assert 'amount' in txn
        assert txn['amount'] > 0
        assert 'timestamp' in txn
        assert 'customer_id' in txn
        
    print(f"✓ Validated {len(transactions)} transactions")

# Check distributions
def check_fraud_rate(transactions, expected_rate=0.02, tolerance=0.005):
    fraud_count = sum(1 for txn in transactions if txn['is_fraud'])
    actual_rate = fraud_count / len(transactions)
    
    assert abs(actual_rate - expected_rate) < tolerance, \
        f"Fraud rate {actual_rate:.3f} outside tolerance"
    
    print(f"✓ Fraud rate {actual_rate:.3f} within expected range")
```

#### 5. Performance Monitoring

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

@timing_decorator
def generate_dataset():
    # Your code here
    pass
```

---

## Summary

This guide covered:

- **Core concepts:** Architecture, features, data model
- **Usage:** Configuration, generators, fraud detection, database, APIs, analytics
- **Advanced topics:** Performance, custom patterns, integration, best practices

**Next Steps:**
- [API Reference](../api/API_REFERENCE.md) - Detailed API documentation
- [FAQ](FAQ.md) - Common questions and answers
- [Examples](../../examples/) - Runnable code examples

For support, visit our [GitHub repository](https://github.com/ssuptrey/SynFinance).
