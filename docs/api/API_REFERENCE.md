# API Reference

Complete reference for all SynFinance APIs: REST, GraphQL, WebSocket, and Python SDK.

## Table of Contents

- [REST API](#rest-api)
- [GraphQL API](#graphql-api)
- [WebSocket API](#websocket-api)
- [Python SDK](#python-sdk)

---

## REST API

Base URL: `http://localhost:8000/api/v1`

### Authentication

```http
# API Key (if enabled)
Authorization: Bearer YOUR_API_KEY

# Basic Auth (if enabled)
Authorization: Basic base64(username:password)
```

### Common Response Format

```json
{
    "success": true,
    "data": {...},
    "message": "Operation completed successfully",
    "timestamp": "2024-11-04T12:00:00Z"
}
```

### Error Response

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid parameter: count must be positive",
        "details": {...}
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

---

### Endpoints

#### Generate Customers

```http
POST /api/v1/generate/customers
```

**Request Body:**
```json
{
    "count": 1000,
    "demographics": {
        "age_range": [25, 45],
        "income_bracket": "middle",
        "cities": ["New York", "Los Angeles"]
    },
    "save_to_db": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "count": 1000,
        "customers": [
            {
                "customer_id": "CUST_00001",
                "name": "John Smith",
                "email": "john.smith@example.com",
                "city": "New York",
                "state": "NY"
            }
        ]
    }
}
```

---

#### Generate Merchants

```http
POST /api/v1/generate/merchants
```

**Request Body:**
```json
{
    "count": 500,
    "categories": ["retail", "dining", "travel"],
    "regions": ["New York", "California"],
    "save_to_db": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "count": 500,
        "merchants": [
            {
                "merchant_id": "MERCH_00001",
                "name": "Acme Coffee Shop",
                "category": "dining",
                "city": "New York"
            }
        ]
    }
}
```

---

#### Generate Transactions

```http
POST /api/v1/generate/transactions
```

**Request Body:**
```json
{
    "count": 10000,
    "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
    },
    "patterns": {
        "temporal": true,
        "geographic": true,
        "fraud_rate": 0.02
    },
    "save_to_db": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "count": 10000,
        "transactions": [
            {
                "transaction_id": "TXN_00001",
                "customer_id": "CUST_00001",
                "merchant_id": "MERCH_00001",
                "amount": 45.50,
                "timestamp": "2024-06-15T14:30:00Z",
                "is_fraud": false
            }
        ],
        "stats": {
            "total_amount": 450000.00,
            "avg_amount": 45.00,
            "fraud_count": 200
        }
    }
}
```

---

#### Get Customers

```http
GET /api/v1/customers
```

**Query Parameters:**
- `limit` (int, default: 100): Number of results
- `offset` (int, default: 0): Offset for pagination
- `city` (string, optional): Filter by city
- `state` (string, optional): Filter by state
- `income_bracket` (string, optional): Filter by income bracket

**Response:**
```json
{
    "success": true,
    "data": {
        "count": 100,
        "total": 10000,
        "customers": [...]
    }
}
```

---

#### Get Customer by ID

```http
GET /api/v1/customers/{customer_id}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "customer_id": "CUST_00001",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "+1-555-0123",
        "address": {...},
        "demographics": {...},
        "transaction_count": 150,
        "total_spent": 7500.00
    }
}
```

---

#### Get Transactions

```http
GET /api/v1/transactions
```

**Query Parameters:**
- `limit` (int, default: 100)
- `offset` (int, default: 0)
- `customer_id` (string, optional)
- `merchant_id` (string, optional)
- `start_date` (date, optional)
- `end_date` (date, optional)
- `min_amount` (float, optional)
- `max_amount` (float, optional)
- `is_fraud` (boolean, optional)

**Response:**
```json
{
    "success": true,
    "data": {
        "count": 100,
        "total": 100000,
        "transactions": [...]
    }
}
```

---

#### Score Transactions (Fraud Detection)

```http
POST /api/v1/fraud/score
```

**Request Body:**
```json
{
    "transactions": [
        {
            "transaction_id": "TXN_00001",
            "customer_id": "CUST_00001",
            "merchant_id": "MERCH_00001",
            "amount": 9999.99,
            "timestamp": "2024-11-04T03:00:00Z"
        }
    ],
    "threshold": 0.75
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "results": [
            {
                "transaction_id": "TXN_00001",
                "fraud_score": 0.92,
                "is_fraud": true,
                "risk_factors": [
                    "High amount",
                    "Unusual time",
                    "First transaction with merchant"
                ]
            }
        ],
        "summary": {
            "total": 1,
            "flagged": 1,
            "avg_score": 0.92
        }
    }
}
```

---

#### Get Analytics Summary

```http
GET /api/v1/analytics/summary
```

**Query Parameters:**
- `start_date` (date, required)
- `end_date` (date, required)

**Response:**
```json
{
    "success": true,
    "data": {
        "period": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        "customers": {
            "total": 10000,
            "new": 1500,
            "active": 8500
        },
        "transactions": {
            "total": 100000,
            "volume": 4500000.00,
            "avg_amount": 45.00
        },
        "fraud": {
            "count": 2000,
            "rate": 0.02,
            "total_amount": 180000.00
        }
    }
}
```

---

#### Get Fraud Statistics

```http
GET /api/v1/analytics/fraud-stats
```

**Response:**
```json
{
    "success": true,
    "data": {
        "total_fraud": 2000,
        "fraud_by_type": {
            "card_testing": 600,
            "account_takeover": 400,
            "synthetic_identity": 500,
            "velocity_abuse": 500
        },
        "fraud_by_hour": {...},
        "high_risk_merchants": [...]
    }
}
```

---

## GraphQL API

Endpoint: `http://localhost:8000/graphql`

### Schema Overview

```graphql
type Customer {
    customerId: ID!
    name: String!
    email: String!
    phone: String
    city: String!
    state: String!
    country: String!
    dateOfBirth: Date!
    transactions: [Transaction!]!
}

type Merchant {
    merchantId: ID!
    name: String!
    category: String!
    mcc: String!
    city: String!
    state: String!
    riskScore: Float!
}

type Transaction {
    transactionId: ID!
    customer: Customer!
    merchant: Merchant!
    amount: Float!
    currency: String!
    timestamp: DateTime!
    status: String!
    fraudScore: Float!
    isFraud: Boolean!
    fraudType: String
}

type Query {
    customers(limit: Int, offset: Int, city: String): [Customer!]!
    customer(customerId: ID!): Customer
    merchants(limit: Int, category: String): [Merchant!]!
    transactions(
        limit: Int
        customerId: ID
        startDate: Date
        endDate: Date
    ): [Transaction!]!
    analytics: Analytics!
}

type Mutation {
    generateCustomers(count: Int!, demographics: DemographicsInput): GenerateResult!
    generateMerchants(count: Int!, categories: [String!]): GenerateResult!
    generateTransactions(count: Int!, config: TransactionConfigInput): GenerateResult!
}

type Analytics {
    totalCustomers: Int!
    totalTransactions: Int!
    totalVolume: Float!
    fraudRate: Float!
    topMerchants(limit: Int): [MerchantStats!]!
}
```

---

### Example Queries

#### Get Customers with Transactions

```graphql
query GetCustomers {
    customers(limit: 10, city: "New York") {
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
```

---

#### Get Single Customer

```graphql
query GetCustomer {
    customer(customerId: "CUST_00001") {
        customerId
        name
        email
        phone
        city
        state
        transactions {
            transactionId
            amount
            merchant {
                name
                category
            }
            isFraud
        }
    }
}
```

---

#### Get Transactions with Filters

```graphql
query GetTransactions {
    transactions(
        limit: 100
        customerId: "CUST_00001"
        startDate: "2024-01-01"
        endDate: "2024-12-31"
    ) {
        transactionId
        amount
        timestamp
        customer {
            name
        }
        merchant {
            name
            category
        }
        fraudScore
        isFraud
    }
}
```

---

#### Get Analytics

```graphql
query GetAnalytics {
    analytics {
        totalCustomers
        totalTransactions
        totalVolume
        fraudRate
        topMerchants(limit: 5) {
            merchantId
            name
            transactionCount
            totalAmount
            avgAmount
        }
    }
}
```

---

### Example Mutations

#### Generate Customers

```graphql
mutation GenerateCustomers {
    generateCustomers(
        count: 1000
        demographics: {
            ageRange: { min: 25, max: 45 }
            incomeBracket: MIDDLE
            cities: ["New York", "Los Angeles"]
        }
    ) {
        success
        count
        message
    }
}
```

---

#### Generate Transactions

```graphql
mutation GenerateTransactions {
    generateTransactions(
        count: 10000
        config: {
            dateRange: {
                start: "2024-01-01"
                end: "2024-12-31"
            }
            fraudRate: 0.02
            patterns: {
                temporal: true
                geographic: true
            }
        }
    ) {
        success
        count
        message
    }
}
```

---

## WebSocket API

Endpoint: `ws://localhost:8000/ws`

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transactions');

ws.onopen = () => {
    console.log('Connected to WebSocket');
    
    // Subscribe to transaction stream
    ws.send(JSON.stringify({
        action: 'subscribe',
        filters: {
            fraud_score_min: 0.75
        }
    }));
};

ws.onmessage = (event) => {
    const transaction = JSON.parse(event.data);
    console.log('Received transaction:', transaction);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from WebSocket');
};
```

---

### Subscribe to Transactions

**Send:**
```json
{
    "action": "subscribe",
    "channel": "transactions",
    "filters": {
        "fraud_score_min": 0.75,
        "amount_min": 1000,
        "categories": ["retail", "dining"]
    }
}
```

**Receive:**
```json
{
    "type": "transaction",
    "data": {
        "transaction_id": "TXN_00001",
        "customer_id": "CUST_00001",
        "merchant_id": "MERCH_00001",
        "amount": 1500.00,
        "timestamp": "2024-11-04T12:30:00Z",
        "fraud_score": 0.85,
        "is_fraud": true
    },
    "timestamp": "2024-11-04T12:30:00Z"
}
```

---

### Unsubscribe

**Send:**
```json
{
    "action": "unsubscribe",
    "channel": "transactions"
}
```

---

### Real-time Fraud Alerts

**Subscribe:**
```json
{
    "action": "subscribe",
    "channel": "fraud_alerts",
    "filters": {
        "severity": "high"
    }
}
```

**Receive:**
```json
{
    "type": "fraud_alert",
    "data": {
        "alert_id": "ALERT_00001",
        "transaction_id": "TXN_00001",
        "severity": "high",
        "fraud_score": 0.95,
        "risk_factors": [
            "Unusual transaction amount",
            "New merchant",
            "Foreign IP address"
        ],
        "recommended_action": "block"
    },
    "timestamp": "2024-11-04T12:30:00Z"
}
```

---

## Python SDK

### Installation

```bash
pip install synfinance
```

### Core Modules

#### Generators

```python
from synfinance import CustomerGenerator, MerchantGenerator, TransactionGenerator
from synfinance.config import load_config

config = load_config()

# Customer Generator
customer_gen = CustomerGenerator(config)
customers = customer_gen.generate_batch(1000)

# Merchant Generator
merchant_gen = MerchantGenerator(config)
merchants = merchant_gen.generate_batch(500)

# Transaction Generator
txn_gen = TransactionGenerator(config, customers, merchants)
transactions = txn_gen.generate_batch(10000)
```

---

#### Database Manager

```python
from synfinance.database import DatabaseManager

db = DatabaseManager(config)

# Bulk insert
db.bulk_insert('customers', customers)
db.bulk_insert('merchants', merchants)
db.bulk_insert('transactions', transactions)

# Query
results = db.query("""
    SELECT customer_id, COUNT(*) as txn_count
    FROM transactions
    GROUP BY customer_id
    ORDER BY txn_count DESC
    LIMIT 10
""")

# Context manager
with DatabaseManager(config) as db:
    db.bulk_insert('transactions', transactions)
```

---

#### Fraud Detection

```python
from synfinance.fraud import FraudDetector, FraudPatternLibrary

# Fraud Detector
detector = FraudDetector(config)
scores = detector.score_batch(transactions)
predictions = detector.predict(transactions, threshold=0.75)

# Pattern Library
fraud_lib = FraudPatternLibrary(config)
fraud_txns = fraud_lib.generate_pattern(
    pattern='card_testing',
    count=100
)
```

---

#### Analytics

```python
from synfinance.analytics import StatisticalAnalyzer, Visualizer

# Statistical Analysis
analyzer = StatisticalAnalyzer(config)
stats = analyzer.describe(transactions)
correlations = analyzer.correlation_matrix(transactions)

# Visualization
viz = Visualizer(config)
viz.plot_time_series(transactions, output='chart.png')
viz.plot_fraud_distribution(transactions, output='fraud.png')
```

---

#### Reporting

```python
from synfinance.reporting import ReportGenerator

report_gen = ReportGenerator(config)

# Executive report
report_gen.create_executive_report(
    output_file='reports/executive.html',
    date_range=('2024-01-01', '2024-12-31')
)

# Fraud report
report_gen.create_fraud_report(
    output_file='reports/fraud.pdf',
    transactions=transactions
)
```

---

### Configuration

```python
from synfinance.config import load_config, validate_config

# Load config
config = load_config('config/production.yaml')

# Validate
validate_config(config)

# Access values
db_host = config['database']['host']
batch_size = config['generators']['default_batch_size']
```

---

### Error Handling

```python
from synfinance.exceptions import (
    GeneratorError,
    DatabaseError,
    ValidationError,
    ConfigError
)

try:
    customers = customer_gen.generate_batch(1000)
except GeneratorError as e:
    print(f"Generation failed: {e}")
except ValidationError as e:
    print(f"Invalid data: {e}")
```

---

### Performance Utilities

```python
from synfinance.performance import BatchProcessor, AsyncProcessor, Profiler

# Batch processing
processor = BatchProcessor(config, batch_size=10000)
processor.process_in_batches(large_dataset)

# Async processing
import asyncio
async_processor = AsyncProcessor(config)
results = asyncio.run(async_processor.generate_all())

# Profiling
profiler = Profiler(config)
with profiler.profile_memory():
    transactions = txn_gen.generate_batch(1000000)
```

---

### Integration

```python
from synfinance.integration import DataExporter, DataImporter

# Export
exporter = DataExporter(config)
exporter.to_csv(transactions, 'output/transactions.csv')
exporter.to_json(transactions, 'output/transactions.json')
exporter.to_parquet(transactions, 'output/transactions.parquet')

# Import
importer = DataImporter(config)
data = importer.from_csv('input/data.csv')
```

---

## API Rate Limits

| Endpoint | Rate Limit | Burst |
|----------|-----------|-------|
| /generate/* | 10 req/min | 20 |
| /customers | 100 req/min | 200 |
| /transactions | 100 req/min | 200 |
| /fraud/* | 50 req/min | 100 |
| GraphQL | 50 req/min | 100 |
| WebSocket | 1 connection/client | - |

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Support

For API support:
- **Documentation:** [docs.synfinance.io](https://docs.synfinance.io)
- **GitHub Issues:** [github.com/ssuptrey/SynFinance/issues](https://github.com/ssuptrey/SynFinance/issues)
- **Email:** support@synfinance.io

---

**Last Updated:** November 4, 2024  
**API Version:** 1.0.0
