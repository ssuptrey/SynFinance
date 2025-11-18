# SynFinance - Actual Module Reference

**Last Verified:** November 4, 2024  
**Purpose:** Documents the ACTUAL implemented modules and their real import paths

---

## Verified Working Modules

### Configuration (`src/config/`)

```python
from src.config import ConfigManager, AppConfig

# Load configuration
config_manager = ConfigManager()
config = config_manager.load()  # Loads from config/default.yaml
```

**Available Classes:**
- `ConfigManager` - Main configuration manager
- `AppConfig` - Application configuration model
- `ServerConfig` - Server settings
- `DatabaseConfig` - Database connection settings
- `CacheConfig` - Cache configuration
- `GenerationConfig` - Data generation settings
- `MLConfig` - Machine learning settings
\
- `MonitoringConfig` - Monitoring and logging
- `SecurityConfig` - Security settings
- `EnvLoader` - Environment variable loader
- `ConfigWatcher` - Hot-reload configuration watcher

---

### Generators (`src/generators/`)

**Verified Working:**

```python
# Merchant Generator
from src.generators.merchant_generator import MerchantGenerator
merchant_gen = MerchantGenerator()
merchants = merchant_gen.generate(count=100)

# Transaction Generator
from src.generators.transaction_core import TransactionGenerator
txn_gen = TransactionGenerator()
transactions = txn_gen.generate(count=1000)

# Geographic Generator
from src.generators.geographic_generator import GeographicGenerator
geo_gen = GeographicGenerator()

# Temporal Generator
from src.generators.temporal_generator import TemporalGenerator
temp_gen = TemporalGenerator()

# Fraud Patterns
from src.generators.fraud_patterns import FraudPatternGenerator
fraud_gen = FraudPatternGenerator()

# ML Features
from src.generators.ml_features import MLFeatureGenerator
feature_gen = MLFeatureGenerator()
```

**Available Modules:**
- `merchant_generator.py` - Generate merchant profiles
- `transaction_core.py` - Core transaction generation
- `geographic_generator.py` - Geographic patterns
- `temporal_generator.py` - Temporal patterns
- `fraud_patterns.py` - Fraud pattern generation
- `fraud_network.py` - Fraud network analysis
- `ml_features.py` - ML feature engineering
- `combined_ml_features.py` - Combined feature sets
- `anomaly_patterns.py` - Anomaly detection patterns
- `anomaly_ml_features.py` - Anomaly ML features
- `advanced_schema_generator.py` - Advanced schema generation
- `ml_dataset_generator.py` - ML dataset generation

---

### Fraud Detection (`src/fraud/`)

**Verified Working:**

```python
# Fraud Scoring Engine
from src.fraud.scoring_engine import FraudScoringEngine
scorer = FraudScoringEngine()
score = scorer.score_transaction(transaction)

# Pattern Detector
from src.fraud.pattern_detector import PatternDetector
detector = PatternDetector()

# Behavioral Analyzer
from src.fraud.behavioral_analyzer import BehavioralAnalyzer
analyzer = BehavioralAnalyzer()

# Velocity Checker
from src.fraud.velocity_checker import VelocityChecker
velocity = VelocityChecker()

# Decision Engine
from src.fraud.decision_engine import DecisionEngine
decision = DecisionEngine()
```

**Available Modules:**
- `scoring_engine.py` - Fraud scoring
- `pattern_detector.py` - Pattern detection
- `behavioral_analyzer.py` - Behavioral analysis
- `velocity_checker.py` - Velocity checking
- `decision_engine.py` - Decision making
- `model_deployer.py` - Model deployment

---

### Database (`src/database/`)

**Verified Working:**

```python
# Database Manager
from src.database.db_manager import DatabaseManager
db = DatabaseManager()
db.connect()
db.bulk_insert('transactions', transactions)
db.disconnect()

# Models (SQLAlchemy ORM)
from src.database.models import Customer, Merchant, Transaction

# Repositories
from src.database.repositories import CustomerRepository, TransactionRepository
```

**Available Modules:**
- `db_manager.py` - Database connection and operations
- `models.py` - SQLAlchemy ORM models
- `repositories.py` - Data access layer

---

### Analytics (`src/analytics/`)

**Verified Working:**

```python
# Statistical Analyzer
from src.analytics.statistical_analyzer import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze(transactions)

# Correlation Analyzer
from src.analytics.correlation_analyzer import CorrelationAnalyzer
corr = CorrelationAnalyzer()

# Visualization
from src.analytics.visualization import Visualizer
viz = Visualizer()

# Data Profiler
from src.analytics.data_profiler import DataProfiler
profiler = DataProfiler()

# Trend Analyzer
from src.analytics.trend_analyzer import TrendAnalyzer
trends = TrendAnalyzer()
```

**Available Modules:**
- `statistical_analyzer.py` - Statistical analysis
- `correlation_analyzer.py` - Correlation analysis
- `visualization.py` - Data visualization
- `data_profiler.py` - Data profiling
- `trend_analyzer.py` - Trend analysis
- `advanced_analytics.py` - Advanced analytics
- `distribution_fitter.py` - Distribution fitting
- `statistical_tests.py` - Statistical tests
- `dashboard.py` - Analytics dashboard
- `analysis_report.py` - Analysis reporting

---

### Performance (`src/performance/`)

**Verified Working:**

```python
# Batch Processor
from src.performance.optimizer import BatchProcessor
processor = BatchProcessor(batch_size=10000)

# Query Optimizer
from src.performance.query_optimizer import QueryOptimizer
optimizer = QueryOptimizer()

# Profiler
from src.performance.profiler import Profiler
profiler = Profiler()

# Metrics Collector
from src.performance.metrics_collector import MetricsCollector
metrics = MetricsCollector()

# Load Tester
from src.performance.load_tester import LoadTester
tester = LoadTester()
```

**Available Modules:**
- `optimizer.py` - Batch processing and optimization
- `query_optimizer.py` - Query optimization
- `profiler.py` - Performance profiling
- `metrics_collector.py` - Metrics collection
- `load_tester.py` - Load testing
- `cache_manager.py` - Cache management
- `parallel_generator.py` - Parallel data generation
- `streaming_generator.py` - Streaming generation
- `benchmarks.py` - Performance benchmarks

---

### Reporting (`src/reporting/`)

**Verified Working:**

```python
# HTML Report Generator
from src.reporting.html_generator import HTMLReportGenerator
html_gen = HTMLReportGenerator()
html_gen.generate_report(data, output='report.html')

# PDF Exporter
from src.reporting.pdf_exporter import PDFExporter
pdf = PDFExporter()

# Excel Generator
from src.reporting.excel_generator import ExcelGenerator
excel = ExcelGenerator()

# Dataset Comparator
from src.reporting.dataset_comparator import DatasetComparator
comparator = DatasetComparator()
```

**Available Modules:**
- `html_generator.py` - HTML report generation
- `pdf_exporter.py` - PDF export
- `excel_generator.py` - Excel report generation
- `dataset_comparator.py` - Dataset comparison
- `templates/` - Report templates directory

---

### API (`src/api/`)

**Verified Working:**

```python
# FastAPI Application
import src.api.app
# Run with: uvicorn src.api.app:app --reload

# API Server
from src.api.api_server import APIServer
server = APIServer()

# Health Checks
from src.api.health import health_check

# Metrics
from src.api.metrics import MetricsEndpoint
```

**Available Modules:**
- `app.py` - Main FastAPI application
- `api_server.py` - API server setup
- `health.py` - Health check endpoints
- `metrics.py` - Metrics endpoints
- `middleware.py` - API middleware
- `logging_config.py` - Logging configuration
- `tracing.py` - Distributed tracing
- `fraud_detection_api.py` - Fraud detection API endpoints
- `api_client.py` - API client
- `graphql/` - GraphQL API (directory)
- `websocket/` - WebSocket API (directory)
- `tenants/` - Multi-tenancy support (directory)
- `versioning/` - API versioning (directory)

---

### Machine Learning (`src/ml/`)

```python
# Base Model
from src.ml.base_model import BaseMLModel

# Model Registry
from src.ml.model_registry import ModelRegistry
registry = ModelRegistry()

# Model Optimization
from src.ml.model_optimization import ModelOptimizer
optimizer = ModelOptimizer()
```

**Available Modules:**
- `base_model.py` - Base ML model class
- `model_registry.py` - Model registration and versioning
- `model_optimization.py` - Model optimization
- `ensemble/` - Ensemble models (directory)
- `models/` - Specific ML models (directory)

---

### Other Modules

**Utils:**
- `src/utils/` - Utility functions

**Quality:**
- `src/quality/` - Data quality checks

**Monitoring:**
- `src/monitoring/` - System monitoring

**Observability:**
- `src/observability/` - Observability features

**Resilience:**
- `src/resilience/` - Resilience patterns

**Tenancy:**
- `src/tenancy/` - Multi-tenancy support

**Visualizations:**
- `src/visualizations/` - Visualization utilities

**CLI:**
- `src/cli/` - Command-line interface

---

## Correct Usage Examples

### Generate Complete Dataset

```python
from src.config import ConfigManager
from src.generators.merchant_generator import MerchantGenerator
from src.generators.transaction_core import TransactionGenerator
from src.database.db_manager import DatabaseManager

# Load config
config_mgr = ConfigManager()
config = config_mgr.load()

# Generate merchants
merchant_gen = MerchantGenerator()
merchants = merchant_gen.generate(count=500)

# Generate transactions
txn_gen = TransactionGenerator()
transactions = txn_gen.generate(count=10000)

# Save to database
db = DatabaseManager()
db.connect()
db.bulk_insert('merchants', merchants)
db.bulk_insert('transactions', transactions)
db.disconnect()
```

### Fraud Detection

```python
from src.fraud.scoring_engine import FraudScoringEngine
from src.fraud.pattern_detector import PatternDetector

# Initialize fraud detection
scorer = FraudScoringEngine()
detector = PatternDetector()

# Score transactions
for txn in transactions:
    score = scorer.score_transaction(txn)
    patterns = detector.detect_patterns(txn)
    
    if score > 0.75:
        print(f"High fraud risk: {txn['id']}, Score: {score}")
```

### Analytics and Reporting

```python
from src.analytics.statistical_analyzer import StatisticalAnalyzer
from src.reporting.html_generator import HTMLReportGenerator

# Analyze data
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze(transactions)

# Generate report
report_gen = HTMLReportGenerator()
report_gen.generate_report(stats, output='analysis_report.html')
```

### Performance Optimization

```python
from src.performance.optimizer import BatchProcessor
from src.performance.profiler import Profiler

# Use batch processing for large datasets
processor = BatchProcessor(batch_size=10000)

with Profiler() as prof:
    for batch in processor.process_in_batches(large_dataset):
        # Process batch
        pass

print(f"Peak memory: {prof.peak_memory_mb}MB")
```

---

## Notes

1. **Config Loading:** Use `ConfigManager().load()` not `load_config()`
2. **Database:** Use `DatabaseManager` from `src.database.db_manager`
3. **Generators:** Import from specific modules in `src.generators/`
4. **API:** Main app is in `src.api.app` - run with uvicorn
5. **All imports verified:** Every import listed above has been tested and works

---

## Missing/Not Implemented

These were mentioned in docs but don't exist yet:
- `src.core.config.load_config()` - Use `ConfigManager` instead
- `CustomerGenerator` class - Use merchant/transaction generators
- Simple `load_config()` function - Use `ConfigManager().load()`
- `from synfinance import ...` - Not packaged yet, use `from src...`

---

**Use this reference for accurate code examples and imports!**
