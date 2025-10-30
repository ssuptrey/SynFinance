# SynFinance Project Structure

Complete directory structure and file organization for the SynFinance synthetic transaction data generator.

**Last Updated**: October 30, 2025
**Version**: 1.0.0 (Production/Stable)
**Status**: Week 7 Complete - Production-Grade Enterprise System

## Project Overview

SynFinance is a production-grade Python-based synthetic financial transaction data generator designed for the Indian market. It creates realistic customer profiles and transactions with advanced behavioral patterns, temporal dynamics, geographic consistency, merchant ecosystems, fraud detection, anomaly detection, ML features, database integration, resilience patterns, and professional CLI tools.

**Key Features**:
- 15 fraud pattern types with ML optimization
- 69 combined ML features (fraud + anomaly + interaction)
- SQLAlchemy 2.0 database integration with PostgreSQL
- Professional CLI with 20+ commands (Click + Rich UI)
- Resilience framework (Circuit Breaker, Retry, Rate Limiter, Health Checker)
- Prometheus monitoring + Grafana dashboards
- Multi-environment configuration management
- Automated quality assurance framework
- Structured logging and distributed tracing
- 800+ tests passing (96.9% success rate)

---

## Directory Structure

```
SynFinance/
├── src/                                # Source code (20,000+ lines)
│   ├── __init__.py
│   ├── app.py                          # Streamlit application
│   ├── config.py                       # Configuration settings
│   ├── constants.py                    # Global constants
│   ├── customer_generator.py          # Customer profile generation
│   ├── customer_profile.py            # Customer profile class
│   ├── data_generator.py              # Main data generation orchestrator
│   │
│   ├── analytics/                      # Advanced analytics (Week 6)
│   │   ├── __init__.py
│   │   ├── correlation_analyzer.py     # Correlation analysis
│   │   ├── feature_importance_analyzer.py  # Feature importance
│   │   ├── model_performance_analyzer.py   # Model metrics
│   │   └── statistical_tests.py        # Statistical testing
│   │
│   ├── api/                            # FastAPI server (Week 6)
│   │   ├── __init__.py
│   │   ├── app.py                      # API server
│   │   ├── client.py                   # API client
│   │   └── schemas.py                  # Pydantic schemas
│   │
│   ├── cli/                            # CLI tools (Week 7 Day 6 - 882 lines)
│   │   ├── __init__.py
│   │   ├── main_cli.py                 # Main CLI entry point (50 lines)
│   │   ├── generate_commands.py        # Data generation commands (240 lines)
│   │   ├── model_commands.py           # ML model commands (220 lines)
│   │   ├── database_commands.py        # Database commands (150 lines)
│   │   └── system_commands.py          # System commands (150 lines)
│   │
│   ├── config/                         # Configuration (Week 7 Day 2)
│   │   ├── __init__.py
│   │   ├── config_manager.py           # Configuration management
│   │   ├── environment.py              # Environment settings
│   │   ├── schema.py                   # Config schema validation
│   │   └── validator.py                # Config validators
│   │
│   ├── database/                       # Database layer (Week 7 Day 5 - 1,577 lines)
│   │   ├── __init__.py
│   │   ├── models.py                   # SQLAlchemy 2.0 models (620 lines)
│   │   ├── db_manager.py               # Database manager (580 lines)
│   │   └── repositories.py             # Repository pattern (580 lines)
│   │
│   ├── generators/                     # Specialized generators
│   │   ├── __init__.py
│   │   ├── advanced_schema_generator.py    # Advanced schema features
│   │   ├── anomaly_patterns.py         # Anomaly detection (Week 5)
│   │   ├── fraud_patterns.py           # Fraud detection (Week 4)
│   │   ├── geographic_generator.py     # Geographic patterns
│   │   ├── merchant_generator.py       # Merchant ecosystem
│   │   ├── temporal_generator.py       # Temporal patterns
│   │   └── transaction_core.py         # Core transaction logic
│   │
│   ├── ml/                             # ML framework (Week 5-6)
│   │   ├── __init__.py
│   │   ├── anomaly_features.py         # Anomaly ML features
│   │   ├── combined_features.py        # Combined features (69 total)
│   │   ├── dataset_preparation.py      # Dataset prep pipeline
│   │   ├── feature_generator.py        # Feature engineering
│   │   ├── fraud_features.py           # Fraud ML features
│   │   └── model_optimization.py       # Hyperparameter tuning, ensembles
│   │
│   ├── models/                         # Data models
│   │   ├── __init__.py
│   │   └── transaction.py              # Transaction data model
│   │
│   ├── monitoring/                     # Monitoring (Week 7 Day 1 - 4,500 lines)
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Metrics definitions
│   │   ├── prometheus_collector.py     # Prometheus integration
│   │   ├── grafana_dashboard.py        # Grafana dashboards
│   │   └── alerts.py                   # Alert rules
│   │
│   ├── observability/                  # Observability (Week 7 Day 4 - 2,216 lines)
│   │   ├── __init__.py
│   │   ├── logger.py                   # Structured logging
│   │   ├── tracer.py                   # Distributed tracing
│   │   ├── context.py                  # Context management
│   │   └── formatter.py                # Log formatters
│   │
│   ├── performance/                    # Performance (Week 6 Days 1-2)
│   │   ├── __init__.py
│   │   ├── parallel_generator.py       # Parallel generation (45K txn/sec)
│   │   ├── streaming_generator.py      # Streaming generation
│   │   ├── cache.py                    # LRU caching
│   │   └── benchmarks.py               # Performance benchmarking
│   │
│   ├── qa/                             # Quality Assurance (Week 7 Day 3 - 3,473 lines)
│   │   ├── __init__.py
│   │   ├── test_framework.py           # Testing framework
│   │   ├── data_validator.py           # Data validation
│   │   ├── quality_metrics.py          # Quality metrics
│   │   └── test_runner.py              # Test runner
│   │
│   ├── resilience/                     # Resilience Framework (Week 7 Day 7 - 1,441 lines)
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py          # Circuit Breaker pattern (368 lines)
│   │   ├── retry_handler.py            # Retry with exponential backoff (280 lines)
│   │   ├── rate_limiter.py             # Token bucket rate limiter (330 lines)
│   │   └── health_checker.py           # Kubernetes-style health probes (390 lines)
│   │
│   └── utils/                          # Utility modules
│       ├── __init__.py
│       ├── geographic_data.py          # City/region data
│       ├── indian_data.py              # Indian market data
│       └── merchant_data.py            # Merchant data
│
├── tests/                              # Test suite (800+ tests passing)
│   ├── __init__.py
│   ├── README.md                       # Test documentation
│   │
│   ├── analytics/                      # Analytics tests (Week 6)
│   │   ├── __init__.py
│   │   ├── test_correlation.py         # Correlation tests
│   │   ├── test_feature_importance.py  # Feature importance tests
│   │   ├── test_model_performance.py   # Model performance tests
│   │   └── test_statistical_tests.py   # Statistical tests
│   │
│   ├── api/                            # API tests (Week 6)
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_api_client.py
│   │   └── test_api_integration.py
│   │
│   ├── cli/                            # CLI tests (Week 7 Day 7)
│   │   ├── __init__.py
│   │   └── test_cli_commands.py        # 30 tests (13 passing, 17 import mocking limitations)
│   │
│   ├── config/                         # Configuration tests (Week 7 Day 2)
│   │   ├── __init__.py
│   │   └── test_config_*.py            # 42 tests
│   │
│   ├── database/                       # Database tests (Week 7 Day 5)
│   │   ├── __init__.py
│   │   └── test_database.py            # 14 tests (models, manager, repositories)
│   │
│   ├── deployment/                     # Deployment tests (Week 6)
│   │   ├── __init__.py
│   │   └── test_docker.py              # 18 tests (4 passing, 14 require Docker)
│   │
│   ├── generators/                     # Generator tests
│   │   ├── __init__.py
│   │   ├── test_advanced_schema.py     # 30 tests
│   │   ├── test_anomaly_patterns.py    # 30 tests
│   │   ├── test_fraud_patterns.py      # 100 tests
│   │   ├── test_geographic_patterns.py # 15 tests
│   │   ├── test_merchant_ecosystem.py  # 54 tests
│   │   └── test_temporal_patterns.py   # 18 tests
│   │
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   └── test_customer_integration.py  # 14 tests
│   │
│   ├── ml/                             # ML tests (Week 5)
│   │   ├── __init__.py
│   │   ├── test_anomaly_features.py    # 23 tests
│   │   ├── test_combined_features.py   # 21 tests
│   │   ├── test_fraud_features.py
│   │   └── test_dataset_preparation.py
│   │
│   ├── monitoring/                     # Monitoring tests (Week 7 Day 1)
│   │   ├── __init__.py
│   │   └── test_monitoring_*.py        # 85 tests
│   │
│   ├── observability/                  # Observability tests (Week 7 Day 4)
│   │   ├── __init__.py
│   │   └── test_observability_*.py     # 31 tests
│   │
│   ├── performance/                    # Performance tests (Week 6)
│   │   ├── __init__.py
│   │   ├── test_parallel.py
│   │   ├── test_streaming.py
│   │   └── test_cache.py
│   │
│   ├── qa/                             # QA tests (Week 7 Day 3)
│   │   ├── __init__.py
│   │   └── test_qa_*.py                # 74 tests
│   │
│   ├── test_resilience.py              # Resilience tests (Week 7 Day 7)
│   │                                   # 33 tests (Circuit Breaker, Retry, Rate Limiter, Health Checker)
│   │
│   └── unit/                           # Unit tests
│       ├── __init__.py
│       └── test_data_quality.py        # 13 tests
│
├── migrations/                         # Alembic migrations (Week 7 Day 5)
│   ├── env.py                          # Migration environment
│   ├── README                          # Migration documentation
│   ├── script.py.mako                  # Migration template
│   └── versions/                       # Migration versions
│
├── config/                             # Configuration files (Week 7 Day 2)
│   ├── default.yaml                    # Base configuration
│   ├── development.yaml                # Dev environment
│   ├── production.yaml                 # Production environment
│   ├── staging.yaml                    # Staging environment
│   ├── test.yaml                       # Test environment
│   └── schema.json                     # Config schema validation
│
├── monitoring/                         # Monitoring configuration (Week 7 Day 1)
│   ├── grafana/
│   │   └── dashboards/                 # Grafana dashboard JSON
│   │       └── synfinance_dashboard.json
│   └── prometheus/
│       └── prometheus.yml              # Prometheus configuration
│
├── deploy/                             # Deployment files (Week 6)
│   ├── docker/
│   │   ├── Dockerfile                  # Production Docker image
│   │   ├── Dockerfile.dev              # Development Docker image
│   │   └── docker-compose.yml          # Docker Compose orchestration
│   └── kubernetes/                     # Kubernetes manifests
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       ├── configmap.yaml
│       └── secrets.yaml
│
├── examples/                           # Example scripts
│   ├── README.md                       # Examples documentation
│   ├── analyze_anomaly_patterns.py
│   ├── analyze_fraud_patterns.py
│   ├── api_demo.py
│   ├── api_integration_example.py      # Week 6 API integration (380 lines)
│   ├── batch_processing_example.py     # Week 6 batch processing (430 lines)
│   ├── complete_ml_pipeline.py         # Week 6 complete pipeline (850 lines)
│   ├── demo_all_fraud_patterns.py
│   ├── demo_analytics_dashboard.py
│   ├── demo_geographic_patterns.py
│   ├── demo_merchant_ecosystem.py
│   ├── demo_observability.py
│   ├── demo_qa_framework.py
│   ├── fraud_detection_tutorial.ipynb  # Jupyter notebook tutorial
│   ├── fraud_detection_tutorial.py
│   ├── generate_anomaly_dataset.py
│   ├── generate_anomaly_ml_features.py
│   ├── generate_combined_features.py
│   ├── generate_fraud_training_data.py
│   ├── monitoring_demo.py
│   ├── optimize_fraud_models.py
│   ├── performance_demo.py
│   ├── real_time_monitoring.py         # Week 6 real-time monitoring (570 lines)
│   ├── run_customer_test.py
│   └── train_fraud_detector.py
│
├── scripts/                            # Utility scripts
│   ├── README.md                       # Scripts documentation
│   ├── analyze_correlations.py
│   ├── analyze_variance.py
│   ├── deploy.sh                       # Deployment script
│   ├── generate_week3_dataset.py
│   ├── health_check.sh                 # Health check script
│   ├── refactor_script.py             # Code refactoring utility
│   ├── rollback.sh                    # Rollback script
│   ├── run.bat                        # Windows runner
│   ├── run.sh                         # Unix/Linux runner
│   └── validate_data_quality.py
│
├── docs/                               # Documentation (50+ documents)
│   ├── INDEX.md                       # Documentation index
│   ├── ORGANIZATION.md                # Documentation organization
│   ├── RECOVERY_REPORT_OCT21.md       # Recovery documentation
│   ├── STRUCTURE.md                   # Structure documentation
│   │
│   ├── guides/                        # User guides
│   │   ├── INTEGRATION_GUIDE.md       # API integration guide
│   │   ├── QUICK_REFERENCE.md         # Quick reference
│   │   ├── QUICKSTART.md              # 5-minute quickstart
│   │   └── WEEK1_GUIDE.md             # Week 1 tutorial
│   │
│   ├── technical/                     # Technical documentation
│   │   ├── ARCHITECTURE.md            # System architecture
│   │   ├── CHANGES.md                 # Change log
│   │   ├── CUSTOMER_SCHEMA.md         # Customer schema reference
│   │   ├── DESIGN_GUIDE.md            # Design patterns
│   │   ├── FIELD_REFERENCE.md         # Field reference (50 fields)
│   │   │
│   │   ├── deployment/                # Deployment documentation (Week 6)
│   │   │   └── PRODUCTION_GUIDE.md    # Production deployment guide (1,100 lines)
│   │   │
│   │   ├── fraud/                     # Fraud documentation (Week 4)
│   │   │   ├── FRAUD_PATTERNS.md
│   │   │   └── FRAUD_TECHNICAL.md
│   │   │
│   │   └── ml/                        # ML documentation (Week 5)
│   │       ├── ML_FEATURES.md
│   │       └── ANOMALY_FEATURES.md
│   │
│   ├── progress/                      # Progress reports
│   │   ├── README.md                  # Progress documentation index
│   │   ├── WEEK1_COMPLETION_SUMMARY.md
│   │   ├── WEEK2_DAY1-2_SUMMARY.md
│   │   ├── WEEK2_DAY3-4_SUMMARY.md
│   │   ├── WEEK2_DAY5-7_SUMMARY.md
│   │   ├── WEEK3_DAY1_COMPLETE.md
│   │   ├── WEEK3_DAY2-3_ANALYSIS.md
│   │   ├── WEEK3_DAY2-3_COMPLETE.md
│   │   ├── WEEK4_DAY1-2_COMPLETE.md
│   │   ├── WEEK4_DAY3-4_COMPLETE.md
│   │   ├── WEEK5_DAY1-7_COMPLETE.md
│   │   ├── WEEK6_DAY1-2_COMPLETE.md
│   │   ├── WEEK6_DAY3-4_COMPLETE.md
│   │   ├── WEEK6_DAY5_COMPLETE.md
│   │   ├── WEEK6_DAY6_COMPLETE.md
│   │   ├── WEEK6_DAY7_COMPLETE.md
│   │   ├── WEEK7_DAY5_COMPLETE.md     # Database integration (Week 7)
│   │   ├── WEEK7_DAY6_COMPLETE.md     # CLI tools (Week 7)
│   │   └── WEEK7_COMPLETE.md          # Week 7 comprehensive summary
│   │
│   └── planning/                      # Planning documents
│       ├── ROADMAP.md                 # 12-week roadmap
│       ├── BUSINESS_PLAN.md           # Business strategy
│       ├── ASSESSMENT_SUMMARY.md
│       └── WEEK7_DETAILED_PLAN.md
│
├── data/                               # Sample data (gitignored)
├── output/                             # Generated outputs (gitignored)
│   ├── analytics/                     # Analytics output
│   ├── combined_features/             # Combined features
│   └── qa_demo/                       # QA demo output
│
├── models/                             # Trained ML models (gitignored)
│
├── alembic.ini                         # Alembic configuration (Week 7)
├── docker-compose.yml                  # Docker Compose (Week 6)
├── Dockerfile                          # Production Dockerfile (Week 6)
├── Dockerfile.dev                      # Development Dockerfile (Week 6)
├── .dockerignore                       # Docker ignore file
├── .github/                            # GitHub Actions CI/CD (Week 6)
│   └── workflows/
│       ├── test.yml                    # Test workflow
│       ├── build.yml                   # Build workflow
│       └── deploy.yml                  # Deploy workflow
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── setup.cfg                           # Setup configuration
├── pyproject.toml                      # Project configuration
├── pytest.ini                          # Pytest configuration
├── MANIFEST.in                         # Package manifest
├── README.md                           # Project README
├── LICENSE                             # License file
├── CHANGELOG.md                        # Change log
├── CONTRIBUTING.md                     # Contribution guidelines
├── PROJECT_STRUCTURE.md                # This file
└── DOCUMENTATION_COMPLETE.md           # Documentation status
│   │   ├── ASSESSMENT_SUMMARY.md      # (empty)
│   │   ├── BUSINESS_PLAN.md           # (empty)
│   │   └── ROADMAP.md                 # (empty)
│   │
│   └── archive/                       # Archived documentation
│       ├── README.md                  # (empty)
│       ├── PROJECT_STRUCTURE.md       # Old structure (29KB)
│       ├── PROJECT_VALIDATION.md      # (empty)
│       ├── README.md                  # (empty)
│       └── REFACTORING_COMPLETE.md    # (empty)
│
├── data/                              # Data files (empty)
├── output/                            # Generated output
│   └── customer_validation_stats.json
│
├── .venv/                             # Virtual environment
├── __pycache__/                       # Python cache
│
├── CONTRIBUTING.md                    # Contribution guidelines
├── DOCUMENTATION_COMPLETE.md          # Documentation status
├── LICENSE                            # MIT License
├── PROJECT_STRUCTURE.md               # This file
├── README.md                          # Project README
└── requirements.txt                   # Python dependencies
```

---

## Key Files

### Source Code

#### Main Entry Points
- **`src/app.py`** - Main application entry point (Streamlit UI)
- **`src/data_generator.py`** - High-level data generation API
- **`src/config.py`** - Global configuration settings

#### Customer Generation
- **`src/customer_profile.py`** - CustomerProfile class (23 fields, 5 enums)
- **`src/customer_generator.py`** - Customer generation logic

#### Transaction Generation
- **`src/generators/transaction_core.py`** - Core transaction generation
- **`src/generators/temporal_generator.py`** - Temporal patterns (Week 2 Day 1-2)
- **`src/generators/geographic_generator.py`** - Geographic patterns (Week 2 Day 3-4)
- **`src/generators/merchant_generator.py`** - Merchant ecosystem (Week 2 Day 5-7)
- **`src/generators/advanced_schema_generator.py`** - Advanced features (Week 3)

#### Data & Utilities
- **`src/utils/indian_data.py`** - Indian market data (names, occupations, etc.)
- **`src/utils/geographic_data.py`** - City/region data (20 cities, 3 tiers)
- **`src/utils/merchant_data.py`** - Merchant data (40+ chains, categories)
- **`src/models/transaction.py`** - Transaction data model

### Tests

#### Integration Tests
- **`tests/integration/test_customer_integration.py`** - Week 1 integration (14 tests)

#### Generator Tests
- **`tests/generators/test_temporal_patterns.py`** - Week 2 Day 1-2 (18 tests)
- **`tests/generators/test_geographic_patterns.py`** - Week 2 Day 3-4 (15 tests)
- **`tests/generators/test_merchant_ecosystem.py`** - Week 2 Day 5-7 (21 tests)

**Total:** 68 tests, 68 passing (100%)

### Documentation

#### Getting Started
- **`docs/guides/QUICKSTART.md`** - 5-minute quickstart guide
- **`docs/guides/INTEGRATION_GUIDE.md`** - API integration guide
- **`docs/guides/QUICK_REFERENCE.md`** - Quick reference with code snippets

#### Technical
- **`docs/technical/ARCHITECTURE.md`** - System architecture overview
- **`docs/technical/CUSTOMER_SCHEMA.md`** - Complete customer schema reference
- **`docs/technical/WEEK*_SUMMARY.md`** - Weekly implementation summaries

#### Progress
- **`docs/progress/WEEK*_COMPLETE.md`** - Detailed weekly progress reports
- **`docs/progress/README.md`** - Progress documentation index

---

## File Statistics

### Source Code
- **Lines of Code:** ~8,500 (excluding tests)
- **Python Files:** 18
- **Modules:** 4 main (customer, transaction, generators, utils)

### Tests
- **Test Files:** 7
- **Test Cases:** 68 (100% passing)
- **Lines of Test Code:** ~4,200

### Documentation
- **Markdown Files:** 30
- **Total Documentation:** 152+ KB
- **With Content:** 19 files (63%)
- **Empty/Placeholder:** 11 files (37%)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                     User Application                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              data_generator.generate_dataset()          │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────────┐  ┌────────────────────────────────┐
│ CustomerGenerator    │  │  TransactionGenerator          │
│ .generate_customers()│  │  .generate_transactions()      │
└──────────┬───────────┘  └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ TemporalPatternGenerator       │
           │              │ .apply_temporal_multiplier()   │
           │              └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ GeographicPatternGenerator     │
           │              │ .select_transaction_city()     │
           │              └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ MerchantEcosystemGenerator     │
           │              │ .select_merchant()             │
           │              └────────┬───────────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────────────────────────────────────────────┐
│              Customer Profiles + Transactions            │
│                 (pandas DataFrame)                       │
└──────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
data_generator
├── customer_generator
│   ├── customer_profile
│   └── utils/indian_data
│
└── generators/transaction_core
    ├── generators/temporal_generator
    │   └── utils/indian_data (festivals)
    │
    ├── generators/geographic_generator
    │   └── utils/geographic_data
    │
    ├── generators/merchant_generator
    │   └── utils/merchant_data
    │
    └── generators/advanced_schema_generator
        └── models/transaction
```

---

## Configuration

### Environment Variables
```bash
PYTHONPATH=e:\SynFinance\src
```

### Dependencies (requirements.txt)
```
streamlit>=1.28.0
pandas>=2.0.0
faker>=20.0.0
numpy>=1.24.0
xlsxwriter>=3.1.0
pytest>=7.4.0 (dev)
```

---

## Version History

### v0.3.0 (Week 2 Complete - October 2025)
- ✅ Temporal patterns (18 tests)
- ✅ Geographic patterns (15 tests)
- ✅ Merchant ecosystem (21 tests)
- ✅ Advanced schema features
- ✅ 68/68 tests passing

### v0.2.0 (Week 1 Complete - October 2025)
- ✅ Customer profile generation (23 fields)
- ✅ Transaction integration (14 tests)
- ✅ Indian market patterns
- ✅ Data validation

### v0.1.0 (Initial Release)
- Basic customer and transaction generation
- Streamlit UI
- CSV/Excel export

---

## Future Additions

### Planned Features
- Unit test suite (`tests/unit/`)
- Planning documentation (`docs/planning/`)
- Additional example scripts
- Performance optimization tools
- Data visualization utilities

### Planned Documentation
- `docs/guides/WEEK1_GUIDE.md` - Detailed Week 1 tutorial
- `docs/technical/CHANGES.md` - Complete change log
- `docs/technical/DESIGN_GUIDE.md` - Design patterns guide
- `docs/planning/ROADMAP.md` - Product roadmap
- `docs/planning/BUSINESS_PLAN.md` - Market strategy

---

## Contributing

When adding new files:
1. Follow the established directory structure
2. Update this document with new file locations
3. Add appropriate documentation in `docs/`
4. Write tests in `tests/` matching the file structure
5. Update `README.md` if adding user-facing features

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

**Last Updated:** October 21, 2025  
**Project Status:** Active Development  
**Test Coverage:** 68/68 tests passing (100%)
