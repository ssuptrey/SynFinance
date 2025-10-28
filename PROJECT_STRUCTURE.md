# SynFinance Project Structure

Complete directory structure and file organization for the SynFinance synthetic transaction data generator.

## Project Overview

SynFinance is a Python-based synthetic financial transaction data generator designed for the Indian market. It creates realistic customer profiles and transactions with advanced behavioral patterns, temporal dynamics, geographic consistency, and merchant ecosystems.

---

## Directory Structure

```
SynFinance/
├── src/                                # Source code
│   ├── __init__.py
│   ├── app.py                          # Main application entry point
│   ├── config.py                       # Configuration settings
│   ├── customer_generator.py          # Customer profile generation
│   ├── customer_profile.py            # Customer profile class
│   ├── data_generator.py              # Main data generation orchestrator
│   │
│   ├── generators/                     # Specialized generators
│   │   ├── __init__.py
│   │   ├── advanced_schema_generator.py    # Advanced schema features
│   │   ├── geographic_generator.py         # Geographic patterns
│   │   ├── merchant_generator.py           # Merchant ecosystem
│   │   ├── temporal_generator.py           # Temporal patterns
│   │   └── transaction_core.py             # Core transaction logic
│   │
│   ├── models/                         # Data models
│   │   ├── __init__.py
│   │   └── transaction.py              # Transaction data model
│   │
│   └── utils/                          # Utility modules
│       ├── __init__.py
│       ├── geographic_data.py          # City/region data
│       ├── indian_data.py              # Indian market data
│       └── merchant_data.py            # Merchant data
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── README.md                       # Test documentation
│   ├── test_customer_generation.py    # Legacy tests
│   │
│   ├── unit/                           # Unit tests
│   │   └── __init__.py
│   │
│   ├── generators/                     # Generator tests
│   │   ├── __init__.py
│   │   ├── test_geographic_patterns.py
│   │   ├── test_merchant_ecosystem.py
│   │   └── test_temporal_patterns.py
│   │
│   └── integration/                    # Integration tests
│       ├── __init__.py
│       └── test_customer_integration.py
│
├── examples/                           # Example scripts
│   ├── README.md                       # Examples documentation
│   ├── demo_geographic_patterns.py
│   ├── demo_merchant_ecosystem.py
│   └── run_customer_test.py
│
├── scripts/                            # Utility scripts
│   ├── README.md                       # Scripts documentation
│   ├── refactor_script.py             # Code refactoring utility
│   ├── run.bat                        # Windows runner
│   └── run.sh                         # Unix/Linux runner
│
├── docs/                               # Documentation
│   ├── INDEX.md                       # Documentation index
│   ├── ORGANIZATION.md                # Documentation organization
│   ├── RECOVERY_REPORT_OCT21.md       # Recovery documentation
│   │
│   ├── guides/                        # User guides
│   │   ├── INTEGRATION_GUIDE.md       # API integration guide
│   │   ├── QUICK_REFERENCE.md         # Quick reference
│   │   ├── QUICKSTART.md              # 5-minute quickstart
│   │   └── WEEK1_GUIDE.md             # Week 1 tutorial (empty)
│   │
│   ├── technical/                     # Technical documentation
│   │   ├── ARCHITECTURE.md            # System architecture
│   │   ├── CHANGES.md                 # Change log (empty)
│   │   ├── CUSTOMER_SCHEMA.md         # Customer schema reference
│   │   ├── DESIGN_GUIDE.md            # Design patterns (empty)
│   │   ├── WEEK1_COMPLETION_SUMMARY.md
│   │   ├── WEEK1_PROGRESS.md          # (empty)
│   │   ├── WEEK2_DAY1-2_SUMMARY.md
│   │   ├── WEEK2_DAY3-4_SUMMARY.md
│   │   └── WEEK2_DAY5-7_SUMMARY.md
│   │
│   ├── progress/                      # Progress reports
│   │   ├── README.md                  # Progress documentation index
│   │   ├── EMOJI_REMOVAL_COMPLETE.md
│   │   ├── STRUCTURE_REORGANIZATION_COMPLETE.md
│   │   ├── WEEK2_DAY3-4_COMPLETE.md
│   │   ├── WEEK3_DAY1_COMPLETE.md
│   │   └── WEEK3_DAY1_PROGRESS.md
│   │
│   ├── planning/                      # Planning documents
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
