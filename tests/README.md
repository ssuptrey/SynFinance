# SynFinance Test Suite

This directory contains all unit and integration tests for the SynFinance synthetic transaction data generation system.

**Current Status:** 211 tests (209 passing - 98.9% pass rate)

## Directory Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── fraud/                      # Fraud pattern detection tests (100 tests)
│   │   ├── test_base_patterns.py           # 10 base fraud patterns (26 tests)
│   │   ├── test_advanced_patterns.py       # 5 advanced patterns (29 tests)
│   │   ├── test_combinations.py            # Combination system (13 tests)
│   │   ├── test_network_analysis.py        # Network analysis (22 tests)
│   │   └── test_cross_pattern_stats.py     # Cross-pattern stats (10 tests)
│   │
│   ├── data_quality/               # Data quality validation tests (13 tests)
│   │   ├── test_variance.py                # Column variance tests
│   │   └── test_geographic_variance.py     # Geographic distribution tests
│   │
│   └── test_customer_generation.py # Customer profile generation (0 tests)
│
├── generators/                     # Generator component tests (84 tests)
│   ├── test_advanced_schema.py             # Advanced schema features (30 tests)
│   ├── test_geographic_patterns.py         # Geographic patterns (15 tests)
│   ├── test_merchant_ecosystem.py          # Merchant system (21 tests)
│   └── test_temporal_patterns.py           # Temporal patterns (18 tests)
│
├── integration/                    # Integration tests (14 tests)
│   └── test_customer_integration.py        # End-to-end customer workflow
│
└── README.md                       # This file
```

## Test Categories

### 1. Fraud Detection Tests (`unit/fraud/`)

**Purpose:** Validate fraud pattern detection, combination, and analysis

**Files:**
- `test_base_patterns.py` - 10 base fraud patterns (Card Cloning, Account Takeover, etc.)
- `test_advanced_patterns.py` - 5 advanced patterns (Transaction Replay, Card Testing, etc.)
- `test_combinations.py` - Chained, coordinated, and progressive fraud
- `test_network_analysis.py` - Fraud rings and temporal clustering
- `test_cross_pattern_stats.py` - Co-occurrence and isolation statistics

**Coverage:** 100 tests covering all 15 fraud patterns and analysis systems

### 2. Data Quality Tests (`unit/data_quality/`)

**Purpose:** Ensure synthetic data meets quality and variance requirements

**Files:**
- `test_variance.py` - Column variance, distribution, and diversity
- `test_geographic_variance.py` - Geographic distribution validation

**Coverage:** 13 tests ensuring realistic data generation

### 3. Generator Tests (`generators/`)

**Purpose:** Test transaction generation components

**Files:**
- `test_advanced_schema.py` - Card types, transaction status, device info
- `test_geographic_patterns.py` - City tiers, cost of living, merchant density
- `test_merchant_ecosystem.py` - Merchant IDs, loyalty, subcategories
- `test_temporal_patterns.py` - Time patterns, salary cycles, festivals

**Coverage:** 84 tests for all generation components

### 4. Integration Tests (`integration/`)

**Purpose:** Test end-to-end workflows

**Files:**
- `test_customer_integration.py` - Customer profile to transaction pipeline

**Coverage:** 14 tests for complete system integration

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run by Category
```bash
# Fraud detection tests only
pytest tests/unit/fraud/ -v

# Data quality tests only
pytest tests/unit/data_quality/ -v

# Generator tests only
pytest tests/generators/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Run Specific Test File
```bash
# Base fraud patterns
pytest tests/unit/fraud/test_base_patterns.py -v

# Advanced patterns
pytest tests/unit/fraud/test_advanced_patterns.py -v

# Network analysis
pytest tests/unit/fraud/test_network_analysis.py -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage by module
pytest tests/ --cov=src --cov-report=term-missing
```

### Run Tests Matching Pattern
```bash
# Run all fraud-related tests
pytest tests/ -k "fraud" -v

# Run all combination tests
pytest tests/ -k "combination" -v

# Run all network tests
pytest tests/ -k "network" -v
```

## Test Naming Conventions

### File Naming
- `test_<component>.py` - Tests for a specific component
- Descriptive names indicating what's being tested
- Grouped by functionality in subdirectories

### Class Naming
- `TestClassName` - Groups related tests
- Descriptive class names (e.g., `TestTransactionReplayPattern`)

### Method Naming
- `test_<specific_behavior>` - Describes what's being tested
- Clear, descriptive names (e.g., `test_replay_detection_with_similar_transaction`)

## Coverage Goals

- **Overall Target:** 95%+ code coverage
- **Critical Paths:** 100% coverage (fraud detection, data generation)
- **Edge Cases:** All error conditions validated
- **Integration:** End-to-end workflows tested

**Current Coverage:**
- Fraud patterns: 100%
- Network analysis: 100%
- Generators: 95%+
- Overall: 98.9% test pass rate

## Adding New Tests

### For New Fraud Patterns
1. Add tests to `tests/unit/fraud/test_advanced_patterns.py`
2. Include pattern initialization, detection logic, confidence calculation
3. Test edge cases and error handling

### For New Generators
1. Add tests to appropriate file in `tests/generators/`
2. Test configuration, output format, and data quality

### For Integration Scenarios
1. Add tests to `tests/integration/`
2. Test complete workflows from input to output

## Test Fixtures

Common fixtures available across all tests:
- `DummyCustomer` - Mock customer profiles
- `create_transaction()` - Generate test transactions
- `create_history()` - Generate transaction history

## Continuous Integration

Tests are automatically run on:
- Every commit to main branch
- All pull requests
- Nightly builds

**CI Requirements:**
- All tests must pass (≥95% pass rate)
- Coverage must not decrease
- No new linting errors

## Troubleshooting

### Import Errors
Ensure you're running from project root:
```bash
cd E:\SynFinance
pytest tests/
```

### Test Discovery Issues
Verify `__init__.py` exists in all test directories.

### Slow Tests
Run specific test files instead of full suite during development.

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass before committing
3. Update this README if adding new test categories
4. Maintain or improve coverage percentage

---

**Last Updated:** October 26, 2025  
**Total Tests:** 211  
**Pass Rate:** 98.9% (209/211)  
**Coverage:** 95%+
