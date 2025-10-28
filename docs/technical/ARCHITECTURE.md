# SynFinance Modular Architecture Documentation

**Date:** October 21, 2025
**Status:** COMPLETE

---

## Overview

SynFinance is designed as a modular, extensible synthetic financial data generator for the Indian market. The architecture supports:
- Realistic customer profiles
- Advanced transaction schema (43 fields)
- Geographic, temporal, and merchant ecosystem modeling
- Risk indicators and state tracking
- Comprehensive test coverage

---

## Folder Structure

```
src/
  app.py                # Main entry point
  config.py             # Configuration constants
  customer_generator.py # Customer profile generation
  customer_profile.py   # Customer schema and segments
  data_generator.py     # Orchestrates data generation
  generators/
    advanced_schema_generator.py   # Week 3: 43-field schema
    geographic_generator.py        # Week 2: Geographic patterns
    merchant_generator.py          # Week 2: Merchant ecosystem
    temporal_generator.py          # Week 2: Temporal patterns
    transaction_core.py            # Transaction logic
    __init__.py
  models/
    transaction.py      # Transaction dataclass (43 fields)
    __init__.py
  utils/
    geographic_data.py  # City/state/region mappings
    indian_data.py      # Merchant/category/city tier data
    merchant_data.py    # Merchant lists and details
    __init__.py
```

---

## Key Modules

### 1. Customer Profile System
- **File:** `customer_profile.py`
- Defines 7 customer segments, income brackets, occupations, risk profiles
- Used by `customer_generator.py` to create realistic profiles

### 2. Customer Generator
- **File:** `customer_generator.py`
- Generates customer profiles with demographic, economic, and behavioral attributes
- Uses segment distributions and config data

### 3. Transaction Model
- **File:** `models/transaction.py`
- 43-field dataclass with core, location, device, risk, and legacy fields
- Methods for dict/csv/legacy export

### 4. Advanced Schema Generator
- **File:** `generators/advanced_schema_generator.py`
- Implements field generation logic for all 43 fields
- Handles risk indicators and state tracking

### 5. Geographic Pattern Generator
- **File:** `generators/geographic_generator.py`
- Implements 80/15/5 city distribution, cost-of-living, proximity groups
- Integrates with transaction generation

### 6. Temporal Pattern Generator
- **File:** `generators/temporal_generator.py`
- Models hour-of-day, day-of-week, salary/festival effects
- Used for realistic time-based transaction patterns

### 7. Merchant Ecosystem Generator
- **File:** `generators/merchant_generator.py`
- Creates merchant pools by city/tier
- Distinguishes chain vs local merchants
- Tracks customer loyalty and merchant reputation

### 8. Data Generator
- **File:** `data_generator.py`
- Orchestrates customer and transaction generation
- Integrates all generators for full synthetic dataset

---

## Design Principles

- **Modularity:** Each generator is a standalone class with clear responsibilities
- **Extensibility:** New fields, segments, or patterns can be added with minimal changes
- **Configurability:** All city, merchant, and category data in config/utils
- **Testability:** 68+ tests across all modules, 100% passing
- **Performance:** 17,200+ transactions/sec sustained
- **Backward Compatibility:** Legacy fields and methods maintained for older code

---

## Data Flow Diagram

```
CustomerGenerator → CustomerProfile
         ↓
TransactionGenerator → Transaction (43 fields)
         ↓
AdvancedSchemaGenerator
         ↓
GeographicPatternGenerator
         ↓
TemporalPatternGenerator
         ↓
MerchantGenerator
         ↓
DataGenerator (orchestrates all)
```

---

## Integration Points

- **Customer profiles** feed into transaction generation
- **Geographic and temporal patterns** modify transaction fields
- **Merchant ecosystem** selects merchants and tracks loyalty
- **Risk indicators** calculated from state tracking and transaction history
- **All modules tested independently and in integration

---

## Evolution Timeline

- **Week 1:** Customer profile system, basic transaction schema (23 fields)
- **Week 2:** Geographic, temporal, merchant ecosystem modules, schema expansion (24 fields)
- **Week 3:** Advanced schema (43 fields), risk indicators, state tracking

---

## Code Statistics

- **Total lines:** ~3,200
- **Classes:** 19
- **Methods:** 74
- **Tests:** 68 (100% passing)

---

## Example: Transaction Generation

```python
customer = CustomerGenerator().generate_customer()
transaction = TransactionGenerator().generate_transaction(customer, datetime.now())
```

**Transaction fields populated by:**
- Customer profile attributes
- Geographic/temporal/merchant generators
- Risk indicators and state tracking

---

## Extending the System

- **Add new cities:** Update `config.py` and `geographic_data.py`
- **Add new merchant categories:** Update `merchant_data.py` and `merchant_generator.py`
- **Add new risk indicators:** Extend `advanced_schema_generator.py` and `transaction.py`
- **Add new customer segments:** Update `customer_profile.py` and `customer_generator.py`

---

## Testing & Validation

- All modules have dedicated test files
- Integration tests validate end-to-end data generation
- Performance benchmarks run after each major change

---

## Lessons Learned

- Manual documentation is critical (avoid batch scripts)
- Modular design enables rapid recovery from corruption
- Comprehensive tests ensure code integrity

---

**Architecture documentation complete.**
