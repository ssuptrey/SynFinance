# Technical Documentation Index

This folder contains technical specifications, architecture documentation, and detailed schema references.

## Folder Structure

```
docs/technical/
├── fraud/              # Fraud pattern documentation
│   └── FRAUD_PATTERNS.md
├── schemas/            # Data schema documentation
│   ├── CUSTOMER_SCHEMA.md
│   └── FIELD_REFERENCE.md
├── ARCHITECTURE.md     # System architecture
├── CHANGES.md          # Technical changes log
├── DESIGN_GUIDE.md     # Design principles
├── ENTERPRISE_READINESS.md  # Enterprise readiness checklist
└── INDEX.md            # This file
```

## Core Documentation

### Architecture & Design

- **[ARCHITECTURE.md](ARCHITECTURE.md)**
  - System architecture overview
  - Component relationships
  - Data flow diagrams
  - Design patterns used

- **[DESIGN_GUIDE.md](DESIGN_GUIDE.md)**
  - Design principles
  - Code organization
  - Naming conventions
  - Best practices

- **[CHANGES.md](CHANGES.md)**
  - Technical changes log
  - Breaking changes
  - Migration guides
  - Deprecation notices

- **[ENTERPRISE_READINESS.md](ENTERPRISE_READINESS.md)**
  - Enterprise feature checklist
  - Production readiness criteria
  - Performance benchmarks
  - Security considerations

## Schema Documentation

Located in `schemas/` subfolder:

### [schemas/CUSTOMER_SCHEMA.md](schemas/CUSTOMER_SCHEMA.md)

**Customer Profile Specifications**
- 23 customer profile fields
- 7 customer segments
- 6 income brackets
- 8 occupation types
- Behavioral attributes
- Risk profiles

**Key Sections:**
- CustomerProfile dataclass structure
- Segment definitions and distributions
- Income bracket ranges
- Occupation mapping
- Helper methods and utilities

### [schemas/FIELD_REFERENCE.md](schemas/FIELD_REFERENCE.md)

**Complete Field Reference (50 Fields)**
- 45 base transaction fields
- 5 fraud detection fields
- Field categories and organization
- Data types and ranges
- Generation logic
- Quality metrics (entropy, CV)

**Field Categories:**
1. Core Transaction (8 fields)
2. Customer Demographics (7 fields)
3. Merchant Details (5 fields)
4. Location Context (3 fields)
5. Device & Channel (6 fields)
6. Payment Details (3 fields)
7. Temporal Context (4 fields)
8. Risk & Fraud (5 fields)
9. Customer Profile (2 fields)
10. Derived Metrics (2 fields)
11. Fraud Labels (5 fields) - NEW in v0.4.0

## Fraud Pattern Documentation

Located in `fraud/` subfolder:

### [fraud/FRAUD_PATTERNS.md](fraud/FRAUD_PATTERNS.md)

**Comprehensive Fraud Pattern Library (v0.5.0)**

**10 Base Fraud Pattern Types:**
1. Card Cloning - Impossible travel detection
2. Account Takeover - Behavioral deviation
3. Merchant Collusion - Round amounts
4. Velocity Abuse - High frequency
5. Amount Manipulation - Structuring
6. Refund Fraud - Excessive refunds
7. Stolen Card - Inactivity spike
8. Synthetic Identity - Limited history
9. First Party Fraud - Bust-out
10. Friendly Fraud - Chargebacks

**5 Advanced Fraud Patterns (NEW v0.5.0):**
1. Transaction Replay - Duplicate detection
2. Card Testing - Small test transactions
3. Mule Account - Money laundering patterns
4. Shipping Fraud - Address manipulation
5. Loyalty Abuse - Points/rewards exploitation

**Additional Features (NEW v0.5.0):**
- Fraud combination system (chained, coordinated, progressive)
- Fraud network analysis (merchant/location/device rings)
- Temporal clustering (coordinated attacks)
- Cross-pattern statistics tracking (co-occurrence matrix)

**Key Sections:**
- Architecture overview
- Pattern specifications (15 types)
- Detection logic and thresholds
- Confidence calculation formulas
- Combination strategies
- Network analysis methods
- Usage examples
- ML training integration
- Performance characteristics
- Best practices
- Troubleshooting guide

**Features:**
- Confidence scoring (0.0-1.0)
- Severity classification (low/medium/high/critical)
- Evidence tracking (JSON serialized)
- History-aware detection
- Configurable rates (0.5-2%)
- Real-time statistics
- Network graph generation
- Pattern isolation tracking (95%+ target)

## Quick Links

### For Developers
- [ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
- [schemas/FIELD_REFERENCE.md](schemas/FIELD_REFERENCE.md) - All field specifications
- [DESIGN_GUIDE.md](DESIGN_GUIDE.md) - Coding standards

### For Data Scientists
- [fraud/FRAUD_PATTERNS.md](fraud/FRAUD_PATTERNS.md) - Fraud detection patterns
- [schemas/FIELD_REFERENCE.md](schemas/FIELD_REFERENCE.md) - ML feature reference
- [schemas/CUSTOMER_SCHEMA.md](schemas/CUSTOMER_SCHEMA.md) - Customer segments

### For Enterprise Teams
- [ENTERPRISE_READINESS.md](ENTERPRISE_READINESS.md) - Production checklist
- [CHANGES.md](CHANGES.md) - Version changes
- [ARCHITECTURE.md](ARCHITECTURE.md) - Scalability considerations

## Related Documentation

- **User Guides:** [docs/guides/](../guides/)
  - [INTEGRATION_GUIDE.md](../guides/INTEGRATION_GUIDE.md)
  - [QUICK_REFERENCE.md](../guides/QUICK_REFERENCE.md)
  - [QUICKSTART.md](../guides/QUICKSTART.md)

- **Planning:** [docs/planning/](../planning/)
  - [ROADMAP.md](../planning/ROADMAP.md)
  - [BUSINESS_PLAN.md](../planning/BUSINESS_PLAN.md)

- **Progress:** [docs/progress/](../progress/)
  - Weekly progress reports by week

## Version History

| Version | Date | Key Technical Changes |
|---------|------|----------------------|
| 0.4.0 | Oct 21, 2025 | Fraud pattern library (10 types, 2,162 lines) |
| 0.3.2 | Oct 21, 2025 | Variance analysis, 45 fields, 111 tests |
| 0.3.1 | Oct 20, 2025 | Merchant ecosystem (164+ merchants) |
| 0.3.0 | Oct 19, 2025 | Geographic patterns (20 cities, 3 tiers) |
| 0.2.0 | Oct 15, 2025 | Temporal patterns (24-hour cycles) |
| 0.1.0 | Oct 13, 2025 | Customer profile system (7 segments) |

---

**Last Updated:** October 21, 2025  
**Version:** 0.4.0  
**Total Documentation:** 340+ KB (50 files)
