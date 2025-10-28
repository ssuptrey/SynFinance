# Contributing to SynFinance

Thank you for your interest in contributing to SynFinance! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment tool (venv, conda, etc.)
- Git

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SynFinance.git
   cd SynFinance
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

---

## Project Structure

```
SynFinance/
├── src/                        # Main source code
│   ├── generators/            # Data generation modules
│   ├── models/                # Data models (Transaction, Customer, etc.)
│   └── utils/                 # Utility modules (data, helpers)
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── generators/            # Generator-specific tests
├── examples/                  # Example scripts and demos
├── scripts/                   # Utility scripts (build, deploy, etc.)
├── docs/                      # Documentation
│   ├── guides/                # User guides
│   ├── technical/             # Technical documentation
│   ├── planning/              # Roadmap and planning
│   ├── progress/              # Weekly progress summaries
│   └── archive/               # Legacy documentation
├── data/                      # Sample data (not in git)
└── output/                    # Generated outputs (not in git)
```

### Key Files

- **src/generators/transaction_core.py** - Main transaction generator
- **src/models/transaction.py** - Transaction dataclass (43 fields)
- **src/customer_profile.py** - Customer profile system
- **tests/integration/test_customer_integration.py** - Integration tests

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test additions/improvements

### 2. Make Your Changes

- Follow the [Code Standards](#code-standards)
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/integration/test_customer_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Update Documentation

- Update relevant docs in `docs/`
- Add docstrings to new functions/classes
- Update README.md if adding major features

### 5. Commit Your Changes

See [Commit Messages](#commit-messages) for guidelines.

---

## Code Standards

### Python Style Guide

Follow **PEP 8** with these specifics:

- **Line Length:** 100 characters (relaxed from 79)
- **Indentation:** 4 spaces (no tabs)
- **Imports:** Group and sort (stdlib, third-party, local)
- **Naming:**
  - Classes: `PascalCase`
  - Functions/Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`

### Type Hints

Use type hints for all functions:

```python
from typing import Dict, List, Optional
from datetime import datetime

def generate_transaction(
    customer_id: str,
    date: datetime,
    amount: float
) -> Dict[str, any]:
    """
    Generate a transaction with given parameters.
    
    Args:
        customer_id: Unique customer identifier
        date: Transaction timestamp
        amount: Transaction amount in INR
        
    Returns:
        Dictionary containing transaction details
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_risk_score(
    amount: float,
    time_since_last: Optional[int],
    is_first_merchant: bool
) -> float:
    """Calculate fraud risk score for a transaction.
    
    Args:
        amount: Transaction amount in INR
        time_since_last: Minutes since last transaction (None if first)
        is_first_merchant: True if first time with this merchant
        
    Returns:
        Risk score between 0.0 (low risk) and 1.0 (high risk)
        
    Examples:
        >>> calculate_risk_score(1000.0, 120, False)
        0.35
        >>> calculate_risk_score(50000.0, 5, True)
        0.92
    """
    pass
```

### Error Handling

Use specific exceptions with clear messages:

```python
# Good
if amount < 0:
    raise ValueError(f"Amount must be positive, got {amount}")

# Bad
if amount < 0:
    raise Exception("Invalid amount")
```

### Code Organization

- **One class per file** (exceptions for small helper classes)
- **Group related functions** together
- **Limit function length** to ~50 lines
- **Extract complex logic** into helper functions
- **Use constants** instead of magic numbers

---

## Testing Guidelines

### Test Structure

```python
"""Test module for AdvancedSchemaGenerator."""
import pytest
from src.generators.advanced_schema_generator import AdvancedSchemaGenerator

class TestAdvancedSchemaGenerator:
    """Tests for AdvancedSchemaGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for tests."""
        return AdvancedSchemaGenerator(seed=42)
    
    def test_generate_card_type_credit(self, generator):
        """Test credit card generation for high income."""
        card_type = generator.generate_card_type("Credit Card", "HIGH")
        assert card_type == "Credit"
    
    def test_generate_card_type_invalid_payment(self, generator):
        """Test card type for non-card payment modes."""
        card_type = generator.generate_card_type("UPI", "MEDIUM")
        assert card_type == "NA"
```

### Test Coverage Goals

- **Unit Tests:** >80% coverage for all modules
- **Integration Tests:** All major workflows covered
- **Edge Cases:** Test boundary conditions, null values, invalid inputs

### Test Naming

- **Format:** `test_<what>_<condition>`
- **Examples:**
  - `test_transaction_generation_with_customer`
  - `test_merchant_selection_for_category`
  - `test_risk_score_high_amount`

### Assertions

Use descriptive assertion messages:

```python
# Good
assert len(transactions) == 100, f"Expected 100 transactions, got {len(transactions)}"

# Also good (for complex checks)
assert card_type in ["Credit", "Debit", "NA"], \
    f"Invalid card type: {card_type}. Must be Credit, Debit, or NA"
```

### Fixtures

Use pytest fixtures for common test data:

```python
@pytest.fixture
def sample_customer():
    """Create a sample customer for testing."""
    return CustomerProfile(
        customer_id="CUST_001",
        age=30,
        income_bracket="MEDIUM",
        city="Mumbai"
    )
```

---

## Documentation

### Code Documentation

- **All public functions** must have docstrings
- **All classes** must have docstrings
- **Complex algorithms** need inline comments
- **Use examples** in docstrings when helpful

### User Documentation

When adding features that affect users:

1. Update `docs/guides/INTEGRATION_GUIDE.md`
2. Add examples to `examples/`
3. Update `README.md` if it's a major feature
4. Consider adding a Jupyter notebook tutorial

### Technical Documentation

For architectural changes:

1. Update `docs/technical/ARCHITECTURE.md`
2. Document design decisions in `docs/technical/DESIGN_GUIDE.md`
3. Add to weekly progress in `docs/progress/`

---

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat:** New feature
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, no logic change)
- **refactor:** Code refactoring
- **test:** Adding or updating tests
- **chore:** Maintenance tasks

### Examples

```
feat(schema): Add 19 new fields to transaction schema

- Implemented Transaction dataclass with 43 fields
- Created AdvancedSchemaGenerator for new field generation
- Added state tracking for risk indicators
- All 68 tests passing

Closes #123
```

```
fix(merchant): Fix merchant loyalty calculation for new customers

Previously crashed when customer had no transaction history.
Now returns 0.5 (neutral) for new customers.

Fixes #456
```

```
docs(readme): Update installation instructions

Added Windows-specific setup steps and troubleshooting section.
```

### Best Practices

- **First line:** 50 characters or less
- **Body:** Wrap at 72 characters
- **Use imperative mood:** "Add feature" not "Added feature"
- **Reference issues:** Use "Fixes #123" or "Closes #456"

---

## Pull Request Process

### Before Submitting

1. [OK] All tests pass (`pytest tests/ -v`)
2. [OK] Code follows style guide
3. [OK] New code has tests (aim for >80% coverage)
4. [OK] Documentation is updated
5. [OK] Commit messages are clear
6. [OK] Branch is up-to-date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** must pass (tests, linting)
2. **At least one approving review** required
3. **All conversations resolved** before merge
4. **Squash commits** for cleaner history

---

## Development Tips

### Running Specific Tests

```bash
# Test a specific file
pytest tests/generators/test_advanced_schema.py -v

# Test a specific class
pytest tests/generators/test_advanced_schema.py::TestAdvancedSchemaGenerator -v

# Test a specific method
pytest tests/generators/test_advanced_schema.py::TestAdvancedSchemaGenerator::test_card_type -v

# Run tests with keyword filter
pytest tests/ -k "merchant" -v
```

### Debugging

```python
# Use pytest's built-in debugger
pytest tests/test_file.py --pdb

# Add breakpoints in code
import pdb; pdb.set_trace()

# Print during tests (use -s flag)
pytest tests/test_file.py -s -v
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Questions or Issues?

- **Bug Reports:** Open an issue on GitHub
- **Feature Requests:** Open an issue with [Feature Request] tag
- **Questions:** Use GitHub Discussions
- **Security Issues:** Email security@synfinance.com (do not open public issue)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- No harassment or discrimination

---

## License

By contributing to SynFinance, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing to SynFinance!**

Your contributions help make financial data generation better for everyone.
