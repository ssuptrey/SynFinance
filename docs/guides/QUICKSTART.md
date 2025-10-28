# SynFinance Quickstart Guide

**Get up and running with SynFinance in 5 minutes**

---

## Prerequisites

- Python 3.8+
- pip package manager
- 500MB free disk space

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/SynFinance.git
cd SynFinance
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit==1.28.0
- pandas==2.1.1
- faker==19.12.0
- numpy==1.26.0
- xlsxwriter==3.1.9
- pytest (for testing)

---

## Quick Test

### Run Tests
```bash
pytest tests/ -v
```

**Expected:** 67/68 tests passing (98.5%)

---

## Generate Your First Dataset

### Method 1: Python Script
```python
from src.data_generator import generate_realistic_dataset

# Generate 1000 transactions
transactions = generate_realistic_dataset(
    num_customers=100,
    transactions_per_customer=10,
    start_date="2025-01-01",
    days=30
)

# Save to CSV
transactions.to_csv("output/transactions.csv", index=False)
print(f"Generated {len(transactions)} transactions")
```

### Method 2: Streamlit App
```bash
streamlit run src/app.py
```

Navigate to `http://localhost:8501` and use the GUI to:
1. Set number of customers
2. Choose date range
3. Generate transactions
4. Download CSV/Excel

---

## Basic Usage Examples

### Create a Customer
```python
from src.customer_generator import CustomerGenerator

gen = CustomerGenerator(seed=42)
customer = gen.generate_customer()

print(customer.customer_id)  # CUST0000001
print(customer.segment)      # CustomerSegment.YOUNG_PROFESSIONAL
print(customer.city)         # Mumbai
```

### Generate Transactions
```python
from src.generators.transaction_core import TransactionGenerator
from datetime import datetime

txn_gen = TransactionGenerator(seed=42)
transaction = txn_gen.generate_transaction(customer, datetime.now())

print(transaction["Transaction_ID"])  # TXN_20251021_000001
print(transaction["Amount"])          # 1850.50
print(transaction["Category"])        # Food & Dining
```

### Generate Multiple Customers
```python
customers = [gen.generate_customer() for _ in range(100)]
print(f"Generated {len(customers)} customers")
```

---

## Output Fields

Each transaction includes **43 fields**:

**Core Fields (10):**
- Transaction_ID, Date, Time, Amount, Category, Merchant, Payment_Mode, Customer_ID, Is_Fraud, Merchant_ID

**Location Fields (5):**
- City, Home_City, Location_Type, City_Tier, State/Region

**Device Fields (4):**
- Device_Type, App_Version, Browser_Type, OS

**Risk Indicators (5):**
- Distance_From_Home, Time_Since_Last_Txn, Is_First_Transaction_With_Merchant, Daily_Transaction_Count, Daily_Transaction_Amount

**Plus 19 more** including customer demographics, merchant details, temporal features

---

## Next Steps

1. **Explore Examples:** See `examples/` folder for demos
2. **Read Week 1 Guide:** [WEEK1_GUIDE.md](WEEK1_GUIDE.md) for detailed tutorial
3. **Integration Guide:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for API reference
4. **Architecture:** [../technical/ARCHITECTURE.md](../technical/ARCHITECTURE.md) for design details

---

## Common Issues

### Import Errors
**Problem:** `ModuleNotFoundError: No module named 'faker'`  
**Solution:** Run `pip install -r requirements.txt`

### Test Failures
**Problem:** Tests fail with import errors  
**Solution:** Ensure you're in the project root and virtual environment is activated

### Performance Issues
**Problem:** Slow generation for large datasets  
**Solution:** Use batch generation with `generate_realistic_dataset()` - optimized for 17,200+ txn/sec

---

## Support

- **Documentation:** [docs/INDEX.md](../INDEX.md)
- **Issues:** GitHub Issues
- **Tests:** `pytest tests/ -v`

---

**Ready to generate enterprise-grade synthetic financial data!**
