# SynFinance Customer Schema Documentation

**Comprehensive Customer Profile Structure**

---

## Overview

The CustomerProfile dataclass defines all characteristics that influence transaction patterns, including demographics, economic status, behavioral traits, and preferences.

---

## Schema Version

**Version:** 1.0  
**Fields:** 23  
**Enums:** 5  
**Last Updated:** October 7, 2025

---

## CustomerProfile Dataclass

### Core Identity
```python
customer_id: str  # Format: CUST0000001
```

### Demographics (5 fields)
```python
age: int                  # 18-75 years
gender: str               # Male/Female/Other
city: str                 # Home city (20 Indian cities)
state: str                # Home state
region: str               # North/South/East/West/Central
```

### Economic Profile (3 fields)
```python
income_bracket: IncomeBracket  # Enum: LOW to PREMIUM
occupation: Occupation         # Enum: 8 types
monthly_income: float          # INR amount
```

### Customer Segment (1 field)
```python
segment: CustomerSegment  # Enum: 7 segments
```

### Behavioral Attributes (2 fields)
```python
risk_profile: RiskProfile          # CONSERVATIVE/MODERATE/AGGRESSIVE
digital_savviness: DigitalSavviness  # LOW/MEDIUM/HIGH
```

### Spending Behavior (4 fields)
```python
avg_transaction_amount: float        # Average transaction value
monthly_transaction_count: int       # Transaction frequency
preferred_categories: List[str]      # Top 3-5 categories
preferred_payment_modes: List[str]   # Preferred payment methods
```

### Time Patterns (2 fields)
```python
preferred_shopping_hours: List[int]  # Hours of day (0-23)
weekend_shopper: bool                # More active on weekends?
```

### Loyalty & Habits (3 fields)
```python
merchant_loyalty: float      # 0.0-1.0: likelihood to revisit merchants
brand_conscious: bool        # Prefers established brands?
impulse_buyer: bool          # Makes spontaneous purchases?
```

### Geographic Behavior (2 fields)
```python
travels_frequently: bool             # Transactions outside home city
online_shopping_preference: float    # 0.0-1.0: online vs offline
```

---

## Enumerations

### 1. CustomerSegment (7 values)
```python
class CustomerSegment(Enum):
    YOUNG_PROFESSIONAL = "Young Professional"        # 20%
    FAMILY_ORIENTED = "Family Oriented"              # 25%
    BUDGET_CONSCIOUS = "Budget Conscious"            # 20%
    TECH_SAVVY_MILLENNIAL = "Tech-Savvy Millennial"  # 15%
    AFFLUENT_SHOPPER = "Affluent Shopper"            # 8%
    SENIOR_CONSERVATIVE = "Senior Conservative"      # 7%
    STUDENT = "Student"                              # 5%
```

### 2. IncomeBracket (6 values)
```python
class IncomeBracket(Enum):
    LOW = "Low"                    # < ₹3 lakhs/year (₹10k-25k/month)
    LOWER_MIDDLE = "Lower Middle"  # ₹3-6 lakhs/year (₹25k-50k/month)
    MIDDLE = "Middle"              # ₹6-12 lakhs/year (₹50k-1L/month)
    UPPER_MIDDLE = "Upper Middle"  # ₹12-25 lakhs/year (₹1L-2L/month)
    HIGH = "High"                  # ₹25-50 lakhs/year (₹2L-4L/month)
    PREMIUM = "Premium"            # > ₹50 lakhs/year (₹4L-10L/month)
```

### 3. Occupation (8 values)
```python
class Occupation(Enum):
    STUDENT = "Student"
    SALARIED_EMPLOYEE = "Salaried Employee"
    BUSINESS_OWNER = "Business Owner"
    FREELANCER = "Freelancer"
    PROFESSIONAL = "Professional"           # Doctor, Lawyer, CA
    GOVERNMENT_EMPLOYEE = "Government Employee"
    RETIRED = "Retired"
    HOMEMAKER = "Homemaker"
```

### 4. RiskProfile (3 values)
```python
class RiskProfile(Enum):
    CONSERVATIVE = "Conservative"  # Prefers safe transactions
    MODERATE = "Moderate"          # Balanced approach
    AGGRESSIVE = "Aggressive"      # High-value, varied merchants
```

### 5. DigitalSavviness (3 values)
```python
class DigitalSavviness(Enum):
    LOW = "Low"        # Prefers cash, basic UPI
    MEDIUM = "Medium"  # Uses cards, UPI, some apps
    HIGH = "High"      # Uses all payment modes, wallets
```

---

## Segment Profiles

### Young Professional
- **Age Range:** 25-35
- **Income:** Middle to High
- **Occupations:** Salaried, Professional, Freelancer
- **Digital Savviness:** High/Medium
- **Categories:** Food & Dining, Entertainment, Travel, Shopping
- **Payment Modes:** UPI, Credit Card, Digital Wallet
- **Avg Transaction:** ₹500-3,000
- **Monthly Transactions:** 40-80
- **Characteristics:** Tech-savvy, impulse buyer, brand conscious, travels

### Family Oriented
- **Age Range:** 35-50
- **Income:** Middle to Upper Middle
- **Occupations:** Salaried, Business Owner, Government
- **Categories:** Groceries, Education, Healthcare, Utilities
- **Avg Transaction:** ₹800-2,500
- **Monthly Transactions:** 50-100
- **Characteristics:** Planned spender, loyal, weekend shopper

### Budget Conscious
- **Age Range:** 22-60
- **Income:** Low to Middle
- **Occupations:** Salaried, Homemaker, Student
- **Categories:** Groceries, Utilities, Transportation
- **Avg Transaction:** ₹200-800
- **Monthly Transactions:** 30-60
- **Characteristics:** Value-conscious, low online preference

### Tech-Savvy Millennial
- **Age Range:** 20-32
- **Income:** Middle to Upper Middle
- **Occupations:** Salaried, Freelancer, Student
- **Digital Savviness:** HIGH
- **Categories:** Electronics, Entertainment, Food & Dining
- **Avg Transaction:** ₹400-2,500
- **Monthly Transactions:** 60-120
- **Characteristics:** Early adopter, high online (85%), impulse buyer

### Affluent Shopper
- **Age Range:** 30-55
- **Income:** High to Premium
- **Occupations:** Business Owner, Professional
- **Categories:** Shopping, Travel, Food & Dining, Entertainment
- **Avg Transaction:** ₹2,000-8,000
- **Monthly Transactions:** 50-90
- **Characteristics:** High spending power, brand conscious

### Senior Conservative
- **Age Range:** 55-75
- **Income:** Middle to High
- **Occupations:** Retired, Business Owner, Government
- **Digital Savviness:** Low/Medium
- **Categories:** Healthcare, Groceries, Utilities
- **Avg Transaction:** ₹500-2,000
- **Monthly Transactions:** 20-40
- **Characteristics:** Careful spender, low online (25%)

### Student
- **Age Range:** 18-25
- **Income:** Low to Lower Middle
- **Occupations:** Student
- **Digital Savviness:** High/Medium
- **Categories:** Food & Dining, Education, Entertainment
- **Avg Transaction:** ₹150-600
- **Monthly Transactions:** 40-80
- **Characteristics:** Limited budget, high online (75%), impulse buyer

---

## Helper Methods

### get_spending_power() -> float
Returns relative spending power (0.0-1.0) based on income bracket.

### is_high_value_customer() -> bool
Returns True if income is High/Premium and avg transaction > ₹2,000.

### get_fraud_vulnerability_score() -> float
Calculates fraud risk (0.0-1.0) based on age, digital savviness, value, travel patterns.

---

## Usage Example

```python
from src.customer_generator import CustomerGenerator

gen = CustomerGenerator(seed=42)
customer = gen.generate_customer()

# Access fields
print(customer.customer_id)           # CUST0000001
print(customer.segment)                # CustomerSegment.YOUNG_PROFESSIONAL
print(customer.age)                    # 29
print(customer.city)                   # Mumbai
print(customer.monthly_income)         # 120000.0
print(customer.preferred_categories)   # ["Food & Dining", "Entertainment", "Travel"]
print(customer.get_spending_power())   # 0.7
print(customer.is_high_value_customer())  # False
```

---

## Field Relationships

- **Segment determines:** Income bracket, occupation, digital savviness, categories, payment modes
- **Age influences:** Occupation, risk profile, digital savviness
- **Income bracket determines:** Monthly income range, avg transaction amount
- **Occupation influences:** Preferred shopping hours, time patterns
- **Region determines:** Available cities for transactions

---

## Validation Rules

1. **Age:** 18-75 years
2. **Monthly Income:** Within bracket range
3. **Preferred Categories:** 3-5 from TRANSACTION_CATEGORIES
4. **Preferred Payment Modes:** 2-4 from PAYMENT_MODES
5. **Merchant Loyalty:** 0.0-1.0
6. **Online Shopping Preference:** 0.0-1.0

---

**For implementation details, see [../technical/ARCHITECTURE.md](ARCHITECTURE.md)**
