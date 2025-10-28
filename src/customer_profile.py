"""
Customer Profile System for SynFinance
Defines customer demographics, behavioral attributes, and segments for realistic transaction generation

Week 1, Day 1-2: Customer Profile Schema Design
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import random


class CustomerSegment(Enum):
    """7 distinct customer segments with unique behavioral patterns"""
    YOUNG_PROFESSIONAL = "Young Professional"
    FAMILY_ORIENTED = "Family Oriented"
    BUDGET_CONSCIOUS = "Budget Conscious"
    TECH_SAVVY_MILLENNIAL = "Tech-Savvy Millennial"
    AFFLUENT_SHOPPER = "Affluent Shopper"
    SENIOR_CONSERVATIVE = "Senior Conservative"
    STUDENT = "Student"


class IncomeBracket(Enum):
    """Income levels aligned with Indian economic segments"""
    LOW = "Low"              # < ₹3 lakhs/year
    LOWER_MIDDLE = "Lower Middle"  # ₹3-6 lakhs/year
    MIDDLE = "Middle"        # ₹6-12 lakhs/year
    UPPER_MIDDLE = "Upper Middle"  # ₹12-25 lakhs/year
    HIGH = "High"            # ₹25-50 lakhs/year
    PREMIUM = "Premium"      # > ₹50 lakhs/year


class Occupation(Enum):
    """Common occupation types in India"""
    STUDENT = "Student"
    SALARIED_EMPLOYEE = "Salaried Employee"
    BUSINESS_OWNER = "Business Owner"
    FREELANCER = "Freelancer"
    PROFESSIONAL = "Professional"  # Doctor, Lawyer, CA, etc.
    GOVERNMENT_EMPLOYEE = "Government Employee"
    RETIRED = "Retired"
    HOMEMAKER = "Homemaker"


class RiskProfile(Enum):
    """Financial risk-taking behavior"""
    CONSERVATIVE = "Conservative"  # Prefers safe transactions, low amounts
    MODERATE = "Moderate"          # Balanced approach
    AGGRESSIVE = "Aggressive"      # High-value transactions, varied merchants


class DigitalSavviness(Enum):
    """Comfort level with digital payment technologies"""
    LOW = "Low"        # Prefers cash, basic UPI
    MEDIUM = "Medium"  # Uses cards, UPI, some apps
    HIGH = "High"      # Uses all payment modes, comfortable with wallets


@dataclass
class CustomerProfile:
    """
    Comprehensive customer profile with demographics and behavioral attributes
    
    This class defines all characteristics that influence transaction patterns:
    - Demographics (age, gender, location)
    - Economic status (income, occupation)
    - Behavioral traits (risk profile, digital savviness)
    - Preferences (payment modes, spending categories)
    """
    
    # Core Identity
    customer_id: str
    
    # Demographics
    age: int                    # 18-75 years
    gender: str                 # Male/Female/Other
    city: str                   # Home city
    state: str                  # Home state
    region: str                 # North/South/East/West/Central
    
    # Economic Profile
    income_bracket: IncomeBracket
    occupation: Occupation
    monthly_income: float       # Approximate monthly income in INR
    
    # Customer Segment
    segment: CustomerSegment
    
    # Behavioral Attributes
    risk_profile: RiskProfile
    digital_savviness: DigitalSavviness
    
    # Spending Behavior
    avg_transaction_amount: float       # Average transaction value
    monthly_transaction_count: int      # How often they transact
    preferred_categories: List[str]     # Top 3-5 categories
    preferred_payment_modes: List[str]  # Preferred payment methods
    
    # Time Patterns
    preferred_shopping_hours: List[int]  # Hours of day (0-23)
    weekend_shopper: bool                # More active on weekends?
    
    # Loyalty & Habits
    merchant_loyalty: float     # 0.0-1.0: likelihood to revisit merchants
    brand_conscious: bool       # Prefers established brands?
    impulse_buyer: bool         # Makes spontaneous purchases?
    
    # Geographic Behavior
    travels_frequently: bool    # Transactions outside home city
    online_shopping_preference: float  # 0.0-1.0: online vs offline
    
    def __str__(self) -> str:
        return (f"Customer {self.customer_id}: {self.segment.value}, "
                f"Age {self.age}, {self.income_bracket.value} Income, "
                f"{self.city}")
    
    def get_spending_power(self) -> float:
        """Calculate relative spending power (0.0-1.0)"""
        income_multipliers = {
            IncomeBracket.LOW: 0.15,
            IncomeBracket.LOWER_MIDDLE: 0.30,
            IncomeBracket.MIDDLE: 0.50,
            IncomeBracket.UPPER_MIDDLE: 0.70,
            IncomeBracket.HIGH: 0.85,
            IncomeBracket.PREMIUM: 1.0
        }
        return income_multipliers.get(self.income_bracket, 0.5)
    
    def is_high_value_customer(self) -> bool:
        """Determine if customer is high-value based on income and spending"""
        return (self.income_bracket in [IncomeBracket.HIGH, IncomeBracket.PREMIUM] 
                and self.avg_transaction_amount > 2000)
    
    def get_fraud_vulnerability_score(self) -> float:
        """
        Calculate fraud vulnerability (0.0-1.0)
        Higher scores indicate more vulnerable to fraud
        """
        score = 0.0
        
        # Age factors
        if self.age < 25 or self.age > 65:
            score += 0.2
        
        # Digital savviness (less savvy = more vulnerable)
        if self.digital_savviness == DigitalSavviness.LOW:
            score += 0.3
        elif self.digital_savviness == DigitalSavviness.MEDIUM:
            score += 0.1
        
        # High-value customers are attractive targets
        if self.is_high_value_customer():
            score += 0.2
        
        # Frequent travelers face more risk
        if self.travels_frequently:
            score += 0.1
        
        return min(score, 1.0)


# Customer Segment Definitions
SEGMENT_PROFILES = {
    CustomerSegment.YOUNG_PROFESSIONAL: {
        "description": "25-35 years old, urban, tech-savvy, high spending on lifestyle",
        "age_range": (25, 35),
        "income_brackets": [IncomeBracket.MIDDLE, IncomeBracket.UPPER_MIDDLE, IncomeBracket.HIGH],
        "occupations": [Occupation.SALARIED_EMPLOYEE, Occupation.PROFESSIONAL, Occupation.FREELANCER],
        "digital_savviness": [DigitalSavviness.HIGH, DigitalSavviness.MEDIUM],
        "preferred_categories": ["Food & Dining", "Entertainment", "Travel", "Shopping", "Health & Fitness"],
        "preferred_payment_modes": ["UPI", "Credit Card", "Digital Wallet"],
        "avg_transaction_range": (500, 3000),
        "monthly_transactions": (40, 80),
        "online_preference": 0.7,
        "weekend_shopper": True,
        "impulse_buyer": True,
        "brand_conscious": True,
        "travels_frequently": True
    },
    
    CustomerSegment.FAMILY_ORIENTED: {
        "description": "35-50 years old, married with kids, focused on household and education",
        "age_range": (35, 50),
        "income_brackets": [IncomeBracket.MIDDLE, IncomeBracket.UPPER_MIDDLE],
        "occupations": [Occupation.SALARIED_EMPLOYEE, Occupation.BUSINESS_OWNER, Occupation.GOVERNMENT_EMPLOYEE],
        "digital_savviness": [DigitalSavviness.MEDIUM, DigitalSavviness.HIGH],
        "preferred_categories": ["Groceries", "Education", "Healthcare", "Utilities", "Shopping"],
        "preferred_payment_modes": ["UPI", "Debit Card", "Credit Card"],
        "avg_transaction_range": (800, 2500),
        "monthly_transactions": (50, 100),
        "online_preference": 0.5,
        "weekend_shopper": True,
        "impulse_buyer": False,
        "brand_conscious": True,
        "travels_frequently": False
    },
    
    CustomerSegment.BUDGET_CONSCIOUS: {
        "description": "All ages, lower-middle income, careful spending, value-conscious",
        "age_range": (22, 60),
        "income_brackets": [IncomeBracket.LOW, IncomeBracket.LOWER_MIDDLE, IncomeBracket.MIDDLE],
        "occupations": [Occupation.SALARIED_EMPLOYEE, Occupation.HOMEMAKER, Occupation.STUDENT],
        "digital_savviness": [DigitalSavviness.LOW, DigitalSavviness.MEDIUM],
        "preferred_categories": ["Groceries", "Utilities", "Transportation", "Healthcare"],
        "preferred_payment_modes": ["UPI", "Cash", "Debit Card"],
        "avg_transaction_range": (200, 800),
        "monthly_transactions": (30, 60),
        "online_preference": 0.3,
        "weekend_shopper": False,
        "impulse_buyer": False,
        "brand_conscious": False,
        "travels_frequently": False
    },
    
    CustomerSegment.TECH_SAVVY_MILLENNIAL: {
        "description": "20-32 years old, early adopters, high digital engagement",
        "age_range": (20, 32),
        "income_brackets": [IncomeBracket.MIDDLE, IncomeBracket.UPPER_MIDDLE],
        "occupations": [Occupation.SALARIED_EMPLOYEE, Occupation.FREELANCER, Occupation.STUDENT],
        "digital_savviness": [DigitalSavviness.HIGH],
        "preferred_categories": ["Electronics", "Entertainment", "Food & Dining", "Travel", "Shopping"],
        "preferred_payment_modes": ["UPI", "Digital Wallet", "Credit Card", "BNPL"],
        "avg_transaction_range": (400, 2500),
        "monthly_transactions": (60, 120),
        "online_preference": 0.85,
        "weekend_shopper": True,
        "impulse_buyer": True,
        "brand_conscious": False,
        "travels_frequently": True
    },
    
    CustomerSegment.AFFLUENT_SHOPPER: {
        "description": "30-55 years old, high income, luxury spending, premium services",
        "age_range": (30, 55),
        "income_brackets": [IncomeBracket.HIGH, IncomeBracket.PREMIUM],
        "occupations": [Occupation.BUSINESS_OWNER, Occupation.PROFESSIONAL, Occupation.SALARIED_EMPLOYEE],
        "digital_savviness": [DigitalSavviness.HIGH, DigitalSavviness.MEDIUM],
        "preferred_categories": ["Shopping", "Travel", "Food & Dining", "Entertainment", "Health & Fitness"],
        "preferred_payment_modes": ["Credit Card", "UPI", "Digital Wallet"],
        "avg_transaction_range": (2000, 8000),
        "monthly_transactions": (50, 90),
        "online_preference": 0.6,
        "weekend_shopper": True,
        "impulse_buyer": True,
        "brand_conscious": True,
        "travels_frequently": True
    },
    
    CustomerSegment.SENIOR_CONSERVATIVE: {
        "description": "55-75 years old, retired or nearing retirement, careful spenders",
        "age_range": (55, 75),
        "income_brackets": [IncomeBracket.MIDDLE, IncomeBracket.UPPER_MIDDLE, IncomeBracket.HIGH],
        "occupations": [Occupation.RETIRED, Occupation.BUSINESS_OWNER, Occupation.GOVERNMENT_EMPLOYEE],
        "digital_savviness": [DigitalSavviness.LOW, DigitalSavviness.MEDIUM],
        "preferred_categories": ["Healthcare", "Groceries", "Utilities", "Travel", "Shopping"],
        "preferred_payment_modes": ["Debit Card", "Cash", "UPI"],
        "avg_transaction_range": (500, 2000),
        "monthly_transactions": (20, 40),
        "online_preference": 0.25,
        "weekend_shopper": False,
        "impulse_buyer": False,
        "brand_conscious": True,
        "travels_frequently": False
    },
    
    CustomerSegment.STUDENT: {
        "description": "18-25 years old, limited income, tech-savvy, value-conscious",
        "age_range": (18, 25),
        "income_brackets": [IncomeBracket.LOW, IncomeBracket.LOWER_MIDDLE],
        "occupations": [Occupation.STUDENT],
        "digital_savviness": [DigitalSavviness.HIGH, DigitalSavviness.MEDIUM],
        "preferred_categories": ["Food & Dining", "Education", "Entertainment", "Shopping", "Transportation"],
        "preferred_payment_modes": ["UPI", "Digital Wallet", "Debit Card"],
        "avg_transaction_range": (150, 600),
        "monthly_transactions": (40, 80),
        "online_preference": 0.75,
        "weekend_shopper": True,
        "impulse_buyer": True,
        "brand_conscious": False,
        "travels_frequently": False
    }
}


# Segment Distribution (should total ~100%)
SEGMENT_DISTRIBUTION = {
    CustomerSegment.YOUNG_PROFESSIONAL: 0.20,      # 20%
    CustomerSegment.FAMILY_ORIENTED: 0.25,         # 25%
    CustomerSegment.BUDGET_CONSCIOUS: 0.20,        # 20%
    CustomerSegment.TECH_SAVVY_MILLENNIAL: 0.15,   # 15%
    CustomerSegment.AFFLUENT_SHOPPER: 0.08,        # 8%
    CustomerSegment.SENIOR_CONSERVATIVE: 0.07,     # 7%
    CustomerSegment.STUDENT: 0.05                  # 5%
}


def get_income_range(bracket: IncomeBracket) -> Tuple[float, float]:
    """Get monthly income range for income bracket"""
    ranges = {
        IncomeBracket.LOW: (10000, 25000),           # ₹10k-25k/month
        IncomeBracket.LOWER_MIDDLE: (25000, 50000),  # ₹25k-50k/month
        IncomeBracket.MIDDLE: (50000, 100000),       # ₹50k-1L/month
        IncomeBracket.UPPER_MIDDLE: (100000, 200000),# ₹1L-2L/month
        IncomeBracket.HIGH: (200000, 400000),        # ₹2L-4L/month
        IncomeBracket.PREMIUM: (400000, 1000000)     # ₹4L-10L/month
    }
    return ranges.get(bracket, (50000, 100000))


def calculate_risk_profile(income_bracket: IncomeBracket, age: int, occupation: Occupation) -> RiskProfile:
    """Determine risk profile based on demographics"""
    # Young + high income = aggressive
    if age < 35 and income_bracket in [IncomeBracket.HIGH, IncomeBracket.PREMIUM]:
        return RiskProfile.AGGRESSIVE
    
    # Business owners tend to be moderate-aggressive
    if occupation == Occupation.BUSINESS_OWNER:
        return RiskProfile.MODERATE if age > 45 else RiskProfile.AGGRESSIVE
    
    # Seniors and government employees are conservative
    if age > 55 or occupation in [Occupation.RETIRED, Occupation.GOVERNMENT_EMPLOYEE]:
        return RiskProfile.CONSERVATIVE
    
    # Students with limited income are conservative
    if occupation == Occupation.STUDENT:
        return RiskProfile.CONSERVATIVE
    
    # Default to moderate
    return RiskProfile.MODERATE


if __name__ == "__main__":
    """Test the schema definitions"""
    print("=== SynFinance Customer Profile Schema ===\n")
    
    print("Customer Segments Defined:")
    for segment in CustomerSegment:
        profile = SEGMENT_PROFILES[segment]
        print(f"\n{segment.value}:")
        print(f"  Description: {profile['description']}")
        print(f"  Age Range: {profile['age_range']}")
        print(f"  Distribution: {SEGMENT_DISTRIBUTION[segment]*100:.1f}%")
    
    print("\n\nIncome Brackets:")
    for bracket in IncomeBracket:
        income_range = get_income_range(bracket)
        print(f"  {bracket.value}: ₹{income_range[0]:,.0f} - ₹{income_range[1]:,.0f}/month")
    
    print("\n\nOccupations:")
    for occ in Occupation:
        print(f"  - {occ.value}")
    
    print("\n\n[COMPLETE] Customer Profile Schema Design Complete!")
    print("Next: Implement CustomerGenerator class to create realistic customers")
