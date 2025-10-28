"""
Configuration file for SynFinance
Contains constants and configuration for synthetic data generation
"""

# Indian cities for transaction generation
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara"
]

# Transaction categories
TRANSACTION_CATEGORIES = [
    "Groceries", "Electronics", "Clothing", "Food & Dining",
    "Travel", "Entertainment", "Healthcare", "Education",
    "Utilities", "Home & Garden", "Automotive", "Shopping",
    "Insurance", "Investments", "Bills & Recharge"
]

# Payment modes
PAYMENT_MODES = [
    "Credit Card", "Debit Card", "UPI", "Net Banking",
    "Cash", "Wallet", "EMI"
]

# Merchant name templates
MERCHANT_TYPES = {
    "Groceries": ["Store", "Mart", "Supermarket", "Bazaar", "Kirana"],
    "Electronics": ["Electronics", "Tech World", "Digital Hub", "Gadget Store"],
    "Clothing": ["Fashion", "Apparel", "Boutique", "Wear", "Trends"],
    "Food & Dining": ["Restaurant", "Cafe", "Dhaba", "Kitchen", "Biryani House"],
    "Travel": ["Tours", "Travels", "Airways", "Railway", "Cab Services"],
    "Entertainment": ["Cinema", "Multiplex", "Gaming Zone", "Theatre"],
    "Healthcare": ["Hospital", "Clinic", "Pharmacy", "Medical Store"],
    "Education": ["Academy", "Classes", "Institute", "School", "College"],
    "Utilities": ["Services", "Utility", "Provider"],
    "Home & Garden": ["Home Store", "Furniture", "Decor", "Hardware"],
    "Automotive": ["Motors", "Auto Parts", "Service Center", "Fuel Station"],
    "Shopping": ["Mall", "Plaza", "Shopping Center", "Retail"],
    "Insurance": ["Insurance Co", "Life Insurance", "General Insurance"],
    "Investments": ["Securities", "Mutual Funds", "Investment Services"],
    "Bills & Recharge": ["Recharge", "Bill Pay", "Utility Services"]
}

# Amount ranges for different categories (min, max)
AMOUNT_RANGES = {
    "Groceries": (200, 5000),
    "Electronics": (2000, 100000),
    "Clothing": (500, 15000),
    "Food & Dining": (150, 3000),
    "Travel": (300, 50000),
    "Entertainment": (200, 2000),
    "Healthcare": (300, 20000),
    "Education": (1000, 50000),
    "Utilities": (500, 10000),
    "Home & Garden": (1000, 50000),
    "Automotive": (500, 100000),
    "Shopping": (500, 25000),
    "Insurance": (5000, 100000),
    "Investments": (10000, 500000),
    "Bills & Recharge": (100, 5000)
}

# Date range (default: last 1 year)
DEFAULT_DATE_RANGE_DAYS = 365
