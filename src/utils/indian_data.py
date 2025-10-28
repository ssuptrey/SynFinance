"""
Indian market data for realistic transaction generation

This module contains core India-specific data:
- Festival calendar with spending multipliers
- 100+ realistic merchant names across 12 categories
- UPI payment handles
- Common chain merchants for loyalty patterns

For geographic data, see: geographic_data.py
For merchant ecosystem data, see: merchant_data.py
"""

# Import geographic and merchant data for backward compatibility
from src.utils.geographic_data import (
    CITY_TIERS,
    COST_OF_LIVING_MULTIPLIERS,
    MERCHANT_DENSITY,
    TOURIST_DESTINATIONS,
    CITY_PROXIMITY_GROUPS
)

from src.utils.merchant_data import (
    MERCHANT_SUBCATEGORIES,
    CHAIN_MERCHANT_DETAILS,
    CATEGORY_LOYALTY_SCORES
)

# ============================================================================
# INDIAN FESTIVALS
# ============================================================================

# Indian festivals and special days that affect spending
INDIAN_FESTIVALS = {
    "Diwali": {
        "month": [10, 11],  # October-November (varies by lunar calendar)
        "spending_multiplier": 2.5, 
        "categories": ["Shopping", "Electronics", "Groceries", "Gifts"]
    },
    "Holi": {
        "month": [3],  # March
        "spending_multiplier": 1.8, 
        "categories": ["Shopping", "Food & Dining", "Groceries"]
    },
    "Eid": {
        "month": [4, 5, 6],  # Varies (Eid al-Fitr typically April-June, Eid al-Adha July-August)
        "spending_multiplier": 2.0, 
        "categories": ["Shopping", "Food & Dining", "Gifts"]
    },
    "Christmas": {
        "month": [12],  # December
        "spending_multiplier": 1.7, 
        "categories": ["Shopping", "Entertainment", "Travel"]
    },
    "Durga Puja": {
        "month": [9, 10],  # September-October
        "spending_multiplier": 2.2, 
        "categories": ["Shopping", "Food & Dining", "Entertainment"]
    },
    "Raksha Bandhan": {
        "month": [8],  # August
        "spending_multiplier": 1.5, 
        "categories": ["Gifts", "Shopping", "Food & Dining"]
    },
    "Onam": {
        "month": [8, 9],  # August-September
        "spending_multiplier": 1.9, 
        "categories": ["Shopping", "Food & Dining", "Groceries"]
    },
    "Pongal": {
        "month": [1],  # January
        "spending_multiplier": 1.6, 
        "categories": ["Groceries", "Shopping", "Food & Dining"]
    },
}

# ============================================================================
# INDIAN MERCHANTS (100+ realistic names across 12 categories)
# ============================================================================

INDIAN_MERCHANTS = {
    "Groceries": [
        "Big Bazaar", "Reliance Fresh", "More Supermarket", "D-Mart", "Spencer's",
        "Nilgiris", "Hypercity", "Star Bazaar", "Aditya Birla Retail", "Nature's Basket",
        "Local Kirana Store", "Super Market", "Fresh Mart", "Daily Needs", "Grocery Store"
    ],
    "Food & Dining": [
        "Zomato Order", "Swiggy Order", "Domino's Pizza", "McDonald's", "KFC",
        "Cafe Coffee Day", "Starbucks", "Haldiram's", "Barbeque Nation", "Pizza Hut",
        "Subway", "Burger King", "Wow! Momo", "Chai Point", "Local Restaurant",
        "Udupi Restaurant", "Biryani House", "Dhaba", "South Indian Restaurant"
    ],
    "Shopping": [
        "Flipkart", "Amazon India", "Myntra", "Ajio", "Shoppers Stop",
        "Pantaloons", "Westside", "Max Fashion", "Lifestyle", "Central",
        "FabIndia", "Bata", "Nike Store", "Adidas Store", "Decathlon"
    ],
    "Entertainment": [
        "BookMyShow", "PVR Cinemas", "INOX", "Carnival Cinemas", "Cinepolis",
        "Netflix India", "Amazon Prime Video", "Hotstar", "Sony Liv", "Zee5",
        "Game Parlour", "Timezone", "Smaaash", "Fun City"
    ],
    "Transportation": [
        "Uber", "Ola Cabs", "Rapido", "Indian Railways", "IRCTC",
        "Petrol Pump", "HP Petrol", "BPCL", "Indian Oil", "Metro Card Recharge",
        "Delhi Metro", "Mumbai Local", "Bus Pass", "Auto Fare"
    ],
    "Healthcare": [
        "Apollo Pharmacy", "MedPlus", "Wellness Forever", "1mg", "PharmEasy",
        "Apollo Hospital", "Fortis Hospital", "Max Hospital", "Clinic",
        "Diagnostic Center", "PathLab", "Dr. Lal PathLabs", "Metropolis"
    ],
    "Utilities": [
        "Electricity Bill", "Water Bill", "Gas Bill", "Broadband Bill",
        "Jio Recharge", "Airtel Recharge", "Vi Recharge", "BSNL Recharge",
        "DTH Recharge", "Tata Sky", "Dish TV", "Airtel DTH"
    ],
    "Travel": [
        "MakeMyTrip", "Goibibo", "Cleartrip", "Yatra", "EaseMyTrip",
        "OYO Rooms", "Treebo Hotels", "FabHotels", "Zostel", "Airbnb",
        "RedBus", "Travel Agency", "Tour Package", "Hotel Booking"
    ],
    "Education": [
        "School Fee", "College Fee", "Tuition Classes", "Coaching Center",
        "Byjus", "Unacademy", "Vedantu", "Course Purchase", "Book Store",
        "Oxford Bookstore", "Crossword", "Stationery Shop", "Online Course"
    ],
    "Electronics": [
        "Croma", "Reliance Digital", "Vijay Sales", "Poorvika", "Sangeetha Mobiles",
        "Amazon Electronics", "Flipkart Electronics", "Apple Store", "Samsung Store",
        "Mi Store", "OnePlus Store", "Sony Center"
    ],
    "Health & Fitness": [
        "Cult.fit", "Gold's Gym", "Snap Fitness", "Talwalkars", "Anytime Fitness",
        "Yoga Classes", "Fitness First", "Sports Equipment", "Decathlon Sports",
        "Nike Running", "Fittr App", "HealthifyMe"
    ],
    "Gifts": [
        "Archies Gallery", "Ferns N Petals", "Gift Shop", "Handicrafts Store",
        "Amazon Gifts", "Flipkart Gifts", "Flower Shop", "Cake Shop",
        "Chocolate Shop", "Hallmark Store", "Gift Voucher"
    ]
}

# ============================================================================
# UPI & PAYMENT DATA
# ============================================================================

# UPI payment patterns (India-specific)
UPI_HANDLES = ["@paytm", "@phonepe", "@googlepay", "@amazonpay", "@ybl", "@okhdfcbank", "@okicici", "@okaxis"]

# Merchant loyalty - commonly visited merchants for repeat transactions
CHAIN_MERCHANTS = [
    "Big Bazaar", "D-Mart", "Reliance Fresh", "More Supermarket",
    "Zomato Order", "Swiggy Order", "Domino's Pizza", "McDonald's",
    "Uber", "Ola Cabs", "Amazon India", "Flipkart"
]

# ============================================================================
# RE-EXPORTED FOR BACKWARD COMPATIBILITY
# ============================================================================
# Geographic and merchant data has been moved to separate modules for better organization:
# - geographic_data.py: City tiers, COL multipliers, proximity groups
# - merchant_data.py: Subcategories, chain details, loyalty scores
#
# These are imported at the top and re-exported here for backward compatibility.
# 
# Available exports:
# - CITY_TIERS, COST_OF_LIVING_MULTIPLIERS, MERCHANT_DENSITY
# - TOURIST_DESTINATIONS, CITY_PROXIMITY_GROUPS
# - MERCHANT_SUBCATEGORIES, CHAIN_MERCHANT_DETAILS, CATEGORY_LOYALTY_SCORES
# ============================================================================
