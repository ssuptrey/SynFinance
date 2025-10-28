"""
Merchant ecosystem data for Indian market

This module contains:
- Merchant categories and subcategories
- Chain merchant details (regional availability, reputation)
- Category-based loyalty patterns
"""

# ============================================================================
# MERCHANT SUBCATEGORIES
# ============================================================================

# Merchant categories with detailed subcategories
MERCHANT_SUBCATEGORIES = {
    "Groceries": {
        "Supermarket": ["Big Bazaar", "D-Mart", "Reliance Fresh", "More Supermarket", "Spencer's", "Hypercity", "Star Bazaar"],
        "Kirana Store": ["Local Kirana Store", "Super Market", "Daily Needs", "Grocery Store", "Neighbourhood Store"],
        "Departmental Store": ["Aditya Birla Retail", "Nature's Basket", "Nilgiris", "Fresh Mart", "Organic Store"]
    },
    "Food & Dining": {
        "Food Delivery": ["Zomato Order", "Swiggy Order", "Uber Eats", "Food Panda"],
        "Fast Food": ["Domino's Pizza", "McDonald's", "KFC", "Burger King", "Pizza Hut", "Subway", "Wow! Momo"],
        "Cafe": ["Cafe Coffee Day", "Starbucks", "Chai Point", "Blue Tokai", "Third Wave Coffee"],
        "Fine Dining": ["Barbeque Nation", "Mainland China", "Punjab Grill", "The Yellow Chilli"],
        "QSR": ["Haldiram's", "MTR", "Sagar Ratna", "Saravana Bhavan"],
        "Local Restaurant": ["Local Restaurant", "Udupi Restaurant", "Biryani House", "Dhaba", "South Indian Restaurant"]
    },
    "Shopping": {
        "E-commerce": ["Flipkart", "Amazon India", "Myntra", "Ajio", "Meesho", "Snapdeal"],
        "Apparel Store": ["Pantaloons", "Westside", "Max Fashion", "Lifestyle", "Central", "FabIndia"],
        "Footwear": ["Bata", "Metro Shoes", "Relaxo", "Nike Store", "Adidas Store", "Puma"],
        "Department Store": ["Shoppers Stop", "Lifestyle", "Westside"],
        "Sports & Outdoor": ["Decathlon", "Nike", "Adidas", "Puma", "Sports Shop"]
    },
    "Entertainment": {
        "Cinema": ["PVR Cinemas", "INOX", "Carnival Cinemas", "Cinepolis", "Miraj Cinemas"],
        "OTT Platform": ["Netflix India", "Amazon Prime Video", "Hotstar", "Sony Liv", "Zee5", "JioCinema"],
        "Event Booking": ["BookMyShow", "Paytm Insider", "MeraEvents"],
        "Gaming": ["Game Parlour", "Timezone", "Smaaash", "Fun City", "Amoeba"]
    },
    "Transportation": {
        "Cab Service": ["Uber", "Ola Cabs", "Rapido", "Blu Smart"],
        "Public Transport": ["Indian Railways", "IRCTC", "Delhi Metro", "Mumbai Local", "Metro Card Recharge", "Bus Pass"],
        "Fuel": ["HP Petrol", "BPCL", "Indian Oil", "Reliance Petrol", "Shell", "Nayara Energy"],
        "Two-Wheeler": ["Auto Fare", "Bike Taxi", "Bounce", "Yulu"]
    },
    "Healthcare": {
        "Pharmacy": ["Apollo Pharmacy", "MedPlus", "Wellness Forever", "1mg", "PharmEasy", "NetMeds"],
        "Hospital": ["Apollo Hospital", "Fortis Hospital", "Max Hospital", "Manipal Hospital", "Cloudnine"],
        "Clinic": ["Clinic", "Polyclinic", "Family Clinic", "Practo Consultation"],
        "Diagnostics": ["Dr. Lal PathLabs", "Metropolis", "Thyrocare", "PathLab", "Diagnostic Center"]
    },
    "Utilities": {
        "Electricity": ["Electricity Bill", "BESCOM", "Tata Power", "Adani Power"],
        "Water & Gas": ["Water Bill", "Gas Bill", "Indane Gas", "HP Gas", "Bharat Gas"],
        "Telecom": ["Jio Recharge", "Airtel Recharge", "Vi Recharge", "BSNL Recharge"],
        "Internet & TV": ["Broadband Bill", "DTH Recharge", "Tata Sky", "Dish TV", "Airtel DTH", "JioFiber"]
    },
    "Travel": {
        "Flight Booking": ["MakeMyTrip", "Goibibo", "Cleartrip", "Yatra", "EaseMyTrip", "IndiGo"],
        "Hotel Booking": ["OYO Rooms", "Treebo Hotels", "FabHotels", "Zostel", "Airbnb", "MakeMyTrip Hotels"],
        "Bus Booking": ["RedBus", "Abhibus", "Paytm Bus", "MakeMyTrip Bus"],
        "Travel Agency": ["Thomas Cook", "Cox & Kings", "SOTC", "Travel Agency", "Tour Package"]
    },
    "Education": {
        "School & College": ["School Fee", "College Fee", "University Fee", "Exam Fee"],
        "Coaching": ["Tuition Classes", "Coaching Center", "Allen", "Aakash", "FIITJEE"],
        "Online Learning": ["Byjus", "Unacademy", "Vedantu", "Coursera", "Udemy", "Skillshare"],
        "Books & Stationery": ["Oxford Bookstore", "Crossword", "Amazon Books", "Flipkart Books", "Stationery Shop"]
    },
    "Electronics": {
        "Electronics Retail": ["Croma", "Reliance Digital", "Vijay Sales", "Poorvika", "Sangeetha Mobiles"],
        "E-commerce": ["Amazon Electronics", "Flipkart Electronics"],
        "Brand Stores": ["Apple Store", "Samsung Store", "Mi Store", "OnePlus Store", "Sony Center", "LG Plaza"]
    },
    "Health & Fitness": {
        "Gym": ["Gold's Gym", "Snap Fitness", "Talwalkars", "Anytime Fitness", "Fitness First"],
        "Fitness App": ["Cult.fit", "Fittr App", "HealthifyMe", "MyFitnessPal"],
        "Yoga & Wellness": ["Yoga Classes", "Meditation Center", "Wellness Center"],
        "Sports Equipment": ["Decathlon Sports", "Nike Running", "Sports Shop"]
    },
    "Gifts": {
        "Gift Store": ["Archies Gallery", "Hallmark Store", "Gift Shop", "Handicrafts Store"],
        "Flowers & Cakes": ["Ferns N Petals", "FlowerAura", "Flower Shop", "Cake Shop", "Winni"],
        "Online Gifts": ["Amazon Gifts", "Flipkart Gifts", "IGP Gifts"],
        "Sweets & Chocolates": ["Haldiram's", "Chocolate Shop", "Sweet Shop", "Cadbury Store"]
    }
}

# ============================================================================
# CHAIN MERCHANT DETAILS
# ============================================================================

# Chain merchants with regional availability
# Format: {merchant_name: {"regions": [regions], "tier_availability": [tiers], "reputation": score}}
CHAIN_MERCHANT_DETAILS = {
    # Pan-India chains (all regions, all tiers)
    "Big Bazaar": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.85},
    "D-Mart": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.88},
    "Reliance Fresh": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.82},
    "More Supermarket": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.78},
    
    # Regional grocery chains
    "Spencer's": {"regions": ["South", "East"], "tiers": [1, 2], "reputation": 0.83},
    "Nilgiris": {"regions": ["South"], "tiers": [1, 2], "reputation": 0.80},
    "Star Bazaar": {"regions": ["West"], "tiers": [1, 2], "reputation": 0.81},
    "Nature's Basket": {"regions": ["North", "South", "West"], "tiers": [1], "reputation": 0.90},
    
    # Food delivery (pan-India)
    "Zomato Order": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.86},
    "Swiggy Order": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.87},
    "Domino's Pizza": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.84},
    "McDonald's": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.89},
    "KFC": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.85},
    
    # Cafes
    "Cafe Coffee Day": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.75},
    "Starbucks": {"regions": ["North", "South", "West"], "tiers": [1, 2], "reputation": 0.92},
    
    # E-commerce (pan-India, all tiers)
    "Amazon India": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.91},
    "Flipkart": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.90},
    "Myntra": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.88},
    
    # Apparel
    "Pantaloons": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.79},
    "Westside": {"regions": ["North", "South", "West"], "tiers": [1, 2], "reputation": 0.82},
    "Max Fashion": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.77},
    
    # Cinemas
    "PVR Cinemas": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.87},
    "INOX": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.86},
    "Cinepolis": {"regions": ["North", "West"], "tiers": [1, 2], "reputation": 0.85},
    
    # Transportation
    "Uber": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.83},
    "Ola Cabs": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.82},
    "Indian Railways": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.70},
    
    # Healthcare
    "Apollo Pharmacy": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.88},
    "MedPlus": {"regions": ["South", "North"], "tiers": [1, 2, 3], "reputation": 0.84},
    "1mg": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2, 3], "reputation": 0.87},
    
    # Electronics
    "Croma": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.83},
    "Reliance Digital": {"regions": ["North", "South", "East", "West", "Central"], "tiers": [1, 2], "reputation": 0.84},
    
    # Fitness
    "Cult.fit": {"regions": ["North", "South", "West"], "tiers": [1, 2], "reputation": 0.85},
    "Gold's Gym": {"regions": ["North", "South", "East", "West"], "tiers": [1, 2], "reputation": 0.80}
}

# ============================================================================
# CATEGORY LOYALTY PATTERNS
# ============================================================================

# Category-based loyalty patterns
# Higher score = more likely to return to same merchant
CATEGORY_LOYALTY_SCORES = {
    "Groceries": 0.75,  # High loyalty (local kirana/supermarket)
    "Food & Dining": 0.45,  # Medium-low loyalty (variety seeking)
    "Shopping": 0.35,  # Low loyalty (price comparison)
    "Entertainment": 0.30,  # Low loyalty (variety seeking)
    "Transportation": 0.60,  # Medium loyalty (preferred app)
    "Healthcare": 0.70,  # High loyalty (trusted pharmacy/doctor)
    "Utilities": 0.95,  # Very high (same provider)
    "Travel": 0.40,  # Low loyalty (price comparison)
    "Education": 0.90,  # Very high (same school/coaching)
    "Electronics": 0.35,  # Low loyalty (price/availability)
    "Health & Fitness": 0.65,  # Medium-high (gym membership)
    "Gifts": 0.40  # Medium-low (occasion-based)
}
