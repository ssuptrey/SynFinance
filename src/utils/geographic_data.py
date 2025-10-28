"""
Geographic data for Indian cities and regions

This module contains:
- City tier classifications
- Cost-of-living multipliers
- Merchant density by tier
- Tourist destinations
- City proximity groups
"""

# ============================================================================
# CITY TIERS AND CLASSIFICATIONS
# ============================================================================

# City tiers based on population, economic activity, infrastructure
# Tier 1: Metros (high COL, high merchant density)
# Tier 2: Major cities (medium COL, medium merchant density)
# Tier 3: Smaller cities (lower COL, lower merchant density)

CITY_TIERS = {
    # Tier 1: Metros (8 cities)
    "Mumbai": 1,
    "Delhi": 1,
    "Bangalore": 1,
    "Hyderabad": 1,
    "Chennai": 1,
    "Kolkata": 1,
    "Pune": 1,
    "Ahmedabad": 1,
    
    # Tier 2: Major cities (7 cities)
    "Jaipur": 2,
    "Lucknow": 2,
    "Surat": 2,
    "Chandigarh": 2,
    "Indore": 2,
    "Kochi": 2,
    "Nagpur": 2,
    
    # Tier 3: Smaller cities (5 cities)
    "Bhopal": 3,
    "Visakhapatnam": 3,
    "Patna": 3,
    "Vadodara": 3,
    "Coimbatore": 3
}

# Cost-of-living multipliers by tier (applied to transaction amounts)
COST_OF_LIVING_MULTIPLIERS = {
    1: 1.3,   # Tier 1 metros: 30% higher prices
    2: 1.0,   # Tier 2 cities: baseline
    3: 0.8    # Tier 3 cities: 20% lower prices
}

# Merchant density by tier (percentage of merchants available)
MERCHANT_DENSITY = {
    1: 1.0,   # 100% merchant availability (all chains + local)
    2: 0.8,   # 80% merchant availability (most chains, some missing)
    3: 0.6    # 60% merchant availability (fewer chains, more local)
}

# ============================================================================
# TOURIST DESTINATIONS
# ============================================================================

# Tourist/travel destinations (higher probability for distant transactions)
TOURIST_DESTINATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Jaipur", "Kochi"
]

# ============================================================================
# CITY PROXIMITY GROUPS
# ============================================================================

# Proximity groups: Cities that are "nearby" to each other
# Based on geographical proximity, state boundaries, and travel patterns
CITY_PROXIMITY_GROUPS = {
    # Maharashtra cluster
    "Mumbai": ["Pune", "Nagpur", "Surat", "Vadodara"],
    "Pune": ["Mumbai", "Surat", "Nagpur"],
    "Nagpur": ["Mumbai", "Pune", "Bhopal", "Indore"],
    
    # North cluster (NCR + nearby)
    "Delhi": ["Chandigarh", "Jaipur", "Lucknow"],
    "Chandigarh": ["Delhi", "Jaipur", "Lucknow"],
    "Jaipur": ["Delhi", "Chandigarh", "Ahmedabad", "Indore"],
    "Lucknow": ["Delhi", "Chandigarh", "Patna"],
    
    # South cluster
    "Bangalore": ["Chennai", "Hyderabad", "Kochi", "Coimbatore"],
    "Chennai": ["Bangalore", "Hyderabad", "Coimbatore", "Kochi"],
    "Hyderabad": ["Bangalore", "Chennai", "Visakhapatnam", "Nagpur"],
    "Kochi": ["Bangalore", "Chennai", "Coimbatore"],
    "Coimbatore": ["Bangalore", "Chennai", "Kochi"],
    
    # East cluster
    "Kolkata": ["Patna", "Bhopal", "Visakhapatnam"],
    "Patna": ["Kolkata", "Lucknow", "Bhopal"],
    
    # West cluster
    "Ahmedabad": ["Surat", "Vadodara", "Mumbai", "Jaipur", "Indore"],
    "Surat": ["Ahmedabad", "Vadodara", "Mumbai", "Pune"],
    "Vadodara": ["Ahmedabad", "Surat", "Mumbai", "Indore"],
    
    # Central cluster
    "Indore": ["Bhopal", "Nagpur", "Ahmedabad", "Vadodara", "Jaipur"],
    "Bhopal": ["Indore", "Nagpur", "Patna", "Kolkata"],
    
    # Coastal Andhra
    "Visakhapatnam": ["Hyderabad", "Chennai", "Kolkata"]
}
