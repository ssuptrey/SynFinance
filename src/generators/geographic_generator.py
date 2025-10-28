"""
Geographic Pattern Generator for SynFinance
Implements realistic geographic consistency patterns for Indian transactions

Week 2, Days 3-4: Geographic Consistency
Features:
- 80/15/5 city distribution (home/nearby/distant)
- Proximity rules for Indian cities
- Cost-of-living adjustments per city
- City-specific merchant density
"""

import random
from typing import Dict, List, Tuple, Optional
from src.customer_profile import CustomerProfile
from src.customer_generator import CustomerGenerator


class GeographicPatternGenerator:
    """
    Generates realistic geographic patterns for transactions
    
    Key features:
    - 80% transactions in home city
    - 15% transactions in nearby cities (same region/state)
    - 5% transactions in distant cities (travel scenarios)
    - Cost-of-living adjustments per city tier
    - Merchant density based on city tier
    """
    
    # City tiers (based on population, economic activity, infrastructure)
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
    
    # Cost-of-living multipliers by tier
    # Applied to transaction amounts to reflect local purchasing power
    COST_OF_LIVING_MULTIPLIERS = {
        1: 1.3,   # Tier 1 metros: 30% higher prices
        2: 1.0,   # Tier 2 cities: baseline
        3: 0.8    # Tier 3 cities: 20% lower prices
    }
    
    # Merchant density (percentage of merchants available in each tier)
    # Tier 1: All merchants available
    # Tier 2: 80% of merchants (missing some premium/niche brands)
    # Tier 3: 60% of merchants (mostly local, fewer chains)
    MERCHANT_DENSITY = {
        1: 1.0,   # 100% merchant availability
        2: 0.8,   # 80% merchant availability
        3: 0.6    # 60% merchant availability
    }
    
    # Tourist/travel destinations (more likely for distant city transactions)
    TOURIST_DESTINATIONS = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
        "Kolkata", "Jaipur", "Kochi"
    ]
    
    # Proximity groups: Cities that are "nearby" to each other
    # Based on geographical proximity, state, and travel patterns
    PROXIMITY_GROUPS = {
        # Maharashtra cluster
        "Mumbai": ["Pune", "Nagpur", "Surat", "Vadodara"],
        "Pune": ["Mumbai", "Surat", "Nagpur"],
        "Nagpur": ["Mumbai", "Pune", "Bhopal", "Indore"],
        
        # North cluster (NCR + nearby)
        "Delhi": ["Chandigarh", "Jaipur", "Lucknow", "Agra"],
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
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize geographic pattern generator"""
        if seed is not None:
            random.seed(seed)
    
    def get_city_tier(self, city: str) -> int:
        """Get tier for a city (1=metro, 2=major, 3=smaller)"""
        return self.CITY_TIERS.get(city, 2)  # Default to tier 2
    
    def get_cost_of_living_multiplier(self, city: str) -> float:
        """
        Get cost-of-living multiplier for a city
        
        Returns:
            Multiplier to apply to transaction amounts
            Tier 1 metros: 1.3x (30% higher prices)
            Tier 2 cities: 1.0x (baseline)
            Tier 3 cities: 0.8x (20% lower prices)
        """
        tier = self.get_city_tier(city)
        return self.COST_OF_LIVING_MULTIPLIERS[tier]
    
    def get_merchant_density(self, city: str) -> float:
        """
        Get merchant density for a city
        
        Returns:
            Percentage of merchants available (0.6 to 1.0)
            Used to filter merchant lists for smaller cities
        """
        tier = self.get_city_tier(city)
        return self.MERCHANT_DENSITY[tier]
    
    def is_merchant_available(self, city: str, merchant: str) -> bool:
        """
        Check if a merchant is available in a given city
        
        Logic:
        - Tier 1: All merchants available
        - Tier 2: 80% probability (some premium merchants missing)
        - Tier 3: 60% probability (fewer chains, more local)
        
        Note: This is probabilistic. Chain merchants (Big Bazaar, Reliance, etc.)
        have higher availability across all tiers.
        """
        density = self.get_merchant_density(city)
        
        # Chain merchants have higher availability across all tiers
        chain_merchants = [
            "Big Bazaar", "Reliance Fresh", "D-Mart", "More Supermarket",
            "Zomato", "Swiggy", "Uber", "Ola", "Amazon", "Flipkart",
            "Domino's", "McDonald's", "KFC", "Pizza Hut"
        ]
        
        # Check if merchant name contains any chain merchant keywords
        is_chain = any(chain in merchant for chain in chain_merchants)
        
        if is_chain:
            # Chain merchants: higher availability
            # Tier 1: 100%, Tier 2: 95%, Tier 3: 80%
            availability = min(1.0, density + 0.2)
        else:
            # Local/premium merchants: use standard density
            availability = density
        
        return random.random() < availability
    
    def get_nearby_cities(self, home_city: str, customer: CustomerProfile) -> List[str]:
        """
        Get list of nearby cities for a customer
        
        Priority order:
        1. Cities in proximity group (pre-defined nearby cities)
        2. Cities in same region (from CustomerGenerator.CITY_STATE_MAP)
        3. Cities in same state (if available)
        
        Args:
            home_city: Customer's home city
            customer: Customer profile with region info
        
        Returns:
            List of nearby cities
        """
        nearby = []
        
        # 1. Add proximity group cities (most likely)
        if home_city in self.PROXIMITY_GROUPS:
            nearby.extend(self.PROXIMITY_GROUPS[home_city])
        
        # 2. Add other cities in same region (if not already included)
        same_region_cities = [
            city for city, (state, region) in CustomerGenerator.CITY_STATE_MAP.items()
            if region == customer.region and city != home_city and city not in nearby
        ]
        nearby.extend(same_region_cities)
        
        # Remove duplicates and return
        return list(set(nearby))
    
    def get_distant_cities(self, home_city: str, customer: CustomerProfile) -> List[str]:
        """
        Get list of distant cities (different regions)
        
        Used for travel scenarios (5% of transactions)
        Prioritizes tourist destinations
        
        Args:
            home_city: Customer's home city
            customer: Customer profile with region info
        
        Returns:
            List of distant cities
        """
        # Cities in different regions
        distant = [
            city for city, (state, region) in CustomerGenerator.CITY_STATE_MAP.items()
            if region != customer.region and city != home_city
        ]
        
        # Prioritize tourist destinations if customer travels frequently
        if customer.travels_frequently:
            # Sort with tourist destinations first
            tourist = [city for city in distant if city in self.TOURIST_DESTINATIONS]
            non_tourist = [city for city in distant if city not in self.TOURIST_DESTINATIONS]
            return tourist + non_tourist
        
        return distant
    
    def select_transaction_city(self, customer: CustomerProfile) -> Tuple[str, str]:
        """
        Select city for a transaction based on 80/15/5 distribution
        
        Returns:
            Tuple of (city, location_type)
            location_type: "home", "nearby", or "travel"
        
        Distribution:
        - 80% home city
        - 15% nearby cities (same region/proximity group)
        - 5% distant cities (travel scenarios)
        """
        rand = random.random()
        
        # 80% home city
        if rand < 0.80:
            return (customer.city, "home")
        
        # 15% nearby cities
        elif rand < 0.95:
            nearby_cities = self.get_nearby_cities(customer.city, customer)
            
            if nearby_cities:
                city = random.choice(nearby_cities)
                return (city, "nearby")
            else:
                # Fallback to home city if no nearby cities
                return (customer.city, "home")
        
        # 5% distant cities (travel)
        else:
            distant_cities = self.get_distant_cities(customer.city, customer)
            
            if distant_cities:
                # Weight tourist destinations higher for frequent travelers
                if customer.travels_frequently and len(distant_cities) > 3:
                    # 70% chance of tourist destination, 30% other
                    if random.random() < 0.7:
                        tourist_options = [c for c in distant_cities if c in self.TOURIST_DESTINATIONS]
                        if tourist_options:
                            city = random.choice(tourist_options)
                        else:
                            city = random.choice(distant_cities)
                    else:
                        city = random.choice(distant_cities)
                else:
                    city = random.choice(distant_cities)
                
                return (city, "travel")
            else:
                # Fallback to home city
                return (customer.city, "home")
    
    def apply_cost_of_living_adjustment(self, amount: float, city: str) -> float:
        """
        Apply cost-of-living adjustment to transaction amount
        
        Args:
            amount: Base transaction amount
            city: City where transaction occurs
        
        Returns:
            Adjusted amount based on city's cost-of-living
        
        Examples:
            Mumbai (Tier 1): ₹1000 → ₹1300 (30% higher)
            Indore (Tier 2): ₹1000 → ₹1000 (baseline)
            Patna (Tier 3): ₹1000 → ₹800 (20% lower)
        """
        multiplier = self.get_cost_of_living_multiplier(city)
        return round(amount * multiplier, 2)
    
    def get_geographic_breakdown(self, customer: CustomerProfile, 
                                 city: str, location_type: str) -> Dict[str, any]:
        """
        Get detailed geographic breakdown for a transaction
        
        Returns dictionary with:
        - city: Transaction city
        - location_type: "home", "nearby", or "travel"
        - home_city: Customer's home city
        - city_tier: 1, 2, or 3
        - cost_of_living_multiplier: 0.8 to 1.3
        - merchant_density: 0.6 to 1.0
        - is_same_region: Boolean
        - distance_category: "local", "regional", or "long_distance"
        """
        home_city = customer.city
        tier = self.get_city_tier(city)
        col_multiplier = self.get_cost_of_living_multiplier(city)
        merchant_density = self.get_merchant_density(city)
        
        # Check if same region
        home_region = CustomerGenerator.CITY_STATE_MAP.get(home_city, (None, None))[1]
        city_region = CustomerGenerator.CITY_STATE_MAP.get(city, (None, None))[1]
        is_same_region = (home_region == city_region)
        
        # Distance category
        if location_type == "home":
            distance_category = "local"
        elif location_type == "nearby":
            distance_category = "regional"
        else:  # travel
            distance_category = "long_distance"
        
        return {
            "city": city,
            "location_type": location_type,
            "home_city": home_city,
            "city_tier": tier,
            "cost_of_living_multiplier": col_multiplier,
            "merchant_density": merchant_density,
            "is_same_region": is_same_region,
            "distance_category": distance_category
        }


if __name__ == "__main__":
    """Test geographic pattern generation"""
    print("=== Geographic Pattern Generator Test ===\n")
    
    # Test city tiers
    print("City Tiers:")
    for city, tier in sorted(GeographicPatternGenerator.CITY_TIERS.items(), 
                            key=lambda x: (x[1], x[0])):
        print(f"  {city}: Tier {tier}")
    
    print("\nCost of Living Multipliers:")
    geo_gen = GeographicPatternGenerator()
    for city in ["Mumbai", "Indore", "Patna"]:
        mult = geo_gen.get_cost_of_living_multiplier(city)
        print(f"  {city}: {mult}x")
    
    print("\nMerchant Density:")
    for city in ["Mumbai", "Indore", "Patna"]:
        density = geo_gen.get_merchant_density(city)
        print(f"  {city}: {density*100:.0f}% availability")
    
    print("\n[TEST COMPLETE]")
