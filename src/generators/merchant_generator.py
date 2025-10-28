"""
Merchant ecosystem generator for realistic merchant patterns

This module creates and manages a merchant ecosystem with:
- Unique merchant identifiers
- Chain vs local merchant distinction
- City-specific merchant availability
- Merchant reputation scores
- Customer-merchant loyalty tracking
"""

import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

from src.customer_profile import CustomerProfile
from src.utils.indian_data import (
    MERCHANT_SUBCATEGORIES,
    CHAIN_MERCHANT_DETAILS,
    CATEGORY_LOYALTY_SCORES,
    CITY_TIERS,
    INDIAN_MERCHANTS
)


@dataclass
class Merchant:
    """Represents a merchant in the ecosystem"""
    merchant_id: str
    name: str
    category: str
    subcategory: str
    city: str
    merchant_type: str  # "chain" or "local"
    reputation: float  # 0.0 to 1.0
    region: str  # North/South/East/West/Central
    tier: int  # City tier (1, 2, or 3)
    
    def to_dict(self) -> Dict:
        """Convert merchant to dictionary"""
        return {
            "Merchant_ID": self.merchant_id,
            "Merchant_Name": self.name,
            "Merchant_Category": self.category,
            "Merchant_Subcategory": self.subcategory,
            "Merchant_City": self.city,
            "Merchant_Type": self.merchant_type,
            "Merchant_Reputation": self.reputation,
            "Merchant_Region": self.region,
            "Merchant_Tier": self.tier
        }


class MerchantGenerator:
    """
    Generate and manage merchant ecosystem with realistic patterns
    
    Features:
    - Unique merchant IDs (MER_{CATEGORY}_{CITY}_{NUM})
    - Chain merchants available across multiple cities
    - Local merchants specific to each city
    - Merchant reputation scores
    - City-tier based merchant density
    - Customer-merchant loyalty tracking
    
    Usage:
        merchant_gen = MerchantGenerator(seed=42)
        merchant = merchant_gen.select_merchant(customer, category, city)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize merchant generator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Merchant pools by city
        self.merchant_pools: Dict[str, List[Merchant]] = {}
        
        # Customer loyalty tracking (customer_id -> {category: [merchant_ids]})
        self.customer_favorites: Dict[str, Dict[str, List[str]]] = {}
        
        # Merchant ID counter by city and category
        self.merchant_counters: Dict[Tuple[str, str], int] = {}
        
        # Region mapping for cities (from customer_generator.py logic)
        self.city_region_map = {
            "Mumbai": "West", "Pune": "West", "Ahmedabad": "West", "Surat": "West", "Vadodara": "West", "Nagpur": "Central",
            "Delhi": "North", "Chandigarh": "North", "Jaipur": "North", "Lucknow": "North",
            "Bangalore": "South", "Chennai": "South", "Hyderabad": "South", "Kochi": "South", "Coimbatore": "South",
            "Kolkata": "East", "Patna": "East", "Visakhapatnam": "East",
            "Indore": "Central", "Bhopal": "Central"
        }
    
    # ============================================================================
    # MERCHANT ID GENERATION
    # ============================================================================
    
    def generate_merchant_id(self, category: str, city: str) -> str:
        """
        Generate unique merchant ID
        
        Format: MER_{CATEGORY}_{CITY}_{NUM}
        Example: MER_GRO_MUM_001
        
        Args:
            category: Merchant category (Groceries, Food & Dining, etc.)
            city: City name
            
        Returns:
            Unique merchant ID
        """
        # Category abbreviation (first 3 letters)
        cat_abbr = category.replace(" & ", "").replace(" ", "")[:3].upper()
        
        # City abbreviation (first 3 letters)
        city_abbr = city[:3].upper()
        
        # Get counter for this city-category combo
        key = (city, category)
        if key not in self.merchant_counters:
            self.merchant_counters[key] = 0
        
        self.merchant_counters[key] += 1
        num = self.merchant_counters[key]
        
        return f"MER_{cat_abbr}_{city_abbr}_{num:03d}"
    
    # ============================================================================
    # MERCHANT POOL CREATION
    # ============================================================================
    
    def get_or_create_merchant_pool(self, city: str) -> List[Merchant]:
        """
        Get or create merchant pool for a city
        
        Creates realistic number of merchants based on city tier:
        - Tier 1: 800-1200 merchants
        - Tier 2: 400-600 merchants
        - Tier 3: 200-300 merchants
        
        Args:
            city: City name
            
        Returns:
            List of Merchant objects for the city
        """
        if city in self.merchant_pools:
            return self.merchant_pools[city]
        
        # Get city tier and region
        tier = CITY_TIERS.get(city, 2)
        region = self.city_region_map.get(city, "Central")
        
        # Determine merchant count based on tier
        merchant_count_ranges = {
            1: (800, 1200),
            2: (400, 600),
            3: (200, 300)
        }
        min_count, max_count = merchant_count_ranges[tier]
        target_merchant_count = random.randint(min_count, max_count)
        
        merchants = []
        
        # Add chain merchants (available in this city)
        chain_merchants = self._create_chain_merchants(city, region, tier)
        merchants.extend(chain_merchants)
        
        # Add local merchants to reach target count
        local_count = target_merchant_count - len(chain_merchants)
        local_merchants = self._create_local_merchants(city, region, tier, local_count)
        merchants.extend(local_merchants)
        
        self.merchant_pools[city] = merchants
        return merchants
    
    def _create_chain_merchants(self, city: str, region: str, tier: int) -> List[Merchant]:
        """Create chain merchants available in this city"""
        chain_merchants = []
        
        for merchant_name, details in CHAIN_MERCHANT_DETAILS.items():
            # Check if merchant operates in this region and tier
            if region in details["regions"] and tier in details["tiers"]:
                # Determine category and subcategory
                category, subcategory = self._find_merchant_category(merchant_name)
                
                if category and subcategory:
                    merchant_id = self.generate_merchant_id(category, city)
                    reputation = details["reputation"]
                    
                    merchant = Merchant(
                        merchant_id=merchant_id,
                        name=merchant_name,
                        category=category,
                        subcategory=subcategory,
                        city=city,
                        merchant_type="chain",
                        reputation=reputation,
                        region=region,
                        tier=tier
                    )
                    chain_merchants.append(merchant)
        
        return chain_merchants
    
    def _create_local_merchants(self, city: str, region: str, tier: int, count: int) -> List[Merchant]:
        """Create local merchants specific to this city"""
        local_merchants = []
        
        # Get all categories
        categories = list(MERCHANT_SUBCATEGORIES.keys())
        
        for _ in range(count):
            # Select random category (weighted by popularity)
            category = random.choice(categories)
            
            # Select random subcategory
            subcategories = list(MERCHANT_SUBCATEGORIES[category].keys())
            subcategory = random.choice(subcategories)
            
            # Get merchant names from subcategory
            merchant_names = MERCHANT_SUBCATEGORIES[category][subcategory]
            
            # Prefer local-sounding names for local merchants
            local_preference = [m for m in merchant_names if "Local" in m or "Store" in m or "Shop" in m]
            
            if local_preference and random.random() < 0.6:
                merchant_name = random.choice(local_preference)
            else:
                merchant_name = random.choice(merchant_names)
            
            # Generate unique ID
            merchant_id = self.generate_merchant_id(category, city)
            
            # Local merchants have varied reputation (0.4 to 0.85)
            reputation = round(random.uniform(0.4, 0.85), 2)
            
            merchant = Merchant(
                merchant_id=merchant_id,
                name=merchant_name,
                category=category,
                subcategory=subcategory,
                city=city,
                merchant_type="local",
                reputation=reputation,
                region=region,
                tier=tier
            )
            local_merchants.append(merchant)
        
        return local_merchants
    
    def _find_merchant_category(self, merchant_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Find category and subcategory for a merchant name"""
        for category, subcategories in MERCHANT_SUBCATEGORIES.items():
            for subcategory, merchants in subcategories.items():
                if merchant_name in merchants:
                    return category, subcategory
        return None, None
    
    # ============================================================================
    # MERCHANT SELECTION WITH LOYALTY
    # ============================================================================
    
    def select_merchant(
        self,
        customer: CustomerProfile,
        category: str,
        city: str,
        is_online: bool = False
    ) -> Merchant:
        """
        Select merchant based on customer loyalty and city availability
        
        Loyalty behavior:
        - 60-80% chance to return to favorite merchant (if exists)
        - Higher loyalty for groceries, healthcare, utilities
        - Lower loyalty for shopping, entertainment
        - Chain merchants preferred for online transactions
        
        Args:
            customer: Customer profile
            category: Transaction category
            city: Transaction city
            is_online: Whether transaction is online
            
        Returns:
            Selected Merchant object
        """
        # Get or create merchant pool for this city
        merchants = self.get_or_create_merchant_pool(city)
        
        # Filter merchants by category
        category_merchants = [m for m in merchants if m.category == category]
        
        if not category_merchants:
            # Fallback: create a merchant if none exist
            return self._create_fallback_merchant(category, city)
        
        # Check for customer loyalty
        customer_id = customer.customer_id
        loyalty_score = CATEGORY_LOYALTY_SCORES.get(category, 0.5)
        
        # Initialize customer favorites if not exists
        if customer_id not in self.customer_favorites:
            self.customer_favorites[customer_id] = {}
        
        # Check if customer has favorite merchants in this category
        if category in self.customer_favorites[customer_id]:
            favorite_ids = self.customer_favorites[customer_id][category]
            
            # Find favorite merchants still in pool
            favorite_merchants = [m for m in category_merchants if m.merchant_id in favorite_ids]
            
            # Return to favorite with probability based on loyalty score
            if favorite_merchants and random.random() < loyalty_score:
                return random.choice(favorite_merchants)
        
        # New merchant selection
        # For online transactions, prefer chain merchants and e-commerce
        if is_online:
            online_merchants = [
                m for m in category_merchants
                if m.merchant_type == "chain" or "Online" in m.subcategory or "E-commerce" in m.subcategory
            ]
            if online_merchants:
                category_merchants = online_merchants
        
        # Weight selection by reputation
        weights = [m.reputation for m in category_merchants]
        selected_merchant = random.choices(category_merchants, weights=weights, k=1)[0]
        
        # Add to customer favorites (maintain 3-5 favorites per category)
        if category not in self.customer_favorites[customer_id]:
            self.customer_favorites[customer_id][category] = []
        
        if selected_merchant.merchant_id not in self.customer_favorites[customer_id][category]:
            self.customer_favorites[customer_id][category].append(selected_merchant.merchant_id)
            
            # Limit to 5 favorites per category (remove oldest if needed)
            if len(self.customer_favorites[customer_id][category]) > 5:
                self.customer_favorites[customer_id][category].pop(0)
        
        return selected_merchant
    
    def _create_fallback_merchant(self, category: str, city: str) -> Merchant:
        """Create a fallback merchant if none exist in the pool"""
        tier = CITY_TIERS.get(city, 2)
        region = self.city_region_map.get(city, "Central")
        
        # Try to find a merchant name from the category
        if category in MERCHANT_SUBCATEGORIES:
            subcategories = list(MERCHANT_SUBCATEGORIES[category].keys())
            subcategory = random.choice(subcategories)
            merchant_names = MERCHANT_SUBCATEGORIES[category][subcategory]
            merchant_name = random.choice(merchant_names)
        else:
            # Fallback to old system
            subcategory = "General"
            if category in INDIAN_MERCHANTS:
                merchant_name = random.choice(INDIAN_MERCHANTS[category])
            else:
                merchant_name = f"{category} Store"
        
        merchant_id = self.generate_merchant_id(category, city)
        reputation = round(random.uniform(0.5, 0.8), 2)
        
        merchant = Merchant(
            merchant_id=merchant_id,
            name=merchant_name,
            category=category,
            subcategory=subcategory,
            city=city,
            merchant_type="local",
            reputation=reputation,
            region=region,
            tier=tier
        )
        
        # Add to pool
        if city not in self.merchant_pools:
            self.merchant_pools[city] = []
        self.merchant_pools[city].append(merchant)
        
        return merchant
    
    # ============================================================================
    # STATISTICS & ANALYSIS
    # ============================================================================
    
    def get_merchant_stats(self, city: str) -> Dict:
        """Get merchant statistics for a city"""
        merchants = self.get_or_create_merchant_pool(city)
        
        chain_count = sum(1 for m in merchants if m.merchant_type == "chain")
        local_count = sum(1 for m in merchants if m.merchant_type == "local")
        
        avg_reputation = np.mean([m.reputation for m in merchants])
        
        category_distribution = {}
        for m in merchants:
            category_distribution[m.category] = category_distribution.get(m.category, 0) + 1
        
        return {
            "total_merchants": len(merchants),
            "chain_merchants": chain_count,
            "local_merchants": local_count,
            "chain_percentage": round(chain_count / len(merchants) * 100, 1) if merchants else 0,
            "average_reputation": round(avg_reputation, 2),
            "category_distribution": category_distribution,
            "city_tier": CITY_TIERS.get(city, 2)
        }
    
    def get_customer_loyalty_stats(self, customer_id: str) -> Dict:
        """Get loyalty statistics for a customer"""
        if customer_id not in self.customer_favorites:
            return {
                "total_favorite_categories": 0,
                "total_favorite_merchants": 0,
                "favorites_by_category": {}
            }
        
        favorites = self.customer_favorites[customer_id]
        total_favorites = sum(len(merchants) for merchants in favorites.values())
        
        return {
            "total_favorite_categories": len(favorites),
            "total_favorite_merchants": total_favorites,
            "favorites_by_category": {cat: len(merchants) for cat, merchants in favorites.items()}
        }
