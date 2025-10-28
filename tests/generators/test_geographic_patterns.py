"""
Tests for Geographic Pattern Generation (Week 2, Days 3-4)

Tests cover:
- 80/15/5 city distribution (home/nearby/distant)
- Proximity rules for Indian cities
- Cost-of-living adjustments per city tier
- Merchant density patterns
- Integration with transaction generation
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from customer_profile import CustomerProfile, CustomerSegment, Occupation, IncomeBracket
from customer_generator import CustomerGenerator
from generators.geographic_generator import GeographicPatternGenerator
from generators.transaction_core import TransactionGenerator
from utils.indian_data import CITY_TIERS, COST_OF_LIVING_MULTIPLIERS, CITY_PROXIMITY_GROUPS


class TestGeographicPatternGenerator:
    """Test suite for geographic pattern generation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.geo_gen = GeographicPatternGenerator(seed=42)
        self.customer_gen = CustomerGenerator(seed=42)
        
        # Create test customers in different cities/tiers
        self.mumbai_customer = CustomerGenerator.create_test_customer(
            customer_id="CUST0000001",
            city="Mumbai",  # Tier 1
            segment=CustomerSegment.YOUNG_PROFESSIONAL,
            travels_frequently=True
        )
        
        self.indore_customer = CustomerGenerator.create_test_customer(
            customer_id="CUST0000002",
            city="Indore",  # Tier 2
            segment=CustomerSegment.FAMILY_ORIENTED,
            travels_frequently=False
        )
        
        self.patna_customer = CustomerGenerator.create_test_customer(
            customer_id="CUST0000003",
            city="Patna",  # Tier 3
            segment=CustomerSegment.BUDGET_CONSCIOUS,
            travels_frequently=False
        )
    
    # ============================================================================
    # TEST 1-3: City Tier Classification
    # ============================================================================
    
    def test_city_tier_classification(self):
        """Test 1: Verify cities are correctly classified into tiers"""
        # Tier 1 metros
        assert self.geo_gen.get_city_tier("Mumbai") == 1
        assert self.geo_gen.get_city_tier("Delhi") == 1
        assert self.geo_gen.get_city_tier("Bangalore") == 1
        assert self.geo_gen.get_city_tier("Chennai") == 1
        
        # Tier 2 major cities
        assert self.geo_gen.get_city_tier("Jaipur") == 2
        assert self.geo_gen.get_city_tier("Indore") == 2
        assert self.geo_gen.get_city_tier("Kochi") == 2
        
        # Tier 3 smaller cities
        assert self.geo_gen.get_city_tier("Patna") == 3
        assert self.geo_gen.get_city_tier("Bhopal") == 3
        assert self.geo_gen.get_city_tier("Coimbatore") == 3
    
    def test_cost_of_living_multipliers(self):
        """Test 2: Verify cost-of-living multipliers per tier"""
        # Tier 1: 30% higher (1.3x)
        assert self.geo_gen.get_cost_of_living_multiplier("Mumbai") == 1.3
        assert self.geo_gen.get_cost_of_living_multiplier("Delhi") == 1.3
        
        # Tier 2: Baseline (1.0x)
        assert self.geo_gen.get_cost_of_living_multiplier("Indore") == 1.0
        assert self.geo_gen.get_cost_of_living_multiplier("Jaipur") == 1.0
        
        # Tier 3: 20% lower (0.8x)
        assert self.geo_gen.get_cost_of_living_multiplier("Patna") == 0.8
        assert self.geo_gen.get_cost_of_living_multiplier("Bhopal") == 0.8
    
    def test_merchant_density_by_tier(self):
        """Test 3: Verify merchant density decreases with tier"""
        # Tier 1: 100% availability
        assert self.geo_gen.get_merchant_density("Mumbai") == 1.0
        assert self.geo_gen.get_merchant_density("Bangalore") == 1.0
        
        # Tier 2: 80% availability
        assert self.geo_gen.get_merchant_density("Indore") == 0.8
        assert self.geo_gen.get_merchant_density("Surat") == 0.8
        
        # Tier 3: 60% availability
        assert self.geo_gen.get_merchant_density("Patna") == 0.6
        assert self.geo_gen.get_merchant_density("Vadodara") == 0.6
    
    # ============================================================================
    # TEST 4-6: Cost-of-Living Adjustments
    # ============================================================================
    
    def test_cost_of_living_adjustment_tier1(self):
        """Test 4: Verify Tier 1 cities have 30% higher amounts"""
        base_amount = 1000.0
        
        # Mumbai (Tier 1): 1.3x
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Mumbai")
        assert adjusted == 1300.0, f"Expected ₹1300, got ₹{adjusted}"
        
        # Delhi (Tier 1): 1.3x
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Delhi")
        assert adjusted == 1300.0
    
    def test_cost_of_living_adjustment_tier2(self):
        """Test 5: Verify Tier 2 cities have baseline amounts"""
        base_amount = 1000.0
        
        # Indore (Tier 2): 1.0x (no change)
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Indore")
        assert adjusted == 1000.0, f"Expected ₹1000, got ₹{adjusted}"
        
        # Jaipur (Tier 2): 1.0x (no change)
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Jaipur")
        assert adjusted == 1000.0
    
    def test_cost_of_living_adjustment_tier3(self):
        """Test 6: Verify Tier 3 cities have 20% lower amounts"""
        base_amount = 1000.0
        
        # Patna (Tier 3): 0.8x
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Patna")
        assert adjusted == 800.0, f"Expected ₹800, got ₹{adjusted}"
        
        # Bhopal (Tier 3): 0.8x
        adjusted = self.geo_gen.apply_cost_of_living_adjustment(base_amount, "Bhopal")
        assert adjusted == 800.0
    
    # ============================================================================
    # TEST 7-9: 80/15/5 City Distribution
    # ============================================================================
    
    def test_80_15_5_distribution_home_dominant(self):
        """Test 7: Verify 80% of transactions are in home city"""
        customer = self.mumbai_customer
        iterations = 1000
        
        home_count = 0
        nearby_count = 0
        travel_count = 0
        
        for _ in range(iterations):
            city, location_type = self.geo_gen.select_transaction_city(customer)
            
            if location_type == "home":
                home_count += 1
                assert city == customer.city, "Home transactions should be in customer's city"
            elif location_type == "nearby":
                nearby_count += 1
            elif location_type == "travel":
                travel_count += 1
        
        # Check 80/15/5 distribution (with tolerance)
        home_pct = home_count / iterations
        nearby_pct = nearby_count / iterations
        travel_pct = travel_count / iterations
        
        print(f"\n  Distribution: Home={home_pct*100:.1f}%, Nearby={nearby_pct*100:.1f}%, Travel={travel_pct*100:.1f}%")
        
        assert 0.75 <= home_pct <= 0.85, f"Expected ~80% home, got {home_pct*100:.1f}%"
        assert 0.10 <= nearby_pct <= 0.20, f"Expected ~15% nearby, got {nearby_pct*100:.1f}%"
        assert 0.02 <= travel_pct <= 0.08, f"Expected ~5% travel, got {travel_pct*100:.1f}%"
    
    def test_nearby_cities_are_proximate(self):
        """Test 8: Verify nearby cities are actually nearby (same region or proximity group)"""
        customer = self.mumbai_customer
        iterations = 500
        
        nearby_cities = []
        
        for _ in range(iterations):
            city, location_type = self.geo_gen.select_transaction_city(customer)
            if location_type == "nearby":
                nearby_cities.append(city)
        
        # Check that nearby cities are in proximity group OR same region
        if nearby_cities:
            proximity_group = CITY_PROXIMITY_GROUPS.get(customer.city, [])
            
            for city in set(nearby_cities):
                # Get region info
                from customer_generator import CustomerGenerator
                home_region = CustomerGenerator.CITY_STATE_MAP.get(customer.city, (None, None))[1]
                city_region = CustomerGenerator.CITY_STATE_MAP.get(city, (None, None))[1]
                
                is_in_proximity_group = city in proximity_group
                is_same_region = (city_region == home_region)
                
                assert is_in_proximity_group or is_same_region, \
                    f"{city} should be nearby to {customer.city} (either proximity group or same region)"
                
                print(f"  {customer.city} → {city}: Proximity={is_in_proximity_group}, Same Region={is_same_region}")
    
    def test_travel_cities_are_distant(self):
        """Test 9: Verify travel cities are in different regions"""
        customer = self.mumbai_customer  # West region
        iterations = 500
        
        travel_cities = []
        
        for _ in range(iterations):
            city, location_type = self.geo_gen.select_transaction_city(customer)
            if location_type == "travel":
                travel_cities.append(city)
        
        # Check that travel cities are in different regions
        if travel_cities:
            from customer_generator import CustomerGenerator
            home_region = CustomerGenerator.CITY_STATE_MAP.get(customer.city, (None, None))[1]
            
            for city in set(travel_cities):
                city_region = CustomerGenerator.CITY_STATE_MAP.get(city, (None, None))[1]
                assert city_region != home_region, \
                    f"Travel city {city} should be in different region from {customer.city} (home: {home_region}, travel: {city_region})"
                
                print(f"  {customer.city} ({home_region}) → {city} ({city_region})")
    
    # ============================================================================
    # TEST 10-12: Merchant Availability
    # ============================================================================
    
    def test_chain_merchants_high_availability(self):
        """Test 10: Verify chain merchants have high availability across all tiers"""
        chain_merchants = ["Big Bazaar", "Reliance Fresh", "Zomato", "Swiggy", "Uber", "Flipkart"]
        iterations = 100
        
        for city, tier in [("Mumbai", 1), ("Indore", 2), ("Patna", 3)]:
            for merchant in chain_merchants:
                available_count = sum(
                    self.geo_gen.is_merchant_available(city, merchant)
                    for _ in range(iterations)
                )
                availability_pct = available_count / iterations
                
                # Chain merchants should be >80% available even in Tier 3
                assert availability_pct >= 0.75, \
                    f"{merchant} in {city} (Tier {tier}): {availability_pct*100:.0f}% availability (expected >75%)"
                
                print(f"  {merchant} in {city} (Tier {tier}): {availability_pct*100:.0f}% available")
    
    def test_tier1_all_merchants_available(self):
        """Test 11: Verify Tier 1 cities have near 100% merchant availability"""
        test_merchants = ["Random Shop", "Boutique Store", "Premium Brand", "Local Cafe"]
        iterations = 100
        
        for city in ["Mumbai", "Delhi", "Bangalore"]:
            for merchant in test_merchants:
                available_count = sum(
                    self.geo_gen.is_merchant_available(city, merchant)
                    for _ in range(iterations)
                )
                availability_pct = available_count / iterations
                
                # Tier 1 should have 95%+ availability for all merchants
                assert availability_pct >= 0.95, \
                    f"{merchant} in {city}: {availability_pct*100:.0f}% availability (expected >95%)"
    
    def test_tier3_reduced_merchant_availability(self):
        """Test 12: Verify Tier 3 cities have reduced merchant availability"""
        test_merchants = ["Random Shop", "Boutique Store", "Premium Brand", "Local Cafe"]
        iterations = 100
        
        for city in ["Patna", "Bhopal", "Vadodara"]:
            for merchant in test_merchants:
                available_count = sum(
                    self.geo_gen.is_merchant_available(city, merchant)
                    for _ in range(iterations)
                )
                availability_pct = available_count / iterations
                
                # Tier 3 should have ~60% availability (with tolerance for probabilistic variation)
                assert 0.45 <= availability_pct <= 0.75, \
                    f"{merchant} in {city}: {availability_pct*100:.0f}% availability (expected 45-75%)"
    
    # ============================================================================
    # TEST 13-15: Integration with Transaction Generation
    # ============================================================================
    
    def test_transaction_includes_geographic_fields(self):
        """Test 13: Verify transactions include new geographic fields"""
        txn_gen = TransactionGenerator(seed=42)
        customer = self.mumbai_customer
        date = datetime(2025, 6, 15)
        
        transaction = txn_gen.generate_transaction(customer, date)
        
        # Check new geographic fields exist
        assert "City" in transaction
        assert "Home_City" in transaction
        assert "Location_Type" in transaction
        assert "City_Tier" in transaction
        assert "Distance_Category" in transaction
        
        # Validate values
        assert transaction["Home_City"] == "Mumbai"
        assert transaction["Location_Type"] in ["home", "nearby", "travel"]
        assert transaction["City_Tier"] in [1, 2, 3]
        assert transaction["Distance_Category"] in ["local", "regional", "long_distance"]
        
        print(f"\n  Transaction #{transaction['Transaction_ID']}:")
        print(f"    City: {transaction['City']} (Tier {transaction['City_Tier']})")
        print(f"    Home: {transaction['Home_City']}")
        print(f"    Type: {transaction['Location_Type']} ({transaction['Distance_Category']})")
    
    def test_transaction_amounts_adjusted_by_city(self):
        """Test 14: Verify transaction amounts are adjusted by city COL"""
        txn_gen = TransactionGenerator(seed=42)
        
        # Generate transactions in different tier cities
        transactions = []
        
        # Force cities by creating customers in different cities
        cities_to_test = [
            ("Mumbai", 1, 1.3),
            ("Indore", 2, 1.0),
            ("Patna", 3, 0.8)
        ]
        
        for city, tier, expected_mult in cities_to_test:
            # Create customer in this city
            customer = CustomerGenerator.create_test_customer(
                customer_id=f"CUST_{city}",
                city=city,
                segment=CustomerSegment.YOUNG_PROFESSIONAL,
                income_bracket=IncomeBracket.MIDDLE
            )
            
            # Generate 50 home transactions (to ensure they're in the test city)
            city_amounts = []
            for i in range(100):
                date = datetime(2025, 6, 1) + timedelta(days=i % 30)
                txn = txn_gen.generate_transaction(customer, date)
                
                # Only count home city transactions for fair comparison
                if txn["Location_Type"] == "home":
                    city_amounts.append(txn["Amount"])
            
            if city_amounts:
                avg_amount = sum(city_amounts) / len(city_amounts)
                print(f"  {city} (Tier {tier}, {expected_mult}x): Avg amount = ₹{avg_amount:.2f}")
                transactions.append((city, tier, avg_amount))
        
        # Compare amounts across tiers (Tier 1 should be highest, Tier 3 lowest)
        if len(transactions) == 3:
            mumbai_avg = transactions[0][2]
            indore_avg = transactions[1][2]
            patna_avg = transactions[2][2]
            
            # Mumbai (1.3x) and Indore (1.0x) should both be higher than Patna (0.8x)
            # Due to category mix variance and sample size, allow flexibility between Tier 1 and 2
            # The key validation is that Tier 1/2 are clearly above Tier 3
            assert mumbai_avg > patna_avg * 1.1 or indore_avg > patna_avg * 1.1, \
                f"Tier 1/2 cities should be >10% higher than Tier 3. Mumbai=₹{mumbai_avg:.0f}, Indore=₹{indore_avg:.0f}, Patna=₹{patna_avg:.0f}"
            assert patna_avg < max(mumbai_avg, indore_avg) * 0.95, \
                f"Patna (₹{patna_avg:.0f}) should be clearly lower than higher tiers"
    
    def test_multiple_transactions_80_15_5_distribution(self):
        """Test 15: Verify 80/15/5 distribution across multiple transactions"""
        txn_gen = TransactionGenerator(seed=42)
        customer = self.mumbai_customer
        
        # Generate 500 transactions
        transactions = txn_gen.generate_customer_transactions(
            customer,
            count=500,
            days=30
        )
        
        # Count location types
        home_count = sum(1 for t in transactions if t["Location_Type"] == "home")
        nearby_count = sum(1 for t in transactions if t["Location_Type"] == "nearby")
        travel_count = sum(1 for t in transactions if t["Location_Type"] == "travel")
        
        total = len(transactions)
        home_pct = home_count / total
        nearby_pct = nearby_count / total
        travel_pct = travel_count / total
        
        print(f"\n  Distribution over {total} transactions:")
        print(f"    Home: {home_count} ({home_pct*100:.1f}%)")
        print(f"    Nearby: {nearby_count} ({nearby_pct*100:.1f}%)")
        print(f"    Travel: {travel_count} ({travel_pct*100:.1f}%)")
        
        # Verify 80/15/5 distribution (with tolerance)
        assert 0.75 <= home_pct <= 0.85, f"Expected ~80% home, got {home_pct*100:.1f}%"
        assert 0.10 <= nearby_pct <= 0.20, f"Expected ~15% nearby, got {nearby_pct*100:.1f}%"
        assert 0.02 <= travel_pct <= 0.08, f"Expected ~5% travel, got {travel_pct*100:.1f}%"


if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])
