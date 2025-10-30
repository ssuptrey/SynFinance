"""
Customer Generator for SynFinance
Generates realistic customer profiles based on defined segments and demographics

Week 1, Day 3-4: Customer Generator Implementation
"""

import random
from typing import List, Tuple
from datetime import datetime

from src.customer_profile import (
    CustomerProfile,
    CustomerSegment,
    IncomeBracket,
    Occupation,
    RiskProfile,
    DigitalSavviness,
    SEGMENT_PROFILES,
    SEGMENT_DISTRIBUTION,
    get_income_range,
    calculate_risk_profile
)

from src.constants import INDIAN_CITIES, TRANSACTION_CATEGORIES, PAYMENT_MODES


class CustomerGenerator:
    """
    Generates realistic customer profiles with consistent behavioral patterns
    """
    
    # Indian states mapped to regions
    INDIAN_STATES = {
        "North": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Uttarakhand", "Himachal Pradesh", "Jammu & Kashmir"],
        "South": ["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Telangana"],
        "East": ["West Bengal", "Bihar", "Odisha", "Jharkhand", "Assam"],
        "West": ["Maharashtra", "Gujarat", "Rajasthan", "Goa"],
        "Central": ["Madhya Pradesh", "Chhattisgarh"]
    }
    
    # City to state mapping (for existing 20 cities in config)
    CITY_STATE_MAP = {
        "Mumbai": ("Maharashtra", "West"),
        "Delhi": ("Delhi", "North"),
        "Bangalore": ("Karnataka", "South"),
        "Hyderabad": ("Telangana", "South"),
        "Chennai": ("Tamil Nadu", "South"),
        "Kolkata": ("West Bengal", "East"),
        "Pune": ("Maharashtra", "West"),
        "Ahmedabad": ("Gujarat", "West"),
        "Jaipur": ("Rajasthan", "West"),
        "Lucknow": ("Uttar Pradesh", "North"),
        "Surat": ("Gujarat", "West"),
        "Chandigarh": ("Chandigarh", "North"),
        "Indore": ("Madhya Pradesh", "Central"),
        "Kochi": ("Kerala", "South"),
        "Bhopal": ("Madhya Pradesh", "Central"),
        "Nagpur": ("Maharashtra", "West"),
        "Visakhapatnam": ("Andhra Pradesh", "South"),
        "Patna": ("Bihar", "East"),
        "Vadodara": ("Gujarat", "West"),
        "Coimbatore": ("Tamil Nadu", "South")
    }
    
    def __init__(self, seed: int = None):
        """Initialize customer generator with optional seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
        
        self.customer_counter = 0
        self.generated_customers: List[CustomerProfile] = []
    
    def generate_customer_id(self) -> str:
        """Generate unique customer ID"""
        self.customer_counter += 1
        return f"CUST{self.customer_counter:07d}"
    
    def select_segment(self) -> CustomerSegment:
        """Select customer segment based on distribution weights"""
        segments = list(SEGMENT_DISTRIBUTION.keys())
        weights = list(SEGMENT_DISTRIBUTION.values())
        return random.choices(segments, weights=weights, k=1)[0]
    
    def generate_age(self, segment: CustomerSegment) -> int:
        """Generate age within segment's range"""
        profile = SEGMENT_PROFILES[segment]
        age_min, age_max = profile["age_range"]
        return random.randint(age_min, age_max)
    
    def generate_gender(self) -> str:
        """Generate gender with realistic distribution"""
        # Approximate Indian distribution: 52% Male, 48% Female, <1% Other
        return random.choices(
            ["Male", "Female", "Other"],
            weights=[0.52, 0.47, 0.01],
            k=1
        )[0]
    
    def generate_location(self) -> Tuple[str, str, str]:
        """
        Generate city, state, and region
        Returns: (city, state, region)
        """
        city = random.choice(INDIAN_CITIES)
        state, region = self.CITY_STATE_MAP.get(city, ("Unknown", "Central"))
        return city, state, region
    
    def select_income_bracket(self, segment: CustomerSegment) -> IncomeBracket:
        """Select income bracket from segment's options"""
        profile = SEGMENT_PROFILES[segment]
        return random.choice(profile["income_brackets"])
    
    def generate_monthly_income(self, bracket: IncomeBracket) -> float:
        """Generate specific monthly income within bracket range"""
        min_income, max_income = get_income_range(bracket)
        # Use log-normal distribution for realistic income spread
        mean = (min_income + max_income) / 2
        std = (max_income - min_income) / 4
        income = random.gauss(mean, std)
        # Clamp to bracket range
        return max(min_income, min(max_income, income))
    
    def select_occupation(self, segment: CustomerSegment, age: int) -> Occupation:
        """Select occupation appropriate for segment and age"""
        profile = SEGMENT_PROFILES[segment]
        available_occupations = profile["occupations"]
        
        # Age-based filtering
        if age < 22:
            # Young people are usually students
            return Occupation.STUDENT if Occupation.STUDENT in available_occupations else random.choice(available_occupations)
        elif age > 60:
            # Older people likely retired or business owners
            if Occupation.RETIRED in available_occupations:
                return Occupation.RETIRED if random.random() < 0.6 else random.choice(available_occupations)
        
        return random.choice(available_occupations)
    
    def select_digital_savviness(self, segment: CustomerSegment, age: int) -> DigitalSavviness:
        """Determine digital savviness based on segment and age"""
        profile = SEGMENT_PROFILES[segment]
        available_levels = profile["digital_savviness"]
        
        # Age-based adjustment
        if age < 30:
            # Younger people skew higher
            if DigitalSavviness.HIGH in available_levels:
                return DigitalSavviness.HIGH if random.random() < 0.7 else random.choice(available_levels)
        elif age > 60:
            # Older people skew lower
            if DigitalSavviness.LOW in available_levels:
                return DigitalSavviness.LOW if random.random() < 0.6 else random.choice(available_levels)
        
        return random.choice(available_levels)
    
    def generate_spending_behavior(
        self, 
        segment: CustomerSegment, 
        income_bracket: IncomeBracket,
        monthly_income: float
    ) -> Tuple[float, int]:
        """
        Generate average transaction amount and monthly transaction count
        Returns: (avg_transaction_amount, monthly_transaction_count)
        """
        profile = SEGMENT_PROFILES[segment]
        
        # Base ranges from segment profile
        amt_min, amt_max = profile["avg_transaction_range"]
        count_min, count_max = profile["monthly_transactions"]
        
        # Adjust based on actual income
        income_multiplier = monthly_income / 100000  # Baseline at 1L/month
        income_multiplier = max(0.5, min(2.0, income_multiplier))  # Clamp between 0.5x and 2x
        
        # Generate values
        avg_amount = random.uniform(amt_min, amt_max) * income_multiplier
        monthly_count = random.randint(count_min, count_max)
        
        return round(avg_amount, 2), monthly_count
    
    def select_preferred_categories(self, segment: CustomerSegment) -> List[str]:
        """Select 3-5 preferred categories for the customer"""
        profile = SEGMENT_PROFILES[segment]
        preferred = profile["preferred_categories"]
        
        # Select 3-5 categories, prioritizing segment preferences
        num_categories = random.randint(3, 5)
        selected = random.sample(preferred, min(len(preferred), num_categories))
        
        # Maybe add 1-2 random categories
        if random.random() < 0.3 and num_categories < 5:
            other_categories = [cat for cat in TRANSACTION_CATEGORIES if cat not in selected]
            if other_categories:
                selected.append(random.choice(other_categories))
        
        return selected
    
    def select_preferred_payment_modes(
        self, 
        segment: CustomerSegment,
        digital_savviness: DigitalSavviness
    ) -> List[str]:
        """Select preferred payment modes based on segment and digital savviness"""
        profile = SEGMENT_PROFILES[segment]
        preferred = profile["preferred_payment_modes"]
        
        # Adjust based on digital savviness
        if digital_savviness == DigitalSavviness.LOW:
            # Prefer cash and debit card
            modes = ["Cash", "Debit Card", "UPI"] if "UPI" in preferred else ["Cash", "Debit Card"]
        elif digital_savviness == DigitalSavviness.HIGH:
            # Use all modern modes
            modes = preferred.copy()
            # High chance of using digital wallet
            if "Digital Wallet" not in modes:
                modes.append("Digital Wallet")
        else:
            # Medium: use segment preferences
            modes = preferred.copy()
        
        # Return 2-4 modes
        num_modes = random.randint(2, min(4, len(modes)))
        return random.sample(modes, num_modes)
    
    def generate_time_patterns(
        self, 
        occupation: Occupation,
        segment: CustomerSegment
    ) -> List[int]:
        """
        Generate preferred shopping hours based on occupation
        Returns: List of preferred hours (0-23)
        """
        if occupation == Occupation.STUDENT:
            # Students: afternoons and evenings
            return [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        
        elif occupation in [Occupation.SALARIED_EMPLOYEE, Occupation.PROFESSIONAL]:
            # Working professionals: early morning, lunch, late evening
            return [7, 8, 12, 13, 18, 19, 20, 21, 22]
        
        elif occupation == Occupation.BUSINESS_OWNER:
            # Business owners: varied, throughout the day
            return [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        elif occupation == Occupation.RETIRED:
            # Retirees: morning and early afternoon
            return [8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        elif occupation == Occupation.HOMEMAKER:
            # Homemakers: morning and afternoon
            return [9, 10, 11, 12, 13, 14, 15, 16, 17]
        
        else:
            # Default: afternoon to evening
            return [12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    def generate_loyalty_traits(
        self,
        segment: CustomerSegment,
        age: int
    ) -> Tuple[float, bool, bool]:
        """
        Generate merchant loyalty, brand consciousness, and impulse buying
        Returns: (merchant_loyalty, brand_conscious, impulse_buyer)
        """
        profile = SEGMENT_PROFILES[segment]
        
        # Base values from segment
        brand_conscious = profile["brand_conscious"]
        impulse_buyer = profile["impulse_buyer"]
        
        # Merchant loyalty influenced by age and segment
        if age > 50:
            merchant_loyalty = random.uniform(0.6, 0.9)  # Older people more loyal
        elif segment == CustomerSegment.TECH_SAVVY_MILLENNIAL:
            merchant_loyalty = random.uniform(0.2, 0.5)  # Millennials explore more
        else:
            merchant_loyalty = random.uniform(0.4, 0.7)  # Moderate loyalty
        
        return merchant_loyalty, brand_conscious, impulse_buyer
    
    def generate_geographic_behavior(
        self,
        segment: CustomerSegment,
        income_bracket: IncomeBracket
    ) -> Tuple[bool, float]:
        """
        Generate travel frequency and online shopping preference
        Returns: (travels_frequently, online_shopping_preference)
        """
        profile = SEGMENT_PROFILES[segment]
        
        travels_frequently = profile["travels_frequently"]
        online_preference = profile["online_preference"]
        
        # Add some variation (±20%)
        online_preference = max(0.0, min(1.0, online_preference + random.uniform(-0.2, 0.2)))
        
        return travels_frequently, online_preference
    
    def generate_customer(self, segment: CustomerSegment = None) -> CustomerProfile:
        """
        Generate a complete customer profile
        
        Args:
            segment: Optional specific segment, otherwise randomly selected
        
        Returns:
            CustomerProfile with all fields populated
        """
        # Select segment
        if segment is None:
            segment = self.select_segment()
        
        # Generate core identity
        customer_id = self.generate_customer_id()
        
        # Demographics
        age = self.generate_age(segment)
        gender = self.generate_gender()
        city, state, region = self.generate_location()
        
        # Economic profile
        income_bracket = self.select_income_bracket(segment)
        monthly_income = self.generate_monthly_income(income_bracket)
        occupation = self.select_occupation(segment, age)
        
        # Behavioral attributes
        risk_profile = calculate_risk_profile(income_bracket, age, occupation)
        digital_savviness = self.select_digital_savviness(segment, age)
        
        # Spending behavior
        avg_amount, monthly_count = self.generate_spending_behavior(
            segment, income_bracket, monthly_income
        )
        
        # Preferences
        preferred_categories = self.select_preferred_categories(segment)
        preferred_payment_modes = self.select_preferred_payment_modes(segment, digital_savviness)
        
        # Time patterns
        preferred_hours = self.generate_time_patterns(occupation, segment)
        profile = SEGMENT_PROFILES[segment]
        weekend_shopper = profile["weekend_shopper"]
        
        # Loyalty traits
        merchant_loyalty, brand_conscious, impulse_buyer = self.generate_loyalty_traits(segment, age)
        
        # Geographic behavior
        travels_frequently, online_preference = self.generate_geographic_behavior(segment, income_bracket)
        
        # Create customer profile
        customer = CustomerProfile(
            customer_id=customer_id,
            age=age,
            gender=gender,
            city=city,
            state=state,
            region=region,
            income_bracket=income_bracket,
            occupation=occupation,
            monthly_income=monthly_income,
            segment=segment,
            risk_profile=risk_profile,
            digital_savviness=digital_savviness,
            avg_transaction_amount=avg_amount,
            monthly_transaction_count=monthly_count,
            preferred_categories=preferred_categories,
            preferred_payment_modes=preferred_payment_modes,
            preferred_shopping_hours=preferred_hours,
            weekend_shopper=weekend_shopper,
            merchant_loyalty=merchant_loyalty,
            brand_conscious=brand_conscious,
            impulse_buyer=impulse_buyer,
            travels_frequently=travels_frequently,
            online_shopping_preference=online_preference
        )
        
        self.generated_customers.append(customer)
        return customer
    
    def generate_customers(self, count: int) -> List[CustomerProfile]:
        """
        Generate multiple customers
        
        Args:
            count: Number of customers to generate
        
        Returns:
            List of CustomerProfile objects
        """
        customers = []
        for _ in range(count):
            customer = self.generate_customer()
            customers.append(customer)
        
        return customers
    
    def get_segment_distribution(self) -> dict:
        """Get actual distribution of generated customers by segment"""
        if not self.generated_customers:
            return {}
        
        distribution = {}
        total = len(self.generated_customers)
        
        for segment in CustomerSegment:
            count = sum(1 for c in self.generated_customers if c.segment == segment)
            distribution[segment.value] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        return distribution
    
    @staticmethod
    def create_test_customer(
        occupation: str = "Salaried Employee",
        customer_id: str = "TEST001",
        age: int = 30,
        segment: CustomerSegment = CustomerSegment.YOUNG_PROFESSIONAL,
        digital_savviness: DigitalSavviness = DigitalSavviness.MEDIUM,
        income_bracket: IncomeBracket = IncomeBracket.MIDDLE,
        city: str = "Mumbai",
        **overrides
    ) -> CustomerProfile:
        """
        Create a deterministic customer for testing purposes.
        
        This method bypasses random generation and creates customers with
        specific attributes. Useful for unit tests and integration tests
        that need predictable customer profiles.
        
        Args:
            occupation: Customer occupation
            customer_id: Unique customer ID
            age: Customer age
            segment: Customer segment
            digital_savviness: Digital savviness level
            income_bracket: Income bracket
            city: Home city
            **overrides: Override any other CustomerProfile attributes
            
        Returns:
            CustomerProfile with specified attributes
            
        Example:
            >>> # Create a student customer for testing
            >>> student = CustomerGenerator.create_test_customer(
            ...     occupation="Student",
            ...     age=20,
            ...     segment=CustomerSegment.STUDENT,
            ...     income_bracket=IncomeBracket.LOWER_MIDDLE
            ... )
        """
        # Get city-state mapping
        state, region = CustomerGenerator.CITY_STATE_MAP.get(city, ("Maharashtra", "West"))
        
        # Set defaults based on segment
        profile_defaults = SEGMENT_PROFILES.get(segment, SEGMENT_PROFILES[CustomerSegment.YOUNG_PROFESSIONAL])
        
        # Calculate monthly income
        income_range = get_income_range(income_bracket)
        monthly_income = (income_range[0] + income_range[1]) / 2  # Use midpoint
        
        # Convert occupation string to Occupation enum
        occupation_enum = Occupation.SALARIED_EMPLOYEE  # Default
        for occ in Occupation:
            if occ.value == occupation:
                occupation_enum = occ
                break
        
        # Calculate risk profile (correct parameter order: income_bracket, age, occupation)
        risk_profile = calculate_risk_profile(income_bracket, age, occupation_enum)
        
        # Set average transaction amount based on income
        avg_transaction_amount = monthly_income * 0.05  # 5% of monthly income
        
        # Set monthly transaction count based on segment
        monthly_txn_count = profile_defaults.get("monthly_transactions", 30)
        
        # Build customer profile with defaults
        customer_data = {
            'customer_id': customer_id,
            'age': age,
            'gender': overrides.get('gender', 'Male'),
            'income_bracket': income_bracket,
            'occupation': occupation_enum,  # Use enum, not string
            'city': city,
            'state': state,
            'region': region,
            'segment': segment,
            'digital_savviness': digital_savviness,
            'monthly_income': monthly_income,
            'risk_profile': risk_profile,
            'avg_transaction_amount': avg_transaction_amount,
            'monthly_transaction_count': monthly_txn_count,
            'preferred_categories': overrides.get('preferred_categories', profile_defaults.get("preferred_categories", [])),
            'preferred_payment_modes': overrides.get('preferred_payment_modes', profile_defaults.get("payment_modes", [])),
            'preferred_shopping_hours': overrides.get('preferred_shopping_hours', [9, 10, 18, 19, 20]),
            'weekend_shopper': overrides.get('weekend_shopper', True),
            'merchant_loyalty': overrides.get('merchant_loyalty', 0.7),
            'brand_conscious': overrides.get('brand_conscious', segment in [CustomerSegment.AFFLUENT_SHOPPER, CustomerSegment.YOUNG_PROFESSIONAL]),
            'impulse_buyer': overrides.get('impulse_buyer', segment in [CustomerSegment.TECH_SAVVY_MILLENNIAL, CustomerSegment.STUDENT]),
            'travels_frequently': overrides.get('travels_frequently', segment in [CustomerSegment.YOUNG_PROFESSIONAL, CustomerSegment.AFFLUENT_SHOPPER]),
            'online_shopping_preference': overrides.get('online_shopping_preference', digital_savviness == DigitalSavviness.HIGH),
        }
        
        # Apply any additional overrides
        for key, value in overrides.items():
            if key not in customer_data:  # Only add if not already set
                customer_data[key] = value
        
        return CustomerProfile(**customer_data)
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics about generated customers"""
        if not self.generated_customers:
            return {"error": "No customers generated yet"}
        
        total = len(self.generated_customers)
        
        stats = {
            "total_customers": total,
            "segment_distribution": self.get_segment_distribution(),
            "age_stats": {
                "min": min(c.age for c in self.generated_customers),
                "max": max(c.age for c in self.generated_customers),
                "avg": sum(c.age for c in self.generated_customers) / total
            },
            "income_stats": {
                "min": min(c.monthly_income for c in self.generated_customers),
                "max": max(c.monthly_income for c in self.generated_customers),
                "avg": sum(c.monthly_income for c in self.generated_customers) / total
            },
            "gender_distribution": {},
            "digital_savviness_distribution": {},
            "region_distribution": {}
        }
        
        # Gender distribution
        for gender in ["Male", "Female", "Other"]:
            count = sum(1 for c in self.generated_customers if c.gender == gender)
            stats["gender_distribution"][gender] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        # Digital savviness
        for level in DigitalSavviness:
            count = sum(1 for c in self.generated_customers if c.digital_savviness == level)
            stats["digital_savviness_distribution"][level.value] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        # Region distribution
        for region in ["North", "South", "East", "West", "Central"]:
            count = sum(1 for c in self.generated_customers if c.region == region)
            stats["region_distribution"][region] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        return stats


if __name__ == "__main__":
    """Test customer generation"""
    print("=== SynFinance Customer Generator Test ===\n")
    
    # Create generator
    generator = CustomerGenerator(seed=42)
    
    # Generate 100 test customers
    print("Generating 100 customers...")
    customers = generator.generate_customers(100)
    
    print(f"\n[SUCCESS] Generated {len(customers)} customers\n")
    
    # Show first 5 customers
    print("Sample Customers:")
    print("-" * 80)
    for customer in customers[:5]:
        print(f"\n{customer}")
        print(f"  Income: ₹{customer.monthly_income:,.0f}/month ({customer.income_bracket.value})")
        print(f"  Occupation: {customer.occupation.value}")
        print(f"  Digital: {customer.digital_savviness.value} | Risk: {customer.risk_profile.value}")
        print(f"  Avg Transaction: ₹{customer.avg_transaction_amount:,.2f}")
        print(f"  Preferred Categories: {', '.join(customer.preferred_categories[:3])}")
        print(f"  Payment Modes: {', '.join(customer.preferred_payment_modes)}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("CUSTOMER STATISTICS")
    print("=" * 80)
    
    stats = generator.get_statistics()
    
    print(f"\nTotal Customers: {stats['total_customers']}")
    
    print("\nSegment Distribution:")
    for segment, data in stats["segment_distribution"].items():
        print(f"  {segment}: {data['count']} ({data['percentage']:.1f}%)")
    
    print(f"\nAge Range: {stats['age_stats']['min']}-{stats['age_stats']['max']} years (avg: {stats['age_stats']['avg']:.1f})")
    print(f"Income Range: ₹{stats['income_stats']['min']:,.0f}-₹{stats['income_stats']['max']:,.0f}/month (avg: ₹{stats['income_stats']['avg']:,.0f})")
    
    print("\nDigital Savviness:")
    for level, data in stats["digital_savviness_distribution"].items():
        print(f"  {level}: {data['count']} ({data['percentage']:.1f}%)")
    
    print("\nRegion Distribution:")
    for region, data in stats["region_distribution"].items():
        print(f"  {region}: {data['count']} ({data['percentage']:.1f}%)")
    
    print("\n[COMPLETE] Customer Generator Test Complete!")
