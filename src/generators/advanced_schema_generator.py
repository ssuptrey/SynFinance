"""
Advanced Schema Generator for SynFinance - Week 3

This module generates the 12+ new fields introduced in Week 3:
- Card type (Credit/Debit/NA)
- Transaction status (Approved/Declined/Pending)
- Transaction channel (POS/Online/ATM/Mobile)
- State and Region
- Customer age groups
- Device information (type, app version, browser, OS)
- Risk indicators (distance, timing, velocity)

Version: 2.0 (Week 3)
Created: October 19, 2025
"""

import random
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

# State to region mapping for Indian states
STATE_TO_REGION = {
    # North
    "Delhi": "North", "Punjab": "North", "Haryana": "North",
    "Rajasthan": "North", "Uttarakhand": "North", "Himachal Pradesh": "North",
    "Uttar Pradesh": "North", "Jammu and Kashmir": "North",
    
    # South
    "Tamil Nadu": "South", "Karnataka": "South", "Kerala": "South",
    "Andhra Pradesh": "South", "Telangana": "South", "Puducherry": "South",
    
    # East
    "West Bengal": "East", "Odisha": "East", "Bihar": "East",
    "Jharkhand": "East", "Assam": "East", "Sikkim": "East",
    "Tripura": "East", "Meghalaya": "East", "Manipur": "East",
    "Mizoram": "East", "Nagaland": "East", "Arunachal Pradesh": "East",
    
    # West
    "Maharashtra": "West", "Gujarat": "West", "Goa": "West",
    "Madhya Pradesh": "West", "Chhattisgarh": "West",
    
    # Central (overlap with North/West)
    "Madhya Pradesh": "Central",
}

# City to state mapping (20 major cities)
CITY_TO_STATE = {
    # Tier 1 (Metro)
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bangalore": "Karnataka",
    "Hyderabad": "Telangana", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal",
    "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
    
    # Tier 2 (Major)
    "Jaipur": "Rajasthan", "Surat": "Gujarat", "Lucknow": "Uttar Pradesh",
    "Kanpur": "Uttar Pradesh", "Nagpur": "Maharashtra",
    "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    
    # Tier 3 (Smaller)
    "Patna": "Bihar", "Ludhiana": "Punjab", "Agra": "Uttar Pradesh",
    "Varanasi": "Uttar Pradesh", "Meerut": "Uttar Pradesh",
}

# Browser types with realistic distribution
BROWSERS = {
    "Chrome": 0.65,  # 65%
    "Firefox": 0.12,
    "Safari": 0.10,
    "Edge": 0.08,
    "Opera": 0.03,
    "Other": 0.02,
}

# App versions (realistic versioning)
APP_VERSIONS = [
    "5.2.1", "5.2.0", "5.1.9", "5.1.8", "5.1.7",
    "5.0.5", "5.0.4", "4.9.2", "4.9.1", "4.8.8"
]

# Operating systems with realistic Indian market distribution
OS_DISTRIBUTION = {
    "Android": 0.72,  # 72% - dominant in India
    "iOS": 0.06,      # 6% - premium segment
    "Windows": 0.15,  # 15% - web transactions
    "Other": 0.07,    # 7% - Linux, etc.
}


class AdvancedSchemaGenerator:
    """
    Generate advanced transaction fields (Week 3)
    
    This class handles generation of 12+ new fields:
    - Card type based on payment mode
    - Transaction status with realistic decline rates
    - Transaction channel based on customer behavior
    - State and region from city
    - Customer age groups from age
    - Device information (type, app, browser, OS)
    - Risk indicators (calculated from transaction context)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed"""
        if seed is not None:
            random.seed(seed)
    
    # ========================================================================
    # TIME FIELDS
    # ========================================================================
    
    def generate_time_fields(self, transaction_datetime: datetime) -> Dict[str, Any]:
        """
        Generate day_of_week and hour from datetime
        
        Args:
            transaction_datetime: Transaction datetime object
        
        Returns:
            Dict with day_of_week and hour
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return {
            "day_of_week": days[transaction_datetime.weekday()],
            "hour": transaction_datetime.hour,
        }
    
    # ========================================================================
    # CARD TYPE
    # ========================================================================
    
    def generate_card_type(self, payment_mode: str, customer_income_bracket: str) -> str:
        """
        Generate card type based on payment mode and income
        
        Rules:
        - Credit card: Higher income brackets, premium customers
        - Debit card: Middle and lower income brackets
        - NA: Non-card payments (UPI, Cash, Net Banking)
        
        Args:
            payment_mode: Payment method
            customer_income_bracket: Customer income level
        
        Returns:
            Card type: "Credit", "Debit", or "NA"
        """
        # Non-card payments
        if payment_mode in ["UPI", "Cash", "Wallet", "Net Banking"]:
            return "NA"
        
        # Card payments - determine Credit vs Debit
        if payment_mode in ["Credit Card", "Debit Card", "Card"]:
            # Higher income = more credit cards
            credit_probability = {
                "LOW": 0.05,
                "LOWER_MIDDLE": 0.15,
                "MIDDLE": 0.30,
                "UPPER_MIDDLE": 0.55,
                "HIGH": 0.75,
                "PREMIUM": 0.85,
            }.get(customer_income_bracket, 0.30)
            
            return "Credit" if random.random() < credit_probability else "Debit"
        
        return "NA"
    
    # ========================================================================
    # TRANSACTION STATUS
    # ========================================================================
    
    def generate_transaction_status(
        self,
        amount: float,
        payment_mode: str,
        is_online: bool,
        customer_digital_savviness: str
    ) -> str:
        """
        Generate transaction status with realistic decline/pending rates
        
        Decline factors:
        - High amount (>Rs.50,000): +5% decline rate
        - Online transaction: +2% decline rate
        - Low digital savviness: +3% decline rate
        - Cash payment: 0% decline (always approved)
        
        Typical rates:
        - Approved: 95-98%
        - Declined: 1-4%
        - Pending: 1%
        
        Args:
            amount: Transaction amount
            payment_mode: Payment method
            is_online: Whether transaction is online
            customer_digital_savviness: Customer's digital level
        
        Returns:
            Status: "Approved", "Declined", or "Pending"
        """
        # Cash always approved
        if payment_mode == "Cash":
            return "Approved"
        
        # Calculate decline probability
        decline_rate = 0.01  # Base 1%
        
        # High amount increases decline risk
        if amount > 50000:
            decline_rate += 0.05
        elif amount > 20000:
            decline_rate += 0.02
        
        # Online has slightly higher decline
        if is_online:
            decline_rate += 0.02
        
        # Low digital savviness = more errors/declines
        if customer_digital_savviness == "LOW":
            decline_rate += 0.03
        elif customer_digital_savviness == "MEDIUM":
            decline_rate += 0.01
        
        # Generate status
        rand = random.random()
        
        if rand < decline_rate:
            return "Declined"
        elif rand < decline_rate + 0.01:  # 1% pending
            return "Pending"
        else:
            return "Approved"
    
    # ========================================================================
    # TRANSACTION CHANNEL
    # ========================================================================
    
    def generate_transaction_channel(
        self,
        is_online: bool,
        payment_mode: str,
        category: str
    ) -> str:
        """
        Generate transaction channel based on context
        
        Channels:
        - POS: Physical store transactions
        - Online: E-commerce, food delivery
        - ATM: Cash withdrawals
        - Mobile: In-app purchases, mobile payments
        
        Args:
            is_online: Whether transaction is online
            payment_mode: Payment method
            category: Transaction category
        
        Returns:
            Channel: "POS", "Online", "ATM", or "Mobile"
        """
        # ATM for cash-related transactions
        if category in ["ATM Withdrawal", "Cash Deposit"]:
            return "ATM"
        
        # Online transactions
        if is_online:
            # Mobile app vs web
            if payment_mode == "UPI" or random.random() < 0.60:  # 60% mobile for online
                return "Mobile"
            else:
                return "Online"
        
        # Physical store transactions
        return "POS"
    
    # ========================================================================
    # LOCATION: STATE & REGION
    # ========================================================================
    
    def get_state_and_region(self, city: str) -> Tuple[str, str]:
        """
        Get state and region from city name
        
        Args:
            city: City name
        
        Returns:
            Tuple of (state, region)
        """
        state = CITY_TO_STATE.get(city, "Unknown")
        region = STATE_TO_REGION.get(state, "Central")
        return state, region
    
    # ========================================================================
    # CUSTOMER CONTEXT: AGE GROUP
    # ========================================================================
    
    def get_age_group(self, age: int) -> str:
        """
        Convert age to age group
        
        Groups: 18-25, 26-35, 36-45, 46-55, 56-65, 66+
        
        Args:
            age: Customer age
        
        Returns:
            Age group string
        """
        if age < 26:
            return "18-25"
        elif age < 36:
            return "26-35"
        elif age < 46:
            return "36-45"
        elif age < 56:
            return "46-55"
        elif age < 66:
            return "56-65"
        else:
            return "66+"
    
    # ========================================================================
    # DEVICE & CHANNEL INFORMATION
    # ========================================================================
    
    def generate_device_info(
        self,
        transaction_channel: str,
        customer_age: int,
        customer_digital_savviness: str
    ) -> Dict[str, Optional[str]]:
        """
        Generate device information based on transaction channel
        
        Returns:
        - device_type: Mobile, Web, POS, ATM
        - app_version: For mobile (e.g., "5.2.1")
        - browser_type: For web (Chrome, Firefox, etc.)
        - os: Android, iOS, Windows, Other, NA
        
        Args:
            transaction_channel: Transaction channel
            customer_age: Customer age (younger = more mobile)
            customer_digital_savviness: Digital level
        
        Returns:
            Dict with device_type, app_version, browser_type, os
        """
        device_info = {
            "device_type": transaction_channel,
            "app_version": None,
            "browser_type": None,
            "os": "NA",
        }
        
        # Mobile transactions
        if transaction_channel == "Mobile":
            device_info["device_type"] = "Mobile"
            device_info["app_version"] = random.choice(APP_VERSIONS)
            
            # Younger users + high digital = more iOS
            ios_probability = 0.06  # Base 6% (Indian market)
            if customer_age < 35 and customer_digital_savviness == "HIGH":
                ios_probability = 0.20
            
            device_info["os"] = "iOS" if random.random() < ios_probability else "Android"
        
        # Web transactions
        elif transaction_channel == "Online":
            device_info["device_type"] = "Web"
            device_info["browser_type"] = random.choices(
                list(BROWSERS.keys()),
                weights=list(BROWSERS.values())
            )[0]
            device_info["os"] = "Windows" if random.random() < 0.75 else "Other"
        
        # POS transactions
        elif transaction_channel == "POS":
            device_info["device_type"] = "POS"
            device_info["os"] = "NA"
        
        # ATM transactions
        elif transaction_channel == "ATM":
            device_info["device_type"] = "ATM"
            device_info["os"] = "NA"
        
        return device_info
    
    # ========================================================================
    # RISK INDICATORS: DISTANCE CALCULATION
    # ========================================================================
    
    def calculate_distance_from_home(
        self,
        home_city: str,
        transaction_city: str,
        city_tier_distances: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate approximate distance from home city to transaction city
        
        Uses simplified distance matrix based on city tiers and regions
        
        Args:
            home_city: Customer's home city
            transaction_city: Transaction city
            city_tier_distances: Optional distance matrix
        
        Returns:
            Distance in kilometers (0 if same city)
        """
        # Same city = 0 distance
        if home_city == transaction_city:
            return 0.0
        
        # Get states and regions
        home_state, home_region = self.get_state_and_region(home_city)
        txn_state, txn_region = self.get_state_and_region(transaction_city)
        
        # Same state, different city
        if home_state == txn_state:
            return random.uniform(50, 300)  # 50-300 km within state
        
        # Same region, different state
        if home_region == txn_region:
            return random.uniform(300, 800)  # 300-800 km within region
        
        # Different regions
        return random.uniform(800, 2500)  # 800-2500 km across India
