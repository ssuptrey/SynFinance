"""
Core transaction generation logic with customer behavioral consistency

This module contains the TransactionGenerator class which creates realistic
financial transactions based on customer profiles and Indian market patterns.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Generator, Any
import numpy as np
import pandas as pd
from faker import Faker

from src.customer_profile import CustomerProfile
from src.customer_generator import CustomerGenerator
from src.config import (
    TRANSACTION_CATEGORIES,
    PAYMENT_MODES,
    AMOUNT_RANGES,
    MERCHANT_TYPES,
    INDIAN_CITIES,
    DEFAULT_DATE_RANGE_DAYS
)
from src.utils.indian_data import (
    INDIAN_FESTIVALS,
    INDIAN_MERCHANTS,
    UPI_HANDLES,
    CHAIN_MERCHANTS
)
from src.generators.temporal_generator import TemporalPatternGenerator
from src.generators.geographic_generator import GeographicPatternGenerator
from src.generators.merchant_generator import MerchantGenerator


class TransactionGenerator:
    """
    Generate synthetic financial transactions with customer behavioral consistency
    
    Architecture:
    - Customer-aware: Uses CustomerProfile to generate realistic behavior
    - Indian market focused: UPI, festivals, regional patterns
    - Scalable: Generator pattern for memory efficiency
    - Developer-friendly: Clear methods, well-documented
    
    Usage:
        generator = TransactionGenerator()
        customer = CustomerGenerator().generate_customer()
        transactions = generator.generate_customer_transactions(customer, count=100)
    """
    
    def __init__(self, locale: str = 'en_IN', seed: Optional[int] = None):
        """
        Initialize the transaction generator
        
        Args:
            locale: Faker locale (default: Indian English)
            seed: Random seed for reproducibility
        """
        self.fake = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # Temporal pattern generator (Week 2, Day 1-2)
        self.temporal_gen = TemporalPatternGenerator(seed=seed)
        
        # Geographic pattern generator (Week 2, Day 3-4)
        self.geographic_gen = GeographicPatternGenerator(seed=seed)
        
        # Merchant ecosystem generator (Week 2, Day 5-7)
        self.merchant_gen = MerchantGenerator(seed=seed)
        
        # Advanced schema generator (Week 3, Day 1-3)
        from src.generators.advanced_schema_generator import AdvancedSchemaGenerator
        self.advanced_gen = AdvancedSchemaGenerator(seed=seed)
        
        # Merchant memory for loyalty behavior (DEPRECATED - now in merchant_gen)
        self.merchant_memory: Dict[str, List[str]] = {}  # customer_id -> list of visited merchants
        
        # Transaction counter for IDs
        self.transaction_counter = 0
        
        # ====================================================================
        # WEEK 3: Transaction State Tracking for Risk Indicators
        # ====================================================================
        
        # Track last transaction time per customer (for time_since_last_txn)
        self.customer_last_txn: Dict[str, datetime] = {}
        
        # Track daily transaction counts per customer (resets daily)
        # Format: {customer_id: {date_str: count}}
        self.daily_txn_counts: Dict[str, Dict[str, int]] = {}
        
        # Track daily transaction amounts per customer (resets daily)
        # Format: {customer_id: {date_str: total_amount}}
        self.daily_txn_amounts: Dict[str, Dict[str, float]] = {}
        
        # Track customer-merchant relationships (for is_first_transaction)
        # Format: {customer_id: set(merchant_ids)}
        self.customer_merchants: Dict[str, set[str]] = {}
    
    # ============================================================================
    # CORE TRANSACTION GENERATION - Customer-aware
    # ============================================================================
    
    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        self.transaction_counter += 1
        return f"TXN{self.transaction_counter:010d}"
    
    # ============================================================================
    # WEEK 3: STATE TRACKING METHODS - Risk Indicators
    # ============================================================================
    
    def _update_transaction_state(
        self,
        customer_id: str,
        merchant_id: str,
        transaction_datetime: datetime,
        amount: float
    ) -> Dict[str, Any]:
        """
        Update transaction state tracking and calculate risk indicators
        
        Tracks:
        - Last transaction time (for time_since_last_txn)
        - Daily transaction counts (for daily_transaction_count)
        - Daily transaction amounts (for daily_transaction_amount)
        - Customer-merchant relationships (for is_first_transaction_with_merchant)
        
        Args:
            customer_id: Customer identifier
            merchant_id: Merchant identifier
            transaction_datetime: Current transaction datetime
            amount: Transaction amount
        
        Returns:
            Dict with risk indicators:
            - time_since_last_txn: Minutes since last transaction (None if first)
            - is_first_transaction_with_merchant: Boolean
            - daily_transaction_count: Running count today
            - daily_transaction_amount: Running total today
        """
        date_str = transaction_datetime.strftime("%Y-%m-%d")
        
        # ====================================================================
        # 1. Calculate time_since_last_txn
        # ====================================================================
        time_since_last = None
        if customer_id in self.customer_last_txn:
            last_txn_time = self.customer_last_txn[customer_id]
            time_diff = transaction_datetime - last_txn_time
            time_since_last = time_diff.total_seconds() / 60.0  # Convert to minutes
        
        # Update last transaction time
        self.customer_last_txn[customer_id] = transaction_datetime
        
        # ====================================================================
        # 2. Calculate daily_transaction_count
        # ====================================================================
        if customer_id not in self.daily_txn_counts:
            self.daily_txn_counts[customer_id] = {}
        
        if date_str not in self.daily_txn_counts[customer_id]:
            self.daily_txn_counts[customer_id][date_str] = 0
        
        self.daily_txn_counts[customer_id][date_str] += 1
        daily_count = self.daily_txn_counts[customer_id][date_str]
        
        # ====================================================================
        # 3. Calculate daily_transaction_amount
        # ====================================================================
        if customer_id not in self.daily_txn_amounts:
            self.daily_txn_amounts[customer_id] = {}
        
        if date_str not in self.daily_txn_amounts[customer_id]:
            self.daily_txn_amounts[customer_id][date_str] = 0.0
        
        # Add current amount to daily total
        self.daily_txn_amounts[customer_id][date_str] += amount
        daily_amount = self.daily_txn_amounts[customer_id][date_str]
        
        # ====================================================================
        # 4. Check if first transaction with merchant
        # ====================================================================
        if customer_id not in self.customer_merchants:
            self.customer_merchants[customer_id] = set()
        
        is_first_with_merchant = merchant_id not in self.customer_merchants[customer_id]
        
        # Add merchant to customer's history
        self.customer_merchants[customer_id].add(merchant_id)
        
        return {
            "time_since_last_txn": time_since_last,
            "is_first_transaction_with_merchant": is_first_with_merchant,
            "daily_transaction_count": daily_count,
            "daily_transaction_amount": daily_amount,
        }
    
    def reset_daily_state(self):
        """
        Reset daily state tracking
        
        Call this when generating transactions for a new day to clear
        daily counters. Useful for multi-day dataset generation.
        """
        self.daily_txn_counts.clear()
        self.daily_txn_amounts.clear()
    
    def reset_all_state(self):
        """
        Reset all state tracking
        
        Call this to start fresh with a new dataset.
        Clears all customer history, daily counters, and merchant relationships.
        """
        self.customer_last_txn.clear()
        self.daily_txn_counts.clear()
        self.daily_txn_amounts.clear()
        self.customer_merchants.clear()
        self.merchant_memory.clear()
        self.transaction_counter = 0
    
    def select_transaction_category(self, customer: CustomerProfile) -> str:
        """
        Select transaction category based on customer preferences
        
        Uses customer's preferred_categories with 70% probability,
        otherwise random category for variety
        """
        if random.random() < 0.7 and customer.preferred_categories:
            return random.choice(customer.preferred_categories)
        return random.choice(TRANSACTION_CATEGORIES)
    
    def select_payment_mode(self, customer: CustomerProfile, amount: float) -> str:
        """
        Select payment mode based on customer's digital savviness and preferences
        
        Indian market patterns:
        - UPI dominates for amounts < ₹2000
        - Credit cards for high amounts (if customer uses them)
        - Cash declining but still used by LOW digital savviness
        """
        # For small amounts, UPI is dominant in India
        if amount < 500 and "UPI" in customer.preferred_payment_modes:
            if random.random() < 0.8:  # 80% UPI for small amounts
                return "UPI"
        
        # For large amounts, credit cards preferred (if available)
        if amount > 5000 and "Credit Card" in customer.preferred_payment_modes:
            if random.random() < 0.6:  # 60% credit card for large amounts
                return "Credit Card"
        
        # Otherwise use customer's preferred modes
        if customer.preferred_payment_modes:
            return random.choice(customer.preferred_payment_modes)
        
        # Fallback
        return random.choice(PAYMENT_MODES)
    
    def select_merchant(
        self, 
        customer: CustomerProfile, 
        category: str,
        city: str = None,
        is_online: bool = False
    ) -> Tuple[str, bool]:
        """
        Select merchant based on customer loyalty and category
        
        Week 2 Day 5-7: Enhanced with MerchantGenerator
        - Uses Merchant objects with IDs, reputation, subcategories
        - Chain vs local merchant logic
        - City-specific merchant availability
        - Returns merchant name for backward compatibility
        
        Returns:
            Tuple of (merchant_name, is_repeat_merchant)
        
        Merchant loyalty:
        - 60-95% repeat based on category (groceries high, shopping low)
        - Uses MerchantGenerator's loyalty tracking system
        """
        # Use customer's home city if not specified
        if city is None:
            city = customer.city
        
        # Get merchant object from merchant generator (Week 2 Day 5-7)
        merchant_obj = self.merchant_gen.select_merchant(customer, category, city, is_online)
        
        # Check if this is a repeat merchant (customer has favorited this merchant)
        customer_id = customer.customer_id
        is_repeat = False
        
        if customer_id in self.merchant_gen.customer_favorites:
            if category in self.merchant_gen.customer_favorites[customer_id]:
                if merchant_obj.merchant_id in self.merchant_gen.customer_favorites[customer_id][category]:
                    is_repeat = True
        
        # Store merchant object for later use in transaction (for new fields)
        self._last_merchant_obj = merchant_obj
        
        return merchant_obj.name, is_repeat
    
    def _generate_generic_merchant(self, category: str, city: str) -> str:
        """Generate a generic merchant name"""
        prefixes = ["New", "Royal", "Prime", "Star", "Golden", "Supreme", "Elite", "Grand", "Metro", "Smart"]
        return f"{random.choice(prefixes)} {category} - {city}"
    
    def calculate_transaction_amount(
        self,
        customer: CustomerProfile,
        category: str,
        date: datetime = None,
        is_weekend: bool = False,
        is_festival: bool = False
    ) -> float:
        """
        Calculate transaction amount based on customer profile and context
        
        Factors:
        - Customer's average transaction amount
        - Customer's income level
        - Category baseline (from config)
        - Weekend/festival multipliers (DEPRECATED - use date param)
        - Temporal multipliers (day-of-week, salary day, festival) [NEW - Week 2]
        - Impulse buying behavior
        
        Args:
            customer: Customer profile
            category: Transaction category
            date: Transaction date (for temporal patterns) [NEW - Week 2]
            is_weekend: DEPRECATED - use date param instead
            is_festival: DEPRECATED - use date param instead
        """
        # Base amount from customer profile
        base_amount = customer.avg_transaction_amount
        
        # Category-specific adjustment
        if category in AMOUNT_RANGES:
            cat_min, cat_max = AMOUNT_RANGES[category]
            cat_avg = (cat_min + cat_max) / 2
            
            # Blend customer average with category average (70/30)
            base_amount = 0.7 * base_amount + 0.3 * cat_avg
            
            # Clamp to category range (with some flexibility)
            base_amount = np.clip(base_amount, cat_min * 0.5, cat_max * 1.5)
        
        # Income-based variance (higher income = more variance)
        income_factor = customer.get_spending_power()
        variance = base_amount * 0.3 * (0.5 + income_factor)
        
        # Generate amount with some randomness
        amount = np.random.normal(base_amount, variance)
        
        # Apply temporal multipliers (Week 2 enhancement)
        if date is not None:
            temporal_multiplier, breakdown = self.temporal_gen.get_combined_temporal_multiplier(
                customer, date, INDIAN_FESTIVALS
            )
            amount *= temporal_multiplier
        else:
            # Legacy: Use old weekend/festival flags
            if is_weekend and customer.weekend_shopper:
                amount *= random.uniform(1.1, 1.3)  # Weekend shoppers spend 10-30% more
            
            if is_festival:
                amount *= random.uniform(1.3, 1.8)  # Festival spending boost
        
        # Impulse buying (independent of temporal patterns)
        if customer.impulse_buyer and random.random() < 0.2:
            amount *= random.uniform(1.2, 2.0)  # Impulse purchases are higher
        
        # Ensure positive and reasonable
        amount = max(50, amount)  # Minimum ₹50
        
        # Round to realistic values
        if amount < 100:
            amount = round(amount, 0)
        elif amount < 1000:
            amount = round(amount / 10) * 10  # Round to nearest 10
        else:
            amount = round(amount / 50) * 50  # Round to nearest 50
        
        return float(amount)
    
    def generate_transaction_datetime(
        self,
        customer: CustomerProfile,
        date: datetime
    ) -> datetime:
        """
        Generate transaction time based on customer's temporal patterns
        
        Week 2 Enhancement: Uses occupation-based hour-of-day distributions
        - Considers weekday vs weekend patterns
        - Different distributions by occupation (salaried, student, retired, etc.)
        - Realistic peak hours (breakfast, lunch, evening)
        
        Old behavior: Used customer's preferred_shopping_hours list
        New behavior: Uses probability distributions from TemporalPatternGenerator
        """
        # Use temporal pattern generator for realistic hour selection (Week 2)
        hour = self.temporal_gen.select_transaction_hour(customer, date)
        
        # Add minute/second variation
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return date.replace(hour=hour, minute=minute, second=second)
    
    def select_transaction_city(self, customer: CustomerProfile) -> Tuple[str, str]:
        """
        Select city for transaction based on customer's travel behavior
        
        Week 2 Day 3-4: Enhanced with geographic patterns
        - 80% home city
        - 15% nearby cities (proximity groups, same region)
        - 5% distant cities (travel scenarios)
        
        Returns:
            Tuple of (city, location_type) where location_type is "home", "nearby", or "travel"
        """
        return self.geographic_gen.select_transaction_city(customer)
    
    def is_online_transaction(self, customer: CustomerProfile, category: str) -> bool:
        """
        Determine if transaction is online based on customer preference and category
        
        Some categories are always online (e.g., streaming services)
        Some are rarely online (e.g., utilities, healthcare)
        """
        online_likely_categories = ["Electronics", "Shopping", "Entertainment", "Travel", "Education"]
        offline_likely_categories = ["Healthcare", "Utilities", "Transportation"]
        
        if category in online_likely_categories:
            threshold = customer.online_shopping_preference * 1.2  # Boost for online categories
        elif category in offline_likely_categories:
            threshold = customer.online_shopping_preference * 0.5  # Reduce for offline categories
        else:
            threshold = customer.online_shopping_preference
        
        return random.random() < min(threshold, 0.95)  # Max 95% online
    
    # ============================================================================
    # MAIN GENERATION METHODS
    # ============================================================================
    
    def generate_transaction(
        self,
        customer: CustomerProfile,
        date: datetime,
        is_weekend: bool = False,
        is_festival: bool = False
    ) -> Dict:
        """
        Generate a single transaction for a customer with 43 fields
        
        Week 3 Enhancement: Complete advanced schema with all risk indicators
        
        Args:
            customer: CustomerProfile object
            date: Base date for transaction
            is_weekend: DEPRECATED - calculated from date
            is_festival: DEPRECATED - calculated from date via INDIAN_FESTIVALS
        
        Returns:
            Dictionary with 43 transaction fields (Transaction.to_dict())
        """
        # Import Transaction dataclass
        from src.models.transaction import Transaction
        
        # ====================================================================
        # STEP 1: Generate datetime and basic attributes
        # ====================================================================
        transaction_datetime = self.generate_transaction_datetime(customer, date)
        category = self.select_transaction_category(customer)
        amount = self.calculate_transaction_amount(customer, category, date=transaction_datetime)
        
        # ====================================================================
        # STEP 2: Geographic location and online status
        # ====================================================================
        city, location_type = self.select_transaction_city(customer)
        is_online = self.is_online_transaction(customer, category)
        
        # Apply cost-of-living adjustment based on city
        amount = self.geographic_gen.apply_cost_of_living_adjustment(amount, city)
        
        # ====================================================================
        # STEP 3: Payment and merchant selection
        # ====================================================================
        payment_mode = self.select_payment_mode(customer, amount)
        merchant, is_repeat = self.select_merchant(customer, category, city=city, is_online=is_online)
        
        # Get merchant object (stored by select_merchant)
        merchant_obj = self._last_merchant_obj
        
        # ====================================================================
        # STEP 4: Week 3 - Generate advanced schema fields
        # ====================================================================
        
        # Time fields (day_of_week, hour)
        time_fields = self.advanced_gen.generate_time_fields(transaction_datetime)
        
        # Card type (Credit/Debit/NA)
        card_type = self.advanced_gen.generate_card_type(
            payment_mode,
            customer.income_bracket.value
        )
        
        # Transaction status (Approved/Declined/Pending)
        transaction_status = self.advanced_gen.generate_transaction_status(
            amount,
            payment_mode,
            is_online,
            customer.digital_savviness.value
        )
        
        # Transaction channel (POS/Online/ATM/Mobile)
        transaction_channel = self.advanced_gen.generate_transaction_channel(
            is_online,
            payment_mode,
            category
        )
        
        # State and region
        state, region = self.advanced_gen.get_state_and_region(city)
        
        # Customer age group
        customer_age_group = self.advanced_gen.get_age_group(customer.age)
        
        # Device information (device_type, app_version, browser_type, os)
        device_info = self.advanced_gen.generate_device_info(
            transaction_channel,
            customer.age,
            customer.digital_savviness.value
        )
        
        # Distance from home
        distance_from_home = self.advanced_gen.calculate_distance_from_home(
            customer.city,
            city
        )
        
        # ====================================================================
        # STEP 5: Update transaction state and calculate risk indicators
        # ====================================================================
        risk_indicators = self._update_transaction_state(
            customer.customer_id,
            merchant_obj.merchant_id,
            transaction_datetime,
            amount
        )
        
        # ====================================================================
        # STEP 6: Get geographic breakdown (Week 2 compatibility)
        # ====================================================================
        geo_breakdown = self.geographic_gen.get_geographic_breakdown(customer, city, location_type)
        is_weekend = transaction_datetime.weekday() >= 5
        
        # ====================================================================
        # STEP 7: Build Transaction object with all 43 fields
        # ====================================================================
        transaction = Transaction(
            # Core Fields (10)
            transaction_id=self.generate_transaction_id(),
            customer_id=customer.customer_id,
            merchant_id=merchant_obj.merchant_id,
            date=transaction_datetime.strftime("%Y-%m-%d"),
            time=transaction_datetime.strftime("%H:%M:%S"),
            day_of_week=time_fields["day_of_week"],
            hour=time_fields["hour"],
            amount=amount,
            currency="INR",
            merchant_name=merchant,
            
            # Transaction Details (9)
            category=category,
            subcategory=merchant_obj.subcategory,
            payment_mode=payment_mode,
            card_type=card_type,
            transaction_status=transaction_status,
            transaction_channel=transaction_channel,
            merchant_type=merchant_obj.merchant_type,
            merchant_reputation=merchant_obj.reputation,
            is_online=is_online,
            
            # Location Fields (5)
            city=city,
            state=state,
            region=region,
            merchant_city=merchant_obj.city,
            location_type=location_type,
            
            # Customer Context (3)
            customer_age_group=customer_age_group,
            customer_income_bracket=customer.income_bracket.value,
            customer_segment=customer.segment.value,
            
            # Device & Channel (4)
            device_type=device_info["device_type"] or "POS",
            app_version=device_info["app_version"],
            browser_type=device_info["browser_type"],
            os=device_info["os"] or "NA",
            
            # Risk Indicators (5)
            distance_from_home=distance_from_home,
            time_since_last_txn=risk_indicators["time_since_last_txn"],
            is_first_transaction_with_merchant=risk_indicators["is_first_transaction_with_merchant"],
            daily_transaction_count=risk_indicators["daily_transaction_count"],
            daily_transaction_amount=risk_indicators["daily_transaction_amount"],
            
            # Backward compatibility fields (Week 2)
            home_city=customer.city,
            city_tier=geo_breakdown["city_tier"],
            distance_category=geo_breakdown["distance_category"],
            is_weekend=is_weekend,
            is_repeat_merchant=is_repeat,
            customer_age=customer.age,
            customer_digital_savviness=customer.digital_savviness.value,
        )
        
        # Return as dictionary for backward compatibility with Week 1-2 tests
        return transaction.to_legacy_dict()
    
    def generate_customer_transactions(
        self,
        customer: CustomerProfile,
        count: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30
    ) -> List[Dict]:
        """
        Generate multiple transactions for a single customer over a time period
        
        Args:
            customer: CustomerProfile object
            count: Number of transactions (if None, uses customer's monthly_transaction_count)
            start_date: Start date for transactions
            end_date: End date for transactions
            days: Number of days to generate over (default 30)
        
        Returns:
            List of transaction dictionaries
        """
        # Determine transaction count
        if count is None:
            # Use customer's typical monthly count with some variation
            count = int(np.random.normal(
                customer.monthly_transaction_count,
                customer.monthly_transaction_count * 0.15
            ))
            count = max(1, count)  # At least 1 transaction
        
        # Set date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        # Generate random dates within range
        time_delta = end_date - start_date
        transactions = []
        
        for _ in range(count):
            # Random day within range
            random_days = random.random() * time_delta.days
            transaction_date = start_date + timedelta(days=random_days)
            
            # Check if weekend
            is_weekend = transaction_date.weekday() >= 5  # Saturday=5, Sunday=6
            
            # Check if festival (simplified - you can expand this)
            is_festival = False  # TODO: Implement festival calendar
            
            transaction = self.generate_transaction(
                customer, transaction_date, is_weekend, is_festival
            )
            transactions.append(transaction)
        
        # Sort by datetime
        transactions.sort(key=lambda x: f"{x['Date']} {x['Time']}")
        
        return transactions
    
    def generate_dataset(
        self,
        customers: List[CustomerProfile],
        transactions_per_customer: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Generate complete transaction dataset for multiple customers
        
        Args:
            customers: List of CustomerProfile objects
            transactions_per_customer: Fixed count per customer (or None for varied)
            start_date: Start date
            end_date: End date
            days: Number of days
        
        Returns:
            DataFrame with all transactions
        """
        all_transactions = []
        
        for customer in customers:
            customer_transactions = self.generate_customer_transactions(
                customer,
                count=transactions_per_customer,
                start_date=start_date,
                end_date=end_date,
                days=days
            )
            all_transactions.extend(customer_transactions)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Sort by date and time
        df = df.sort_values(['Date', 'Time']).reset_index(drop=True)
        
        return df
    
    def generate_dataset_streaming(
        self,
        customers: List[CustomerProfile],
        transactions_per_customer: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
        chunk_size: int = 10000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate dataset in chunks for memory efficiency (scalable for millions of records)
        
        Args:
            customers: List of CustomerProfile objects
            transactions_per_customer: Fixed count per customer
            start_date: Start date
            end_date: End date
            days: Number of days
            chunk_size: Number of transactions per chunk
        
        Yields:
            DataFrame chunks
        """
        buffer = []
        
        for customer in customers:
            customer_transactions = self.generate_customer_transactions(
                customer,
                count=transactions_per_customer,
                start_date=start_date,
                end_date=end_date,
                days=days
            )
            buffer.extend(customer_transactions)
            
            # Yield chunk when buffer is full
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                df_chunk = pd.DataFrame(chunk)
                df_chunk = df_chunk.sort_values(['Date', 'Time']).reset_index(drop=True)
                yield df_chunk
        
        # Yield remaining transactions
        if buffer:
            df_chunk = pd.DataFrame(buffer)
            df_chunk = df_chunk.sort_values(['Date', 'Time']).reset_index(drop=True)
            yield df_chunk
    
    # ============================================================================
    # LEGACY METHODS (kept for backward compatibility - NOT customer-aware)
    # ============================================================================
    
    def generate_date(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> datetime:
        """Generate a random date within the specified range (legacy method)"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_DATE_RANGE_DAYS)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        random_seconds = random.randint(0, 86400)  # seconds in a day
        
        return start_date + timedelta(days=random_days, seconds=random_seconds)
    
    def generate_merchant(self, category: str) -> str:
        """Generate a merchant name based on category (legacy method)"""
        if category in INDIAN_MERCHANTS:
            return random.choice(INDIAN_MERCHANTS[category])
        elif category in MERCHANT_TYPES:
            merchant_type = random.choice(MERCHANT_TYPES[category])
            if random.random() > 0.5:
                city = random.choice(INDIAN_CITIES)
                return f"{city} {merchant_type}"
            else:
                prefix = random.choice(["New", "Royal", "Prime", "Star", "Golden", 
                                       "Supreme", "Elite", "Grand", "Metro", "Smart"])
                return f"{prefix} {merchant_type}"
        return self.fake.company()
    
    def generate_amount(self, category: str) -> float:
        """Generate transaction amount based on category (legacy method)"""
        if category in AMOUNT_RANGES:
            min_amount, max_amount = AMOUNT_RANGES[category]
            mean_amount = (min_amount + max_amount) / 2
            std_dev = (max_amount - min_amount) / 4
            
            amount = np.random.lognormal(np.log(mean_amount), 0.5)
            amount = np.clip(amount, min_amount, max_amount)
            return round(amount, 2)
        return round(random.uniform(100, 10000), 2)
    
    def generate_transactions(self, num_records: int) -> pd.DataFrame:
        """
        Generate multiple transaction records (legacy method - NOT customer-aware)
        
        DEPRECATED: Use generate_dataset() with CustomerProfile objects instead
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            DataFrame with generated transactions
        """
        transactions = []
        
        for _ in range(num_records):
            category = random.choice(TRANSACTION_CATEGORIES)
            transaction = {
                'Transaction_ID': self.generate_transaction_id(),
                'Date': self.generate_date(),
                'Merchant': self.generate_merchant(category),
                'Category': category,
                'Amount': self.generate_amount(category),
                'Mode': random.choice(PAYMENT_MODES),
                'City': random.choice(INDIAN_CITIES)
            }
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    
    # ============================================================================
    # EXPORT METHODS
    # ============================================================================
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = 'transactions.csv') -> str:
        """Export DataFrame to CSV"""
        df.to_csv(filename, index=False)
        return filename
    
    def export_to_json(self, df: pd.DataFrame, filename: str = 'transactions.json') -> str:
        """Export DataFrame to JSON"""
        df.to_json(filename, orient='records', indent=2)
        return filename
    
    def export_to_excel(self, df: pd.DataFrame, filename: str = 'transactions.xlsx') -> str:
        """Export DataFrame to Excel"""
        df.to_excel(filename, index=False, engine='xlsxwriter')
        return filename
