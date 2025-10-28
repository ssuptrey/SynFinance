"""
Transaction dataclass for SynFinance - Week 3 Advanced Schema

This module defines the Transaction dataclass with 36+ fields covering:
- Core transaction details (10 fields)
- Transaction details (9 fields)
- Location fields (5 fields)
- Customer context (3 fields)
- Device & channel (4 fields)
- Risk indicators (5 fields)

Version: 2.0 (Week 3)
Created: October 19, 2025
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Transaction:
    """
    Comprehensive transaction record with 36+ fields
    
    This dataclass represents a single financial transaction with all attributes
    needed for fraud detection, behavioral analytics, and pattern recognition.
    
    Field Categories:
    1. Core Fields (10): Basic transaction identification
    2. Transaction Details (9): Payment and status information
    3. Location Fields (5): Geographic context
    4. Customer Context (3): Customer attributes
    5. Device & Channel (4): Technical context
    6. Risk Indicators (5): Fraud detection signals
    
    Usage:
        txn = Transaction(
            transaction_id="TXN0000000001",
            customer_id="CUST0000001",
            # ... other fields
        )
        
        # Convert to dict for pandas/CSV
        txn_dict = txn.to_dict()
    """
    
    # ========================================================================
    # CORE FIELDS (10)
    # ========================================================================
    
    transaction_id: str
    """Unique transaction identifier (format: TXN0000000001)"""
    
    customer_id: str
    """Customer identifier (format: CUST0000001)"""
    
    merchant_id: str
    """Merchant identifier (format: MER_GRO_MUM_001)"""
    
    date: str
    """Transaction date (format: YYYY-MM-DD)"""
    
    time: str
    """Transaction time (format: HH:MM:SS)"""
    
    day_of_week: str
    """Day name (Monday, Tuesday, etc.)"""
    
    hour: int
    """Hour of transaction (0-23)"""
    
    amount: float
    """Transaction amount in INR"""
    
    currency: str = "INR"
    """Transaction currency (default: INR)"""
    
    merchant_name: str = ""
    """Human-readable merchant name"""
    
    # ========================================================================
    # TRANSACTION DETAILS (9)
    # ========================================================================
    
    category: str = ""
    """Transaction category (Groceries, Dining, etc.)"""
    
    subcategory: str = ""
    """Detailed subcategory (Supermarket, Fast Food, etc.)"""
    
    payment_mode: str = ""
    """Payment method (UPI, Card, Cash, etc.)"""
    
    card_type: str = "NA"
    """Card type: Credit, Debit, or NA (for non-card payments)"""
    
    transaction_status: str = "Approved"
    """Transaction status: Approved, Declined, Pending"""
    
    transaction_channel: str = "POS"
    """Channel: POS, Online, ATM, Mobile"""
    
    merchant_type: str = ""
    """Merchant type: chain or local"""
    
    merchant_reputation: float = 0.0
    """Merchant reputation score (0.0-1.0)"""
    
    is_online: bool = False
    """Whether transaction was online"""
    
    # ========================================================================
    # LOCATION FIELDS (5)
    # ========================================================================
    
    city: str = ""
    """Transaction city"""
    
    state: str = ""
    """Transaction state"""
    
    region: str = ""
    """Geographic region: North, South, East, West, Central"""
    
    merchant_city: str = ""
    """Merchant's city (may differ from transaction city for online)"""
    
    location_type: str = ""
    """Location type: home, nearby, travel"""
    
    # ========================================================================
    # CUSTOMER CONTEXT (3)
    # ========================================================================
    
    customer_age_group: str = ""
    """Age group: 18-25, 26-35, 36-45, 46-55, 56-65, 66+"""
    
    customer_income_bracket: str = ""
    """Income bracket: LOW, LOWER_MIDDLE, MIDDLE, UPPER_MIDDLE, HIGH, PREMIUM"""
    
    customer_segment: str = ""
    """Customer segment: Young Professional, Student, etc."""
    
    # ========================================================================
    # DEVICE & CHANNEL (4)
    # ========================================================================
    
    device_type: str = "POS"
    """Device: Mobile, Web, POS, ATM"""
    
    app_version: Optional[str] = None
    """App version for mobile transactions (e.g., "5.2.1")"""
    
    browser_type: Optional[str] = None
    """Browser for web transactions (Chrome, Firefox, Safari, etc.)"""
    
    os: str = "NA"
    """Operating system: Android, iOS, Windows, Other, NA"""
    
    # ========================================================================
    # RISK INDICATORS (5)
    # ========================================================================
    
    distance_from_home: float = 0.0
    """Distance from customer's home city in km"""
    
    time_since_last_txn: Optional[float] = None
    """Minutes since customer's last transaction (None if first)"""
    
    is_first_transaction_with_merchant: bool = True
    """Whether this is first transaction with this merchant"""
    
    daily_transaction_count: int = 1
    """Customer's transaction count today (running total)"""
    
    daily_transaction_amount: float = 0.0
    """Customer's total spending today in INR (running total)"""
    
    # ========================================================================
    # ADDITIONAL CONTEXT (Backward compatibility with Week 2)
    # ========================================================================
    
    home_city: str = ""
    """Customer's home city"""
    
    city_tier: int = 2
    """City tier: 1 (metro), 2 (major), 3 (smaller)"""
    
    distance_category: str = "local"
    """Distance category: local, regional, long_distance"""
    
    is_weekend: bool = False
    """Whether transaction occurred on weekend"""
    
    is_repeat_merchant: bool = False
    """Whether customer has transacted with merchant before"""
    
    customer_age: int = 30
    """Customer's exact age"""
    
    customer_digital_savviness: str = "MEDIUM"
    """Digital savviness: LOW, MEDIUM, HIGH"""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transaction to dictionary for pandas DataFrame / CSV export
        
        Returns lowercase field names matching dataclass attributes.
        For backward compatibility with Week 1-2 code, use to_legacy_dict()
        
        Returns:
            Dictionary with all transaction fields (lowercase keys)
        """
        return asdict(self)
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with Week 1-2 compatible field names (uppercase)
        
        This ensures backward compatibility with existing tests and code
        that expects field names like 'Transaction_ID', 'Date', 'Time', etc.
        
        Returns:
            Dictionary with uppercase/snake_case field names matching Week 1-2 format
        """
        data = asdict(self)
        
        # Map lowercase dataclass fields to uppercase Week 1-2 format
        legacy_dict = {
            # Core fields
            "Transaction_ID": data["transaction_id"],
            "Customer_ID": data["customer_id"],
            "Merchant_ID": data["merchant_id"],
            "Date": data["date"],
            "Time": data["time"],
            "Day_of_Week": data["day_of_week"],
            "Hour": data["hour"],
            "Amount": data["amount"],
            "Currency": data["currency"],
            "Merchant": data["merchant_name"],  # Note: renamed from merchant_name
            
            # Transaction details
            "Category": data["category"],
            "Subcategory": data["subcategory"],
            "Payment_Mode": data["payment_mode"],
            "Card_Type": data["card_type"],
            "Transaction_Status": data["transaction_status"],
            "Transaction_Channel": data["transaction_channel"],
            "Merchant_Type": data["merchant_type"],
            "Merchant_Reputation": data["merchant_reputation"],
            "Is_Online": data["is_online"],
            
            # Location
            "City": data["city"],
            "State": data["state"],
            "Region": data["region"],
            "Merchant_City": data["merchant_city"],
            "Location_Type": data["location_type"],
            
            # Customer context
            "Customer_Age_Group": data["customer_age_group"],
            "Customer_Income_Bracket": data["customer_income_bracket"],
            "Customer_Segment": data["customer_segment"],
            
            # Device & channel
            "Device_Type": data["device_type"],
            "App_Version": data["app_version"],
            "Browser_Type": data["browser_type"],
            "OS": data["os"],
            
            # Risk indicators
            "Distance_from_Home": data["distance_from_home"],
            "Time_Since_Last_Txn": data["time_since_last_txn"],
            "Is_First_Transaction_with_Merchant": data["is_first_transaction_with_merchant"],
            "Daily_Transaction_Count": data["daily_transaction_count"],
            "Daily_Transaction_Amount": data["daily_transaction_amount"],
            
            # Week 2 backward compatibility
            "Home_City": data["home_city"],
            "City_Tier": data["city_tier"],
            "Distance_Category": data["distance_category"],
            "Is_Weekend": data["is_weekend"],
            "Is_Repeat_Merchant": data["is_repeat_merchant"],
            "Customer_Age": data["customer_age"],
            "Customer_Digital_Savviness": data["customer_digital_savviness"],
            
            # Additional field for Week 2 compatibility
            "Merchant_Category": data["category"],  # Duplicate for compatibility
            "Merchant_Subcategory": data["subcategory"],  # Duplicate for compatibility
        }
        
        return legacy_dict
    
    def to_csv_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with CSV-friendly formatting
        
        - Formats floats to 2 decimal places
        - Converts booleans to Yes/No
        - Handles None values
        
        Returns:
            Dictionary ready for CSV export
        """
        data = self.to_dict()
        
        # Format amount fields
        if 'amount' in data:
            data['amount'] = f"{data['amount']:.2f}"
        if 'daily_transaction_amount' in data:
            data['daily_transaction_amount'] = f"{data['daily_transaction_amount']:.2f}"
        if 'distance_from_home' in data:
            data['distance_from_home'] = f"{data['distance_from_home']:.1f}"
        if 'merchant_reputation' in data:
            data['merchant_reputation'] = f"{data['merchant_reputation']:.2f}"
        
        # Format time fields
        if 'time_since_last_txn' in data and data['time_since_last_txn'] is not None:
            data['time_since_last_txn'] = f"{data['time_since_last_txn']:.1f}"
        
        # Boolean to Yes/No
        for key, value in data.items():
            if isinstance(value, bool):
                data[key] = "Yes" if value else "No"
        
        # Handle None values
        for key, value in data.items():
            if value is None:
                data[key] = ""
        
        return data
    
    @property
    def field_count(self) -> int:
        """Return total number of fields in transaction"""
        return len(asdict(self))
    
    def get_core_fields(self) -> Dict[str, Any]:
        """Get only core transaction fields (10 fields)"""
        return {
            'transaction_id': self.transaction_id,
            'customer_id': self.customer_id,
            'merchant_id': self.merchant_id,
            'date': self.date,
            'time': self.time,
            'day_of_week': self.day_of_week,
            'hour': self.hour,
            'amount': self.amount,
            'currency': self.currency,
            'merchant_name': self.merchant_name,
        }
    
    def get_risk_indicators(self) -> Dict[str, Any]:
        """Get risk indicator fields (5 fields)"""
        return {
            'distance_from_home': self.distance_from_home,
            'time_since_last_txn': self.time_since_last_txn,
            'is_first_transaction_with_merchant': self.is_first_transaction_with_merchant,
            'daily_transaction_count': self.daily_transaction_count,
            'daily_transaction_amount': self.daily_transaction_amount,
        }
    
    def calculate_risk_score(self) -> float:
        """
        Calculate simple risk score based on indicators
        
        Risk factors:
        - High distance from home (+risk)
        - Very quick successive transactions (+risk)
        - First transaction with merchant (+risk)
        - High daily transaction count (+risk)
        - High daily transaction amount (+risk)
        
        Returns:
            Risk score from 0.0 (low risk) to 1.0 (high risk)
        """
        risk_score = 0.0
        
        # Distance risk (>500km = +0.2)
        if self.distance_from_home > 500:
            risk_score += 0.2
        elif self.distance_from_home > 200:
            risk_score += 0.1
        
        # Time risk (< 5 minutes = +0.2)
        if self.time_since_last_txn is not None and self.time_since_last_txn < 5:
            risk_score += 0.2
        
        # New merchant risk (+0.1)
        if self.is_first_transaction_with_merchant:
            risk_score += 0.1
        
        # High velocity risk (>10 txns today = +0.2)
        if self.daily_transaction_count > 10:
            risk_score += 0.2
        elif self.daily_transaction_count > 5:
            risk_score += 0.1
        
        # High amount risk (>Rs.50,000 today = +0.2)
        if self.daily_transaction_amount > 50000:
            risk_score += 0.2
        elif self.daily_transaction_amount > 20000:
            risk_score += 0.1
        
        return min(1.0, risk_score)  # Cap at 1.0


# Field count constant for validation
TRANSACTION_FIELD_COUNT = 36  # Core transaction fields

# Field categories for documentation
FIELD_CATEGORIES = {
    "Core Fields": 10,
    "Transaction Details": 9,
    "Location Fields": 5,
    "Customer Context": 3,
    "Device & Channel": 4,
    "Risk Indicators": 5,
}

# Total including backward compatibility fields
TOTAL_FIELDS_WITH_LEGACY = 43  # 36 new + 7 legacy Week 2 fields
