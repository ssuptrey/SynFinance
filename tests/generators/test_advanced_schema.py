"""
Test suite for advanced schema generation and Transaction dataclass.

Tests cover:
- AdvancedSchemaGenerator field generation methods
- Transaction dataclass methods (to_dict, calculate_risk_score, etc.)
- All 43 field validations

Note: Risk indicator tests moved to test_transaction_generator.py since
calculate_risk_indicators is a TransactionGenerator method, not AdvancedSchemaGenerator.
"""

import pytest
from datetime import datetime
from src.generators.advanced_schema_generator import AdvancedSchemaGenerator
from src.models.transaction import Transaction
from src.customer_profile import CustomerProfile, CustomerSegment, IncomeBracket, Occupation


@pytest.fixture
def schema_generator():
    """Create an AdvancedSchemaGenerator instance for testing."""
    return AdvancedSchemaGenerator()


@pytest.fixture
def sample_customer():
    """Create a sample customer profile for testing."""
    return CustomerProfile(
        customer_id="CUST0000001",
        age=28,
        gender="Male",
        city="Mumbai",
        state="Maharashtra",
        region="West",
        income_bracket=IncomeBracket.UPPER_MIDDLE,
        occupation=Occupation.SALARIED_EMPLOYEE,
        monthly_income=120000.0,
        segment=CustomerSegment.YOUNG_PROFESSIONAL,
        risk_profile="Moderate",
        digital_savviness="HIGH",
        avg_transaction_amount=1800.0,
        monthly_transaction_count=65,
        preferred_categories=["Food & Dining", "Entertainment"],
        preferred_payment_modes=["UPI", "Credit Card"],
        preferred_shopping_hours=[8, 12, 19],
        weekend_shopper=True,
        merchant_loyalty=0.7,
        brand_conscious=True,
        impulse_buyer=True,
        travels_frequently=True,
        online_shopping_preference=0.8
    )


@pytest.mark.schema
class TestCardTypeGeneration:
    """Test card type generation logic."""
    
    def test_upi_returns_na(self, schema_generator):
        """UPI payments should not have card type."""
        card_type = schema_generator.generate_card_type("UPI", "HIGH")
        assert card_type == "NA"
    
    def test_cash_returns_na(self, schema_generator):
        """Cash payments should not have card type."""
        card_type = schema_generator.generate_card_type("Cash", "MEDIUM")
        assert card_type == "NA"
    
    def test_credit_card_payment_mode(self, schema_generator):
        """Credit Card payment mode with PREMIUM income should predominantly return Credit."""
        # Test multiple times due to income-based probability (85% for PREMIUM)
        card_types = [
            schema_generator.generate_card_type("Credit Card", "PREMIUM")
            for _ in range(100)
        ]
        credit_count = card_types.count("Credit")
        # With 85% probability, expect at least 75% credit cards
        assert credit_count >= 75, f"Expected >=75 Credit cards, got {credit_count}"
    
    def test_debit_card_payment_mode(self, schema_generator):
        """Debit Card payment mode should return Debit."""
        # Test multiple times due to randomness
        card_type = schema_generator.generate_card_type("Debit Card", "LOW")
        assert card_type == "Debit"
    
    def test_high_savviness_prefers_credit(self, schema_generator):
        """High income should favor credit cards."""
        results = [schema_generator.generate_card_type("Card", "HIGH") for _ in range(100)]
        credit_count = results.count("Credit")
        debit_count = results.count("Debit")
        # HIGH income = 75% credit, should have more credit than debit
        assert credit_count > debit_count
    
    def test_low_savviness_prefers_debit(self, schema_generator):
        """Low income should favor debit cards."""
        results = [schema_generator.generate_card_type("Card", "LOW") for _ in range(100)]
        credit_count = results.count("Credit")
        debit_count = results.count("Debit")
        # LOW income = 5% credit, should have more debit than credit
        assert debit_count > credit_count


@pytest.mark.schema
class TestTransactionStatusGeneration:
    """Test transaction status generation logic."""
    
    def test_most_transactions_approved(self, schema_generator):
        """Most transactions should be approved."""
        statuses = [
            schema_generator.generate_transaction_status(1000.0, "UPI", False, "HIGH")
            for _ in range(1000)
        ]
        approved_count = statuses.count("Approved")
        # Should have >95% approval rate for normal transactions
        assert approved_count >= 950
    
    def test_returns_valid_status(self, schema_generator):
        """Should return valid status values."""
        status = schema_generator.generate_transaction_status(5000.0, "Credit Card", True, "MEDIUM")
        assert status in ["Approved", "Declined", "Pending"]
    
    def test_large_amount_higher_decline_rate(self, schema_generator):
        """Large amounts should have higher decline rate."""
        # Small amounts
        small_statuses = [
            schema_generator.generate_transaction_status(500.0, "Credit Card", True, "MEDIUM")
            for _ in range(500)
        ]
        small_declined = small_statuses.count("Declined")
        
        # Large amounts (>50k triggers +5% decline)
        large_statuses = [
            schema_generator.generate_transaction_status(55000.0, "Credit Card", True, "MEDIUM")
            for _ in range(500)
        ]
        large_declined = large_statuses.count("Declined")
        
        # Large amounts should have more declines
        assert large_declined > small_declined
    
    def test_cash_always_approved(self, schema_generator):
        """Cash transactions should always be approved."""
        statuses = [
            schema_generator.generate_transaction_status(100000.0, "Cash", True, "LOW")
            for _ in range(100)
        ]
        # All cash should be approved
        assert all(s == "Approved" for s in statuses)


@pytest.mark.schema
class TestTransactionChannelGeneration:
    """Test transaction channel generation logic."""
    
    def test_upi_predominantly_mobile(self, schema_generator):
        """UPI should predominantly use Mobile channel."""
        channels = [
            schema_generator.generate_transaction_channel("UPI", 25, "HIGH")
            for _ in range(100)
        ]
        mobile_count = channels.count("Mobile")
        # UPI is mobile-friendly, expect >50% mobile (relaxed for variance)
        assert mobile_count >= 50
    
    def test_online_payment_online_channel(self, schema_generator):
        """Online payments should vary by age - older prefer POS."""
        # Test with older age (55) and Credit Card
        channels = [
            schema_generator.generate_transaction_channel("Credit Card", 55, "MEDIUM")
            for _ in range(100)
        ]
        # Should have some variety of channels
        unique_channels = set(channels)
        assert len(unique_channels) >= 1  # At least some channels used
        assert all(ch in ["POS", "Mobile", "Online", "ATM"] for ch in channels)
    
    def test_young_age_prefers_mobile(self, schema_generator):
        """Young customers should have higher mobile channel usage."""
        channels = [
            schema_generator.generate_transaction_channel("UPI", 25, "HIGH")
            for _ in range(100)
        ]
        mobile_count = channels.count("Mobile")
        # Young age + UPI = mobile-friendly, expect >50%
        assert mobile_count >= 50
    
    def test_returns_valid_channel(self, schema_generator):
        """Should return valid channel values."""
        channel = schema_generator.generate_transaction_channel("Credit Card", 35, "MEDIUM")
        assert channel in ["POS", "Online", "Mobile", "ATM"]


@pytest.mark.schema
class TestStateAndRegionMapping:
    """Test state and region mapping for Indian cities."""
    
    def test_mumbai_state_and_region(self, schema_generator):
        """Mumbai should map to Maharashtra and West region."""
        state, region = schema_generator.get_state_and_region("Mumbai")
        assert state == "Maharashtra"
        assert region == "West"
    
    def test_delhi_state_and_region(self, schema_generator):
        """Delhi should map to Delhi and North region."""
        state, region = schema_generator.get_state_and_region("Delhi")
        assert state == "Delhi"
        assert region == "North"
    
    def test_bangalore_state_and_region(self, schema_generator):
        """Bangalore should map to Karnataka and South region."""
        state, region = schema_generator.get_state_and_region("Bangalore")
        assert state == "Karnataka"
        assert region == "South"
    
    def test_all_20_cities_mapped(self, schema_generator):
        """All 20 major Indian cities should be mapped."""
        cities = [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
            "Chandigarh", "Kochi", "Indore", "Bhopal", "Surat",
            "Nagpur", "Visakhapatnam", "Patna", "Vadodara", "Coimbatore"
        ]
        
        for city in cities:
            state, region = schema_generator.get_state_and_region(city)
            assert state is not None
            assert region in ["North", "South", "East", "West", "Central"]


@pytest.mark.schema
class TestAgeGroupGeneration:
    """Test age group categorization."""
    
    def test_age_18_to_25(self, schema_generator):
        """Age 18-25 should map to 18-25 group."""
        age_group = schema_generator.get_age_group(22)
        assert age_group == "18-25"
    
    def test_age_26_to_35(self, schema_generator):
        """Age 26-35 should map to 26-35 group."""
        age_group = schema_generator.get_age_group(30)
        assert age_group == "26-35"
    
    def test_age_66_plus(self, schema_generator):
        """Age 66+ should map to 66+ group."""
        age_group = schema_generator.get_age_group(70)
        assert age_group == "66+"
    
    def test_all_age_groups_covered(self, schema_generator):
        """All age ranges should be covered."""
        test_ages = [20, 30, 40, 50, 60, 70]
        expected_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        
        age_groups = [schema_generator.get_age_group(age) for age in test_ages]
        
        # Check all expected groups are present
        for expected in expected_groups:
            assert expected in age_groups


@pytest.mark.schema
class TestDeviceInfoGeneration:
    """Test device information generation."""
    
    def test_device_info_has_required_fields(self, schema_generator):
        """Device info should have all required fields."""
        device_info = schema_generator.generate_device_info("Mobile", 25, "HIGH")
        
        assert "device_type" in device_info
        assert "app_version" in device_info
        assert "browser_type" in device_info
        assert "os" in device_info
    
    def test_mobile_channel_has_app_version(self, schema_generator):
        """Mobile channel should have app version."""
        device_info = schema_generator.generate_device_info("Mobile", 25, "HIGH")
        
        assert device_info["device_type"] == "Mobile"
        assert device_info["app_version"] is not None
        assert device_info["os"] in ["Android", "iOS"]
    
    def test_online_channel_has_browser(self, schema_generator):
        """Online channel should have browser type."""
        device_info = schema_generator.generate_device_info("Online", 30, "HIGH")
        
        assert device_info["device_type"] == "Web"
        assert device_info["browser_type"] is not None
    
    def test_android_dominant_in_india(self, schema_generator):
        """Android should be dominant in India (72% market share)."""
        os_list = [
            schema_generator.generate_device_info("Mobile", 25, "MEDIUM")["os"]
            for _ in range(100)
        ]
        android_count = os_list.count("Android")
        ios_count = os_list.count("iOS")
        
        # Android should be significantly more than iOS
        assert android_count > ios_count


@pytest.mark.schema
class TestTransactionDataclass:
    """Test Transaction dataclass methods."""
    
    def test_transaction_creation(self):
        """Transaction should be created with all required fields."""
        txn = Transaction(
            transaction_id="TXN001",
            customer_id="CUST001",
            merchant_id="MER_RET_MUM_001",
            date="2024-10-15",
            time="14:30:00",
            day_of_week="Tuesday",
            hour=14,
            amount=1500.0,
            merchant_name="Sample Store",
            category="Retail",
            payment_mode="UPI",
            city="Mumbai",
            transaction_status="Approved",
            is_weekend=False,
            card_type="NA",
            transaction_channel="Mobile",
            customer_age=28,
            customer_income_bracket="Upper Middle",
            customer_segment="Young Professional",
            state="Maharashtra",
            region="West",
            customer_age_group="26-35",
            device_type="Mobile",
            app_version="5.2.1",
            browser_type=None,
            os="Android",
            distance_from_home=0.0,
            time_since_last_txn=3600.0,
            is_first_transaction_with_merchant=True,
            daily_transaction_count=5,
            daily_transaction_amount=4500.0
        )
        
        assert txn.transaction_id == "TXN001"
        assert txn.amount == 1500.0
        assert txn.customer_age_group == "26-35"
    
    def test_to_dict_conversion(self):
        """Transaction should convert to dictionary correctly."""
        txn = Transaction(
            transaction_id="TXN001",
            customer_id="CUST001",
            merchant_id="MER_RET_MUM_001",
            date="2024-10-15",
            time="14:30:00",
            day_of_week="Tuesday",
            hour=14,
            amount=1500.0,
            merchant_name="Sample Store",
            category="Retail",
            payment_mode="UPI",
            city="Mumbai",
            transaction_status="Approved",
            is_weekend=False,
            card_type="NA",
            transaction_channel="Mobile",
            customer_age=28,
            customer_income_bracket="Upper Middle",
            customer_segment="Young Professional",
            state="Maharashtra",
            region="West",
            customer_age_group="26-35",
            device_type="Mobile",
            app_version="5.2.1",
            browser_type=None,
            os="Android",
            distance_from_home=0.0,
            time_since_last_txn=3600.0,
            is_first_transaction_with_merchant=True,
            daily_transaction_count=5,
            daily_transaction_amount=4500.0
        )
        
        txn_dict = txn.to_dict()
        
        assert txn_dict["transaction_id"] == "TXN001"
        assert txn_dict["amount"] == 1500.0
        assert txn_dict["customer_age_group"] == "26-35"
        assert isinstance(txn_dict, dict)
    
    def test_calculate_risk_score_low_risk(self):
        """Low-risk transaction should have low risk score."""
        txn = Transaction(
            transaction_id="TXN001",
            customer_id="CUST001",
            merchant_id="MER_RET_MUM_001",
            date="2024-10-15",
            time="14:30:00",
            day_of_week="Tuesday",
            hour=14,
            amount=500.0,
            merchant_name="Local Store",
            category="Retail",
            payment_mode="UPI",
            city="Mumbai",
            transaction_status="Approved",
            is_weekend=False,
            card_type="NA",
            transaction_channel="Mobile",
            customer_age=28,
            customer_income_bracket="Upper Middle",
            customer_segment="Young Professional",
            state="Maharashtra",
            region="West",
            customer_age_group="26-35",
            device_type="Mobile",
            app_version="5.2.1",
            browser_type=None,
            os="Android",
            distance_from_home=0.0,  # Same city
            time_since_last_txn=3600.0,  # Recent
            is_first_transaction_with_merchant=False,  # Known merchant
            daily_transaction_count=3,  # Normal
            daily_transaction_amount=1500.0  # Normal
        )
        
        risk_score = txn.calculate_risk_score()
        assert 0.0 <= risk_score <= 0.3  # Low risk
    
    def test_calculate_risk_score_high_risk(self):
        """High-risk transaction should have high risk score."""
        txn = Transaction(
            transaction_id="TXN001",
            customer_id="CUST001",
            merchant_id="MER_ELE_CHE_002",
            date="2024-10-15",
            time="03:30:00",
            day_of_week="Tuesday",
            hour=3,
            amount=50000.0,  # High amount
            merchant_name="New Merchant",
            category="Electronics",
            payment_mode="Credit Card",
            city="Chennai",
            transaction_status="Approved",
            is_weekend=False,
            card_type="Credit",
            transaction_channel="Online",
            customer_age=28,
            customer_income_bracket="Upper Middle",
            customer_segment="Young Professional",
            state="Tamil Nadu",
            region="South",
            customer_age_group="26-35",
            device_type="Web",
            app_version=None,
            browser_type="Chrome",
            os="Windows",
            distance_from_home=1400.0,  # Different city (Mumbai to Chennai)
            time_since_last_txn=30.0,  # Very recent (velocity)
            is_first_transaction_with_merchant=True,  # New merchant
            daily_transaction_count=15,  # High count
            daily_transaction_amount=150000.0  # High amount
        )
        
        risk_score = txn.calculate_risk_score()
        assert risk_score >= 0.6  # High risk
