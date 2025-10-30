"""
Integration Tests for Customer-Aware Transaction Generation
Week 1, Days 5-7: Customer-Transaction Integration Testing

Tests verify:
1. Transactions match customer behavioral profiles
2. Indian market patterns are realistic
3. Behavioral consistency over time
4. Scalability (performance tests)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from src.customer_profile import CustomerProfile, CustomerSegment, DigitalSavviness, IncomeBracket
from src.customer_generator import CustomerGenerator
from data_generator import TransactionGenerator, generate_realistic_dataset


class TestCustomerTransactionIntegration:
    """Test suite for customer-aware transaction generation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.customer_gen = CustomerGenerator(seed=42)
        self.txn_gen = TransactionGenerator(seed=42)
    
    # ============================================================================
    # TEST 1: BEHAVIORAL CONSISTENCY
    # ============================================================================
    
    def test_transaction_categories_match_customer_preferences(self):
        """Verify that >60% of transactions use customer's preferred categories"""
        customer = self.customer_gen.generate_customer()
        transactions = self.txn_gen.generate_customer_transactions(customer, count=100, days=30)
        
        df = pd.DataFrame(transactions)
        preferred_categories = set(customer.preferred_categories)
        
        # Count transactions in preferred categories
        preferred_count = df[df['Category'].isin(preferred_categories)].shape[0]
        preferred_percentage = (preferred_count / len(df)) * 100
        
        print(f"\n[TEST] Category Preference")
        print(f"  Customer Preferred: {', '.join(customer.preferred_categories)}")
        print(f"  Transactions in Preferred: {preferred_count}/100 ({preferred_percentage:.1f}%)")
        
        assert preferred_percentage >= 60, \
            f"Expected >=60% preferred categories, got {preferred_percentage:.1f}%"
    
    def test_payment_modes_match_digital_savviness(self):
        """Verify payment mode selection matches digital savviness"""
        # Test LOW digital savviness customer
        customer_low = None
        for _ in range(100):  # Generate until we get LOW savviness
            c = self.customer_gen.generate_customer()
            if c.digital_savviness == DigitalSavviness.LOW:
                customer_low = c
                break
        
        if customer_low:
            transactions = self.txn_gen.generate_customer_transactions(customer_low, count=50, days=30)
            df = pd.DataFrame(transactions)
            
            # LOW digital savviness should use more Cash/Debit/basic UPI
            traditional_modes = df[df['Payment_Mode'].isin(['Cash', 'Debit Card', 'UPI'])].shape[0]
            traditional_percentage = (traditional_modes / len(df)) * 100
            
            print(f"\n[TEST] Payment Mode - LOW Digital Savviness")
            print(f"  Traditional payment modes: {traditional_percentage:.1f}%")
            print(f"  Mode distribution: {df['Payment_Mode'].value_counts().to_dict()}")
            
            assert traditional_percentage >= 70, \
                f"Expected >=70% traditional modes for LOW savviness, got {traditional_percentage:.1f}%"
    
    def test_transaction_amounts_match_income_bracket(self):
        """Verify transaction amounts correlate with income bracket"""
        # Generate customers from different income brackets
        customers_by_income = {}
        
        for _ in range(200):
            customer = self.customer_gen.generate_customer()
            bracket = customer.income_bracket
            if bracket not in customers_by_income:
                customers_by_income[bracket] = customer
            if len(customers_by_income) >= 4:  # Get 4 different brackets
                break
        
        # Generate transactions and compare averages
        results = {}
        for bracket, customer in customers_by_income.items():
            transactions = self.txn_gen.generate_customer_transactions(customer, count=50, days=30)
            df = pd.DataFrame(transactions)
            avg_amount = df['Amount'].mean()
            results[bracket] = avg_amount
        
        print(f"\n[TEST] Transaction Amount by Income Bracket")
        for bracket, avg_amount in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {bracket.value}: ₹{avg_amount:,.2f}")
        
        # Verify that higher income brackets have higher transaction amounts
        # (at least Premium > Low)
        if IncomeBracket.PREMIUM in results and IncomeBracket.LOW in results:
            assert results[IncomeBracket.PREMIUM] > results[IncomeBracket.LOW] * 2, \
                "Premium bracket should have significantly higher transaction amounts than Low bracket"
    
    def test_merchant_loyalty_behavior(self):
        """Verify merchant loyalty based on customer profile"""
        # Test high loyalty customer
        customer = None
        for _ in range(100):
            c = self.customer_gen.generate_customer()
            if c.merchant_loyalty > 0.7:  # High loyalty
                customer = c
                break
        
        if customer:
            transactions = self.txn_gen.generate_customer_transactions(customer, count=50, days=30)
            df = pd.DataFrame(transactions)
            
            repeat_percentage = (df['Is_Repeat_Merchant'].sum() / len(df)) * 100
            
            print(f"\n[TEST] Merchant Loyalty")
            print(f"  Customer loyalty score: {customer.merchant_loyalty:.2f}")
            print(f"  Repeat merchant percentage: {repeat_percentage:.1f}%")
            
            # High loyalty customers should have >40% repeat merchants
            assert repeat_percentage >= 40, \
                f"Expected >=40% repeat merchants for high loyalty customer, got {repeat_percentage:.1f}%"
    
    def test_time_patterns_match_occupation(self):
        """Verify transaction times match occupation patterns"""
        # Test salaried employee (should have lunch hour transactions)
        customer = None
        for _ in range(100):
            c = self.customer_gen.generate_customer()
            if c.occupation.value == "Salaried Employee":
                customer = c
                break
        
        if customer:
            transactions = self.txn_gen.generate_customer_transactions(customer, count=100, days=30)
            df = pd.DataFrame(transactions)
            
            # Extract hours
            df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
            
            # Check hour distribution matches occupation pattern
            # Week 2 Enhancement: Uses probability distributions instead of preferred hours list
            
            # For Salaried Employee:
            # - Weekday peaks: 7-9am (before work), 12-2pm (lunch), 6-10pm (after work)
            # - Expected: Higher density in peak hours (6am-10pm), very low at night (0-5am)
            
            peak_hours = set(range(6, 23))  # 6am to 10pm
            night_hours = set(range(0, 6))   # Midnight to 5am
            
            transactions_in_peak = df[df['Hour'].isin(peak_hours)].shape[0]
            transactions_at_night = df[df['Hour'].isin(night_hours)].shape[0]
            
            peak_percentage = (transactions_in_peak / len(df)) * 100
            night_percentage = (transactions_at_night / len(df)) * 100
            
            print(f"\n[TEST] Time Patterns - Salaried Employee (Week 2: Occupation-based distributions)")
            print(f"  Peak hours (6am-10pm): {peak_percentage:.1f}%")
            print(f"  Night hours (0-5am): {night_percentage:.1f}%")
            print(f"  Hour distribution: {df['Hour'].value_counts().sort_index().to_dict()}")
            
            # Week 2: More realistic expectations
            # Most transactions (>85%) should be in peak hours, very few at night (<10%)
            assert peak_percentage >= 85, \
                f"Expected >=85% transactions in peak hours (6am-10pm), got {peak_percentage:.1f}%"
            assert night_percentage <= 10, \
                f"Expected <=10% transactions at night (0-5am), got {night_percentage:.1f}%"
    
    # ============================================================================
    # TEST 2: INDIAN MARKET PATTERNS
    # ============================================================================
    
    def test_upi_dominance_for_small_amounts(self):
        """Verify UPI is dominant for small transactions (<₹500)"""
        df = generate_realistic_dataset(num_customers=50, transactions_per_customer=20, days=30, seed=42)
        
        # Filter small transactions
        small_txns = df[df['Amount'] < 500]
        
        if len(small_txns) > 0:
            upi_percentage = (small_txns['Payment_Mode'] == 'UPI').sum() / len(small_txns) * 100
            
            print(f"\n[TEST] UPI Dominance for Small Amounts")
            print(f"  Small transactions (<₹500): {len(small_txns)}")
            print(f"  UPI usage: {upi_percentage:.1f}%")
            
            # UPI should be >=50% for small amounts in India
            assert upi_percentage >= 50, \
                f"Expected >=50% UPI for small amounts, got {upi_percentage:.1f}%"
    
    def test_indian_merchant_names(self):
        """Verify realistic Indian merchant names"""
        df = generate_realistic_dataset(num_customers=20, transactions_per_customer=50, days=30, seed=42)
        
        # Check for known Indian chains
        indian_chains = [
            'Big Bazaar', 'D-Mart', 'Reliance Fresh', 'Zomato', 'Swiggy', 'Flipkart', 
            'Amazon India', 'Uber', 'Ola', 'Apollo', 'PVR', 'INOX'
        ]
        
        has_indian_merchant = False
        for chain in indian_chains:
            if df['Merchant'].str.contains(chain, case=False).any():
                has_indian_merchant = True
                break
        
        print(f"\n[TEST] Indian Merchant Names")
        print(f"  Sample merchants: {df['Merchant'].head(10).tolist()}")
        print(f"  Contains Indian chains: {has_indian_merchant}")
        
        assert has_indian_merchant, "Dataset should contain realistic Indian merchant names"
    
    def test_geographic_distribution(self):
        """Verify geographic distribution (80% home city)"""
        customers = self.customer_gen.generate_customers(10)
        
        for customer in customers[:3]:  # Test first 3 customers
            # Use 100 transactions for better statistical stability
            # With 50 txns, variance can be 64-84% (mean 77%)
            # With 100 txns, variance is 78-86% (mean 83%)
            transactions = self.txn_gen.generate_customer_transactions(customer, count=100, days=30)
            df = pd.DataFrame(transactions)
            
            home_city_percentage = (df['City'] == customer.city).sum() / len(df) * 100
            
            print(f"\n[TEST] Geographic Distribution - {customer.customer_id}")
            print(f"  Home city: {customer.city}")
            print(f"  Transactions in home city: {home_city_percentage:.1f}%")
            
            # Expected: ~80% in home city (80-15-5 distribution)
            # Threshold: >=70% allows for statistical variance
            assert home_city_percentage >= 70, \
                f"Expected >=70% in home city (80-15-5 distribution), got {home_city_percentage:.1f}%"
    
    # ============================================================================
    # TEST 3: DATA QUALITY
    # ============================================================================
    
    def test_no_missing_values(self):
        """Verify no missing values in required fields (Week 3 update)"""
        df = generate_realistic_dataset(num_customers=20, transactions_per_customer=30, days=30, seed=42)
        
        null_counts = df.isnull().sum()
        
        print(f"\n[TEST] Missing Values Check (Week 3)")
        print(f"  Total rows: {len(df)}")
        print(f"  Null counts: {null_counts[null_counts > 0].to_dict()}")
        
        # Week 3: Some fields are nullable by design
        # - App_Version: Only for mobile transactions
        # - Browser_Type: Only for web transactions
        # - Time_Since_Last_Txn: Null for first transaction per customer
        nullable_fields = {'App_Version', 'Browser_Type', 'Time_Since_Last_Txn'}
        
        # Check required fields have no nulls
        required_fields = [col for col in df.columns if col not in nullable_fields]
        required_nulls = df[required_fields].isnull().sum().sum()
        
        print(f"  Required fields with nulls: {required_nulls}")
        print(f"  Nullable fields (by design): {nullable_fields}")
        
        assert required_nulls == 0, f"Required fields should have no missing values, found {required_nulls}"
        
        # Verify nullable fields have expected patterns
        assert df['App_Version'].notnull().sum() > 0, "Some transactions should have App_Version (mobile)"
        assert df['Browser_Type'].notnull().sum() > 0, "Some transactions should have Browser_Type (web)"
        assert df['Time_Since_Last_Txn'].notnull().sum() > 0, "Most transactions should have Time_Since_Last_Txn"
    
    def test_transaction_id_uniqueness(self):
        """Verify all transaction IDs are unique"""
        df = generate_realistic_dataset(num_customers=50, transactions_per_customer=20, days=30, seed=42)
        
        duplicates = df['Transaction_ID'].duplicated().sum()
        
        print(f"\n[TEST] Transaction ID Uniqueness")
        print(f"  Total transactions: {len(df)}")
        print(f"  Duplicate IDs: {duplicates}")
        
        assert duplicates == 0, "All transaction IDs should be unique"
    
    def test_date_range_consistency(self):
        """Verify transactions are within specified date range"""
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 31)
        
        df = generate_realistic_dataset(
            num_customers=10, 
            transactions_per_customer=20, 
            start_date=start_date,
            end_date=end_date,
            seed=42
        )
        
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        
        print(f"\n[TEST] Date Range Consistency")
        print(f"  Requested range: {start_date.date()} to {end_date.date()}")
        print(f"  Actual range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        assert df['DateTime'].min() >= start_date, "All transactions should be after start_date"
        assert df['DateTime'].max() <= end_date, "All transactions should be before end_date"
    
    def test_amount_validity(self):
        """Verify all transaction amounts are positive and reasonable"""
        df = generate_realistic_dataset(num_customers=30, transactions_per_customer=30, days=30, seed=42)
        
        print(f"\n[TEST] Amount Validity")
        print(f"  Min amount: ₹{df['Amount'].min():,.2f}")
        print(f"  Max amount: ₹{df['Amount'].max():,.2f}")
        print(f"  Mean amount: ₹{df['Amount'].mean():,.2f}")
        
        assert (df['Amount'] > 0).all(), "All amounts should be positive"
        # Week 2 Day 3-4: Tier 3 cities have 0.8x COL adjustment, so min can be ~₹40
        assert df['Amount'].min() >= 40, "Minimum amount should be >=₹40 (accounting for COL adjustment)"
        # Max amount allows for premium purchases with festival/salary day multipliers (up to 3x)
        assert df['Amount'].max() <= 1000000, "Maximum amount should be reasonable (<₹10L)"
    
    # ============================================================================
    # TEST 4: SCALABILITY
    # ============================================================================
    
    def test_large_dataset_generation(self):
        """Test generation of large dataset (10K+ transactions)"""
        import time
        
        start_time = time.time()
        
        df = generate_realistic_dataset(
            num_customers=200,
            transactions_per_customer=50,
            days=30,
            seed=42
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n[TEST] Scalability - Large Dataset")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Generation time: {elapsed_time:.2f} seconds")
        print(f"  Throughput: {len(df)/elapsed_time:.0f} transactions/second")
        
        assert len(df) >= 10000, "Should generate at least 10K transactions"
        assert elapsed_time < 60, "Should complete in under 60 seconds"
    
    def test_streaming_generation(self):
        """Test streaming generation for memory efficiency"""
        customers = self.customer_gen.generate_customers(20)
        
        chunks = []
        for chunk in self.txn_gen.generate_dataset_streaming(
            customers, 
            transactions_per_customer=50, 
            days=30, 
            chunk_size=200
        ):
            chunks.append(chunk)
            print(f"  Generated chunk: {len(chunk)} transactions")
        
        total_transactions = sum(len(chunk) for chunk in chunks)
        
        print(f"\n[TEST] Streaming Generation")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total transactions: {total_transactions:,}")
        
        assert len(chunks) >= 2, "Should generate multiple chunks"
        assert total_transactions >= 1000, "Should generate at least 1000 transactions"


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SynFinance Integration Tests - Week 1 Days 5-7")
    print("=" * 80)
    
    tester = TestCustomerTransactionIntegration()
    tester.setup_method()
    
    # Run all tests
    tests = [
        ("Behavioral Consistency: Categories", tester.test_transaction_categories_match_customer_preferences),
        ("Behavioral Consistency: Payment Modes", tester.test_payment_modes_match_digital_savviness),
        ("Behavioral Consistency: Income-Amount Correlation", tester.test_transaction_amounts_match_income_bracket),
        ("Behavioral Consistency: Merchant Loyalty", tester.test_merchant_loyalty_behavior),
        ("Behavioral Consistency: Time Patterns", tester.test_time_patterns_match_occupation),
        ("Indian Market: UPI Dominance", tester.test_upi_dominance_for_small_amounts),
        ("Indian Market: Merchant Names", tester.test_indian_merchant_names),
        ("Indian Market: Geographic Distribution", tester.test_geographic_distribution),
        ("Data Quality: No Missing Values", tester.test_no_missing_values),
        ("Data Quality: Transaction ID Uniqueness", tester.test_transaction_id_uniqueness),
        ("Data Quality: Date Range", tester.test_date_range_consistency),
        ("Data Quality: Amount Validity", tester.test_amount_validity),
        ("Scalability: Large Dataset", tester.test_large_dataset_generation),
        ("Scalability: Streaming", tester.test_streaming_generation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n\nRunning: {test_name}")
            print("-" * 80)
            test_func()
            print(f"\n[PASS] {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_name}")
            print(f"  Exception: {e}")
            failed += 1
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    
    if failed == 0:
        print("\n[SUCCESS] All integration tests passed!")
        print("Week 1 Days 5-7: Customer-Transaction Integration COMPLETE")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Review and fix issues.")
    
    print("=" * 80)
