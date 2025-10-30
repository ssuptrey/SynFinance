"""
Unit Tests for Temporal Pattern Generator
Week 2, Days 1-2: Temporal Pattern Testing

Tests verify:
1. Hour distributions match occupation patterns
2. Day-of-week multipliers work correctly
3. Salary day patterns (spikes on 1st, 30th, 31st)
4. Pre-salary day reduction (28th-29th)
5. Festival spending multipliers
6. Combined temporal multipliers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from datetime import datetime, date
from collections import Counter

from generators.temporal_generator import TemporalPatternGenerator
from src.customer_profile import CustomerProfile, CustomerSegment, IncomeBracket, DigitalSavviness
from src.customer_generator import CustomerGenerator
from utils.indian_data import INDIAN_FESTIVALS


class TestTemporalPatternGenerator:
    """Test suite for temporal pattern generation"""
    
    def setup_method(self):
        """Setup test fixtures - using deterministic test factory"""
        self.temporal_gen = TemporalPatternGenerator(seed=42)
        
        # Use test factory for deterministic customer creation
        self.salaried_customer = CustomerGenerator.create_test_customer(
            occupation="Salaried Employee",
            customer_id="TEST_SALARIED_001",
            age=30,
            segment=CustomerSegment.YOUNG_PROFESSIONAL,
            income_bracket=IncomeBracket.MIDDLE,
            city="Mumbai"
        )
        
        self.student_customer = CustomerGenerator.create_test_customer(
            occupation="Student",
            customer_id="TEST_STUDENT_001",
            age=20,
            segment=CustomerSegment.STUDENT,
            income_bracket=IncomeBracket.LOW,
            city="Delhi"
        )
        
        self.homemaker_customer = CustomerGenerator.create_test_customer(
            occupation="Homemaker",
            customer_id="TEST_HOMEMAKER_001",
            age=35,
            segment=CustomerSegment.FAMILY_ORIENTED,
            income_bracket=IncomeBracket.MIDDLE,
            city="Bangalore"
        )
    
    # ============================================================================
    # TEST 1: HOUR DISTRIBUTIONS
    # ============================================================================
    
    def test_salaried_employee_weekday_hours(self):
        """Verify salaried employees have morning/lunch/evening peaks on weekdays"""
        test_date = date(2025, 1, 6)  # Monday
        
        hours = []
        for _ in range(500):
            hour = self.temporal_gen.select_transaction_hour(self.salaried_customer, test_date)
            hours.append(hour)
        
        hour_counts = Counter(hours)
        
        # Peak hours for salaried: 7-9am, 12-2pm, 6-10pm
        peak_hours = list(range(7, 10)) + list(range(12, 15)) + list(range(18, 23))
        off_peak_hours = list(range(0, 6)) + [10, 11, 15, 16, 17, 23]
        
        peak_count = sum(hour_counts.get(h, 0) for h in peak_hours)
        off_peak_count = sum(hour_counts.get(h, 0) for h in off_peak_hours)
        
        peak_percentage = (peak_count / len(hours)) * 100
        
        print(f"\n[TEST] Salaried Employee Weekday Hours")
        print(f"  Total transactions: {len(hours)}")
        print(f"  Peak hours (7-9am, 12-2pm, 6-10pm): {peak_percentage:.1f}%")
        print(f"  Top 5 hours: {hour_counts.most_common(5)}")
        
        # Salaried employees should have >70% transactions in peak hours
        assert peak_percentage >= 70, \
            f"Expected >=70% in peak hours, got {peak_percentage:.1f}%"
    
    def test_student_evening_peak(self):
        """Verify students have strong evening peak (6-11pm)"""
        test_date = date(2025, 1, 6)  # Monday
        
        hours = []
        for _ in range(500):
            hour = self.temporal_gen.select_transaction_hour(self.student_customer, test_date)
            hours.append(hour)
        
        hour_counts = Counter(hours)
        
        # Students: Evening peak 6-11pm (after classes)
        evening_hours = list(range(18, 24))
        evening_count = sum(hour_counts.get(h, 0) for h in evening_hours)
        evening_percentage = (evening_count / len(hours)) * 100
        
        print(f"\n[TEST] Student Evening Peak")
        print(f"  Total transactions: {len(hours)}")
        print(f"  Evening hours (6-11pm): {evening_percentage:.1f}%")
        print(f"  Top 5 hours: {hour_counts.most_common(5)}")
        
        # Students should have strong evening preference (>=45% is significant vs 25% random)
        assert evening_percentage >= 45, \
            f"Expected >=45% in evening (6-11pm), got {evening_percentage:.1f}%"
    
    def test_homemaker_morning_peak(self):
        """Verify homemakers have morning peak (6-11am)"""
        test_date = date(2025, 1, 6)  # Monday
        
        hours = []
        for _ in range(500):
            hour = self.temporal_gen.select_transaction_hour(self.homemaker_customer, test_date)
            hours.append(hour)
        
        hour_counts = Counter(hours)
        
        # Homemakers: Morning peak 6-11am
        morning_hours = list(range(6, 12))
        morning_count = sum(hour_counts.get(h, 0) for h in morning_hours)
        morning_percentage = (morning_count / len(hours)) * 100
        
        print(f"\n[TEST] Homemaker Morning Peak")
        print(f"  Total transactions: {len(hours)}")
        print(f"  Morning hours (6-11am): {morning_percentage:.1f}%")
        print(f"  Top 5 hours: {hour_counts.most_common(5)}")
        
        # Homemakers should have >40% transactions in morning
        assert morning_percentage >= 40, \
            f"Expected >=40% in morning (6-11am), got {morning_percentage:.1f}%"
    
    def test_weekend_vs_weekday_distribution(self):
        """Verify weekend distributions differ from weekday"""
        weekday = date(2025, 1, 6)  # Monday
        weekend = date(2025, 1, 11)  # Saturday
        
        weekday_hours = []
        weekend_hours = []
        
        for _ in range(300):
            weekday_hours.append(
                self.temporal_gen.select_transaction_hour(self.salaried_customer, weekday)
            )
            weekend_hours.append(
                self.temporal_gen.select_transaction_hour(self.salaried_customer, weekend)
            )
        
        weekday_counts = Counter(weekday_hours)
        weekend_counts = Counter(weekend_hours)
        
        print(f"\n[TEST] Weekend vs Weekday Distribution")
        print(f"  Weekday top 3: {weekday_counts.most_common(3)}")
        print(f"  Weekend top 3: {weekend_counts.most_common(3)}")
        
        # Distributions should be different
        # Weekend should have more late morning/afternoon activity
        weekend_midday = sum(weekend_counts.get(h, 0) for h in range(10, 16))
        weekday_midday = sum(weekday_counts.get(h, 0) for h in range(10, 16))
        
        weekend_midday_pct = (weekend_midday / len(weekend_hours)) * 100
        weekday_midday_pct = (weekday_midday / len(weekday_hours)) * 100
        
        print(f"  Weekday midday (10am-4pm): {weekday_midday_pct:.1f}%")
        print(f"  Weekend midday (10am-4pm): {weekend_midday_pct:.1f}%")
        
        # Weekend should have more midday activity than weekday
        assert weekend_midday_pct > weekday_midday_pct, \
            "Weekend should have more midday activity than weekday"
    
    # ============================================================================
    # TEST 2: DAY-OF-WEEK MULTIPLIERS
    # ============================================================================
    
    def test_student_weekend_multiplier(self):
        """Verify students spend more on weekends (1.5x multiplier)"""
        weekday = date(2025, 1, 6)  # Monday
        weekend = date(2025, 1, 11)  # Saturday
        
        weekday_mult = self.temporal_gen.get_day_of_week_multiplier(
            self.student_customer, weekday
        )
        weekend_mult = self.temporal_gen.get_day_of_week_multiplier(
            self.student_customer, weekend
        )
        
        print(f"\n[TEST] Student Day-of-Week Multipliers")
        print(f"  Weekday multiplier: {weekday_mult:.2f}x")
        print(f"  Weekend multiplier: {weekend_mult:.2f}x")
        print(f"  Weekend boost: {(weekend_mult/weekday_mult - 1)*100:.1f}%")
        
        # Students should have higher weekend multiplier
        assert weekend_mult > weekday_mult, \
            f"Weekend multiplier ({weekend_mult:.2f}) should be > weekday ({weekday_mult:.2f})"
        
        # Weekend should be at least 1.3x weekday
        assert weekend_mult >= weekday_mult * 1.3, \
            f"Weekend should be >=1.3x weekday, got {weekend_mult/weekday_mult:.2f}x"
    
    def test_young_professional_weekend_boost(self):
        """Verify young professionals spend more on weekends"""
        weekday = date(2025, 1, 6)  # Monday
        weekend = date(2025, 1, 11)  # Saturday
        
        weekday_mult = self.temporal_gen.get_day_of_week_multiplier(
            self.salaried_customer, weekday
        )
        weekend_mult = self.temporal_gen.get_day_of_week_multiplier(
            self.salaried_customer, weekend
        )
        
        print(f"\n[TEST] Young Professional Day-of-Week Multipliers")
        print(f"  Weekday multiplier: {weekday_mult:.2f}x")
        print(f"  Weekend multiplier: {weekend_mult:.2f}x")
        print(f"  Weekend boost: {(weekend_mult/weekday_mult - 1)*100:.1f}%")
        
        # Young professionals should spend more on weekends
        assert weekend_mult > weekday_mult, \
            f"Weekend ({weekend_mult:.2f}) should be > weekday ({weekday_mult:.2f})"
    
    # ============================================================================
    # TEST 3: SALARY DAY PATTERNS
    # ============================================================================
    
    def test_salary_day_detection(self):
        """Verify salary days are correctly detected (1st, 30th, 31st)"""
        # Test 1st of month
        first = date(2025, 1, 1)
        assert self.temporal_gen.is_salary_day(first), \
            "1st should be detected as salary day"
        
        # Test 30th
        thirtieth = date(2025, 1, 30)
        assert self.temporal_gen.is_salary_day(thirtieth), \
            "30th should be detected as salary day"
        
        # Test 31st
        thirty_first = date(2025, 1, 31)
        assert self.temporal_gen.is_salary_day(thirty_first), \
            "31st should be detected as salary day"
        
        # Test non-salary days
        regular_day = date(2025, 1, 15)
        assert not self.temporal_gen.is_salary_day(regular_day), \
            "15th should NOT be detected as salary day"
        
        print(f"\n[TEST] Salary Day Detection")
        print(f"  1st: {self.temporal_gen.is_salary_day(first)} ✓")
        print(f"  30th: {self.temporal_gen.is_salary_day(thirtieth)} ✓")
        print(f"  31st: {self.temporal_gen.is_salary_day(thirty_first)} ✓")
        print(f"  15th: {self.temporal_gen.is_salary_day(regular_day)} ✓")
    
    def test_salary_day_spending_spike(self):
        """Verify spending increases on salary days (1.5x-2.0x)"""
        salary_day = date(2025, 1, 1)  # 1st
        regular_day = date(2025, 1, 15)
        
        salary_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, salary_day
        )
        regular_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, regular_day
        )
        
        print(f"\n[TEST] Salary Day Spending Spike")
        print(f"  Regular day (15th): {regular_mult:.2f}x")
        print(f"  Salary day (1st): {salary_mult:.2f}x")
        print(f"  Increase: {(salary_mult/regular_mult - 1)*100:.1f}%")
        
        # Salary day should have 1.5x-2.0x multiplier
        assert salary_mult >= 1.5, \
            f"Salary day multiplier should be >=1.5x, got {salary_mult:.2f}x"
        assert salary_mult <= 2.0, \
            f"Salary day multiplier should be <=2.0x, got {salary_mult:.2f}x"
        
        # Regular day should be 1.0x
        assert regular_mult == 1.0, \
            f"Regular day should be 1.0x, got {regular_mult:.2f}x"
    
    def test_pre_salary_day_reduction(self):
        """Verify spending reduces before salary day (0.7x on 28th-29th)"""
        pre_salary = date(2025, 1, 28)  # 28th
        regular_day = date(2025, 1, 15)
        
        pre_salary_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, pre_salary
        )
        regular_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, regular_day
        )
        
        print(f"\n[TEST] Pre-Salary Day Reduction")
        print(f"  Regular day (15th): {regular_mult:.2f}x")
        print(f"  Pre-salary day (28th): {pre_salary_mult:.2f}x")
        print(f"  Reduction: {(1 - pre_salary_mult/regular_mult)*100:.1f}%")
        
        # Pre-salary should be 0.7x
        assert pre_salary_mult == 0.7, \
            f"Pre-salary multiplier should be 0.7x, got {pre_salary_mult:.2f}x"
        
        # Should be less than regular day
        assert pre_salary_mult < regular_mult, \
            "Pre-salary spending should be less than regular day"
    
    def test_salary_pattern_cycle(self):
        """Verify full salary cycle: regular → pre-salary → salary → regular"""
        regular = date(2025, 1, 15)
        pre_salary = date(2025, 1, 29)
        salary = date(2025, 2, 1)
        post_salary = date(2025, 2, 5)
        
        regular_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, regular
        )
        pre_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, pre_salary
        )
        salary_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, salary
        )
        post_mult = self.temporal_gen.get_salary_day_multiplier(
            self.salaried_customer, post_salary
        )
        
        print(f"\n[TEST] Salary Pattern Cycle")
        print(f"  Regular (15th): {regular_mult:.2f}x")
        print(f"  Pre-salary (29th): {pre_mult:.2f}x ↓")
        print(f"  Salary day (1st): {salary_mult:.2f}x ↑↑")
        print(f"  Post-salary (5th): {post_mult:.2f}x")
        
        # Verify pattern: regular (1.0) → pre (0.7) → salary (1.5-2.0) → regular (1.0)
        assert regular_mult == 1.0
        assert pre_mult == 0.7
        assert salary_mult >= 1.5
        assert post_mult == 1.0
    
    # ============================================================================
    # TEST 4: FESTIVAL MULTIPLIERS
    # ============================================================================
    
    def test_diwali_spending_boost(self):
        """Verify Diwali has spending boost (1.5x-1.8x)"""
        # Diwali 2025: November 1
        diwali = date(2025, 11, 1)
        regular = date(2025, 2, 15)  # February - no festivals
        
        diwali_mult, diwali_festival = self.temporal_gen.get_festival_multiplier(
            diwali, self.salaried_customer, INDIAN_FESTIVALS
        )
        regular_mult, regular_festival = self.temporal_gen.get_festival_multiplier(
            regular, self.salaried_customer, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Diwali Spending Boost")
        print(f"  Regular day: {regular_mult:.2f}x (festival: {regular_festival or 'None'})")
        print(f"  Diwali: {diwali_mult:.2f}x (festival: {diwali_festival})")
        print(f"  Boost: {(diwali_mult/regular_mult - 1)*100:.1f}%")
        
        # Diwali should have 1.5x-1.8x boost
        assert diwali_mult >= 1.5, \
            f"Diwali should have >=1.5x multiplier, got {diwali_mult:.2f}x"
        
        # Regular day should be 1.0x
        assert regular_mult == 1.0, \
            f"Regular day should be 1.0x, got {regular_mult:.2f}x"
    
    def test_christmas_spending_boost(self):
        """Verify Christmas has spending boost"""
        christmas = date(2025, 12, 25)
        regular = date(2025, 2, 15)  # February - no festivals
        
        christmas_mult, christmas_festival = self.temporal_gen.get_festival_multiplier(
            christmas, self.salaried_customer, INDIAN_FESTIVALS
        )
        regular_mult, regular_festival = self.temporal_gen.get_festival_multiplier(
            regular, self.salaried_customer, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Christmas Spending Boost")
        print(f"  Regular day: {regular_mult:.2f}x (festival: {regular_festival or 'None'})")
        print(f"  Christmas: {christmas_mult:.2f}x (festival: {christmas_festival})")
        print(f"  Boost: {(christmas_mult/regular_mult - 1)*100:.1f}%")
        
        # Christmas should have boost
        assert christmas_mult > 1.0, \
            f"Christmas should have >1.0x multiplier, got {christmas_mult:.2f}x"
    
    def test_holi_spending_boost(self):
        """Verify Holi has spending boost"""
        # Holi 2025: March 14
        holi = date(2025, 3, 14)
        regular = date(2025, 2, 15)  # February - no festivals
        
        holi_mult, holi_festival = self.temporal_gen.get_festival_multiplier(
            holi, self.salaried_customer, INDIAN_FESTIVALS
        )
        regular_mult, regular_festival = self.temporal_gen.get_festival_multiplier(
            regular, self.salaried_customer, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Holi Spending Boost")
        print(f"  Regular day: {regular_mult:.2f}x (festival: {regular_festival or 'None'})")
        print(f"  Holi: {holi_mult:.2f}x (festival: {holi_festival})")
        print(f"  Boost: {(holi_mult/regular_mult - 1)*100:.1f}%")
        
        # Holi should have boost
        assert holi_mult > 1.0, \
            f"Holi should have >1.0x multiplier, got {holi_mult:.2f}x"
    
    # ============================================================================
    # TEST 5: COMBINED TEMPORAL MULTIPLIERS
    # ============================================================================
    
    def test_combined_multipliers_regular_weekday(self):
        """Verify combined multipliers on regular weekday"""
        regular_weekday = date(2025, 2, 18)  # Wednesday in February (no festivals)
        
        combined, breakdown = self.temporal_gen.get_combined_temporal_multiplier(
            self.salaried_customer, regular_weekday, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Combined Multipliers - Regular Weekday")
        print(f"  Date: {regular_weekday} (Wednesday)")
        print(f"  Day-of-week: {breakdown['day_of_week']:.2f}x")
        print(f"  Salary day: {breakdown['salary_day']:.2f}x")
        print(f"  Festival: {breakdown['festival']:.2f}x")
        print(f"  Combined: {combined:.2f}x")
        
        # Regular weekday should be close to 1.0x (slightly less for young professional)
        assert 0.8 <= combined <= 1.1, \
            f"Regular weekday should be ~0.9-1.0x, got {combined:.2f}x"
    
    def test_combined_multipliers_salary_weekend(self):
        """Verify combined multipliers on salary day weekend (highest spending)"""
        # Saturday, 1st of month
        salary_weekend = date(2025, 2, 1)  # Sunday (1st)
        
        combined, breakdown = self.temporal_gen.get_combined_temporal_multiplier(
            self.salaried_customer, salary_weekend, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Combined Multipliers - Salary Weekend")
        print(f"  Date: {salary_weekend} (Sunday, 1st)")
        print(f"  Day-of-week: {breakdown['day_of_week']:.2f}x")
        print(f"  Salary day: {breakdown['salary_day']:.2f}x")
        print(f"  Festival: {breakdown['festival']:.2f}x")
        print(f"  Combined: {combined:.2f}x")
        
        # Salary weekend should be high (1.5x-2.5x)
        assert combined >= 1.5, \
            f"Salary weekend should be >=1.5x, got {combined:.2f}x"
    
    def test_combined_multipliers_pre_salary_weekday(self):
        """Verify combined multipliers on pre-salary weekday (lowest spending)"""
        # Wednesday, 28th in February (no festivals)
        pre_salary_weekday = date(2025, 7, 29)  # Tuesday, 29th in July (no festivals)
        
        combined, breakdown = self.temporal_gen.get_combined_temporal_multiplier(
            self.salaried_customer, pre_salary_weekday, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Combined Multipliers - Pre-Salary Weekday")
        print(f"  Date: {pre_salary_weekday} (Tuesday, 29th)")
        print(f"  Day-of-week: {breakdown['day_of_week']:.2f}x")
        print(f"  Salary day: {breakdown['salary_day']:.2f}x")
        print(f"  Festival: {breakdown['festival']:.2f}x")
        print(f"  Combined: {combined:.2f}x")
        
        # Pre-salary weekday should be low (0.6x-0.8x)
        assert combined <= 0.8, \
            f"Pre-salary weekday should be <=0.8x, got {combined:.2f}x"
    
    def test_combined_multipliers_diwali_salary_weekend(self):
        """Verify combined multipliers on Diwali + Salary + Weekend (maximum spending)"""
        # If Diwali falls on 1st and is a Saturday (hypothetical)
        # Use actual Diwali 2025 date
        diwali = date(2025, 11, 1)  # Diwali Saturday
        
        combined, breakdown = self.temporal_gen.get_combined_temporal_multiplier(
            self.salaried_customer, diwali, INDIAN_FESTIVALS
        )
        
        print(f"\n[TEST] Combined Multipliers - Diwali + Salary + Weekend")
        print(f"  Date: {diwali} (Diwali, 1st, Saturday)")
        print(f"  Day-of-week: {breakdown['day_of_week']:.2f}x")
        print(f"  Salary day: {breakdown['salary_day']:.2f}x")
        print(f"  Festival: {breakdown['festival']:.2f}x")
        print(f"  Combined: {combined:.2f}x")
        
        # Triple boost should be significant (2.0x-3.0x, but capped)
        assert combined >= 2.0, \
            f"Diwali+Salary+Weekend should be >=2.0x, got {combined:.2f}x"
        
        # Should be capped at 3.0x
        assert combined <= 3.0, \
            f"Combined should be capped at 3.0x, got {combined:.2f}x"
    
    def test_combined_multipliers_capping(self):
        """Verify combined multipliers are capped at reasonable ranges"""
        # Test various dates
        test_dates = [
            date(2025, 1, 1),   # Salary day
            date(2025, 1, 15),  # Regular day
            date(2025, 1, 29),  # Pre-salary
            date(2025, 11, 1),  # Diwali + Salary
        ]
        
        print(f"\n[TEST] Combined Multipliers Capping")
        for test_date in test_dates:
            combined, _ = self.temporal_gen.get_combined_temporal_multiplier(
                self.salaried_customer, test_date, INDIAN_FESTIVALS
            )
            
            print(f"  {test_date}: {combined:.2f}x")
            
            # Should be between 0.5x and 3.0x
            assert 0.5 <= combined <= 3.0, \
                f"Multiplier should be in range [0.5, 3.0], got {combined:.2f}x on {test_date}"


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Temporal Pattern Generator Tests - Week 2 Days 1-2")
    print("=" * 80)
    
    test_suite = TestTemporalPatternGenerator()
    
    tests = [
        ("Hour Distribution: Salaried Weekday", test_suite.test_salaried_employee_weekday_hours),
        ("Hour Distribution: Student Evening", test_suite.test_student_evening_peak),
        ("Hour Distribution: Homemaker Morning", test_suite.test_homemaker_morning_peak),
        ("Hour Distribution: Weekend vs Weekday", test_suite.test_weekend_vs_weekday_distribution),
        ("Day-of-Week: Student Weekend", test_suite.test_student_weekend_multiplier),
        ("Day-of-Week: Young Professional", test_suite.test_young_professional_weekend_boost),
        ("Salary Day: Detection", test_suite.test_salary_day_detection),
        ("Salary Day: Spending Spike", test_suite.test_salary_day_spending_spike),
        ("Salary Day: Pre-Salary Reduction", test_suite.test_pre_salary_day_reduction),
        ("Salary Day: Full Cycle", test_suite.test_salary_pattern_cycle),
        ("Festival: Diwali", test_suite.test_diwali_spending_boost),
        ("Festival: Christmas", test_suite.test_christmas_spending_boost),
        ("Festival: Holi", test_suite.test_holi_spending_boost),
        ("Combined: Regular Weekday", test_suite.test_combined_multipliers_regular_weekday),
        ("Combined: Salary Weekend", test_suite.test_combined_multipliers_salary_weekend),
        ("Combined: Pre-Salary Weekday", test_suite.test_combined_multipliers_pre_salary_weekday),
        ("Combined: Diwali+Salary+Weekend", test_suite.test_combined_multipliers_diwali_salary_weekend),
        ("Combined: Capping", test_suite.test_combined_multipliers_capping),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n\nRunning: {test_name}")
        print("-" * 80)
        test_suite.setup_method()
        
        try:
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
        print("\n[SUCCESS] All temporal pattern tests passed!")
        print("Week 2 Days 1-2: Temporal Patterns VALIDATED ✓")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Review and fix issues.")
    
    print("=" * 80)
