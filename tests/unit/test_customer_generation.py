"""
Test script to validate customer generation with 1000 customers
Week 1, Day 3-4: Validation
"""

import pytest
import random

from src.customer_generator import CustomerGenerator
from src.customer_profile import CustomerSegment
import json


def validate_customer_generation():
    """Generate 1000 customers and validate distributions"""
    
    print("=" * 80)
    print("SYNFINANCE CUSTOMER GENERATION VALIDATION")
    print("Generating 1000 customers...")
    print("=" * 80)
    
    # Create generator with seed for reproducibility
    generator = CustomerGenerator(seed=42)
    
    # Generate 1000 customers
    customers = generator.generate_customers(1000)
    
    print(f"\n[SUCCESS] Successfully generated {len(customers)} customers\n")
    
    # Get comprehensive statistics
    stats = generator.get_statistics()
    
    # Display segment distribution
    print("SEGMENT DISTRIBUTION")
    print("-" * 80)
    print(f"{'Segment':<30} {'Count':<10} {'Actual %':<12} {'Expected %':<12}")
    print("-" * 80)
    
    expected_distribution = {
        "Young Professional": 20.0,
        "Family Oriented": 25.0,
        "Budget Conscious": 20.0,
        "Tech-Savvy Millennial": 15.0,
        "Affluent Shopper": 8.0,
        "Senior Conservative": 7.0,
        "Student": 5.0
    }
    
    for segment, data in stats["segment_distribution"].items():
        expected = expected_distribution.get(segment, 0)
        actual = data['percentage']
        deviation = abs(actual - expected)
        status = "[PASS]" if deviation < 5.0 else "[WARN]"
        print(f"{segment:<30} {data['count']:<10} {actual:<12.1f} {expected:<12.1f} {status}")
    
    # Age statistics
    print("\n" + "=" * 80)
    print("AGE DISTRIBUTION")
    print("-" * 80)
    print(f"Minimum Age: {stats['age_stats']['min']}")
    print(f"Maximum Age: {stats['age_stats']['max']}")
    print(f"Average Age: {stats['age_stats']['avg']:.1f}")
    
    # Income statistics
    print("\n" + "=" * 80)
    print("INCOME DISTRIBUTION")
    print("-" * 80)
    print(f"Minimum Monthly Income: ₹{stats['income_stats']['min']:,.0f}")
    print(f"Maximum Monthly Income: ₹{stats['income_stats']['max']:,.0f}")
    print(f"Average Monthly Income: ₹{stats['income_stats']['avg']:,.0f}")
    
    # Gender distribution
    print("\n" + "=" * 80)
    print("GENDER DISTRIBUTION")
    print("-" * 80)
    for gender, data in stats["gender_distribution"].items():
        print(f"{gender:<15} {data['count']:<10} ({data['percentage']:.1f}%)")
    
    # Digital savviness
    print("\n" + "=" * 80)
    print("DIGITAL SAVVINESS DISTRIBUTION")
    print("-" * 80)
    for level, data in stats["digital_savviness_distribution"].items():
        print(f"{level:<15} {data['count']:<10} ({data['percentage']:.1f}%)")
    
    # Region distribution
    print("\n" + "=" * 80)
    print("REGION DISTRIBUTION")
    print("-" * 80)
    for region, data in stats["region_distribution"].items():
        print(f"{region:<15} {data['count']:<10} ({data['percentage']:.1f}%)")
    
    # Sample customers from each segment
    print("\n" + "=" * 80)
    print("SAMPLE CUSTOMERS (One from each segment)")
    print("=" * 80)
    
    for segment in CustomerSegment:
        # Find first customer from this segment
        customer = next((c for c in customers if c.segment == segment), None)
        if customer:
            print(f"\n{segment.value}:")
            print(f"  {customer}")
            print(f"  Age: {customer.age} | Gender: {customer.gender}")
            print(f"  Income: ₹{customer.monthly_income:,.0f}/month ({customer.income_bracket.value})")
            print(f"  Occupation: {customer.occupation.value}")
            print(f"  Digital Savviness: {customer.digital_savviness.value}")
            print(f"  Risk Profile: {customer.risk_profile.value}")
            print(f"  Avg Transaction: ₹{customer.avg_transaction_amount:,.2f}")
            print(f"  Monthly Transactions: {customer.monthly_transaction_count}")
            print(f"  Preferred Categories: {', '.join(customer.preferred_categories)}")
            print(f"  Payment Modes: {', '.join(customer.preferred_payment_modes)}")
            print(f"  Online Preference: {customer.online_shopping_preference*100:.0f}%")
            print(f"  Merchant Loyalty: {customer.merchant_loyalty:.2f}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Segment distribution within 5% of expected
    all_within_range = True
    for segment, data in stats["segment_distribution"].items():
        expected = expected_distribution.get(segment, 0)
        actual = data['percentage']
        if abs(actual - expected) > 5.0:
            all_within_range = False
    checks.append(("Segment distribution within ±5%", all_within_range))
    
    # Check 2: Age range reasonable (18-75)
    age_valid = stats['age_stats']['min'] >= 18 and stats['age_stats']['max'] <= 75
    checks.append(("Age range 18-75", age_valid))
    
    # Check 3: Income range reasonable
    income_valid = (stats['income_stats']['min'] >= 10000 and 
                   stats['income_stats']['max'] <= 1000000)
    checks.append(("Income range ₹10k-₹10L", income_valid))
    
    # Check 4: Gender distribution reasonable
    male_pct = stats['gender_distribution']['Male']['percentage']
    gender_valid = 45 <= male_pct <= 60
    checks.append(("Gender distribution reasonable (45-60% male)", gender_valid))
    
    # Check 5: All customers have required fields
    all_valid = all(
        c.customer_id and c.age and c.city and c.income_bracket and 
        c.occupation and c.segment and c.preferred_categories and 
        c.preferred_payment_modes
        for c in customers
    )
    checks.append(("All customers have required fields", all_valid))
    
    # Display results
    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:<10} {check_name}")
    
    # Overall result
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("SUCCESS: ALL VALIDATION CHECKS PASSED!")
        print("[COMPLETE] Customer generation system is working correctly")
    else:
        print("WARNING: SOME VALIDATION CHECKS FAILED")
        print("Review the distribution and adjust segment weights if needed")
    print("=" * 80)
    
    return all_passed, stats


if __name__ == "__main__":
    success, stats = validate_customer_generation()
    
    # Save statistics to file
    print("\n[INFO] Saving statistics to 'output/customer_validation_stats.json'...")
    
    import os
    os.makedirs("output", exist_ok=True)
    
    with open("output/customer_validation_stats.json", "w") as f:
        # Convert stats to JSON-serializable format
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                json_stats[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        json_stats[key][str(k)] = v
                    else:
                        json_stats[key][str(k)] = v
            else:
                json_stats[key] = value
        
        json.dump(json_stats, f, indent=2)
    
    print("[SUCCESS] Statistics saved!")
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("- Review the validation results above")
    print("- Check output/customer_validation_stats.json for detailed statistics")
    print("- If all checks passed, proceed to integrate with transaction generator")
    print(f"{'='*80}")
