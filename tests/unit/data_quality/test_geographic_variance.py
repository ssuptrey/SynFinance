"""Test geographic distribution variance"""
import sys
sys.path.insert(0, 'e:/SynFinance/src')

from customer_generator import CustomerGenerator
from generators.transaction_core import TransactionGenerator
import pandas as pd

cg = CustomerGenerator(seed=42)
tg = TransactionGenerator(seed=42)

customers = cg.generate_customers(10)

print("Testing Geographic Distribution (80-15-5 pattern)\n")
print("=" * 70)

results_50 = []
results_100 = []

for i, customer in enumerate(customers[:5], 1):
    # Test with 50 transactions
    txns_50 = tg.generate_customer_transactions(customer, count=50, days=30)
    df_50 = pd.DataFrame(txns_50)
    pct_50 = (df_50['City'] == customer.city).sum() / len(df_50) * 100
    results_50.append(pct_50)
    
    # Test with 100 transactions
    txns_100 = tg.generate_customer_transactions(customer, count=100, days=30)
    df_100 = pd.DataFrame(txns_100)
    pct_100 = (df_100['City'] == customer.city).sum() / len(df_100) * 100
    results_100.append(pct_100)
    
    print(f"{i}. {customer.customer_id} ({customer.city})")
    print(f"   50 txns:  {pct_50:5.1f}% home city")
    print(f"   100 txns: {pct_100:5.1f}% home city")

print("\n" + "=" * 70)
print("SUMMARY:")
print(f"  50 transactions  - Mean: {sum(results_50)/len(results_50):.1f}%, "
      f"Min: {min(results_50):.1f}%, Max: {max(results_50):.1f}%")
print(f"  100 transactions - Mean: {sum(results_100)/len(results_100):.1f}%, "
      f"Min: {min(results_100):.1f}%, Max: {max(results_100):.1f}%")
print(f"\nExpected: ~80% (with variance due to randomness)")
print(f"Test threshold: >=70% (allows for statistical variance)")
