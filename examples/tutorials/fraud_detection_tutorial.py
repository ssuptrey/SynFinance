"""
Fraud Detection ML Tutorial - Example Script

This script demonstrates the complete ML pipeline for fraud detection using SynFinance.

Steps:
1. Generate synthetic transaction data with fraud
2. Engineer ML features
3. Create balanced dataset
4. Train fraud detection models
5. Evaluate model performance

Author: SynFinance Development Team
Version: 0.5.0
Date: October 26, 2025
"""

# %% [markdown]
# # Fraud Detection ML Tutorial
# 
# This tutorial demonstrates how to train fraud detection models using SynFinance synthetic data.

# %% Import libraries
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.customer_profile import CustomerProfile, CustomerSegment
from src.customer_generator import CustomerGenerator
from src.generators.transaction_core import TransactionGenerator
from src.generators.fraud_patterns import FraudPatternGenerator, apply_fraud_labels
from src.generators.ml_features import MLFeatureEngineer
from src.generators.ml_dataset_generator import MLDatasetGenerator
import json

# %% [markdown]
# ## Step 1: Generate Synthetic Transaction Data
# 
# First, we'll generate synthetic transactions with fraud patterns.

# %% Generate customers
print("=" * 60)
print("STEP 1: GENERATING CUSTOMERS")
print("=" * 60)

customer_gen = CustomerGenerator(seed=42)
customers = customer_gen.generate_customers(count=100)
print(f"✓ Generated {len(customers)} customers")

# %% Generate transactions
print("\n" + "=" * 60)
print("STEP 2: GENERATING TRANSACTIONS")
print("=" * 60)

txn_gen = TransactionGenerator(seed=42)
df = txn_gen.generate_dataset(customers=customers, transactions_per_customer=10)
print(f"✓ Generated {len(df)} base transactions")

# Convert DataFrame to list of dicts for processing
transactions = df.to_dict('records')

# %% Inject fraud
print("\n" + "=" * 60)
print("STEP 3: INJECTING FRAUD PATTERNS")
print("=" * 60)

fraud_gen = FraudPatternGenerator(fraud_rate=0.05, seed=42)  # 5% fraud rate

# Create customer map for lookups (convert CustomerProfile to simple namespace for fraud generator)
from types import SimpleNamespace

customer_map = {}
for c in customers:
    # Create a simple namespace that supports both dict access and attribute access
    customer_ns = SimpleNamespace(
        Customer_ID=c.customer_id,
        Age=c.age,
        Gender=c.gender,
        City=c.city,
        city=c.city,  # For compatibility with fraud patterns
        State=c.state,
        Region=c.region,
        Occupation=c.occupation.value,
        Income_Bracket=c.income_bracket.value,
        Segment=c.segment.value,
        Digital_Savviness=c.digital_savviness.value,
        digital_savviness=c.digital_savviness,  # For compatibility
        Risk_Profile=c.risk_profile.value,
        preferred_categories=c.preferred_categories if hasattr(c, 'preferred_categories') else [],
        preferred_merchants=c.preferred_merchants if hasattr(c, 'preferred_merchants') else [],
        preferred_payment_modes=c.preferred_payment_modes if hasattr(c, 'preferred_payment_modes') else [],
    )
    customer_map[c.customer_id] = customer_ns

fraud_transactions = []
customer_history_map = {}

for txn in transactions:
    customer_id = txn.get('Customer_ID')
    customer_dict = customer_map.get(customer_id)
    history = customer_history_map.get(customer_id, [])
    
    # Apply fraud
    modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer_dict, history)
    
    # Add fraud labels
    if fraud_indicator:
        modified_txn['Is_Fraud'] = 1
        modified_txn['Fraud_Type'] = fraud_indicator.fraud_type.value
        modified_txn['Fraud_Confidence'] = fraud_indicator.confidence
    else:
        modified_txn['Is_Fraud'] = 0
        modified_txn['Fraud_Type'] = 'None'
        modified_txn['Fraud_Confidence'] = 0.0
    
    fraud_transactions.append(modified_txn)
    
    # Update history
    if customer_id not in customer_history_map:
        customer_history_map[customer_id] = []
    customer_history_map[customer_id].append(modified_txn)

# Get fraud statistics
stats = fraud_gen.get_fraud_statistics()
print(f"✓ Fraud Statistics:")
print(f"  Total Transactions: {stats['total_transactions']}")
print(f"  Fraud Transactions: {stats['total_fraud']}")
print(f"  Fraud Rate: {stats['fraud_rate']:.2%}")
print(f"  Fraud Types Distribution:")
for fraud_type, count in stats['fraud_by_type'].items():
    print(f"    {fraud_type}: {count}")

# %% [markdown]
# ## Step 2: Engineer ML Features
# 
# Extract 32 ML features from transactions.

# %% Engineer features
print("\n" + "=" * 60)
print("STEP 4: ENGINEERING ML FEATURES")
print("=" * 60)

engineer = MLFeatureEngineer()

ml_features = []
for txn in fraud_transactions:
    customer_id = txn.get('Customer_ID')
    customer_ns = customer_map.get(customer_id)
    # Convert SimpleNamespace to dict for ml_features
    customer_dict = vars(customer_ns) if customer_ns else {}
    history = customer_history_map.get(customer_id, [])[:-1]  # All history except current txn
    
    # Engineer features
    features = engineer.engineer_features(txn, customer_dict, history)
    ml_features.append(features.to_dict())

print(f"✓ Engineered {len(ml_features)} feature sets")
print(f"✓ Features per transaction: 32")

# Show sample features
print("\nSample Features (first transaction):")
sample = ml_features[0]
print(f"  Transaction ID: {sample['transaction_id']}")
print(f"  Daily Transaction Count: {sample['daily_txn_count']}")
print(f"  Weekly Transaction Amount: Rs.{sample['weekly_txn_amount']:.2f}")
print(f"  Distance from Home: {sample['distance_from_home']:.2f} km")
print(f"  Category Diversity Score: {sample['category_diversity_score']:.3f}")
print(f"  Is Fraud: {sample['is_fraud']}")
print(f"  Distance from Home: {sample['distance_from_home']:.1f} km")
print(f"  Travel Velocity: {sample['travel_velocity_kmh']:.1f} km/h")
print(f"  Merchant Loyalty Score: {sample['merchant_loyalty_score']:.2f}")
print(f"  Is Fraud: {sample['is_fraud']}")

# %% [markdown]
# ## Step 3: Create Balanced ML Dataset
# 
# Create train/validation/test splits with balanced classes.

# %% Create dataset
print("\n" + "=" * 60)
print("STEP 5: CREATING ML-READY DATASET")
print("=" * 60)

ml_gen = MLDatasetGenerator(seed=42)
split, metadata = ml_gen.create_ml_ready_dataset(
    ml_features,
    balance_strategy='undersample',  # Balance classes
    target_fraud_rate=0.5,            # 50-50 split
    normalize=True,                    # Normalize features
    encode_categorical=True,           # Encode categorical
    train_ratio=0.70,
    val_ratio=0.15
)

print("\n✓ Dataset Statistics:")
stats = split.get_stats()
for key, value in stats.items():
    if 'fraud_rate' in key:
        print(f"  {key}: {value:.2%}")
    else:
        print(f"  {key}: {value}")

# %% Export datasets
print("\n" + "=" * 60)
print("STEP 6: EXPORTING DATASETS")
print("=" * 60)

import os
output_dir = 'output/ml_dataset'
os.makedirs(output_dir, exist_ok=True)

ml_gen.export_to_csv(split.train, f"{output_dir}/train.csv")
ml_gen.export_to_csv(split.validation, f"{output_dir}/validation.csv")
ml_gen.export_to_csv(split.test, f"{output_dir}/test.csv")
ml_gen.export_metadata(f"{output_dir}/metadata.json", metadata)

print(f"✓ Exported datasets to {output_dir}/")
print(f"  - train.csv ({len(split.train)} samples)")
print(f"  - validation.csv ({len(split.validation)} samples)")
print(f"  - test.csv ({len(split.test)} samples)")
print(f"  - metadata.json")

# %% [markdown]
# ## Step 4: Train Fraud Detection Models
# 
# Train and evaluate fraud detection models (requires scikit-learn).
# 
# **Note:** This is a demonstration script. In production, you would:
# - Import scikit-learn, XGBoost, etc.
# - Train actual models
# - Evaluate with proper metrics
# - Save trained models for deployment

# %% Model training placeholder
print("\n" + "=" * 60)
print("STEP 7: MODEL TRAINING (PLACEHOLDER)")
print("=" * 60)

print("""
Next steps for ML model training:

1. Install ML libraries:
   pip install scikit-learn xgboost pandas

2. Load datasets:
   import pandas as pd
   train_df = pd.read_csv('output/ml_dataset/train.csv')
   val_df = pd.read_csv('output/ml_dataset/validation.csv')
   test_df = pd.read_csv('output/ml_dataset/test.csv')

3. Train models:
   from sklearn.ensemble import RandomForestClassifier
   from xgboost import XGBClassifier
   
   # Random Forest
   rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_model.fit(X_train, y_train)
   
   # XGBoost
   xgb_model = XGBClassifier(n_estimators=100, random_state=42)
   xgb_model.fit(X_train, y_train)

4. Evaluate models:
   from sklearn.metrics import classification_report, roc_auc_score
   
   y_pred = rf_model.predict(X_test)
   print(classification_report(y_test, y_pred))
   print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

5. Feature importance:
   import matplotlib.pyplot as plt
   
   feature_importance = rf_model.feature_importances_
   plt.barh(feature_names, feature_importance)
   plt.xlabel('Importance')
   plt.title('Feature Importance for Fraud Detection')
   plt.show()
""")

# %% Summary
print("\n" + "=" * 60)
print("TUTORIAL COMPLETE!")
print("=" * 60)

# Get final fraud statistics
final_stats = fraud_gen.get_fraud_statistics()

print(f"""
Summary:
✓ Generated {len(customers)} customers
✓ Generated {len(fraud_transactions)} transactions ({final_stats['fraud_rate']:.2%} fraud)
✓ Engineered 32 ML features per transaction
✓ Created balanced dataset (50-50 fraud/normal)
✓ Split into train/val/test (70/15/15)
✓ Exported ML-ready datasets to {output_dir}/

Your datasets are ready for training fraud detection models!

Key Features Generated:
- 6 Aggregate features (daily/weekly counts and amounts)
- 6 Velocity features (transaction frequency in 1h/6h/24h)
- 5 Geographic features (distance, travel patterns)
- 5 Temporal features (unusual hours, weekends, holidays)
- 6 Behavioral features (category diversity, merchant loyalty)
- 4 Network features (shared merchants, customer proximity)

Total: 32 features + 1 label (is_fraud)

Next Steps:
1. Install scikit-learn, xgboost, pandas
2. Load the exported datasets
3. Train Random Forest or XGBoost models
4. Evaluate with precision, recall, F1-score, ROC-AUC
5. Analyze feature importance
6. Deploy best model for real-time fraud detection
""")

# %% Feature metadata
print("\n" + "=" * 60)
print("FEATURE METADATA")
print("=" * 60)

feature_metadata = engineer.get_feature_metadata()
print(f"Total Features: {feature_metadata['feature_count']}")
print("\nFeature List:")
for i, feat in enumerate(feature_metadata['features'], 1):
    print(f"{i:2d}. {feat['name']:30s} ({feat['type']:5s}) - {feat['description']}")

print("\n" + "=" * 60)
print("Thank you for using SynFinance!")
print("=" * 60)
