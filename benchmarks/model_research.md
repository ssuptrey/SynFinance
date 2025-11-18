# Fraud Detection Models - Research & Selection

**Date:** November 4, 2024  
**Purpose:** Document 5 fraud detection models for SynFinance benchmark validation

---

## Selection Criteria

1. **Diversity:** Different algorithm families (linear, tree-based, neural)
2. **Proven:** Used in real fraud detection systems or competitions
3. **Reproducible:** Open-source implementations available
4. **Explainable:** Can provide feature importance
5. **Production-ready:** Inference latency <100ms

---

## Model 1: Baseline Logistic Regression

### Description
Simple linear classifier with L2 regularization. Industry standard for baseline.

### Architecture
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,  # Inverse regularization strength
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42
)
```

### Strengths
- Fast training (<1 minute on 500k samples)
- Fast inference (<1ms per transaction)
- Coefficients = feature importance (explainable)
- Well-understood by regulators

### Weaknesses
- Linear decision boundary (can't capture complex patterns)
- Poor with feature interactions
- Lower accuracy than ensemble methods

### Expected Performance
- Recall: 60-70%
- Precision: 50-60%
- F1: 55-65%
- AUC-ROC: 0.75-0.80

### References
- Standard sklearn implementation
- Used as baseline in IEEE-CIS competition
- Industry standard for compliance (GDPR, explainability)

---

## Model 2: Random Forest (Kaggle IEEE-CIS Winner)

### Description
Ensemble of 100+ decision trees. Proven winner in Kaggle fraud competitions.

### Architecture
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=15,  # Tree depth (prevent overfit)
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',  # Feature sampling
    class_weight='balanced',
    n_jobs=-1,  # Parallel training
    random_state=42
)
```

### Strengths
- Handles non-linear patterns well
- Robust to outliers
- Feature importance (Gini importance)
- Works with missing data

### Weaknesses
- Slower inference than logistic (~10ms)
- Larger model size (100MB+)
- Can overfit on small datasets

### Expected Performance
- Recall: 75-85%
- Precision: 65-75%
- F1: 70-80%
- AUC-ROC: 0.85-0.92

### References
- IEEE-CIS Fraud Detection (1st place solution): https://www.kaggle.com/c/ieee-fraud-detection
- Typical production model for fintech

---

## Model 3: XGBoost (Gradient Boosting)

### Description
Gradient boosted trees. Industry standard for fraud detection at PayPal, Stripe, etc.

### Architecture
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,  # Row sampling
    colsample_bytree=0.8,  # Column sampling
    gamma=1,  # Minimum loss reduction
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    scale_pos_weight=19,  # Handle 5% fraud rate (95/5)
    tree_method='hist',  # Fast histogram-based
    random_state=42
)
```

### Strengths
- Best performance on tabular data
- Handles imbalanced data well
- SHAP values for explainability
- Fast inference (~5ms)

### Weaknesses
- Hyperparameter tuning required
- Can overfit if not regularized
- Slower training than Random Forest

### Expected Performance
- Recall: 80-90%
- Precision: 70-80%
- F1: 75-85%
- AUC-ROC: 0.90-0.95

### References
- PayPal fraud detection: https://arxiv.org/abs/1609.08152
- Stripe ML platform: https://stripe.com/blog/payment-api-design
- Production standard for fintech

---

## Model 4: LightGBM (Microsoft)

### Description
Gradient boosted trees optimized for speed and memory. Used by Microsoft, Alibaba.

### Architecture
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=31,  # Leaf-wise growth
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=19,
    random_state=42
)
```

### Strengths
- Faster training than XGBoost (2-3x)
- Lower memory usage
- Handles categorical features natively
- Similar accuracy to XGBoost

### Weaknesses
- Can overfit on small datasets
- Less popular than XGBoost (fewer resources)

### Expected Performance
- Recall: 78-88%
- Precision: 68-78%
- F1: 73-83%
- AUC-ROC: 0.88-0.94

### References
- LightGBM paper: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Microsoft fraud detection system

---

## Model 5: Neural Network (Custom Architecture)

### Description
3-layer feedforward neural network with dropout regularization.

### Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_dim=n_features),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
```

### Strengths
- Can learn complex non-linear patterns
- Scales to large datasets
- Modern ML approach

### Weaknesses
- Requires more data (>100k samples)
- Slower training (GPU needed)
- Black box (harder to explain)
- Inference latency higher (~20ms)

### Expected Performance
- Recall: 75-85%
- Precision: 65-75%
- F1: 70-80%
- AUC-ROC: 0.86-0.92

### References
- Typical deep learning baseline for fraud detection
- Used by large tech companies (Google, Facebook)

---

## Comparison Matrix

| Model | Training Time | Inference (ms) | Explainability | Expected F1 | Production Use |
|-------|--------------|----------------|----------------|-------------|----------------|
| Logistic Regression | <1 min | <1 | ⭐⭐⭐⭐⭐ | 55-65% | Baseline/Compliance |
| Random Forest | ~10 min | ~10 | ⭐⭐⭐⭐ | 70-80% | Common |
| XGBoost | ~15 min | ~5 | ⭐⭐⭐⭐ | 75-85% | Industry Standard |
| LightGBM | ~5 min | ~5 | ⭐⭐⭐⭐ | 73-83% | Microsoft/Alibaba |
| Neural Network | ~30 min | ~20 | ⭐⭐ | 70-80% | Large Tech |

---

## Feature Engineering Requirements

All models require these 30+ features:

### Transaction Features (8)
1. `amount` - Transaction amount
2. `amount_log` - Log-transformed amount
3. `merchant_category` - MCC code (one-hot encoded)
4. `transaction_type` - P2P, P2M, QR code
5. `is_international` - Cross-border flag
6. `currency` - INR, USD, etc.
7. `is_online` - Online vs in-person
8. `payment_method` - UPI, card, wallet

### Temporal Features (10)
9. `hour` - Hour of day (0-23)
10. `day_of_week` - 0=Monday, 6=Sunday
11. `is_weekend` - Boolean
12. `is_night` - 11pm-6am
13. `days_since_account_creation` - Account age
14. `days_since_last_transaction` - Recency
15. `transactions_today` - Count
16. `transactions_this_week` - Count
17. `avg_amount_last_30d` - Rolling average
18. `std_amount_last_30d` - Rolling std dev

### Geographic Features (5)
19. `ip_country` - IP geolocation
20. `city` - Transaction city
21. `distance_from_home` - km from registered address
22. `distance_from_last_txn` - Geographic velocity
23. `new_location` - First time in this city

### Behavioral Features (5)
24. `amount_deviation` - Z-score from personal avg
25. `new_merchant` - First time at this merchant
26. `new_device` - First time from this device
27. `failed_pin_attempts` - Failed UPI PIN count
28. `account_takeover_score` - Behavioral anomaly score

### Network Features (3)
29. `connected_to_fraudster` - Graph analysis
30. `mule_account_score` - Money laundering risk
31. `merchant_fraud_rate` - Historical fraud at merchant

---

## Implementation Plan

### Day 1: Setup (4 hours)
1. Install libraries: `pip install scikit-learn xgboost lightgbm tensorflow`
2. Create model training pipeline skeleton
3. Set up reproducible random seeds
4. Test each model on toy dataset (1000 samples)

### Day 2: Dataset Generation (6 hours)
1. Generate 500k transactions using existing generators
2. Add UPI-specific features
3. Validate realism (distribution checks)
4. Create train/test splits (70/30)
5. Save to parquet format

### Day 3: Training (6 hours)
1. Train Logistic Regression (baseline)
2. Train Random Forest + hyperparameter tuning
3. Train XGBoost + hyperparameter tuning
4. Train LightGBM + hyperparameter tuning
5. Train Neural Network (with early stopping)

### Day 4: Evaluation (6 hours)
1. Calculate metrics for all models
2. Generate ROC curves, PR curves
3. Feature importance analysis
4. Cost analysis (FP vs FN costs)
5. Latency benchmarks

### Day 5: Report Writing (6 hours)
1. Write technical report (10-15 pages)
2. Create visualizations
3. Write 1-page summary
4. Prepare GitHub release
5. Record demo video

---

## Next Steps

1. ✅ Document 5 models
2. ⏳ Install required libraries
3. ⏳ Create model training pipeline
4. ⏳ Generate 500k dataset
5. ⏳ Train all models
6. ⏳ Evaluate and compare
7. ⏳ Write validation report

**Ready to proceed with implementation.**
