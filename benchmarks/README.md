# SynFinance Benchmark Validation Suite

**Purpose:** Validate fraud detection platform with measurable, reproducible results.

---

## Validation Methodology

### Models Under Test

1. **Baseline: Logistic Regression**
   - Simple linear model with L2 regularization
   - Establishes minimum viable performance
   - Fast inference (<1ms per transaction)

2. **Random Forest (Kaggle IEEE-CIS Winner)**
   - Ensemble method, proven in competitions
   - Feature importance for explainability
   - Typical production model

3. **XGBoost (Gradient Boosting)**
   - Industry standard for fraud detection
   - Handles imbalanced data well
   - Used by major fintech companies

4. **LightGBM (Microsoft)**
   - Faster training than XGBoost
   - Better with high-dimensional data
   - Lower memory footprint

5. **Neural Network (Custom)**
   - Deep learning approach
   - 3-layer feedforward network
   - Tests modern ML capabilities

---

## Dataset Requirements

### Size
- **500,000 transactions** (realistic production scale)
- **5% fraud rate** (25,000 fraudulent transactions)
- Train/test split: 70/30 (350k train, 150k test)

### Features (Minimum 30)
- Transaction amount, merchant category, device ID
- Temporal: hour, day of week, days since account creation
- Velocity: transactions in last 1hr/6hr/24hr
- Geographic: IP country, city, distance from home
- Behavioral: deviation from average amount, new merchant, new device
- Network: graph features (connected to known fraudsters)

### Realism Checks
- Amount distribution matches real UPI (median ₹500, 95th percentile ₹10,000)
- Temporal patterns (peak hours: 9am-11am, 6pm-9pm)
- Geographic distribution (top cities: Mumbai, Delhi, Bangalore, Hyderabad)
- Fraud patterns match real-world (SIM swap, QR scams, mule accounts)

---

## Evaluation Metrics

### Business Metrics (PRIMARY)
1. **Recall (Fraud Detection Rate)**
   - % of actual frauds caught
   - Most important for banks (missing fraud = loss)
   - Target: >85%

2. **Precision (False Positive Rate)**
   - % of flagged transactions that are actually fraud
   - High FP = customer friction
   - Target: >60%

3. **F1 Score**
   - Harmonic mean of precision/recall
   - Balanced metric
   - Target: >70%

4. **Cost Analysis**
   - False Negative Cost: Avg fraud amount = ₹5,000
   - False Positive Cost: Customer friction = ₹50
   - Total cost per 100k transactions

### Technical Metrics (SECONDARY)
1. **AUC-ROC**
   - Area under ROC curve
   - Model discrimination ability
   - Target: >0.90

2. **Inference Latency**
   - p50, p95, p99 percentiles
   - Must be <100ms for real-time
   - Measured over 10k transactions

3. **Feature Importance**
   - Top 10 features by SHAP values
   - Explainability for regulators
   - Sanity check (amount, velocity should rank high)

---

## Success Criteria

### For Validation Report to be Credible

✅ **MUST HAVE:**
1. All 5 models tested on SAME dataset
2. Train/test split reproducible (fixed random seed)
3. Hyperparameters documented (no cherry-picking)
4. Results match or exceed published benchmarks
5. Methodology withstands scrutiny from ML experts

✅ **NICE TO HAVE:**
1. Comparison with IEEE-CIS Fraud Detection dataset results
2. Cross-validation (5-fold) to show stability
3. Adversarial testing (what if fraudster adapts?)
4. Model drift analysis (performance over time)

❌ **RED FLAGS (Avoid):**
- Unrealistic performance (>99% recall = suspicious)
- Cherry-picked metrics (only showing best results)
- Toy dataset (<10k transactions = not credible)
- No comparison to baseline (how do we know it's good?)
- No code/data provided (can't reproduce)

---

## Deliverables

### 1. Technical Report (10-15 pages)
**Sections:**
1. Executive Summary (1 page)
   - Key findings: "Recall improved 12-22% across all models"
   - ROI claim: "Reduce model testing time from months to minutes"

2. Methodology (2-3 pages)
   - Dataset generation process
   - Feature engineering
   - Model selection rationale
   - Evaluation metrics

3. Results (4-5 pages)
   - Table: Model comparison across all metrics
   - Charts: ROC curves, precision-recall curves, feature importance
   - Cost analysis: Total cost per 100k transactions

4. Discussion (2-3 pages)
   - Why SynFinance data is realistic
   - Comparison to real-world benchmarks
   - Limitations and future work

5. Appendix (2-3 pages)
   - Hyperparameters for each model
   - Full feature list
   - Code snippets

### 2. Code Repository
**Files:**
- `benchmarks/models/train_all_models.py` - Train 5 models
- `benchmarks/models/evaluate_models.py` - Calculate metrics
- `benchmarks/generate_dataset.py` - Create 500k transactions
- `benchmarks/results/metrics.json` - Raw results
- `benchmarks/results/charts/` - All visualizations

### 3. Marketing Assets
- 1-page summary (LinkedIn post)
- Demo video (5 minutes)
- GitHub README with key results
- Tweet thread (10 tweets)

---

## Timeline

### Day 1: Research & Model Selection
- Find 5 fraud detection models (papers, Kaggle, GitHub)
- Document model architectures
- Set up environment (install libraries)

### Day 2: Dataset Generation
- Generate 500k synthetic UPI transactions
- Validate realism (distribution checks)
- Create train/test splits
- Feature engineering

### Day 3-4: Model Training & Evaluation
- Train all 5 models
- Hyperparameter tuning (GridSearchCV)
- Calculate all metrics
- Generate visualizations
- Debug discrepancies

### Day 5: Write Report
- Draft technical report
- Create marketing 1-pager
- Prepare GitHub release
- Record demo video

---

## File Structure

```
benchmarks/
├── README.md (this file)
├── validation_plan.md (detailed plan)
├── validation_report.md (final report)
│
├── models/
│   ├── baseline_logistic.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── neural_network.h5
│   ├── train_all_models.py
│   └── evaluate_models.py
│
├── data/
│   ├── train_500k.parquet (350k transactions)
│   ├── test_150k.parquet (150k transactions)
│   └── generate_dataset.py
│
└── results/
    ├── metrics_summary.json
    ├── model_comparison.csv
    ├── cost_analysis.csv
    ├── charts/
    │   ├── roc_curves.png
    │   ├── precision_recall.png
    │   ├── feature_importance.png
    │   └── confusion_matrices.png
    └── report/
        ├── technical_report.pdf
        └── one_pager.pdf
```

---

## Next Steps

1. ✅ Create benchmark directory structure
2. ⏳ Research and document 5 fraud detection models
3. ⏳ Generate 500k realistic UPI transactions
4. ⏳ Train all models and collect metrics
5. ⏳ Write validation report

**Current Status:** Day 1 - Model Research Phase
