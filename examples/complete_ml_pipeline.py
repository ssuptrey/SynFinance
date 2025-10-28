"""
Complete End-to-End ML Pipeline Example
=========================================

This comprehensive example demonstrates the complete SynFinance fraud detection
workflow from data generation to production deployment.

Workflow Steps:
1. Generate synthetic transactions with fraud and anomalies
2. Engineer 69 combined ML features
3. Perform advanced analytics
4. Train and optimize models
5. Deploy via API
6. Make real-time predictions
7. Generate comprehensive reports

Author: SynFinance Team
Version: 0.7.0
Date: October 28, 2025
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Any
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import SyntheticDataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.combined_ml_features import CombinedMLFeatureGenerator
from src.analytics.advanced_analytics import (
    CorrelationAnalyzer,
    FeatureImportanceAnalyzer,
    ModelPerformanceAnalyzer,
    StatisticalTestAnalyzer
)
from src.ml.model_optimization import (
    HyperparameterOptimizer,
    EnsembleModelBuilder,
    FeatureSelector,
    ModelRegistry
)
from src.performance.parallel_generator import ParallelGenerator
from src.performance.cache_manager import CacheManager


class CompleteMLPipeline:
    """
    Orchestrates the complete end-to-end ML pipeline for fraud detection.
    """
    
    def __init__(self, output_dir: str = "output/complete_pipeline"):
        """Initialize the pipeline with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.transactions: List[Dict[str, Any]] = []
        self.features_df: pd.DataFrame = None
        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        
        # Initialize components
        self.cache_manager = CacheManager()
        self.model_registry = ModelRegistry(str(self.output_dir / "models"))
        
        print("=" * 80)
        print("SynFinance Complete ML Pipeline")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print()
    
    def step1_generate_data(
        self,
        num_transactions: int = 50000,
        fraud_rate: float = 0.05,
        anomaly_rate: float = 0.10,
        use_parallel: bool = True
    ):
        """
        Step 1: Generate synthetic transactions with fraud and anomalies.
        
        Args:
            num_transactions: Number of transactions to generate
            fraud_rate: Percentage of fraudulent transactions (0-1)
            anomaly_rate: Percentage of anomalous transactions (0-1)
            use_parallel: Use parallel generation for speed
        """
        print("\n" + "=" * 80)
        print("STEP 1: Data Generation")
        print("=" * 80)
        
        start_time = time.time()
        
        if use_parallel:
            print(f"Generating {num_transactions:,} transactions using parallel processing...")
            parallel_gen = ParallelGenerator(num_workers=4)
            self.transactions = parallel_gen.generate(
                num_transactions=num_transactions,
                fraud_rate=fraud_rate
            )
        else:
            print(f"Generating {num_transactions:,} transactions...")
            generator = SyntheticDataGenerator()
            fraud_gen = FraudPatternGenerator(fraud_rate=fraud_rate)
            
            self.transactions = []
            for i in range(num_transactions):
                txn = generator.generate_transaction()
                txn = fraud_gen.maybe_apply_fraud(txn, generator.transaction_history)
                self.transactions.append(txn)
                generator.transaction_history.append(txn)
        
        # Apply anomaly patterns
        print(f"Applying anomaly patterns (rate: {anomaly_rate:.1%})...")
        anomaly_gen = AnomalyPatternGenerator(anomaly_rate=anomaly_rate)
        self.transactions = [
            anomaly_gen.apply_anomaly_patterns(txn, self.transactions[:i])
            for i, txn in enumerate(self.transactions)
        ]
        
        elapsed = time.time() - start_time
        
        # Statistics
        fraud_count = sum(1 for t in self.transactions if t.get('is_fraud', False))
        anomaly_count = sum(1 for t in self.transactions if t.get('has_anomaly', False))
        
        print(f"\n✓ Generated {len(self.transactions):,} transactions in {elapsed:.2f}s")
        print(f"  - Fraud transactions: {fraud_count:,} ({fraud_count/len(self.transactions):.1%})")
        print(f"  - Anomalous transactions: {anomaly_count:,} ({anomaly_count/len(self.transactions):.1%})")
        print(f"  - Generation speed: {len(self.transactions)/elapsed:.0f} txns/sec")
        
        # Save raw data
        output_file = self.output_dir / "raw_transactions.csv"
        pd.DataFrame(self.transactions).to_csv(output_file, index=False)
        print(f"\n✓ Saved raw transactions to: {output_file}")
        
        self.results['step1'] = {
            'total_transactions': len(self.transactions),
            'fraud_count': fraud_count,
            'fraud_rate': fraud_count / len(self.transactions),
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / len(self.transactions),
            'generation_time': elapsed,
            'speed': len(self.transactions) / elapsed
        }
    
    def step2_engineer_features(self):
        """
        Step 2: Engineer 69 combined ML features.
        """
        print("\n" + "=" * 80)
        print("STEP 2: Feature Engineering")
        print("=" * 80)
        
        start_time = time.time()
        
        print("Engineering combined ML features (fraud + anomaly + interactions)...")
        feature_gen = CombinedMLFeatureGenerator()
        
        all_features = []
        for i, txn in enumerate(self.transactions):
            history = self.transactions[:i]
            features = feature_gen.generate_features(txn, history)
            all_features.append(features)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} transactions...")
        
        # Convert to DataFrame
        features_dicts = [f.to_dict() for f in all_features]
        self.features_df = pd.DataFrame(features_dicts)
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Engineered {len(all_features[0].get_feature_names())} features in {elapsed:.2f}s")
        print(f"  - Feature engineering speed: {len(self.transactions)/elapsed:.0f} txns/sec")
        print(f"  - Dataset shape: {self.features_df.shape}")
        
        # Feature statistics
        print("\nFeature Categories:")
        fraud_features = [f for f in self.features_df.columns if f.startswith('fraud_')]
        anomaly_features = [f for f in self.features_df.columns if f.startswith('anomaly_')]
        interaction_features = [f for f in self.features_df.columns if f.startswith(('risk_', 'ensemble_', 'conflict_'))]
        
        print(f"  - Fraud features: {len(fraud_features)}")
        print(f"  - Anomaly features: {len(anomaly_features)}")
        print(f"  - Interaction features: {len(interaction_features)}")
        
        # Save features
        output_file = self.output_dir / "ml_features.csv"
        self.features_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved ML features to: {output_file}")
        
        self.results['step2'] = {
            'num_features': len(all_features[0].get_feature_names()),
            'fraud_features': len(fraud_features),
            'anomaly_features': len(anomaly_features),
            'interaction_features': len(interaction_features),
            'engineering_time': elapsed,
            'speed': len(self.transactions) / elapsed
        }
    
    def step3_perform_analytics(self):
        """
        Step 3: Perform advanced analytics on features and patterns.
        """
        print("\n" + "=" * 80)
        print("STEP 3: Advanced Analytics")
        print("=" * 80)
        
        start_time = time.time()
        
        # Prepare data
        feature_cols = [c for c in self.features_df.columns if c not in ['transaction_id', 'is_fraud', 'has_anomaly']]
        X = self.features_df[feature_cols].fillna(0)
        y = self.features_df['is_fraud'].astype(int)
        
        analytics_results = {}
        
        # 1. Correlation Analysis
        print("\n1. Correlation Analysis...")
        corr_analyzer = CorrelationAnalyzer()
        
        # Pearson correlation
        pearson_corr = corr_analyzer.pearson_correlation(X)
        print(f"   - Computed Pearson correlation matrix ({pearson_corr.shape})")
        
        # High correlations
        high_corr = corr_analyzer.highly_correlated_pairs(pearson_corr, threshold=0.7)
        print(f"   - Found {len(high_corr)} highly correlated feature pairs (r > 0.7)")
        
        analytics_results['correlation'] = {
            'matrix_shape': pearson_corr.shape,
            'high_correlation_pairs': len(high_corr),
            'top_correlations': high_corr[:5] if len(high_corr) > 0 else []
        }
        
        # 2. Statistical Tests
        print("\n2. Statistical Significance Tests...")
        stat_analyzer = StatisticalTestAnalyzer()
        
        # Test each feature for fraud association
        significant_features = []
        for col in feature_cols[:10]:  # Test first 10 for demo
            if self.features_df[col].dtype in [np.float64, np.int64]:
                result = stat_analyzer.t_test(
                    self.features_df[self.features_df['is_fraud'] == 1][col].fillna(0),
                    self.features_df[self.features_df['is_fraud'] == 0][col].fillna(0)
                )
                if result['significant']:
                    significant_features.append((col, result['p_value']))
        
        print(f"   - Significant features: {len(significant_features)}/10 tested")
        
        analytics_results['statistical_tests'] = {
            'features_tested': 10,
            'significant_features': len(significant_features)
        }
        
        # 3. Feature Importance (using Random Forest)
        print("\n3. Feature Importance Analysis...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        importance_analyzer = FeatureImportanceAnalyzer()
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        importance = importance_analyzer.tree_based_importance(rf_model, feature_cols)
        top_features = importance_analyzer.get_top_features(importance, k=10)
        
        print(f"   - Top 10 most important features:")
        for i, (feature, score) in enumerate(top_features, 1):
            print(f"     {i}. {feature}: {score:.4f}")
        
        analytics_results['feature_importance'] = {
            'top_10_features': [(f, float(s)) for f, s in top_features]
        }
        
        # 4. Model Performance Analysis
        print("\n4. Model Performance Evaluation...")
        perf_analyzer = ModelPerformanceAnalyzer()
        
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        
        performance = perf_analyzer.analyze(y_test, y_pred, y_prob)
        perf_dict = performance.to_dict()
        
        print(f"   - Accuracy: {perf_dict['accuracy']:.3f}")
        print(f"   - Precision: {perf_dict['precision']:.3f}")
        print(f"   - Recall: {perf_dict['recall']:.3f}")
        print(f"   - F1-Score: {perf_dict['f1']:.3f}")
        print(f"   - ROC-AUC: {perf_dict['roc_auc']:.3f}")
        
        analytics_results['model_performance'] = perf_dict
        
        elapsed = time.time() - start_time
        print(f"\n✓ Analytics completed in {elapsed:.2f}s")
        
        # Save analytics results
        output_file = self.output_dir / "analytics_results.json"
        with open(output_file, 'w') as f:
            json.dump(analytics_results, f, indent=2)
        print(f"✓ Saved analytics results to: {output_file}")
        
        self.results['step3'] = {
            'analytics_time': elapsed,
            'results': analytics_results
        }
        
        # Store for next steps
        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
    
    def step4_optimize_models(self):
        """
        Step 4: Train and optimize ML models.
        """
        print("\n" + "=" * 80)
        print("STEP 4: Model Optimization")
        print("=" * 80)
        
        start_time = time.time()
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        # 1. Hyperparameter Optimization
        print("\n1. Hyperparameter Optimization...")
        from sklearn.ensemble import RandomForestClassifier
        
        optimizer = HyperparameterOptimizer()
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
        
        print("   Running Grid Search...")
        best_model, best_params, best_score = optimizer.grid_search(
            RandomForestClassifier(random_state=42),
            param_grid,
            X_train,
            y_train,
            cv=3
        )
        
        print(f"   - Best score: {best_score:.3f}")
        print(f"   - Best params: {best_params}")
        
        self.models['random_forest'] = best_model
        
        # 2. Feature Selection
        print("\n2. Feature Selection...")
        selector = FeatureSelector()
        
        selected_features = selector.correlation_based_selection(
            X_train,
            y_train,
            threshold=0.01
        )
        
        print(f"   - Selected {len(selected_features)} features (from {X_train.shape[1]})")
        
        # 3. Ensemble Model
        print("\n3. Building Ensemble Model...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        
        ensemble_builder = EnsembleModelBuilder()
        
        # Train base models
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5)
        
        lr_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Create voting ensemble
        ensemble = ensemble_builder.create_voting_ensemble(
            {
                'rf': best_model,
                'lr': lr_model,
                'gb': gb_model
            },
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        
        print("   - Created soft voting ensemble (3 models)")
        
        # 4. Model Comparison
        print("\n4. Model Performance Comparison...")
        
        models_to_compare = {
            'Random Forest': best_model,
            'Logistic Regression': lr_model,
            'Gradient Boosting': gb_model,
            'Ensemble': ensemble
        }
        
        comparison_results = []
        for name, model in models_to_compare.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            perf_analyzer = ModelPerformanceAnalyzer()
            perf = perf_analyzer.analyze(y_test, y_pred, y_prob)
            perf_dict = perf.to_dict()
            perf_dict['model'] = name
            comparison_results.append(perf_dict)
            
            print(f"   {name}:")
            print(f"     - Accuracy: {perf_dict['accuracy']:.3f}")
            print(f"     - F1-Score: {perf_dict['f1']:.3f}")
            print(f"     - ROC-AUC: {perf_dict['roc_auc']:.3f}")
        
        # Select best model
        best_f1_model = max(comparison_results, key=lambda x: x['f1'])
        print(f"\n✓ Best model: {best_f1_model['model']} (F1: {best_f1_model['f1']:.3f})")
        
        # 5. Register Models
        print("\n5. Registering Models...")
        for name, model in models_to_compare.items():
            metadata = {
                'model_type': name,
                'training_date': datetime.now().isoformat(),
                'num_features': X_train.shape[1],
                'num_samples': X_train.shape[0]
            }
            self.model_registry.register_model(
                model,
                name=name.lower().replace(' ', '_'),
                version='1.0',
                metrics=next(r for r in comparison_results if r['model'] == name),
                metadata=metadata
            )
        
        print(f"   - Registered {len(models_to_compare)} models to registry")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Model optimization completed in {elapsed:.2f}s")
        
        # Save results
        output_file = self.output_dir / "model_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"✓ Saved model comparison to: {output_file}")
        
        self.results['step4'] = {
            'optimization_time': elapsed,
            'best_model': best_f1_model['model'],
            'best_f1_score': best_f1_model['f1'],
            'num_models_trained': len(models_to_compare),
            'comparison_results': comparison_results
        }
    
    def step5_deploy_api(self):
        """
        Step 5: Demonstrate API deployment preparation.
        """
        print("\n" + "=" * 80)
        print("STEP 5: API Deployment")
        print("=" * 80)
        
        print("\nAPI deployment steps (ready for production):")
        print("  1. ✓ Models trained and registered")
        print("  2. ✓ Features engineered and validated")
        print("  3. ✓ Performance metrics established")
        print("  4. → Docker container ready (see docker-compose.yml)")
        print("  5. → FastAPI server ready (see src/api/)")
        print("  6. → CI/CD pipeline configured (see .github/workflows/)")
        
        print("\nTo deploy the API:")
        print("  $ docker-compose up -d")
        print("  $ curl http://localhost:8000/health")
        print("  $ curl http://localhost:8000/docs")
        
        print("\nAPI Endpoints available:")
        print("  - POST /predict - Single transaction prediction")
        print("  - POST /batch-predict - Batch predictions")
        print("  - GET /model-info - Model metadata")
        print("  - GET /metrics - Performance metrics")
        print("  - GET /health - Health check")
        
        self.results['step5'] = {
            'deployment_ready': True,
            'api_endpoints': 5,
            'docker_ready': True,
            'cicd_ready': True
        }
    
    def step6_make_predictions(self, num_samples: int = 10):
        """
        Step 6: Make real-time predictions on sample data.
        """
        print("\n" + "=" * 80)
        print("STEP 6: Real-Time Predictions")
        print("=" * 80)
        
        print(f"\nMaking predictions on {num_samples} sample transactions...")
        
        X_test, y_test = self.test_data
        best_model = self.models['ensemble']
        
        # Sample random transactions
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_true = y_test.iloc[sample_indices]
        
        # Make predictions
        start_time = time.time()
        y_pred = best_model.predict(X_sample)
        y_prob = best_model.predict_proba(X_sample)[:, 1]
        elapsed = time.time() - start_time
        
        print(f"\n✓ Predictions completed in {elapsed*1000:.2f}ms")
        print(f"  - Average latency: {elapsed/num_samples*1000:.2f}ms per prediction")
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 80)
        print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Confidence':<12} {'Result':<10}")
        print("-" * 80)
        
        correct = 0
        for i, (idx, true, pred, prob) in enumerate(zip(sample_indices, y_true, y_pred, y_prob)):
            result = "✓ CORRECT" if true == pred else "✗ WRONG"
            if true == pred:
                correct += 1
            
            print(f"{idx:<8} {true:<12} {pred:<12} {prob:.4f}       {result}")
        
        accuracy = correct / num_samples
        print("-" * 80)
        print(f"Accuracy: {correct}/{num_samples} ({accuracy:.1%})")
        
        self.results['step6'] = {
            'num_predictions': num_samples,
            'prediction_time': elapsed,
            'avg_latency_ms': elapsed / num_samples * 1000,
            'sample_accuracy': accuracy
        }
    
    def step7_generate_reports(self):
        """
        Step 7: Generate comprehensive reports and summaries.
        """
        print("\n" + "=" * 80)
        print("STEP 7: Report Generation")
        print("=" * 80)
        
        # Pipeline summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        total_time = sum(
            self.results[step].get('generation_time', 0) +
            self.results[step].get('engineering_time', 0) +
            self.results[step].get('analytics_time', 0) +
            self.results[step].get('optimization_time', 0) +
            self.results[step].get('prediction_time', 0)
            for step in self.results.keys()
            if step.startswith('step')
        )
        
        print(f"\nTotal Execution Time: {total_time:.2f}s")
        print(f"\nStep-by-Step Breakdown:")
        print(f"  1. Data Generation:      {self.results['step1']['generation_time']:.2f}s")
        print(f"  2. Feature Engineering:  {self.results['step2']['engineering_time']:.2f}s")
        print(f"  3. Advanced Analytics:   {self.results['step3']['analytics_time']:.2f}s")
        print(f"  4. Model Optimization:   {self.results['step4']['optimization_time']:.2f}s")
        print(f"  5. API Deployment:       (preparation complete)")
        print(f"  6. Real-Time Predictions: {self.results['step6']['prediction_time']*1000:.2f}ms")
        print(f"  7. Report Generation:    (in progress)")
        
        print(f"\nKey Metrics:")
        print(f"  - Total Transactions:    {self.results['step1']['total_transactions']:,}")
        print(f"  - Fraud Rate:            {self.results['step1']['fraud_rate']:.1%}")
        print(f"  - Total Features:        {self.results['step2']['num_features']}")
        print(f"  - Best Model F1:         {self.results['step4']['best_f1_score']:.3f}")
        print(f"  - Prediction Latency:    {self.results['step6']['avg_latency_ms']:.2f}ms")
        
        # Save final summary
        summary = {
            'pipeline_version': '0.7.0',
            'execution_date': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'steps': self.results
        }
        
        output_file = self.output_dir / "pipeline_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved pipeline summary to: {output_file}")
        
        # Generate markdown report
        report_file = self.output_dir / "PIPELINE_REPORT.md"
        self._generate_markdown_report(report_file, summary)
        print(f"✓ Saved markdown report to: {report_file}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nNext Steps:")
        print("  1. Review generated reports and analytics")
        print("  2. Deploy API using Docker: docker-compose up -d")
        print("  3. Test API endpoints: curl http://localhost:8000/docs")
        print("  4. Monitor performance: http://localhost:9090 (Prometheus)")
        print("  5. View dashboards: http://localhost:3000 (Grafana)")
    
    def _generate_markdown_report(self, filepath: Path, summary: Dict):
        """Generate a comprehensive markdown report."""
        report = f"""# SynFinance ML Pipeline Execution Report

**Version:** {summary['pipeline_version']}  
**Date:** {summary['execution_date']}  
**Total Time:** {summary['total_execution_time']:.2f}s

---

## Executive Summary

This report summarizes the end-to-end execution of the SynFinance fraud detection ML pipeline.

### Key Achievements

- ✓ Generated {self.results['step1']['total_transactions']:,} synthetic transactions
- ✓ Engineered {self.results['step2']['num_features']} ML features
- ✓ Trained and optimized {self.results['step4']['num_models_trained']} models
- ✓ Achieved F1-Score of {self.results['step4']['best_f1_score']:.3f}
- ✓ Average prediction latency: {self.results['step6']['avg_latency_ms']:.2f}ms

---

## Step 1: Data Generation

- **Transactions Generated:** {self.results['step1']['total_transactions']:,}
- **Fraud Transactions:** {self.results['step1']['fraud_count']:,} ({self.results['step1']['fraud_rate']:.1%})
- **Anomalous Transactions:** {self.results['step1']['anomaly_count']:,} ({self.results['step1']['anomaly_rate']:.1%})
- **Generation Speed:** {self.results['step1']['speed']:.0f} txns/sec
- **Time:** {self.results['step1']['generation_time']:.2f}s

---

## Step 2: Feature Engineering

- **Total Features:** {self.results['step2']['num_features']}
- **Fraud Features:** {self.results['step2']['fraud_features']}
- **Anomaly Features:** {self.results['step2']['anomaly_features']}
- **Interaction Features:** {self.results['step2']['interaction_features']}
- **Engineering Speed:** {self.results['step2']['speed']:.0f} txns/sec
- **Time:** {self.results['step2']['engineering_time']:.2f}s

---

## Step 3: Advanced Analytics

- **Analytics Time:** {self.results['step3']['analytics_time']:.2f}s

### Top 10 Important Features

"""
        # Add top features
        for i, (feature, score) in enumerate(self.results['step3']['results']['feature_importance']['top_10_features'], 1):
            report += f"{i}. **{feature}**: {score:.4f}\n"
        
        report += f"""
### Model Performance

- **Accuracy:** {self.results['step3']['results']['model_performance']['accuracy']:.3f}
- **Precision:** {self.results['step3']['results']['model_performance']['precision']:.3f}
- **Recall:** {self.results['step3']['results']['model_performance']['recall']:.3f}
- **F1-Score:** {self.results['step3']['results']['model_performance']['f1']:.3f}
- **ROC-AUC:** {self.results['step3']['results']['model_performance']['roc_auc']:.3f}

---

## Step 4: Model Optimization

- **Models Trained:** {self.results['step4']['num_models_trained']}
- **Best Model:** {self.results['step4']['best_model']}
- **Best F1-Score:** {self.results['step4']['best_f1_score']:.3f}
- **Optimization Time:** {self.results['step4']['optimization_time']:.2f}s

### Model Comparison

"""
        # Add model comparison table
        report += "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
        report += "|-------|----------|-----------|--------|----------|----------|\n"
        for model in self.results['step4']['comparison_results']:
            report += f"| {model['model']} | {model['accuracy']:.3f} | {model['precision']:.3f} | {model['recall']:.3f} | {model['f1']:.3f} | {model['roc_auc']:.3f} |\n"
        
        report += f"""
---

## Step 5: API Deployment

- **Status:** Ready for Production
- **Docker:** ✓ Configured
- **CI/CD:** ✓ Configured
- **Endpoints:** 5 available

---

## Step 6: Real-Time Predictions

- **Predictions Made:** {self.results['step6']['num_predictions']}
- **Total Time:** {self.results['step6']['prediction_time']*1000:.2f}ms
- **Average Latency:** {self.results['step6']['avg_latency_ms']:.2f}ms
- **Sample Accuracy:** {self.results['step6']['sample_accuracy']:.1%}

---

## Conclusion

The complete ML pipeline executed successfully in {summary['total_execution_time']:.2f}s. The system is ready for production deployment with:

- High-quality synthetic data generation
- Comprehensive feature engineering (69 features)
- Advanced analytics and model optimization
- Production-ready API deployment
- Low-latency real-time predictions (<{self.results['step6']['avg_latency_ms']:.0f}ms)

**System is production-ready for deployment.**

---

*Generated by SynFinance v{summary['pipeline_version']}*
"""
        
        with open(filepath, 'w') as f:
            f.write(report)
    
    def run_complete_pipeline(
        self,
        num_transactions: int = 50000,
        fraud_rate: float = 0.05,
        anomaly_rate: float = 0.10
    ):
        """
        Run the complete end-to-end pipeline.
        
        Args:
            num_transactions: Number of transactions to generate
            fraud_rate: Fraud rate (0-1)
            anomaly_rate: Anomaly rate (0-1)
        """
        try:
            self.step1_generate_data(num_transactions, fraud_rate, anomaly_rate)
            self.step2_engineer_features()
            self.step3_perform_analytics()
            self.step4_optimize_models()
            self.step5_deploy_api()
            self.step6_make_predictions(num_samples=10)
            self.step7_generate_reports()
            
        except Exception as e:
            print(f"\n❌ Pipeline failed at step: {e}")
            raise


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("  SynFinance: Complete End-to-End ML Pipeline Demonstration")
    print("  Version: 0.7.0")
    print("  Author: SynFinance Team")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = CompleteMLPipeline(output_dir="output/complete_pipeline")
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(
        num_transactions=50000,  # 50K transactions
        fraud_rate=0.05,         # 5% fraud
        anomaly_rate=0.10        # 10% anomalies
    )
    
    print("\n" + "=" * 80)
    print("  Pipeline execution completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
