"""
Evaluate All Fraud Detection Models

Calculates metrics, generates visualizations, and compares model performance.
Creates the validation report with measurable ROI claims.

Week 11 Day 3-4: Benchmark Validation - Model Evaluation
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

import tensorflow as tf


class ModelEvaluator:
    """Evaluate and compare all fraud detection models"""
    
    def __init__(self, data_dir='benchmarks/data', models_dir='benchmarks/models', output_dir='benchmarks/results'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        (self.output_dir / 'charts').mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.inference_times = {}
        
        print("=" * 80)
        print("FRAUD DETECTION MODEL EVALUATION")
        print("=" * 80)
    
    def load_data(self):
        """Load test dataset"""
        print("\n[LOADING DATA]")
        
        test_path = self.data_dir / 'test_150k.parquet'
        self.test_df = pd.read_parquet(test_path)
        
        print(f"  Test: {len(self.test_df):,} transactions")
        
        # Load feature names
        with open(self.models_dir / 'feature_names.json', 'r') as f:
            self.feature_cols = json.load(f)
        
        # Re-encode categorical features (if needed)
        categorical_cols = ['city', 'ip_country', 'payment_mode', 'app_version']
        
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            if f'{col}_encoded' not in self.test_df.columns:
                le = LabelEncoder()
                # Fit on unique values
                le.fit(self.test_df[col])
                self.test_df[f'{col}_encoded'] = le.transform(self.test_df[col])
        
        self.X_test = self.test_df[self.feature_cols].values
        self.y_test = self.test_df['is_fraud'].values
        
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Fraud rate: {self.y_test.mean()*100:.2f}%")
    
    def evaluate_model(self, model_name, model, scaler=None, X_test=None):
        """Evaluate single model"""
        print(f"\n[EVALUATING {model_name.upper()}]")
        
        if X_test is None:
            X_test = self.X_test
        
        # Scale if needed
        if scaler is not None:
            X_test_eval = scaler.transform(X_test)
        else:
            X_test_eval = X_test
        
        # Predict
        start_time = time.time()
        
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X_test_eval, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            y_pred = model.predict(X_test_eval)
        
        inference_time = time.time() - start_time
        latency_ms = (inference_time / len(X_test)) * 1000
        
        self.inference_times[model_name] = {
            'total_seconds': inference_time,
            'latency_ms': latency_ms,
            'throughput': len(X_test) / inference_time
        }
        
        print(f"  Inference: {inference_time:.2f}s ({latency_ms:.3f}ms per transaction)")
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Cost analysis (business metrics)
        # False negative cost: Avg fraud amount = ₹5,000
        # False positive cost: Customer friction = ₹50
        fn_cost = fn * 5000
        fp_cost = fp * 50
        total_cost = fn_cost + fp_cost
        cost_per_100k = (total_cost / len(X_test)) * 100_000
        
        print(f"\n  Metrics:")
        print(f"    Accuracy:  {accuracy*100:.2f}%")
        print(f"    Precision: {precision*100:.2f}%")
        print(f"    Recall:    {recall*100:.2f}%")
        print(f"    F1 Score:  {f1*100:.2f}%")
        print(f"    AUC-ROC:   {auc_roc:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {tn:,}")
        print(f"    False Positives: {fp:,}")
        print(f"    False Negatives: {fn:,}")
        print(f"    True Positives:  {tp:,}")
        
        print(f"\n  Cost Analysis (per 100k transactions):")
        print(f"    FN Cost: ₹{fn_cost/len(X_test)*100_000:,.0f} ({fn} frauds missed)")
        print(f"    FP Cost: ₹{fp_cost/len(X_test)*100_000:,.0f} ({fp} false alarms)")
        print(f"    Total:   ₹{cost_per_100k:,.0f}")
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'fn_cost': float(fn_cost),
            'fp_cost': float(fp_cost),
            'total_cost': float(total_cost),
            'cost_per_100k': float(cost_per_100k),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        return y_pred, y_pred_proba
    
    def evaluate_all(self):
        """Evaluate all models"""
        self.load_data()
        
        # 1. Logistic Regression
        with open(self.models_dir / 'logistic_regression.pkl', 'rb') as f:
            lr_data = pickle.load(f)
        self.evaluate_model('logistic_regression', lr_data['model'], lr_data['scaler'])
        
        # 2. Random Forest
        with open(self.models_dir / 'random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        self.evaluate_model('random_forest', rf_model)
        
        # 3. XGBoost
        with open(self.models_dir / 'xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        self.evaluate_model('xgboost', xgb_model)
        
        # 4. LightGBM
        with open(self.models_dir / 'lightgbm.pkl', 'rb') as f:
            lgb_model = pickle.load(f)
        self.evaluate_model('lightgbm', lgb_model)
        
        # 5. Neural Network
        nn_model = tf.keras.models.load_model(self.models_dir / 'neural_network.h5')
        with open(self.models_dir / 'neural_network_scaler.pkl', 'rb') as f:
            nn_scaler = pickle.load(f)
        self.evaluate_model('neural_network', nn_model, nn_scaler)
    
    def generate_comparison_table(self):
        """Generate model comparison table"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall': f"{metrics['recall']*100:.2f}%",
                'F1 Score': f"{metrics['f1']*100:.2f}%",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'Latency (ms)': f"{self.inference_times[model_name]['latency_ms']:.3f}",
                'Cost/100k': f"₹{metrics['cost_per_100k']:,.0f}"
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
        
        # Save to CSV
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        print(f"\n✓ Saved: {self.output_dir / 'model_comparison.csv'}")
        
        # Calculate improvement over baseline
        baseline_recall = self.results['logistic_regression']['recall']
        print(f"\n[IMPROVEMENT OVER BASELINE]")
        for model_name, metrics in self.results.items():
            if model_name != 'logistic_regression':
                recall_improvement = ((metrics['recall'] - baseline_recall) / baseline_recall) * 100
                print(f"  {model_name.replace('_', ' ').title()}: +{recall_improvement:.1f}% recall")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print(f"\n[GENERATING ROC CURVES]")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            y_pred_proba = np.array(metrics['y_pred_proba'])
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = metrics['auc_roc']
            
            label = f"{model_name.replace('_', ' ').title()} (AUC={auc:.4f})"
            plt.plot(fpr, tpr, label=label, linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5000)', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'charts' / 'roc_curves.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves"""
        print(f"\n[GENERATING PRECISION-RECALL CURVES]")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            y_pred_proba = np.array(metrics['y_pred_proba'])
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            avg_prec = metrics['avg_precision']
            
            label = f"{model_name.replace('_', ' ').title()} (AP={avg_prec:.4f})"
            plt.plot(recall, precision, label=label, linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'charts' / 'precision_recall.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print(f"\n[GENERATING CONFUSION MATRICES]")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            
            axes[idx].set_title(f"{model_name.replace('_', ' ').title()}\n" +
                               f"F1={metrics['f1']*100:.1f}%, Recall={metrics['recall']*100:.1f}%",
                               fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide last subplot
        axes[-1].axis('off')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'charts' / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_metrics_comparison(self):
        """Plot metrics comparison bar chart"""
        print(f"\n[GENERATING METRICS COMPARISON]")
        
        metrics_names = ['Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        model_names = [m.replace('_', ' ').title() for m in self.results.keys()]
        
        data = {
            'Precision': [self.results[m]['precision'] for m in self.results.keys()],
            'Recall': [self.results[m]['recall'] for m in self.results.keys()],
            'F1 Score': [self.results[m]['f1'] for m in self.results.keys()],
            'AUC-ROC': [self.results[m]['auc_roc'] for m in self.results.keys()]
        }
        
        df = pd.DataFrame(data, index=model_names)
        
        ax = df.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metrics', fontsize=10)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / 'charts' / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def save_results(self):
        """Save all results to JSON"""
        print(f"\n[SAVING RESULTS]")
        
        # Combine all results
        full_results = {
            'models': self.results,
            'inference_times': self.inference_times,
            'test_samples': len(self.X_test),
            'fraud_rate': float(self.y_test.mean())
        }
        
        output_path = self.output_dir / 'evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_report(self):
        """Generate final evaluation report"""
        self.evaluate_all()
        self.generate_comparison_table()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrices()
        self.plot_metrics_comparison()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("✓ EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  - {self.output_dir / 'model_comparison.csv'}")
        print(f"  - {self.output_dir / 'evaluation_results.json'}")
        print(f"  - {self.output_dir / 'charts' / 'roc_curves.png'}")
        print(f"  - {self.output_dir / 'charts' / 'precision_recall.png'}")
        print(f"  - {self.output_dir / 'charts' / 'confusion_matrices.png'}")
        print(f"  - {self.output_dir / 'charts' / 'metrics_comparison.png'}")
        
        print(f"\nNext step: Write validation report")


def main():
    """Evaluate all models"""
    evaluator = ModelEvaluator()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
