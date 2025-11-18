"""
Train All 5 Fraud Detection Models

Trains baseline through advanced models on 500k synthetic UPI dataset.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network

Week 11 Day 3: Benchmark Validation - Model Training
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    """Train and save all fraud detection models"""
    
    def __init__(self, data_dir='benchmarks/data', output_dir='benchmarks/models'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path('benchmarks/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.training_times = {}
        
        print("=" * 80)
        print("FRAUD DETECTION MODEL TRAINING")
        print("=" * 80)
    
    def load_data(self):
        """Load train and test datasets"""
        print("\n[LOADING DATA]")
        
        train_path = self.data_dir / 'train_500k.parquet'
        test_path = self.data_dir / 'test_150k.parquet'
        
        self.train_df = pd.read_parquet(train_path)
        self.test_df = pd.read_parquet(test_path)
        
        print(f"  Train: {len(self.train_df):,} transactions")
        print(f"  Test: {len(self.test_df):,} transactions")
        print(f"  Features: {len(self.train_df.columns)}")
        
        # Prepare features and labels
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features for training"""
        print("\n[FEATURE ENGINEERING]")
        
        # Identify feature columns (exclude ID columns and target)
        exclude_cols = ['transaction_id', 'customer_id', 'merchant_id', 'is_fraud', 'timestamp']
        
        # Categorical columns
        categorical_cols = ['city', 'ip_country', 'payment_mode', 'app_version']
        
        # Numeric columns
        numeric_cols = [col for col in self.train_df.columns 
                       if col not in exclude_cols + categorical_cols]
        
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        
        # Encode categorical features
        self.label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            self.train_df[f'{col}_encoded'] = le.fit_transform(self.train_df[col])
            self.test_df[f'{col}_encoded'] = le.transform(self.test_df[col])
            self.label_encoders[col] = le
        
        # Final feature list
        self.feature_cols = numeric_cols + [f'{col}_encoded' for col in categorical_cols]
        
        # Extract features and labels
        self.X_train = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['is_fraud'].values
        
        self.X_test = self.test_df[self.feature_cols].values
        self.y_test = self.test_df['is_fraud'].values
        
        print(f"  Final feature count: {len(self.feature_cols)}")
        print(f"  Train shape: {self.X_train.shape}")
        print(f"  Test shape: {self.X_test.shape}")
        print(f"  Class balance: {self.y_train.mean()*100:.2f}% fraud")
        
        # Save feature names
        with open(self.output_dir / 'feature_names.json', 'w') as f:
            json.dump(self.feature_cols, f, indent=2)
    
    def train_logistic_regression(self):
        """Train baseline logistic regression"""
        print("\n" + "=" * 80)
        print("[MODEL 1/5] LOGISTIC REGRESSION (Baseline)")
        print("=" * 80)
        
        start_time = time.time()
        
        # Scale features (important for logistic regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Train model
        print("\nTraining...")
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train_scaled, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['logistic_regression'] = training_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save model and scaler
        with open(self.output_dir / 'logistic_regression.pkl', 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        
        self.models['logistic_regression'] = (model, scaler, X_train_scaled, X_test_scaled)
        
        print("✓ Model saved")
    
    def train_random_forest(self):
        """Train Random Forest classifier"""
        print("\n" + "=" * 80)
        print("[MODEL 2/5] RANDOM FOREST")
        print("=" * 80)
        
        start_time = time.time()
        
        print("\nTraining with 200 trees...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['random_forest'] = training_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save model
        with open(self.output_dir / 'random_forest.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['random_forest'] = (model, None, self.X_train, self.X_test)
        
        print("✓ Model saved")
    
    def train_xgboost(self):
        """Train XGBoost classifier"""
        print("\n" + "=" * 80)
        print("[MODEL 3/5] XGBOOST (Industry Standard)")
        print("=" * 80)
        
        start_time = time.time()
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        print(f"\nTraining with scale_pos_weight={scale_pos_weight:.2f}...")
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=50
        )
        
        training_time = time.time() - start_time
        self.training_times['xgboost'] = training_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save model
        with open(self.output_dir / 'xgboost.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['xgboost'] = (model, None, self.X_train, self.X_test)
        
        print("✓ Model saved")
    
    def train_lightgbm(self):
        """Train LightGBM classifier"""
        print("\n" + "=" * 80)
        print("[MODEL 4/5] LIGHTGBM (Microsoft)")
        print("=" * 80)
        
        start_time = time.time()
        
        # Calculate scale_pos_weight
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        print(f"\nTraining with scale_pos_weight={scale_pos_weight:.2f}...")
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.log_evaluation(50)]
        )
        
        training_time = time.time() - start_time
        self.training_times['lightgbm'] = training_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save model
        with open(self.output_dir / 'lightgbm.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['lightgbm'] = (model, None, self.X_train, self.X_test)
        
        print("✓ Model saved")
    
    def train_neural_network(self):
        """Train neural network classifier"""
        print("\n" + "=" * 80)
        print("[MODEL 5/5] NEURAL NETWORK (Deep Learning)")
        print("=" * 80)
        
        start_time = time.time()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Calculate class weights
        class_weight = {
            0: 1.0,
            1: (self.y_train == 0).sum() / (self.y_train == 1).sum()
        }
        
        print(f"\nBuilding 3-layer network (input_dim={X_train_scaled.shape[1]})...")
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        print("\nTraining...")
        history = model.fit(
            X_train_scaled, self.y_train,
            validation_data=(X_test_scaled, self.y_test),
            epochs=50,
            batch_size=512,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        training_time = time.time() - start_time
        self.training_times['neural_network'] = training_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save model
        model.save(self.output_dir / 'neural_network.h5')
        
        with open(self.output_dir / 'neural_network_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        self.models['neural_network'] = (model, scaler, X_train_scaled, X_test_scaled)
        
        print("✓ Model saved")
    
    def train_all(self):
        """Train all models"""
        self.load_data()
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_neural_network()
        
        # Save training times
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        
        for model_name, training_time in self.training_times.items():
            print(f"  {model_name}: {training_time:.2f} seconds ({training_time/60:.1f} min)")
        
        total_time = sum(self.training_times.values())
        print(f"\n  Total training time: {total_time:.2f} seconds ({total_time/60:.1f} min)")
        
        # Save summary
        summary = {
            'training_times': self.training_times,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': len(self.feature_cols),
            'fraud_rate': float(self.y_train.mean())
        }
        
        with open(self.results_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ ALL MODELS TRAINED")
        print(f"\nNext step: python benchmarks/evaluate_models.py")


def main():
    """Train all models"""
    trainer = ModelTrainer()
    trainer.train_all()


if __name__ == '__main__':
    main()
