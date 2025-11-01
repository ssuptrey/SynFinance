"""
Tests for Machine Learning Models

Tests Random Forest, XGBoost, and Voting Ensemble for fraud detection.
Week 8 Day 3: Ensemble ML Models
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.ml.models.random_forest import RandomForestModel
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.ensemble.voting import VotingEnsemble


@pytest.fixture
def fraud_dataset():
    """Generate synthetic fraud detection dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced: 90% legitimate, 10% fraud
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": [f"feature_{i}" for i in range(20)]
    }


class TestRandomForestModel:
    """Test Random Forest fraud detection model"""
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters"""
        model = RandomForestModel()
        
        assert model.model_name == "RandomForest"
        assert model.model_version == "1.0.0"
        assert not model.is_trained
        assert model.hyperparameters["n_estimators"] == 200
        assert model.hyperparameters["max_depth"] == 20
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters"""
        model = RandomForestModel(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10
        )
        
        assert model.hyperparameters["n_estimators"] == 100
        assert model.hyperparameters["max_depth"] == 15
        assert model.hyperparameters["min_samples_split"] == 10
    
    def test_model_training(self, fraud_dataset):
        """Test model can be trained on fraud data"""
        model = RandomForestModel(n_estimators=50)  # Fewer trees for speed
        
        metrics = model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        assert model.is_trained
        assert model.training_date is not None
        assert "train_accuracy" in metrics
        assert "train_precision" in metrics
        assert "train_recall" in metrics
        assert "train_f1" in metrics
        assert metrics["train_accuracy"] > 0.5
    
    def test_model_training_with_validation(self, fraud_dataset):
        """Test model training with validation data"""
        model = RandomForestModel(n_estimators=50)
        
        metrics = model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            validation_data=(fraud_dataset["X_test"], fraud_dataset["y_test"]),
            feature_names=fraud_dataset["feature_names"]
        )
        
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert metrics["val_accuracy"] > 0.5
    
    def test_model_prediction(self, fraud_dataset):
        """Test model prediction"""
        model = RandomForestModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"]
        )
        
        predictions = model.predict(fraud_dataset["X_test"])
        
        assert len(predictions) == len(fraud_dataset["X_test"])
        assert set(predictions).issubset({0, 1})
    
    def test_model_predict_proba(self, fraud_dataset):
        """Test probability prediction"""
        model = RandomForestModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"]
        )
        
        probas = model.predict_proba(fraud_dataset["X_test"])
        
        assert len(probas) == len(fraud_dataset["X_test"])
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_model_evaluation(self, fraud_dataset):
        """Test model evaluation metrics"""
        model = RandomForestModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"]
        )
        
        metrics = model.evaluate(
            fraud_dataset["X_test"],
            fraud_dataset["y_test"]
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "true_positives" in metrics
        assert "false_positives" in metrics
        assert metrics["accuracy"] > 0.5
    
    def test_feature_importance(self, fraud_dataset):
        """Test feature importance extraction"""
        model = RandomForestModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == 20
        assert all(v >= 0 for v in importance.values())
        assert np.isclose(sum(importance.values()), 1.0)
    
    def test_model_persistence(self, fraud_dataset, tmp_path):
        """Test model save and load"""
        model = RandomForestModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        # Save model
        save_path = tmp_path / "rf_model.pkl"
        model.save(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = RandomForestModel.load(str(save_path))
        assert loaded_model.is_trained
        assert loaded_model.model_name == model.model_name
        
        # Test predictions match
        original_pred = model.predict(fraud_dataset["X_test"])
        loaded_pred = loaded_model.predict(fraud_dataset["X_test"])
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestXGBoostModel:
    """Test XGBoost fraud detection model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = XGBoostModel()
        
        assert model.model_name == "XGBoost"
        assert model.model_version == "1.0.0"
        assert not model.is_trained
        assert model.hyperparameters["n_estimators"] == 200
        assert model.hyperparameters["max_depth"] == 8
        assert model.hyperparameters["learning_rate"] == 0.1
    
    def test_model_training(self, fraud_dataset):
        """Test XGBoost model training"""
        model = XGBoostModel(n_estimators=50)
        
        metrics = model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        assert model.is_trained
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.5
    
    def test_model_training_with_early_stopping(self, fraud_dataset):
        """Test training with early stopping"""
        model = XGBoostModel(n_estimators=100)
        
        metrics = model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            validation_data=(fraud_dataset["X_test"], fraud_dataset["y_test"]),
            early_stopping_rounds=10
        )
        
        assert "val_accuracy" in metrics
        # Early stopping may or may not trigger depending on data
        # Just verify training completed successfully
    
    def test_model_prediction(self, fraud_dataset):
        """Test XGBoost prediction"""
        model = XGBoostModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"]
        )
        
        predictions = model.predict(fraud_dataset["X_test"])
        
        assert len(predictions) == len(fraud_dataset["X_test"])
        assert set(predictions).issubset({0, 1})
    
    def test_feature_importance(self, fraud_dataset):
        """Test XGBoost feature importance"""
        model = XGBoostModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == 20
        assert all(v >= 0 for v in importance.values())
    
    def test_model_persistence(self, fraud_dataset, tmp_path):
        """Test XGBoost save and load"""
        model = XGBoostModel(n_estimators=50)
        model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"]
        )
        
        save_path = tmp_path / "xgb_model.pkl"
        model.save(str(save_path))
        
        loaded_model = XGBoostModel.load(str(save_path))
        assert loaded_model.is_trained
        
        # Test predictions match
        original_pred = model.predict(fraud_dataset["X_test"])
        loaded_pred = loaded_model.predict(fraud_dataset["X_test"])
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestVotingEnsemble:
    """Test Voting Ensemble classifier"""
    
    @pytest.fixture
    def trained_models(self, fraud_dataset):
        """Create and train base models"""
        rf = RandomForestModel(n_estimators=30)
        xgb_model = XGBoostModel(n_estimators=30)
        
        rf.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        xgb_model.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            feature_names=fraud_dataset["feature_names"]
        )
        
        return [rf, xgb_model]
    
    def test_ensemble_initialization(self, trained_models):
        """Test ensemble initialization"""
        ensemble = VotingEnsemble(
            models=trained_models,
            voting="soft"
        )
        
        assert ensemble.model_name == "VotingEnsemble"
        assert ensemble.voting == "soft"
        assert len(ensemble.models) == 2
        assert ensemble.is_trained
    
    def test_ensemble_initialization_with_weights(self, trained_models):
        """Test ensemble with custom weights"""
        ensemble = VotingEnsemble(
            models=trained_models,
            voting="soft",
            weights=[0.6, 0.4]
        )
        
        np.testing.assert_array_almost_equal(
            ensemble.weights,
            np.array([0.6, 0.4])
        )
    
    def test_ensemble_requires_multiple_models(self):
        """Test ensemble requires at least 2 models"""
        rf = RandomForestModel()
        
        with pytest.raises(ValueError, match="at least 2 models"):
            VotingEnsemble(models=[rf])
    
    def test_ensemble_soft_voting_prediction(self, trained_models, fraud_dataset):
        """Test soft voting prediction"""
        ensemble = VotingEnsemble(
            models=trained_models,
            voting="soft"
        )
        
        predictions = ensemble.predict(fraud_dataset["X_test"])
        
        assert len(predictions) == len(fraud_dataset["X_test"])
        assert set(predictions).issubset({0, 1})
    
    def test_ensemble_hard_voting_prediction(self, trained_models, fraud_dataset):
        """Test hard voting prediction"""
        ensemble = VotingEnsemble(
            models=trained_models,
            voting="hard"
        )
        
        predictions = ensemble.predict(fraud_dataset["X_test"])
        
        assert len(predictions) == len(fraud_dataset["X_test"])
        assert set(predictions).issubset({0, 1})
    
    def test_ensemble_predict_proba(self, trained_models, fraud_dataset):
        """Test ensemble probability prediction"""
        ensemble = VotingEnsemble(
            models=trained_models,
            voting="soft"
        )
        
        probas = ensemble.predict_proba(fraud_dataset["X_test"])
        
        assert len(probas) == len(fraud_dataset["X_test"])
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_ensemble_evaluation(self, trained_models, fraud_dataset):
        """Test ensemble evaluation"""
        ensemble = VotingEnsemble(models=trained_models)
        
        metrics = ensemble.evaluate(
            fraud_dataset["X_test"],
            fraud_dataset["y_test"]
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0.5
    
    def test_ensemble_feature_importance(self, trained_models):
        """Test ensemble feature importance averaging"""
        ensemble = VotingEnsemble(models=trained_models)
        
        importance = ensemble.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == 20
        assert all(v >= 0 for v in importance.values())
    
    def test_ensemble_individual_predictions(self, trained_models, fraud_dataset):
        """Test getting individual model predictions"""
        ensemble = VotingEnsemble(models=trained_models)
        
        individual = ensemble.get_model_predictions(fraud_dataset["X_test"])
        
        assert "RandomForest_0" in individual
        assert "XGBoost_1" in individual
        assert "labels" in individual["RandomForest_0"]
        assert "probas" in individual["RandomForest_0"]
    
    def test_ensemble_training(self, fraud_dataset):
        """Test ensemble can train base models"""
        rf = RandomForestModel(n_estimators=30)
        xgb_model = XGBoostModel(n_estimators=30)
        
        ensemble = VotingEnsemble(models=[rf, xgb_model])
        
        metrics = ensemble.train(
            fraud_dataset["X_train"],
            fraud_dataset["y_train"],
            validation_data=(fraud_dataset["X_test"], fraud_dataset["y_test"]),
            feature_names=fraud_dataset["feature_names"]
        )
        
        assert "RandomForest_0_train_accuracy" in metrics
        assert "XGBoost_1_train_accuracy" in metrics
        assert "ensemble_train_accuracy" in metrics
        assert "ensemble_val_accuracy" in metrics
