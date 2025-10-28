"""
Model Optimization Framework

Provides hyperparameter optimization, ensemble methods, and feature selection
for fraud detection models.

Classes:
    HyperparameterOptimizer: Optimize model hyperparameters using Grid/Random/Bayesian search
    EnsembleModelBuilder: Create ensemble models (Voting, Stacking, Bagging)
    FeatureSelector: Select optimal features using various methods
    OptimizationResult: Results from hyperparameter optimization
    EnsembleResult: Results from ensemble model creation
    FeatureSelectionResult: Results from feature selection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import warnings


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, np.ndarray]
    best_estimator: Any
    optimization_method: str
    search_space: Dict[str, List[Any]]
    n_iterations: int
    
    def get_param_importance(self) -> List[Tuple[str, float]]:
        """
        Get parameter importance based on score variance
        
        Returns:
            List of (param_name, importance_score) tuples sorted by importance
        """
        param_importance = []
        
        for param_name in self.search_space.keys():
            param_key = f'param_{param_name}'
            if param_key in self.cv_results:
                # Calculate variance of scores for this parameter
                unique_values = np.unique(self.cv_results[param_key])
                if len(unique_values) > 1:
                    scores_by_param = []
                    for value in unique_values:
                        mask = self.cv_results[param_key] == value
                        scores_by_param.append(np.mean(self.cv_results['mean_test_score'][mask]))
                    
                    importance = np.std(scores_by_param)
                    param_importance.append((param_name, importance))
        
        return sorted(param_importance, key=lambda x: x[1], reverse=True)


@dataclass
class EnsembleResult:
    """Results from ensemble model creation"""
    
    ensemble_model: Any
    ensemble_type: str
    base_models: List[Tuple[str, Any]]
    ensemble_score: float
    individual_scores: Dict[str, float]
    improvement: float
    weights: Optional[List[float]] = None
    
    def get_best_base_model(self) -> Tuple[str, Any, float]:
        """Get the best performing base model"""
        best_name = max(self.individual_scores.items(), key=lambda x: x[1])[0]
        best_model = next(model for name, model in self.base_models if name == best_name)
        best_score = self.individual_scores[best_name]
        return (best_name, best_model, best_score)


@dataclass
class FeatureSelectionResult:
    """Results from feature selection"""
    
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_method: str
    n_features_original: int
    n_features_selected: int
    score_before: float
    score_after: float
    improvement: float
    
    def get_removed_features(self, all_features: List[str]) -> List[str]:
        """Get list of features that were removed"""
        return [f for f in all_features if f not in self.selected_features]
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by score"""
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]


class HyperparameterOptimizer:
    """
    Optimize model hyperparameters using various search strategies
    
    Supports:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling from parameter distributions
    - Bayesian Optimization: Sequential model-based optimization (requires optuna)
    """
    
    def __init__(
        self,
        scoring: str = 'f1',
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
    ):
        """
        Initialize optimizer
        
        Args:
            scoring: Scoring metric ('f1', 'roc_auc', 'precision', 'recall', 'accuracy')
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level
            random_state: Random seed for reproducibility
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
    
    def grid_search(
        self,
        model: Any,
        param_grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> OptimizationResult:
        """
        Perform grid search over parameter grid
        
        Args:
            model: Scikit-learn compatible model
            param_grid: Dictionary mapping parameter names to lists of values
            X_train: Training features
            y_train: Training labels
            
        Returns:
            OptimizationResult with best parameters and score
        """
        print(f"Starting Grid Search with {len(param_grid)} parameters...")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        result = OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            cv_results=grid_search.cv_results_,
            best_estimator=grid_search.best_estimator_,
            optimization_method='grid_search',
            search_space=param_grid,
            n_iterations=len(grid_search.cv_results_['mean_test_score'])
        )
        
        print(f"✓ Grid Search complete!")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best params: {result.best_params}")
        
        return result
    
    def random_search(
        self,
        model: Any,
        param_distributions: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 50
    ) -> OptimizationResult:
        """
        Perform random search over parameter distributions
        
        Args:
            model: Scikit-learn compatible model
            param_distributions: Dictionary mapping parameter names to distributions
            X_train: Training features
            y_train: Training labels
            n_iter: Number of parameter settings sampled
            
        Returns:
            OptimizationResult with best parameters and score
        """
        print(f"Starting Random Search with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        result = OptimizationResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            cv_results=random_search.cv_results_,
            best_estimator=random_search.best_estimator_,
            optimization_method='random_search',
            search_space=param_distributions,
            n_iterations=n_iter
        )
        
        print(f"✓ Random Search complete!")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best params: {result.best_params}")
        
        return result
    
    def bayesian_optimization(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50
    ) -> OptimizationResult:
        """
        Perform Bayesian optimization using Optuna (if available)
        
        Args:
            model_class: Model class (not instance) to optimize
            param_space: Dictionary defining parameter search space
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            OptimizationResult with best parameters and score
            
        Note:
            Requires optuna package. Falls back to random search if not available.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            warnings.warn("Optuna not available. Falling back to random search.")
            return self.random_search(model_class(), param_space, X_train, y_train, n_trials)
        
        print(f"Starting Bayesian Optimization with {n_trials} trials...")
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                else:
                    # Simple list of choices
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Create and evaluate model
            model = model_class(**params, random_state=self.random_state)
            score = cross_val_score(
                model, X_train, y_train,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs
            ).mean()
            
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Train final model with best parameters
        best_model = model_class(**study.best_params, random_state=self.random_state)
        best_model.fit(X_train, y_train)
        
        # Convert trial results to CV results format
        cv_results = {
            'mean_test_score': np.array([trial.value for trial in study.trials]),
            'params': [trial.params for trial in study.trials]
        }
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            cv_results=cv_results,
            best_estimator=best_model,
            optimization_method='bayesian_optimization',
            search_space=param_space,
            n_iterations=n_trials
        )
        
        print(f"✓ Bayesian Optimization complete!")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best params: {result.best_params}")
        
        return result


class EnsembleModelBuilder:
    """
    Create and optimize ensemble models for improved fraud detection
    
    Supports:
    - Voting: Combine multiple models with soft/hard voting
    - Stacking: Use meta-learner on base model predictions
    - Bagging: Bootstrap aggregating for variance reduction
    """
    
    def __init__(
        self,
        scoring: str = 'f1',
        cv: int = 5,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize ensemble builder
        
        Args:
            scoring: Scoring metric for evaluation
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def create_voting_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        voting: str = 'soft'
    ) -> EnsembleResult:
        """
        Create voting ensemble from base models
        
        Args:
            base_models: List of (name, model) tuples
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            voting: 'soft' (predict probabilities) or 'hard' (predict classes)
            
        Returns:
            EnsembleResult with ensemble model and performance
        """
        print(f"Creating Voting Ensemble ({voting} voting) with {len(base_models)} base models...")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting=voting,
            n_jobs=self.n_jobs
        )
        
        # Train ensemble
        voting_clf.fit(X_train, y_train)
        
        # Evaluate ensemble
        if self.scoring == 'f1':
            ensemble_score = f1_score(y_test, voting_clf.predict(X_test))
        elif self.scoring == 'roc_auc':
            if voting == 'soft':
                ensemble_score = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])
            else:
                ensemble_score = roc_auc_score(y_test, voting_clf.predict(X_test))
        else:
            ensemble_score = voting_clf.score(X_test, y_test)
        
        # Evaluate individual models
        individual_scores = {}
        for name, model in base_models:
            model.fit(X_train, y_train)
            if self.scoring == 'f1':
                score = f1_score(y_test, model.predict(X_test))
            elif self.scoring == 'roc_auc':
                score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                score = model.score(X_test, y_test)
            individual_scores[name] = score
        
        avg_base_score = np.mean(list(individual_scores.values()))
        improvement = ensemble_score - avg_base_score
        
        result = EnsembleResult(
            ensemble_model=voting_clf,
            ensemble_type=f'voting_{voting}',
            base_models=base_models,
            ensemble_score=ensemble_score,
            individual_scores=individual_scores,
            improvement=improvement
        )
        
        print(f"✓ Voting Ensemble created!")
        print(f"  Ensemble score: {ensemble_score:.4f}")
        print(f"  Average base score: {avg_base_score:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result
    
    def create_stacking_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        meta_learner: Optional[Any] = None
    ) -> EnsembleResult:
        """
        Create stacking ensemble with meta-learner
        
        Args:
            base_models: List of (name, model) tuples for base layer
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            meta_learner: Meta-learner model (default: LogisticRegression)
            
        Returns:
            EnsembleResult with stacked model and performance
        """
        print(f"Creating Stacking Ensemble with {len(base_models)} base models...")
        
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=self.random_state)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        
        # Train ensemble
        stacking_clf.fit(X_train, y_train)
        
        # Evaluate ensemble
        if self.scoring == 'f1':
            ensemble_score = f1_score(y_test, stacking_clf.predict(X_test))
        elif self.scoring == 'roc_auc':
            ensemble_score = roc_auc_score(y_test, stacking_clf.predict_proba(X_test)[:, 1])
        else:
            ensemble_score = stacking_clf.score(X_test, y_test)
        
        # Evaluate individual models
        individual_scores = {}
        for name, model in base_models:
            model.fit(X_train, y_train)
            if self.scoring == 'f1':
                score = f1_score(y_test, model.predict(X_test))
            elif self.scoring == 'roc_auc':
                score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                score = model.score(X_test, y_test)
            individual_scores[name] = score
        
        avg_base_score = np.mean(list(individual_scores.values()))
        improvement = ensemble_score - avg_base_score
        
        result = EnsembleResult(
            ensemble_model=stacking_clf,
            ensemble_type='stacking',
            base_models=base_models,
            ensemble_score=ensemble_score,
            individual_scores=individual_scores,
            improvement=improvement
        )
        
        print(f"✓ Stacking Ensemble created!")
        print(f"  Ensemble score: {ensemble_score:.4f}")
        print(f"  Average base score: {avg_base_score:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result
    
    def create_bagging_ensemble(
        self,
        base_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0
    ) -> EnsembleResult:
        """
        Create bagging ensemble
        
        Args:
            base_model: Base estimator to bag
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            n_estimators: Number of base estimators
            max_samples: Fraction of samples to draw for each base estimator
            max_features: Fraction of features to draw for each base estimator
            
        Returns:
            EnsembleResult with bagged model and performance
        """
        print(f"Creating Bagging Ensemble with {n_estimators} estimators...")
        
        # Create bagging classifier
        bagging_clf = BaggingClassifier(
            estimator=base_model,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Train ensemble
        bagging_clf.fit(X_train, y_train)
        
        # Evaluate ensemble
        if self.scoring == 'f1':
            ensemble_score = f1_score(y_test, bagging_clf.predict(X_test))
        elif self.scoring == 'roc_auc':
            ensemble_score = roc_auc_score(y_test, bagging_clf.predict_proba(X_test)[:, 1])
        else:
            ensemble_score = bagging_clf.score(X_test, y_test)
        
        # Evaluate base model
        base_model.fit(X_train, y_train)
        if self.scoring == 'f1':
            base_score = f1_score(y_test, base_model.predict(X_test))
        elif self.scoring == 'roc_auc':
            base_score = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])
        else:
            base_score = base_model.score(X_test, y_test)
        
        improvement = ensemble_score - base_score
        
        result = EnsembleResult(
            ensemble_model=bagging_clf,
            ensemble_type='bagging',
            base_models=[('base', base_model)],
            ensemble_score=ensemble_score,
            individual_scores={'base': base_score},
            improvement=improvement
        )
        
        print(f"✓ Bagging Ensemble created!")
        print(f"  Ensemble score: {ensemble_score:.4f}")
        print(f"  Base score: {base_score:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result


class FeatureSelector:
    """
    Select optimal features using various selection methods
    
    Supports:
    - RFE: Recursive Feature Elimination
    - LASSO: L1 regularization-based selection
    - Correlation: Remove highly correlated features
    - Variance: Remove low-variance features
    """
    
    def __init__(
        self,
        scoring: str = 'f1',
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Initialize feature selector
        
        Args:
            scoring: Scoring metric for evaluation
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
    
    def rfe_selection(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features_to_select: Optional[int] = None,
        step: int = 1
    ) -> FeatureSelectionResult:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            model: Estimator with feature_importances_ or coef_ attribute
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            n_features_to_select: Number of features to select (default: half)
            step: Number of features to remove at each iteration
            
        Returns:
            FeatureSelectionResult with selected features
        """
        if n_features_to_select is None:
            n_features_to_select = max(1, X_train.shape[1] // 2)
        
        print(f"RFE: Selecting {n_features_to_select} from {X_train.shape[1]} features...")
        
        # Evaluate with all features
        score_before = cross_val_score(
            model, X_train, y_train,
            scoring=self.scoring,
            cv=self.cv
        ).mean()
        
        # Perform RFE
        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features_to_select,
            step=step
        )
        rfe.fit(X_train, y_train)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Get feature rankings (1 = selected, higher = removed earlier)
        feature_scores = {
            feature_names[i]: 1.0 / rfe.ranking_[i]
            for i in range(len(feature_names))
        }
        
        # Evaluate with selected features
        X_train_selected = X_train[:, selected_mask]
        score_after = cross_val_score(
            model, X_train_selected, y_train,
            scoring=self.scoring,
            cv=self.cv
        ).mean()
        
        improvement = score_after - score_before
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='rfe',
            n_features_original=X_train.shape[1],
            n_features_selected=len(selected_features),
            score_before=score_before,
            score_after=score_after,
            improvement=improvement
        )
        
        print(f"✓ RFE complete!")
        print(f"  Selected: {len(selected_features)} features")
        print(f"  Score before: {score_before:.4f}")
        print(f"  Score after: {score_after:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result
    
    def lasso_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        alpha: float = 0.01
    ) -> FeatureSelectionResult:
        """
        Select features using LASSO (L1) regularization
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            alpha: Regularization strength (higher = more regularization)
            
        Returns:
            FeatureSelectionResult with selected features
        """
        from sklearn.linear_model import LogisticRegression
        
        print(f"LASSO: Selecting features with alpha={alpha}...")
        
        # Create LASSO model
        lasso = LogisticRegression(
            penalty='l1',
            C=1/alpha,  # C = 1/alpha
            solver='liblinear',
            random_state=self.random_state
        )
        
        # Evaluate with all features
        score_before = cross_val_score(
            lasso, X_train, y_train,
            scoring=self.scoring,
            cv=self.cv
        ).mean()
        
        # Fit LASSO
        lasso.fit(X_train, y_train)
        
        # Get feature coefficients
        if hasattr(lasso, 'coef_'):
            coefficients = np.abs(lasso.coef_[0])
        else:
            coefficients = np.abs(lasso.feature_importances_)
        
        # Select features with non-zero coefficients
        selector = SelectFromModel(lasso, prefit=True)
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Get feature scores (absolute coefficients)
        feature_scores = {
            feature_names[i]: float(coefficients[i])
            for i in range(len(feature_names))
        }
        
        # Evaluate with selected features
        if selected_mask.sum() > 0:
            X_train_selected = X_train[:, selected_mask]
            score_after = cross_val_score(
                lasso, X_train_selected, y_train,
                scoring=self.scoring,
                cv=self.cv
            ).mean()
        else:
            score_after = score_before
        
        improvement = score_after - score_before
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='lasso',
            n_features_original=X_train.shape[1],
            n_features_selected=len(selected_features),
            score_before=score_before,
            score_after=score_after,
            improvement=improvement
        )
        
        print(f"✓ LASSO complete!")
        print(f"  Selected: {len(selected_features)} features")
        print(f"  Score before: {score_before:.4f}")
        print(f"  Score after: {score_after:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result
    
    def correlation_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.9
    ) -> FeatureSelectionResult:
        """
        Remove highly correlated features
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            threshold: Correlation threshold (remove if > threshold)
            
        Returns:
            FeatureSelectionResult with selected features
        """
        print(f"Correlation filtering: Removing features with correlation > {threshold}...")
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_train.T)
        
        # Find highly correlated pairs
        selected_mask = np.ones(len(feature_names), dtype=bool)
        removed_features = []
        
        for i in range(len(feature_names)):
            if not selected_mask[i]:
                continue
            for j in range(i + 1, len(feature_names)):
                if not selected_mask[j]:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Remove feature with lower variance
                    if np.var(X_train[:, i]) < np.var(X_train[:, j]):
                        selected_mask[i] = False
                        removed_features.append(feature_names[i])
                        break
                    else:
                        selected_mask[j] = False
                        removed_features.append(feature_names[j])
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Feature scores (inverse of max correlation)
        feature_scores = {}
        for i in range(len(feature_names)):
            max_corr = np.max([abs(corr_matrix[i, j]) for j in range(len(feature_names)) if j != i])
            feature_scores[feature_names[i]] = 1.0 - max_corr
        
        # Evaluate before and after (use dummy model)
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
        
        score_before = cross_val_score(
            dummy_model, X_train, y_train,
            scoring=self.scoring,
            cv=self.cv
        ).mean()
        
        X_train_selected = X_train[:, selected_mask]
        score_after = cross_val_score(
            dummy_model, X_train_selected, y_train,
            scoring=self.scoring,
            cv=self.cv
        ).mean()
        
        improvement = score_after - score_before
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='correlation',
            n_features_original=X_train.shape[1],
            n_features_selected=len(selected_features),
            score_before=score_before,
            score_after=score_after,
            improvement=improvement
        )
        
        print(f"✓ Correlation filtering complete!")
        print(f"  Removed: {len(removed_features)} correlated features")
        print(f"  Selected: {len(selected_features)} features")
        print(f"  Score before: {score_before:.4f}")
        print(f"  Score after: {score_after:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        return result
