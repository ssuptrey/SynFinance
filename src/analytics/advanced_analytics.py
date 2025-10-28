"""
Advanced Analytics Module for Fraud Detection

Provides comprehensive analytics capabilities including:
- Correlation analysis (Pearson, Spearman, Kendall)
- Feature importance scoring (permutation, SHAP, tree-based)
- Model performance metrics (precision, recall, F1, ROC-AUC)
- Statistical significance testing (chi-square, t-tests, ANOVA)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import warnings


@dataclass
class CorrelationResult:
    """Result from correlation analysis"""
    method: str  # 'pearson', 'spearman', 'kendall'
    correlation_matrix: np.ndarray
    feature_names: List[str]
    significant_correlations: List[Tuple[str, str, float]]  # (feature1, feature2, correlation)
    threshold: float = 0.7
    
    def get_highly_correlated_pairs(self) -> List[Tuple[str, str, float]]:
        """Get pairs of features with correlation above threshold"""
        pairs = []
        n = len(self.feature_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(self.correlation_matrix[i, j])
                if corr >= self.threshold:
                    pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        self.correlation_matrix[i, j]
                    ))
        
        # Sort by absolute correlation (descending)
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


@dataclass
class FeatureImportanceResult:
    """Result from feature importance analysis"""
    method: str  # 'permutation', 'shap', 'tree_based', 'mutual_info'
    feature_names: List[str]
    importances: np.ndarray
    importances_std: Optional[np.ndarray] = None
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        indices = np.argsort(self.importances)[::-1][:n]
        return [(self.feature_names[i], self.importances[i]) for i in indices]
    
    def get_bottom_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get bottom N least important features"""
        indices = np.argsort(self.importances)[:n]
        return [(self.feature_names[i], self.importances[i]) for i in indices]


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    confusion_matrix: np.ndarray
    classification_report: str
    
    # ROC curve data
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Precision-recall curve data
    precision_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    recall_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    pr_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1': float(self.f1),
            'roc_auc': float(self.roc_auc),
            'average_precision': float(self.average_precision),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.classification_report,
        }


@dataclass
class StatisticalTestResult:
    """Result from statistical significance test"""
    test_name: str  # 'chi_square', 't_test', 'anova'
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    is_significant: bool = field(init=False)
    alpha: float = 0.05
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.alpha


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    dataset_info: Dict[str, Any]
    correlation_results: Optional[CorrelationResult] = None
    feature_importance_results: Optional[List[FeatureImportanceResult]] = None
    model_metrics: Optional[ModelMetrics] = None
    statistical_tests: Optional[List[StatisticalTestResult]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        report = {
            'dataset_info': self.dataset_info,
        }
        
        if self.correlation_results:
            report['correlation'] = {
                'method': self.correlation_results.method,
                'num_significant_pairs': len(self.correlation_results.get_highly_correlated_pairs()),
                'threshold': self.correlation_results.threshold,
            }
        
        if self.feature_importance_results:
            report['feature_importance'] = [
                {
                    'method': fi.method,
                    'top_5_features': fi.get_top_features(5),
                }
                for fi in self.feature_importance_results
            ]
        
        if self.model_metrics:
            report['model_performance'] = self.model_metrics.to_dict()
        
        if self.statistical_tests:
            report['statistical_tests'] = [
                {
                    'test': test.test_name,
                    'statistic': float(test.statistic),
                    'p_value': float(test.p_value),
                    'significant': test.is_significant,
                }
                for test in self.statistical_tests
            ]
        
        return report


class CorrelationAnalyzer:
    """Analyzes correlations between features"""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize correlation analyzer
        
        Args:
            threshold: Correlation threshold for flagging high correlations
        """
        self.threshold = threshold
    
    def analyze(
        self,
        X: np.ndarray,
        feature_names: List[str],
        method: str = 'pearson'
    ) -> CorrelationResult:
        """
        Compute correlation matrix
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            CorrelationResult with correlation matrix and significant pairs
        """
        if method == 'pearson':
            corr_matrix = np.corrcoef(X, rowvar=False)
        elif method == 'spearman':
            corr_matrix = stats.spearmanr(X)[0]
        elif method == 'kendall':
            # Kendall is slow for large datasets, compute pairwise
            n_features = X.shape[1]
            corr_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tau, _ = stats.kendalltau(X[:, i], X[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = CorrelationResult(
            method=method,
            correlation_matrix=corr_matrix,
            feature_names=feature_names,
            significant_correlations=[],
            threshold=self.threshold,
        )
        
        result.significant_correlations = result.get_highly_correlated_pairs()
        
        return result
    
    def compare_feature_groups(
        self,
        X: np.ndarray,
        feature_names: List[str],
        group1_indices: List[int],
        group2_indices: List[int],
        method: str = 'pearson'
    ) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
        """
        Compute cross-correlation between two feature groups
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            group1_indices: Indices of features in group 1
            group2_indices: Indices of features in group 2
            method: Correlation method
            
        Returns:
            Tuple of (cross_correlation_matrix, top_correlations)
        """
        X1 = X[:, group1_indices]
        X2 = X[:, group2_indices]
        
        # Compute cross-correlation
        if method == 'pearson':
            cross_corr = np.corrcoef(X1.T, X2.T)[:len(group1_indices), len(group1_indices):]
        else:
            # For Spearman/Kendall, compute pairwise
            cross_corr = np.zeros((len(group1_indices), len(group2_indices)))
            for i, idx1 in enumerate(group1_indices):
                for j, idx2 in enumerate(group2_indices):
                    if method == 'spearman':
                        corr, _ = stats.spearmanr(X[:, idx1], X[:, idx2])
                    else:  # kendall
                        corr, _ = stats.kendalltau(X[:, idx1], X[:, idx2])
                    cross_corr[i, j] = corr
        
        # Find top correlations
        top_corr = []
        for i, idx1 in enumerate(group1_indices):
            for j, idx2 in enumerate(group2_indices):
                top_corr.append((
                    feature_names[idx1],
                    feature_names[idx2],
                    cross_corr[i, j]
                ))
        
        top_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return cross_corr, top_corr[:10]  # Return top 10


class FeatureImportanceAnalyzer:
    """Analyzes feature importance using multiple methods"""
    
    def __init__(self, n_repeats: int = 10, random_state: int = 42):
        """
        Initialize feature importance analyzer
        
        Args:
            n_repeats: Number of repeats for permutation importance
            random_state: Random seed for reproducibility
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def permutation_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        scoring: str = 'f1'
    ) -> FeatureImportanceResult:
        """
        Compute permutation importance
        
        Args:
            model: Trained sklearn model
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            scoring: Scoring metric
            
        Returns:
            FeatureImportanceResult
        """
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=scoring
        )
        
        return FeatureImportanceResult(
            method='permutation',
            feature_names=feature_names,
            importances=perm_importance.importances_mean,
            importances_std=perm_importance.importances_std,
        )
    
    def tree_based_importance(
        self,
        model,
        feature_names: List[str]
    ) -> FeatureImportanceResult:
        """
        Get feature importance from tree-based model
        
        Args:
            model: Trained tree-based model (RandomForest, XGBoost, etc.)
            feature_names: List of feature names
            
        Returns:
            FeatureImportanceResult
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        return FeatureImportanceResult(
            method='tree_based',
            feature_names=feature_names,
            importances=model.feature_importances_,
        )
    
    def mutual_information_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> FeatureImportanceResult:
        """
        Compute mutual information between features and target
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            FeatureImportanceResult
        """
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(
            X, y,
            random_state=self.random_state
        )
        
        return FeatureImportanceResult(
            method='mutual_info',
            feature_names=feature_names,
            importances=mi_scores,
        )
    
    def analyze_all(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportanceResult]:
        """
        Compute feature importance using all available methods
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            List of FeatureImportanceResult
        """
        results = []
        
        # Permutation importance (on test set)
        try:
            perm_result = self.permutation_importance(
                model, X_test, y_test, feature_names
            )
            results.append(perm_result)
        except Exception as e:
            warnings.warn(f"Permutation importance failed: {e}")
        
        # Tree-based importance
        try:
            tree_result = self.tree_based_importance(model, feature_names)
            results.append(tree_result)
        except Exception as e:
            warnings.warn(f"Tree-based importance failed: {e}")
        
        # Mutual information (on training set)
        try:
            mi_result = self.mutual_information_importance(
                X_train, y_train, feature_names
            )
            results.append(mi_result)
        except Exception as e:
            warnings.warn(f"Mutual information failed: {e}")
        
        return results


class ModelPerformanceAnalyzer:
    """Analyzes model performance with comprehensive metrics"""
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Compute comprehensive model performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC/PR curves)
            
        Returns:
            ModelMetrics with all performance metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        
        # ROC AUC and curves (requires probabilities)
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            
            # Precision-recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true, y_pred_proba
            )
        else:
            roc_auc = 0.0
            avg_precision = 0.0
            fpr = np.array([])
            tpr = np.array([])
            roc_thresholds = np.array([])
            precision_curve = np.array([])
            recall_curve = np.array([])
            pr_thresholds = np.array([])
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            confusion_matrix=cm,
            classification_report=report,
            fpr=fpr,
            tpr=tpr,
            roc_thresholds=roc_thresholds,
            precision_curve=precision_curve,
            recall_curve=recall_curve,
            pr_thresholds=pr_thresholds,
        )
    
    def compare_models(
        self,
        models_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        y_true: np.ndarray
    ) -> Dict[str, ModelMetrics]:
        """
        Compare multiple models
        
        Args:
            models_dict: Dict of {model_name: (y_pred, y_pred_proba)}
            y_true: True labels
            
        Returns:
            Dict of {model_name: ModelMetrics}
        """
        results = {}
        for name, (y_pred, y_pred_proba) in models_dict.items():
            if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
                # Handle multi-class probabilities
                y_pred_proba = y_pred_proba[:, 1]
            results[name] = self.analyze(y_true, y_pred, y_pred_proba)
        return results


class StatisticalTestAnalyzer:
    """Performs statistical significance tests"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical test analyzer
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def chi_square_test(
        self,
        contingency_table: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform chi-square test of independence
        
        Args:
            contingency_table: 2D contingency table
            
        Returns:
            StatisticalTestResult
        """
        chi2, p_value, dof, _ = chi2_contingency(contingency_table)
        
        return StatisticalTestResult(
            test_name='chi_square',
            statistic=chi2,
            p_value=p_value,
            degrees_of_freedom=dof,
            alpha=self.alpha,
        )
    
    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True
    ) -> StatisticalTestResult:
        """
        Perform independent samples t-test
        
        Args:
            group1: First group samples
            group2: Second group samples
            equal_var: Whether to assume equal variances
            
        Returns:
            StatisticalTestResult
        """
        t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)
        
        dof = len(group1) + len(group2) - 2
        
        return StatisticalTestResult(
            test_name='t_test',
            statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=dof,
            alpha=self.alpha,
        )
    
    def anova_test(
        self,
        *groups: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform one-way ANOVA
        
        Args:
            *groups: Variable number of group samples
            
        Returns:
            StatisticalTestResult
        """
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate degrees of freedom
        k = len(groups)  # number of groups
        n = sum(len(g) for g in groups)  # total samples
        dof = n - k
        
        return StatisticalTestResult(
            test_name='anova',
            statistic=f_stat,
            p_value=p_value,
            degrees_of_freedom=dof,
            alpha=self.alpha,
        )
    
    def test_fraud_vs_normal(
        self,
        feature_values: np.ndarray,
        is_fraud: np.ndarray
    ) -> StatisticalTestResult:
        """
        Test if feature differs significantly between fraud and normal transactions
        
        Args:
            feature_values: Feature values
            is_fraud: Boolean array indicating fraud
            
        Returns:
            StatisticalTestResult from t-test
        """
        fraud_values = feature_values[is_fraud]
        normal_values = feature_values[~is_fraud]
        
        return self.t_test(fraud_values, normal_values)
