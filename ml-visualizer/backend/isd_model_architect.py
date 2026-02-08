"""
Model Architect Module - ISD
Recommends optimal ML architecture based on problem and data characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class ModelArchitectModule:
    """
    Expert system that recommends ML algorithms, loss functions,
    metrics, and validation strategies
    """
    
    def __init__(self, problem_analysis: Dict, data_analysis: Dict):
        self.problem = problem_analysis
        self.data = data_analysis
        
    def recommend(self) -> Dict[str, Any]:
        """
        Generate comprehensive model recommendations
        """
        results = {
            'algorithm_recommendations': [],
            'loss_function': None,
            'evaluation_metrics': [],
            'validation_strategy': None,
            'data_split': None,
            'preprocessing_pipeline': [],
            'hyperparameter_priorities': [],
            'ensemble_strategy': None
        }
        
        # Algorithm recommendations
        algorithms = self._recommend_algorithms()
        results['algorithm_recommendations'] = algorithms
        
        # Loss function
        loss_func = self._recommend_loss_function()
        results['loss_function'] = loss_func
        
        # Evaluation metrics
        metrics = self._recommend_metrics()
        results['evaluation_metrics'] = metrics
        
        # Validation strategy
        validation = self._recommend_validation_strategy()
        results['validation_strategy'] = validation
        
        # Data split strategy
        split = self._recommend_data_split()
        results['data_split'] = split
        
        # Preprocessing pipeline
        preprocessing = self._recommend_preprocessing()
        results['preprocessing_pipeline'] = preprocessing
        
        # Hyperparameter priorities
        hyperparams = self._recommend_hyperparameters(algorithms)
        results['hyperparameter_priorities'] = hyperparams
        
        # Ensemble strategy
        ensemble = self._recommend_ensemble()
        results['ensemble_strategy'] = ensemble
        
        return results
    
    def _recommend_algorithms(self) -> List[Dict[str, Any]]:
        """Recommend algorithms with detailed reasoning"""
        algorithms = []
        problem_type = self.problem['problem_type']
        complexity = self.problem['task_complexity']
        n_samples = complexity['n_samples']
        n_features = complexity['n_features']
        
        if problem_type == 'classification':
            # Logistic Regression
            lr_score = 70
            if n_samples < 1000: lr_score += 15
            if n_features < 20: lr_score += 10
            if self.problem['risk_profile']['requires_interpretability']: lr_score += 15
            
            algorithms.append({
                'name': 'Logistic Regression',
                'family': 'Linear Models',
                'score': min(100, lr_score),
                'reasoning': [
                    'Interpretable coefficients',
                    'Fast training and prediction',
                    'Works well with small datasets',
                    'Provides probability estimates'
                ],
                'when_to_use': 'Linear decision boundaries, need interpretability',
                'limitations': ['Assumes linearity', 'May underfit complex patterns'],
                'sklearn_class': 'LogisticRegression',
                'key_hyperparameters': ['C', 'penalty', 'solver']
            })

            
            # Random Forest
            rf_score = 85
            if n_samples > 1000: rf_score += 10
            if self.data['detailed_analysis']['missing_values']['overall_missing_rate'] > 10: rf_score += 5
            
            algorithms.append({
                'name': 'Random Forest',
                'family': 'Ensemble - Bagging',
                'score': min(100, rf_score),
                'reasoning': ['Non-linear', 'Robust to outliers', 'Handles missing values', 'Feature importance'],
                'when_to_use': 'Complex patterns, mixed feature types',
                'limitations': ['Less interpretable', 'Slower than linear models'],
                'sklearn_class': 'RandomForestClassifier',
                'key_hyperparameters': ['n_estimators', 'max_depth', 'min_samples_split']
            })
            
            # XGBoost
            xgb_score = 90
            if n_samples > 5000: xgb_score += 10
            if n_samples < 500: xgb_score -= 20
            
            algorithms.append({
                'name': 'XGBoost',
                'family': 'Ensemble - Boosting',
                'score': min(100, xgb_score),
                'reasoning': ['State-of-the-art performance', 'Built-in regularization', 'Handles missing values'],
                'when_to_use': 'Maximum accuracy needed, large datasets',
                'limitations': ['Many hyperparameters', 'Can overfit small data'],
                'sklearn_class': 'XGBClassifier',
                'key_hyperparameters': ['learning_rate', 'max_depth', 'n_estimators']
            })
        
        else:  # Regression
            algorithms.append({
                'name': 'Linear Regression',
                'family': 'Linear Models',
                'score': 70,
                'reasoning': ['Simple baseline', 'Interpretable', 'Fast'],
                'when_to_use': 'Linear relationships',
                'limitations': ['Assumes linearity'],
                'sklearn_class': 'LinearRegression',
                'key_hyperparameters': []
            })
            
            algorithms.append({
                'name': 'Random Forest Regressor',
                'family': 'Ensemble',
                'score': 85,
                'reasoning': ['Non-linear', 'Robust', 'Feature importance'],
                'when_to_use': 'Complex patterns',
                'limitations': ['Less interpretable'],
                'sklearn_class': 'RandomForestRegressor',
                'key_hyperparameters': ['n_estimators', 'max_depth']
            })
            
            algorithms.append({
                'name': 'XGBoost Regressor',
                'family': 'Ensemble - Boosting',
                'score': 90,
                'reasoning': ['High accuracy', 'Handles missing values', 'Fast training'],
                'when_to_use': 'Maximum accuracy needed',
                'limitations': ['Many hyperparameters'],
                'sklearn_class': 'XGBRegressor',
                'key_hyperparameters': ['learning_rate', 'max_depth']
            })
        
        algorithms.sort(key=lambda x: x['score'], reverse=True)
        return algorithms
    
    def _recommend_loss_function(self) -> Dict[str, Any]:
        """Recommend appropriate loss function"""
        problem_type = self.problem['problem_type']
        
        if problem_type == 'classification':
            if self.problem.get('n_classes', 2) == 2:
                return {
                    'name': 'Binary Cross-Entropy',
                    'sklearn_equivalent': 'log_loss',
                    'reason': 'Standard for binary classification',
                    'formula': '-[y*log(p) + (1-y)*log(1-p)]'
                }
            else:
                return {
                    'name': 'Categorical Cross-Entropy',
                    'sklearn_equivalent': 'log_loss',
                    'reason': 'Standard for multiclass classification',
                    'formula': '-sum(y_i * log(p_i))'
                }
        else:
            return {
                'name': 'Mean Squared Error (MSE)',
                'sklearn_equivalent': 'squared_error',
                'reason': 'Standard for regression, penalizes large errors',
                'formula': 'mean((y_true - y_pred)^2)'
            }
    
    def _recommend_metrics(self) -> List[Dict[str, Any]]:
        """Recommend evaluation metrics"""
        metrics = []
        problem_type = self.problem['problem_type']
        
        if problem_type == 'classification':
            metrics.extend([
                {'name': 'Accuracy', 'primary': True, 'reason': 'Overall correctness'},
                {'name': 'Precision', 'primary': True, 'reason': 'Minimize false positives'},
                {'name': 'Recall', 'primary': True, 'reason': 'Minimize false negatives'},
                {'name': 'F1-Score', 'primary': True, 'reason': 'Balance precision and recall'},
                {'name': 'ROC-AUC', 'primary': False, 'reason': 'Threshold-independent metric'}
            ])
        else:
            metrics.extend([
                {'name': 'R² Score', 'primary': True, 'reason': 'Variance explained'},
                {'name': 'RMSE', 'primary': True, 'reason': 'Average prediction error'},
                {'name': 'MAE', 'primary': True, 'reason': 'Robust to outliers'},
                {'name': 'MAPE', 'primary': False, 'reason': 'Percentage error'}
            ])
        
        return metrics
    
    def _recommend_validation_strategy(self) -> Dict[str, Any]:
        """Recommend validation strategy"""
        n_samples = self.problem['task_complexity']['n_samples']
        is_time_series = self.problem.get('is_time_series', False)
        
        if is_time_series:
            return {
                'strategy': 'Time Series Split',
                'n_splits': 5,
                'reason': 'Preserves temporal order, prevents data leakage',
                'sklearn_class': 'TimeSeriesSplit'
            }
        elif n_samples < 100:
            return {
                'strategy': 'Leave-One-Out CV',
                'n_splits': n_samples,
                'reason': 'Maximizes training data for small datasets',
                'sklearn_class': 'LeaveOneOut'
            }
        elif n_samples < 1000:
            return {
                'strategy': 'Stratified K-Fold',
                'n_splits': 5,
                'reason': 'Maintains class distribution in each fold',
                'sklearn_class': 'StratifiedKFold'
            }
        else:
            return {
                'strategy': 'Stratified K-Fold',
                'n_splits': 10,
                'reason': 'Standard for large datasets',
                'sklearn_class': 'StratifiedKFold'
            }
    
    def _recommend_data_split(self) -> Dict[str, Any]:
        """Recommend train/test split strategy"""
        n_samples = self.problem['task_complexity']['n_samples']
        
        if n_samples < 100:
            test_size = 0.1
        elif n_samples < 1000:
            test_size = 0.2
        else:
            test_size = 0.2
        
        return {
            'train_size': 1 - test_size,
            'test_size': test_size,
            'validation_size': 0.1,
            'stratify': self.problem['problem_type'] == 'classification',
            'reason': f'Standard split for {n_samples} samples'
        }
    
    def _recommend_preprocessing(self) -> List[Dict[str, Any]]:
        """Recommend preprocessing steps"""
        steps = []
        
        # Missing values
        if self.data['detailed_analysis']['missing_values']['overall_missing_rate'] > 0:
            steps.append({
                'step': 'Imputation',
                'method': 'SimpleImputer(strategy="median")',
                'reason': 'Handle missing values',
                'priority': 'critical'
            })
        
        # Scaling
        steps.append({
            'step': 'Feature Scaling',
            'method': 'StandardScaler()',
            'reason': 'Normalize feature ranges',
            'priority': 'high'
        })
        
        # Encoding
        if self.problem['data_type']['categorical_ratio'] > 0:
            steps.append({
                'step': 'Categorical Encoding',
                'method': 'OneHotEncoder() or LabelEncoder()',
                'reason': 'Convert categorical to numeric',
                'priority': 'critical'
            })
        
        return steps
    
    def _recommend_hyperparameters(self, algorithms: List[Dict]) -> List[Dict[str, Any]]:
        """Recommend hyperparameter tuning priorities"""
        if not algorithms:
            return []
        
        best_algo = algorithms[0]
        return [
            {
                'parameter': param,
                'priority': 'high' if i < 2 else 'medium',
                'tuning_method': 'RandomizedSearchCV or Optuna'
            }
            for i, param in enumerate(best_algo.get('key_hyperparameters', []))
        ]
    
    def _recommend_ensemble(self) -> Dict[str, Any]:
        """Recommend ensemble strategy"""
        n_samples = self.problem['task_complexity']['n_samples']
        
        if n_samples > 5000:
            return {
                'recommended': True,
                'strategy': 'Stacking',
                'reason': 'Large dataset supports ensemble complexity',
                'base_models': ['RandomForest', 'XGBoost', 'LogisticRegression'],
                'meta_model': 'LogisticRegression'
            }
        else:
            return {
                'recommended': False,
                'reason': 'Dataset too small for effective ensembling'
            }
