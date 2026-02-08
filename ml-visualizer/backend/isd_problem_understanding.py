"""
Problem Understanding Module - ISD
Automatically classifies ML problem type and characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ProblemUnderstandingModule:
    """
    Intelligent problem classification system that understands
    the ML task from data characteristics
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target_col = df.columns[-1]
        self.feature_cols = df.columns[:-1].tolist()
        self.n_rows, self.n_cols = df.shape
        
    def analyze(self) -> Dict[str, Any]:
        """
        Comprehensive problem analysis
        """
        results = {
            'problem_type': None,
            'problem_subtype': None,
            'task_complexity': None,
            'data_type': None,
            'risk_profile': None,
            'characteristics': {},
            'recommendations': []
        }
        
        # Classify problem type
        problem_classification = self._classify_problem_type()
        results.update(problem_classification)
        
        # Analyze data type
        data_type_analysis = self._analyze_data_type()
        results['data_type'] = data_type_analysis
        
        # Assess task complexity
        complexity_analysis = self._assess_complexity()
        results['task_complexity'] = complexity_analysis
        
        # Determine risk profile
        risk_analysis = self._assess_risk_profile()
        results['risk_profile'] = risk_analysis
        
        # Generate characteristics
        characteristics = self._extract_characteristics()
        results['characteristics'] = characteristics
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        return results
    
    def _classify_problem_type(self) -> Dict[str, Any]:
        """
        Classify the ML problem type
        """
        target_data = self.df[self.target_col]
        n_unique = target_data.nunique()
        target_dtype = target_data.dtype
        
        # Check for time series
        is_time_series = self._detect_time_series()
        
        # Classification vs Regression
        if target_dtype == 'object' or (target_dtype in ['int64', 'int32'] and n_unique < 20):
            problem_type = 'classification'
            
            # Binary vs Multiclass
            if n_unique == 2:
                problem_subtype = 'binary_classification'
            elif n_unique <= 10:
                problem_subtype = 'multiclass_classification'
            else:
                problem_subtype = 'multilabel_classification'
            
            # Check for multilabel (multiple targets)
            if self._is_multilabel():
                problem_subtype = 'multilabel_classification'
        
        else:
            problem_type = 'regression'
            
            # Check regression subtype
            if is_time_series:
                problem_subtype = 'time_series_regression'
            elif self._is_count_data():
                problem_subtype = 'count_regression'
            elif target_data.min() >= 0 and target_data.max() <= 1:
                problem_subtype = 'bounded_regression'
            else:
                problem_subtype = 'continuous_regression'
        
        # Override if time series
        if is_time_series:
            if problem_type == 'classification':
                problem_subtype = 'time_series_classification'
            else:
                problem_subtype = 'time_series_regression'
        
        return {
            'problem_type': problem_type,
            'problem_subtype': problem_subtype,
            'n_classes': int(n_unique) if problem_type == 'classification' else None,
            'is_time_series': is_time_series
        }
    
    def _detect_time_series(self) -> bool:
        """
        Detect if data has time series characteristics
        """
        # Check for date/time columns
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                date_cols.append(col)
        
        if len(date_cols) > 0:
            return True
        
        # Check if index is datetime
        if isinstance(self.df.index, pd.DatetimeIndex):
            return True
        
        # Check for sequential patterns (heuristic)
        if self.n_rows > 100:
            # Check if first column looks like sequential IDs
            first_col = self.df.iloc[:, 0]
            if first_col.dtype in ['int64', 'int32']:
                diffs = first_col.diff().dropna()
                if (diffs == 1).sum() / len(diffs) > 0.9:
                    return True
        
        return False
    
    def _is_multilabel(self) -> bool:
        """
        Check if problem is multilabel classification
        """
        # Heuristic: check if target contains lists or multiple values
        sample = self.df[self.target_col].iloc[0]
        if isinstance(sample, (list, tuple, set)):
            return True
        
        # Check if target column name suggests multilabel
        if 'multi' in self.target_col.lower() or 'tags' in self.target_col.lower():
            return True
        
        return False
    
    def _is_count_data(self) -> bool:
        """
        Check if target is count data (Poisson regression)
        """
        target_data = self.df[self.target_col]
        
        # Check if all values are non-negative integers
        if target_data.dtype in ['int64', 'int32']:
            if (target_data >= 0).all():
                # Check if distribution looks like counts
                if target_data.max() < 1000 and target_data.mean() < 50:
                    return True
        
        return False
    
    def _analyze_data_type(self) -> Dict[str, Any]:
        """
        Classify the type of data (tabular, text, image, etc.)
        """
        numeric_ratio = len(self.df.select_dtypes(include=[np.number]).columns) / self.n_cols
        categorical_ratio = len(self.df.select_dtypes(include=['object']).columns) / self.n_cols
        
        # Check for text data
        has_text = False
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else ""
            if isinstance(sample, str) and len(sample) > 50:
                has_text = True
                break
        
        # Classify data type
        if has_text:
            data_type = 'text'
            subtype = 'nlp'
        elif numeric_ratio > 0.8:
            data_type = 'tabular'
            subtype = 'numeric_heavy'
        elif categorical_ratio > 0.5:
            data_type = 'tabular'
            subtype = 'categorical_heavy'
        else:
            data_type = 'tabular'
            subtype = 'mixed'
        
        return {
            'primary_type': data_type,
            'subtype': subtype,
            'numeric_ratio': float(numeric_ratio),
            'categorical_ratio': float(categorical_ratio),
            'has_text_features': has_text
        }
    
    def _assess_complexity(self) -> Dict[str, Any]:
        """
        Assess task complexity
        """
        # Feature-to-sample ratio
        feature_ratio = self.n_cols / self.n_rows
        
        # Dimensionality
        if self.n_cols < 10:
            dimensionality = 'low'
        elif self.n_cols < 50:
            dimensionality = 'medium'
        elif self.n_cols < 200:
            dimensionality = 'high'
        else:
            dimensionality = 'very_high'
        
        # Sample size
        if self.n_rows < 100:
            sample_size = 'very_small'
            complexity_level = 'high'
        elif self.n_rows < 1000:
            sample_size = 'small'
            complexity_level = 'medium'
        elif self.n_rows < 10000:
            sample_size = 'medium'
            complexity_level = 'medium'
        elif self.n_rows < 100000:
            sample_size = 'large'
            complexity_level = 'low'
        else:
            sample_size = 'very_large'
            complexity_level = 'low'
        
        # Adjust complexity based on feature ratio
        if feature_ratio > 0.1:
            complexity_level = 'high'
        elif feature_ratio > 0.01:
            if complexity_level == 'low':
                complexity_level = 'medium'
        
        # Check for non-linearity indicators
        non_linearity_score = self._estimate_non_linearity()
        
        return {
            'complexity_level': complexity_level,
            'dimensionality': dimensionality,
            'sample_size': sample_size,
            'feature_ratio': float(feature_ratio),
            'n_features': self.n_cols - 1,
            'n_samples': self.n_rows,
            'non_linearity_score': non_linearity_score,
            'curse_of_dimensionality_risk': 'high' if feature_ratio > 0.1 else 'medium' if feature_ratio > 0.01 else 'low'
        }
    
    def _estimate_non_linearity(self) -> float:
        """
        Estimate non-linearity in the data (0-1 scale)
        """
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2 or self.target_col not in numeric_cols:
                return 0.5  # Unknown
            
            # Sample data for performance
            sample_size = min(1000, len(self.df))
            df_sample = self.df.sample(n=sample_size, random_state=42)
            
            # Try to use mutual information if available
            try:
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                from sklearn.preprocessing import LabelEncoder
                
                X = df_sample[numeric_cols].drop(columns=[self.target_col], errors='ignore')
                y = df_sample[self.target_col]
                
                if len(X.columns) == 0:
                    return 0.5
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Calculate mutual information
                if y.dtype == 'object' or y.nunique() < 20:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y.astype(str))
                    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                
                # Calculate correlations
                correlations = []
                for col in X.columns:
                    if col in numeric_cols:
                        corr = abs(X[col].corr(y if y.dtype != 'object' else pd.Series(y_encoded)))
                        correlations.append(corr)
                
                # Non-linearity score: high MI with low correlation suggests non-linearity
                avg_mi = np.mean(mi_scores) if len(mi_scores) > 0 else 0
                avg_corr = np.mean(correlations) if len(correlations) > 0 else 0
                
                if avg_corr > 0:
                    non_linearity = 1 - (avg_corr / (avg_mi + 1e-10))
                    return float(np.clip(non_linearity, 0, 1))
                else:
                    return 0.5
            except Exception as e:
                print(f"Warning: Could not calculate mutual information ({e}). Using correlation-based estimate.")
                # Fallback: use correlation variance as proxy for non-linearity
                X = df_sample[numeric_cols].drop(columns=[self.target_col], errors='ignore')
                y = df_sample[self.target_col]
                
                if len(X.columns) == 0:
                    return 0.5
                
                X = X.fillna(X.mean())
                
                # Calculate correlation variance (high variance suggests non-linearity)
                correlations = []
                for col in X.columns:
                    try:
                        if y.dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y_num = le.fit_transform(y.astype(str))
                            corr = abs(X[col].corr(pd.Series(y_num)))
                        else:
                            corr = abs(X[col].corr(y))
                        correlations.append(corr)
                    except:
                        continue
                
                if len(correlations) > 0:
                    # High variance in correlations suggests non-linearity
                    corr_variance = np.var(correlations)
                    return float(np.clip(corr_variance * 2, 0, 1))
                else:
                    return 0.5
        except Exception as e:
            print(f"Warning: Non-linearity estimation failed ({e}). Using default value.")
            return 0.5
    
    def _assess_risk_profile(self) -> Dict[str, Any]:
        """
        Assess the risk profile of the ML application
        """
        # Heuristic-based risk assessment
        risk_indicators = []
        risk_level = 'low'
        
        # Check column names for high-risk domains
        high_risk_keywords = ['medical', 'health', 'disease', 'diagnosis', 'patient', 
                             'financial', 'credit', 'loan', 'fraud', 'transaction',
                             'safety', 'critical', 'emergency', 'life', 'death']
        
        medium_risk_keywords = ['price', 'cost', 'revenue', 'profit', 'customer',
                               'user', 'rating', 'score', 'performance']
        
        all_text = ' '.join(self.df.columns).lower()
        
        for keyword in high_risk_keywords:
            if keyword in all_text:
                risk_indicators.append(f'High-risk domain detected: {keyword}')
                risk_level = 'critical'
        
        if risk_level != 'critical':
            for keyword in medium_risk_keywords:
                if keyword in all_text:
                    risk_indicators.append(f'Medium-risk domain detected: {keyword}')
                    risk_level = 'medium'
        
        # Assess based on problem type
        problem_type = self._classify_problem_type()
        if problem_type['problem_type'] == 'classification':
            if problem_type['n_classes'] == 2:
                risk_indicators.append('Binary classification - ensure balanced evaluation')
        
        # Check for sensitive attributes (fairness concerns)
        sensitive_keywords = ['gender', 'race', 'age', 'ethnicity', 'religion', 'sex']
        has_sensitive = any(keyword in all_text for keyword in sensitive_keywords)
        
        if has_sensitive:
            risk_indicators.append('Sensitive attributes detected - fairness concerns')
            if risk_level == 'low':
                risk_level = 'medium'
        
        return {
            'risk_level': risk_level,
            'risk_indicators': risk_indicators,
            'requires_interpretability': risk_level in ['critical', 'medium'],
            'requires_fairness_audit': has_sensitive,
            'regulatory_considerations': risk_level == 'critical'
        }
    
    def _extract_characteristics(self) -> Dict[str, Any]:
        """
        Extract key dataset characteristics
        """
        return {
            'n_samples': self.n_rows,
            'n_features': self.n_cols - 1,
            'n_numeric_features': len(self.df.select_dtypes(include=[np.number]).columns) - 1,
            'n_categorical_features': len(self.df.select_dtypes(include=['object']).columns),
            'target_type': str(self.df[self.target_col].dtype),
            'has_missing_values': bool(self.df.isnull().any().any()),
            'missing_percentage': float((self.df.isnull().sum().sum() / (self.n_rows * self.n_cols)) * 100),
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> list:
        """
        Generate problem-specific recommendations
        """
        recommendations = []
        
        # Problem type recommendations
        if analysis['problem_type'] == 'classification':
            if analysis['problem_subtype'] == 'binary_classification':
                recommendations.append({
                    'category': 'metrics',
                    'recommendation': 'Use ROC-AUC, Precision-Recall curves for evaluation',
                    'reason': 'Binary classification benefits from threshold-independent metrics'
                })
            elif analysis['problem_subtype'] == 'multiclass_classification':
                recommendations.append({
                    'category': 'metrics',
                    'recommendation': 'Use macro/micro F1-score, confusion matrix',
                    'reason': 'Multiclass requires class-wise performance analysis'
                })
        
        # Complexity recommendations
        complexity = analysis['task_complexity']
        if complexity['sample_size'] == 'very_small':
            recommendations.append({
                'category': 'modeling',
                'recommendation': 'Use simple models (Logistic Regression, Decision Trees)',
                'reason': 'Small datasets require low-complexity models to avoid overfitting'
            })
            recommendations.append({
                'category': 'validation',
                'recommendation': 'Use Leave-One-Out or Stratified K-Fold CV',
                'reason': 'Maximize training data usage with small samples'
            })
        elif complexity['sample_size'] == 'very_large':
            recommendations.append({
                'category': 'modeling',
                'recommendation': 'Consider deep learning or gradient boosting',
                'reason': 'Large datasets can support complex models'
            })
        
        if complexity['curse_of_dimensionality_risk'] == 'high':
            recommendations.append({
                'category': 'preprocessing',
                'recommendation': 'Apply dimensionality reduction (PCA, feature selection)',
                'reason': 'High feature-to-sample ratio increases overfitting risk'
            })
        
        # Risk profile recommendations
        risk = analysis['risk_profile']
        if risk['risk_level'] in ['critical', 'medium']:
            recommendations.append({
                'category': 'interpretability',
                'recommendation': 'Use interpretable models or SHAP/LIME explanations',
                'reason': f"{risk['risk_level'].capitalize()}-risk application requires transparency"
            })
        
        if risk['requires_fairness_audit']:
            recommendations.append({
                'category': 'fairness',
                'recommendation': 'Conduct fairness audit across sensitive groups',
                'reason': 'Sensitive attributes present - ensure equitable performance'
            })
        
        # Data type recommendations
        if analysis['data_type']['has_text_features']:
            recommendations.append({
                'category': 'preprocessing',
                'recommendation': 'Use TF-IDF or word embeddings for text features',
                'reason': 'Text data requires specialized encoding'
            })
        
        # Time series recommendations
        if analysis.get('is_time_series'):
            recommendations.append({
                'category': 'validation',
                'recommendation': 'Use time-based split (not random split)',
                'reason': 'Time series requires temporal validation to avoid data leakage'
            })
            recommendations.append({
                'category': 'modeling',
                'recommendation': 'Consider ARIMA, LSTM, or Prophet for forecasting',
                'reason': 'Time series models capture temporal dependencies'
            })
        
        return recommendations
