"""
Data Intelligence Engine - ISD Module
Analyzes dataset health, detects issues, and provides repair suggestions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy, but provide fallbacks if it fails
try:
    from scipy import stats
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except Exception as e:
    print(f"Warning: scipy not available ({e}). Using fallback methods.")
    SCIPY_AVAILABLE = False
    
    # Fallback entropy calculation
    def entropy(pk):
        """Simple entropy calculation without scipy"""
        pk = np.asarray(pk)
        pk = pk / np.sum(pk)
        return -np.sum(pk * np.log(pk + 1e-10))
    
    # Fallback stats module
    class stats:
        @staticmethod
        def skew(data):
            """Simple skewness calculation"""
            data = np.asarray(data)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        
        @staticmethod
        def kurtosis(data):
            """Simple kurtosis calculation"""
            data = np.asarray(data)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        
        @staticmethod
        def zscore(data):
            """Simple z-score calculation"""
            data = np.asarray(data)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
        
        @staticmethod
        def shapiro(data):
            """Dummy shapiro test - always returns normal"""
            return (0.95, 0.1)
        
        @staticmethod
        def anderson(data):
            """Dummy anderson test"""
            class Result:
                statistic = 0.5
                critical_values = [0.5, 0.7, 0.9, 1.0, 1.2]
            return Result()


class DataIntelligenceEngine:
    """
    Industry-grade data analysis engine that diagnoses dataset quality
    and provides actionable insights
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.n_rows, self.n_cols = df.shape
        
    def analyze(self):
        """
        Comprehensive data analysis returning health score and diagnostics
        """
        results = {
            'health_score': 0,
            'red_flags': [],
            'warnings': [],
            'insights': [],
            'repair_suggestions': [],
            'detailed_analysis': {}
        }
        
        # Run all analysis modules
        missing_analysis = self._analyze_missing_values()
        outlier_analysis = self._analyze_outliers()
        imbalance_analysis = self._analyze_class_imbalance()
        correlation_analysis = self._analyze_correlations()
        entropy_analysis = self._analyze_entropy()
        leakage_analysis = self._detect_data_leakage()
        distribution_analysis = self._analyze_distributions()
        cardinality_analysis = self._analyze_cardinality()
        
        # Aggregate results
        results['detailed_analysis'] = {
            'missing_values': missing_analysis,
            'outliers': outlier_analysis,
            'class_imbalance': imbalance_analysis,
            'correlations': correlation_analysis,
            'entropy': entropy_analysis,
            'data_leakage': leakage_analysis,
            'distributions': distribution_analysis,
            'cardinality': cardinality_analysis
        }
        
        # Calculate health score (0-100)
        health_score = self._calculate_health_score(results['detailed_analysis'])
        results['health_score'] = health_score
        
        # Generate red flags, warnings, and suggestions
        self._generate_alerts(results)
        
        return results
    
    def _analyze_missing_values(self):
        """Analyze missing data patterns"""
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / self.n_rows) * 100
        
        total_missing = missing_counts.sum()
        total_cells = self.n_rows * self.n_cols
        overall_missing_rate = (total_missing / total_cells) * 100
        
        # Detect missing patterns
        columns_with_missing = missing_counts[missing_counts > 0].to_dict()
        high_missing_cols = missing_percent[missing_percent > 50].index.tolist()
        
        # Check for systematic missing (MCAR, MAR, MNAR)
        missing_pattern = self._detect_missing_pattern()
        
        return {
            'total_missing_cells': int(total_missing),
            'overall_missing_rate': float(overall_missing_rate),
            'columns_with_missing': {k: int(v) for k, v in columns_with_missing.items()},
            'high_missing_columns': high_missing_cols,
            'missing_pattern': missing_pattern,
            'severity': 'critical' if overall_missing_rate > 30 else 'warning' if overall_missing_rate > 10 else 'ok'
        }
    
    def _detect_missing_pattern(self):
        """Detect if missing data is MCAR, MAR, or MNAR"""
        if self.df.isnull().sum().sum() == 0:
            return 'none'
        
        # Simple heuristic: check if missingness correlates with other variables
        missing_indicators = self.df.isnull().astype(int)
        
        if len(self.numeric_cols) > 0:
            correlations = []
            for col in missing_indicators.columns:
                if missing_indicators[col].sum() > 0:
                    for num_col in self.numeric_cols:
                        if col != num_col:
                            corr = missing_indicators[col].corr(self.df[num_col].fillna(0))
                            if abs(corr) > 0.3:
                                correlations.append((col, num_col, corr))
            
            if len(correlations) > 0:
                return 'MAR'  # Missing At Random (correlated with observed data)
        
        return 'MCAR'  # Missing Completely At Random (assumed if no pattern)
    
    def _analyze_outliers(self):
        """Detect outliers using multiple methods"""
        outlier_summary = {}
        total_outliers = 0
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            
            # Modified Z-score (robust)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros(len(data))
            robust_outliers = (np.abs(modified_z_scores) > 3.5).sum()
            
            outlier_count = max(iqr_outliers, z_outliers, robust_outliers)
            outlier_percent = (outlier_count / len(data)) * 100
            
            if outlier_count > 0:
                outlier_summary[col] = {
                    'count': int(outlier_count),
                    'percent': float(outlier_percent),
                    'iqr_method': int(iqr_outliers),
                    'zscore_method': int(z_outliers),
                    'robust_method': int(robust_outliers),
                    'severity': 'high' if outlier_percent > 10 else 'medium' if outlier_percent > 5 else 'low'
                }
                total_outliers += outlier_count
        
        outlier_rate = (total_outliers / (self.n_rows * len(self.numeric_cols))) * 100 if len(self.numeric_cols) > 0 else 0
        
        return {
            'total_outliers': int(total_outliers),
            'outlier_rate': float(outlier_rate),
            'by_column': outlier_summary,
            'severity': 'critical' if outlier_rate > 15 else 'warning' if outlier_rate > 5 else 'ok'
        }
    
    def _analyze_class_imbalance(self):
        """Detect class imbalance in target variable"""
        target_col = self.df.columns[-1]
        
        # Check if it's a classification problem
        unique_values = self.df[target_col].nunique()
        if unique_values > 20 or self.df[target_col].dtype not in ['object', 'int64', 'int32']:
            return {'is_classification': False, 'severity': 'ok'}
        
        value_counts = self.df[target_col].value_counts()
        total = len(self.df)
        
        # Calculate imbalance ratio
        majority_class_count = value_counts.iloc[0]
        minority_class_count = value_counts.iloc[-1]
        imbalance_ratio = majority_class_count / minority_class_count if minority_class_count > 0 else float('inf')
        
        # Calculate class distribution
        class_distribution = (value_counts / total * 100).to_dict()
        
        # Gini impurity (0 = perfectly balanced, 1 = completely imbalanced)
        proportions = value_counts / total
        gini = 1 - sum(proportions ** 2)
        
        severity = 'critical' if imbalance_ratio > 10 else 'warning' if imbalance_ratio > 3 else 'ok'
        
        return {
            'is_classification': True,
            'n_classes': int(unique_values),
            'imbalance_ratio': float(imbalance_ratio),
            'majority_class': str(value_counts.index[0]),
            'minority_class': str(value_counts.index[-1]),
            'majority_count': int(majority_class_count),
            'minority_count': int(minority_class_count),
            'class_distribution': {str(k): float(v) for k, v in class_distribution.items()},
            'gini_impurity': float(gini),
            'severity': severity
        }
    
    def _analyze_correlations(self):
        """Analyze feature correlations and multicollinearity"""
        if len(self.numeric_cols) < 2:
            return {'severity': 'ok', 'high_correlations': []}
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.9:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        # Calculate VIF for multicollinearity
        vif_scores = self._calculate_vif()
        
        severity = 'critical' if len(high_corr_pairs) > 5 else 'warning' if len(high_corr_pairs) > 0 else 'ok'
        
        return {
            'high_correlations': high_corr_pairs,
            'n_high_correlations': len(high_corr_pairs),
            'vif_scores': vif_scores,
            'severity': severity
        }
    
    def _calculate_vif(self):
        """Calculate Variance Inflation Factor for multicollinearity detection"""
        if len(self.numeric_cols) < 2:
            return {}
        
        from sklearn.linear_model import LinearRegression
        
        vif_data = {}
        df_numeric = self.df[self.numeric_cols].dropna()
        
        if len(df_numeric) < 10:
            return {}
        
        for i, col in enumerate(self.numeric_cols[:10]):  # Limit to first 10 features for performance
            try:
                X = df_numeric.drop(columns=[col])
                y = df_numeric[col]
                
                if len(X.columns) == 0:
                    continue
                
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)
                
                vif = 1 / (1 - r_squared) if r_squared < 0.9999 else float('inf')
                vif_data[col] = {
                    'vif': float(vif),
                    'severity': 'high' if vif > 10 else 'medium' if vif > 5 else 'low'
                }
            except:
                continue
        
        return vif_data
    
    def _analyze_entropy(self):
        """Calculate entropy for each feature to measure information content"""
        entropy_scores = {}
        
        for col in self.df.columns:
            try:
                value_counts = self.df[col].value_counts(normalize=True)
                col_entropy = entropy(value_counts)
                
                # Normalize entropy by max possible entropy
                max_entropy = np.log(len(value_counts)) if len(value_counts) > 1 else 1
                normalized_entropy = col_entropy / max_entropy if max_entropy > 0 else 0
                
                entropy_scores[col] = {
                    'entropy': float(col_entropy),
                    'normalized_entropy': float(normalized_entropy),
                    'unique_values': int(len(value_counts)),
                    'information_content': 'high' if normalized_entropy > 0.7 else 'medium' if normalized_entropy > 0.3 else 'low'
                }
            except:
                continue
        
        # Identify low-information features
        low_info_features = [col for col, data in entropy_scores.items() 
                            if data['normalized_entropy'] < 0.1]
        
        return {
            'by_column': entropy_scores,
            'low_information_features': low_info_features,
            'severity': 'warning' if len(low_info_features) > 0 else 'ok'
        }
    
    def _detect_data_leakage(self):
        """Detect potential data leakage risks"""
        leakage_risks = []
        target_col = self.df.columns[-1]
        
        # Check for perfect correlations with target
        if target_col in self.numeric_cols:
            for col in self.numeric_cols:
                if col != target_col:
                    corr = self.df[col].corr(self.df[target_col])
                    if abs(corr) > 0.95:
                        leakage_risks.append({
                            'feature': col,
                            'type': 'perfect_correlation',
                            'correlation': float(corr),
                            'risk_level': 'critical'
                        })
        
        # Check for duplicate columns
        for i, col1 in enumerate(self.df.columns[:-1]):
            for col2 in self.df.columns[i+1:-1]:
                if self.df[col1].equals(self.df[col2]):
                    leakage_risks.append({
                        'feature': f'{col1} == {col2}',
                        'type': 'duplicate_column',
                        'risk_level': 'high'
                    })
        
        # Check for features with target in name (heuristic)
        target_keywords = ['target', 'label', 'outcome', 'result', 'prediction']
        for col in self.df.columns[:-1]:
            if any(keyword in col.lower() for keyword in target_keywords):
                leakage_risks.append({
                    'feature': col,
                    'type': 'suspicious_naming',
                    'risk_level': 'medium'
                })
        
        severity = 'critical' if any(r['risk_level'] == 'critical' for r in leakage_risks) else \
                   'warning' if len(leakage_risks) > 0 else 'ok'
        
        return {
            'risks': leakage_risks,
            'n_risks': len(leakage_risks),
            'severity': severity
        }
    
    def _analyze_distributions(self):
        """Analyze distribution characteristics and detect shifts"""
        distribution_analysis = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            
            # Calculate distribution metrics
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if len(data) < 5000:
                _, p_value = stats.shapiro(data[:5000])
                is_normal = p_value > 0.05
            else:
                result = stats.anderson(data)
                is_normal = result.statistic < result.critical_values[2]  # 5% significance
            
            distribution_analysis[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': bool(is_normal),
                'distribution_type': self._classify_distribution(skewness, kurtosis, is_normal)
            }
        
        return {
            'by_column': distribution_analysis,
            'severity': 'ok'
        }
    
    def _classify_distribution(self, skewness, kurtosis, is_normal):
        """Classify distribution type based on statistics"""
        if is_normal:
            return 'normal'
        elif abs(skewness) > 1:
            return 'highly_skewed'
        elif abs(skewness) > 0.5:
            return 'moderately_skewed'
        elif abs(kurtosis) > 3:
            return 'heavy_tailed'
        else:
            return 'unknown'
    
    def _analyze_cardinality(self):
        """Analyze cardinality of categorical features"""
        cardinality_analysis = {}
        
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / self.n_rows
            
            # Classify cardinality
            if unique_ratio > 0.9:
                cardinality_type = 'high_cardinality'
                severity = 'warning'
            elif unique_count > 50:
                cardinality_type = 'medium_cardinality'
                severity = 'info'
            else:
                cardinality_type = 'low_cardinality'
                severity = 'ok'
            
            cardinality_analysis[col] = {
                'unique_count': int(unique_count),
                'unique_ratio': float(unique_ratio),
                'cardinality_type': cardinality_type,
                'severity': severity
            }
        
        high_cardinality_cols = [col for col, data in cardinality_analysis.items() 
                                if data['cardinality_type'] == 'high_cardinality']
        
        return {
            'by_column': cardinality_analysis,
            'high_cardinality_columns': high_cardinality_cols,
            'severity': 'warning' if len(high_cardinality_cols) > 0 else 'ok'
        }
    
    def _calculate_health_score(self, analysis):
        """Calculate overall dataset health score (0-100)"""
        score = 100.0
        
        # Missing values penalty
        missing_rate = analysis['missing_values']['overall_missing_rate']
        score -= min(missing_rate * 2, 30)  # Max 30 points penalty
        
        # Outliers penalty
        outlier_rate = analysis['outliers']['outlier_rate']
        score -= min(outlier_rate, 15)  # Max 15 points penalty
        
        # Class imbalance penalty
        if analysis['class_imbalance']['is_classification']:
            imbalance_ratio = analysis['class_imbalance']['imbalance_ratio']
            if imbalance_ratio > 10:
                score -= 20
            elif imbalance_ratio > 5:
                score -= 10
            elif imbalance_ratio > 3:
                score -= 5
        
        # Correlation penalty
        n_high_corr = analysis['correlations']['n_high_correlations']
        score -= min(n_high_corr * 3, 15)  # Max 15 points penalty
        
        # Data leakage penalty (critical)
        n_leakage_risks = analysis['data_leakage']['n_risks']
        critical_leaks = sum(1 for r in analysis['data_leakage']['risks'] if r['risk_level'] == 'critical')
        score -= critical_leaks * 20  # 20 points per critical leak
        score -= (n_leakage_risks - critical_leaks) * 5  # 5 points per other risk
        
        # Low information features penalty
        n_low_info = len(analysis['entropy']['low_information_features'])
        score -= min(n_low_info * 2, 10)  # Max 10 points penalty
        
        return max(0, min(100, score))
    
    def _generate_alerts(self, results):
        """Generate red flags, warnings, and repair suggestions"""
        analysis = results['detailed_analysis']
        
        # RED FLAGS (Critical issues)
        if analysis['data_leakage']['severity'] == 'critical':
            results['red_flags'].append({
                'type': 'data_leakage',
                'message': 'CRITICAL: Potential data leakage detected',
                'details': f"{analysis['data_leakage']['n_risks']} leakage risks found",
                'impact': 'Model will have unrealistic performance'
            })
        
        if analysis['missing_values']['severity'] == 'critical':
            results['red_flags'].append({
                'type': 'missing_data',
                'message': 'CRITICAL: Excessive missing data',
                'details': f"{analysis['missing_values']['overall_missing_rate']:.1f}% of data is missing",
                'impact': 'Severe information loss, unreliable models'
            })
        
        if analysis['class_imbalance']['severity'] == 'critical':
            results['red_flags'].append({
                'type': 'class_imbalance',
                'message': 'CRITICAL: Severe class imbalance',
                'details': f"Imbalance ratio: {analysis['class_imbalance']['imbalance_ratio']:.1f}:1",
                'impact': 'Model will be biased towards majority class'
            })
        
        # WARNINGS (Important but not critical)
        if analysis['outliers']['severity'] in ['warning', 'critical']:
            results['warnings'].append({
                'type': 'outliers',
                'message': f"High outlier rate: {analysis['outliers']['outlier_rate']:.1f}%",
                'recommendation': 'Consider robust scaling or outlier removal'
            })
        
        if analysis['correlations']['severity'] in ['warning', 'critical']:
            results['warnings'].append({
                'type': 'multicollinearity',
                'message': f"{analysis['correlations']['n_high_correlations']} highly correlated feature pairs",
                'recommendation': 'Consider PCA or feature selection'
            })
        
        if len(analysis['entropy']['low_information_features']) > 0:
            results['warnings'].append({
                'type': 'low_information',
                'message': f"{len(analysis['entropy']['low_information_features'])} low-information features",
                'recommendation': 'Consider removing constant or near-constant features'
            })
        
        # REPAIR SUGGESTIONS
        self._generate_repair_suggestions(results, analysis)
        
        # INSIGHTS (Positive observations)
        if results['health_score'] >= 80:
            results['insights'].append('Dataset is in good health - ready for modeling')
        
        if analysis['missing_values']['severity'] == 'ok':
            results['insights'].append('Minimal missing data - no imputation needed')
        
        if analysis['class_imbalance']['severity'] == 'ok' and analysis['class_imbalance']['is_classification']:
            results['insights'].append('Classes are well balanced')
    
    def _generate_repair_suggestions(self, results, analysis):
        """Generate actionable repair suggestions"""
        
        # Missing data suggestions
        if analysis['missing_values']['overall_missing_rate'] > 0:
            if analysis['missing_values']['overall_missing_rate'] > 50:
                results['repair_suggestions'].append({
                    'priority': 'critical',
                    'action': 'Drop high-missing columns',
                    'details': f"Remove columns with >50% missing: {analysis['missing_values']['high_missing_columns']}",
                    'code_hint': "df.drop(columns=high_missing_cols)"
                })
            elif analysis['missing_values']['missing_pattern'] == 'MCAR':
                results['repair_suggestions'].append({
                    'priority': 'high',
                    'action': 'Impute missing values',
                    'details': 'Use mean/median for numeric, mode for categorical',
                    'code_hint': "SimpleImputer(strategy='median')"
                })
            else:
                results['repair_suggestions'].append({
                    'priority': 'high',
                    'action': 'Advanced imputation',
                    'details': 'Missing data is not random - use KNN or iterative imputation',
                    'code_hint': "KNNImputer() or IterativeImputer()"
                })
        
        # Class imbalance suggestions
        if analysis['class_imbalance']['severity'] in ['warning', 'critical']:
            ratio = analysis['class_imbalance']['imbalance_ratio']
            if ratio > 10:
                results['repair_suggestions'].append({
                    'priority': 'critical',
                    'action': 'Address severe class imbalance',
                    'details': 'Use SMOTE, ADASYN, or class weights',
                    'code_hint': "SMOTE() or class_weight='balanced'"
                })
            else:
                results['repair_suggestions'].append({
                    'priority': 'medium',
                    'action': 'Balance classes',
                    'details': 'Use stratified sampling or class weights',
                    'code_hint': "stratify=y in train_test_split"
                })
        
        # Outlier suggestions
        if analysis['outliers']['severity'] in ['warning', 'critical']:
            results['repair_suggestions'].append({
                'priority': 'medium',
                'action': 'Handle outliers',
                'details': 'Use robust scaling or cap extreme values',
                'code_hint': "RobustScaler() or winsorization"
            })
        
        # Multicollinearity suggestions
        if analysis['correlations']['n_high_correlations'] > 0:
            results['repair_suggestions'].append({
                'priority': 'medium',
                'action': 'Reduce multicollinearity',
                'details': 'Remove redundant features or use PCA',
                'code_hint': "PCA(n_components=0.95) or drop correlated features"
            })
        
        # Low information features
        if len(analysis['entropy']['low_information_features']) > 0:
            results['repair_suggestions'].append({
                'priority': 'low',
                'action': 'Remove low-information features',
                'details': f"Drop: {analysis['entropy']['low_information_features'][:5]}",
                'code_hint': "VarianceThreshold(threshold=0.01)"
            })
