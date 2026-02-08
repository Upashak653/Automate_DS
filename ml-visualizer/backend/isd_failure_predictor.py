"""
Failure Prediction Module - ISD
Predicts potential ML project failures before training
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class FailurePredictionModule:
    """
    Predicts overfitting, underfitting, and data sufficiency issues
    """
    
    def __init__(self, problem_analysis: Dict, data_analysis: Dict):
        self.problem = problem_analysis
        self.data = data_analysis
        
    def predict_failures(self) -> Dict[str, Any]:
        """Predict potential failures"""
        results = {
            'overfitting_risk': None,
            'underfitting_risk': None,
            'data_sufficiency': None,
            'feature_relevance': None,
            'overall_risk_score': 0,
            'critical_warnings': [],
            'preventive_actions': []
        }
        
        # Overfitting risk
        overfitting = self._assess_overfitting_risk()
        results['overfitting_risk'] = overfitting
        
        # Underfitting risk
        underfitting = self._assess_underfitting_risk()
        results['underfitting_risk'] = underfitting
        
        # Data sufficiency
        sufficiency = self._assess_data_sufficiency()
        results['data_sufficiency'] = sufficiency
        
        # Feature relevance
        relevance = self._assess_feature_relevance()
        results['feature_relevance'] = relevance
        
        # Calculate overall risk
        risk_score = self._calculate_overall_risk(results)
        results['overall_risk_score'] = risk_score
        
        # Generate warnings and actions
        self._generate_warnings_and_actions(results)
        
        return results
    
    def _assess_overfitting_risk(self) -> Dict[str, Any]:
        """Assess overfitting risk"""
        risk_score = 0
        risk_factors = []
        
        complexity = self.problem['task_complexity']
        n_samples = complexity['n_samples']
        n_features = complexity['n_features']
        feature_ratio = complexity['feature_ratio']
        
        # High feature-to-sample ratio
        if feature_ratio > 0.1:
            risk_score += 40
            risk_factors.append(f'High feature-to-sample ratio: {feature_ratio:.3f}')
        elif feature_ratio > 0.05:
            risk_score += 20
            risk_factors.append(f'Moderate feature-to-sample ratio: {feature_ratio:.3f}')
        
        # Small dataset
        if n_samples < 100:
            risk_score += 30
            risk_factors.append(f'Very small dataset: {n_samples} samples')
        elif n_samples < 500:
            risk_score += 15
            risk_factors.append(f'Small dataset: {n_samples} samples')
        
        # High dimensionality
        if n_features > 100:
            risk_score += 20
            risk_factors.append(f'High dimensionality: {n_features} features')
        
        # Complex model on small data
        if n_samples < 1000 and complexity['non_linearity_score'] > 0.7:
            risk_score += 15
            risk_factors.append('Complex patterns with limited data')
        
        risk_level = 'critical' if risk_score > 70 else 'high' if risk_score > 40 else 'medium' if risk_score > 20 else 'low'
        
        return {
            'risk_score': min(100, risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'probability': f'{min(100, risk_score)}%'
        }
    
    def _assess_underfitting_risk(self) -> Dict[str, Any]:
        """Assess underfitting risk"""
        risk_score = 0
        risk_factors = []
        
        complexity = self.problem['task_complexity']
        non_linearity = complexity['non_linearity_score']
        
        # High non-linearity
        if non_linearity > 0.8:
            risk_score += 30
            risk_factors.append('Highly non-linear relationships detected')
        
        # Low feature count for complex problem
        if complexity['n_features'] < 5 and non_linearity > 0.6:
            risk_score += 25
            risk_factors.append('Few features for complex problem')
        
        # High noise (from outliers)
        outlier_rate = self.data['detailed_analysis']['outliers']['outlier_rate']
        if outlier_rate > 15:
            risk_score += 20
            risk_factors.append(f'High noise level: {outlier_rate:.1f}% outliers')
        
        risk_level = 'high' if risk_score > 50 else 'medium' if risk_score > 25 else 'low'
        
        return {
            'risk_score': min(100, risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'probability': f'{min(100, risk_score)}%'
        }
    
    def _assess_data_sufficiency(self) -> Dict[str, Any]:
        """Assess if data is sufficient"""
        n_samples = self.problem['task_complexity']['n_samples']
        n_features = self.problem['task_complexity']['n_features']
        
        # Rule of thumb: need at least 10 samples per feature
        recommended_samples = n_features * 10
        sufficiency_ratio = n_samples / recommended_samples if recommended_samples > 0 else 1
        
        if sufficiency_ratio >= 1:
            status = 'sufficient'
            message = f'Dataset size is adequate ({n_samples} samples for {n_features} features)'
        elif sufficiency_ratio >= 0.5:
            status = 'marginal'
            message = f'Dataset size is marginal. Recommended: {recommended_samples} samples, have: {n_samples}'
        else:
            status = 'insufficient'
            message = f'Dataset too small. Recommended: {recommended_samples} samples, have: {n_samples}'
        
        return {
            'status': status,
            'sufficiency_ratio': float(sufficiency_ratio),
            'current_samples': n_samples,
            'recommended_samples': recommended_samples,
            'message': message
        }
    
    def _assess_feature_relevance(self) -> Dict[str, Any]:
        """Assess feature relevance issues"""
        issues = []
        
        # Low information features
        low_info = self.data['detailed_analysis']['entropy']['low_information_features']
        if len(low_info) > 0:
            issues.append({
                'type': 'low_information',
                'severity': 'medium',
                'message': f'{len(low_info)} features have low information content',
                'features': low_info[:5]
            })
        
        # High correlation (redundant features)
        high_corr = self.data['detailed_analysis']['correlations']['n_high_correlations']
        if high_corr > 0:
            issues.append({
                'type': 'redundancy',
                'severity': 'medium',
                'message': f'{high_corr} highly correlated feature pairs',
                'impact': 'Multicollinearity may affect model stability'
            })
        
        # Data leakage
        leakage_risks = self.data['detailed_analysis']['data_leakage']['n_risks']
        if leakage_risks > 0:
            issues.append({
                'type': 'data_leakage',
                'severity': 'critical',
                'message': f'{leakage_risks} potential data leakage risks',
                'impact': 'Model will have unrealistic performance'
            })
        
        relevance_score = 100 - (len(issues) * 20)
        
        return {
            'relevance_score': max(0, relevance_score),
            'issues': issues,
            'n_issues': len(issues)
        }
    
    def _calculate_overall_risk(self, results: Dict) -> int:
        """Calculate overall project risk score"""
        overfitting_risk = results['overfitting_risk']['risk_score']
        underfitting_risk = results['underfitting_risk']['risk_score']
        sufficiency = results['data_sufficiency']['sufficiency_ratio']
        relevance = results['feature_relevance']['relevance_score']
        
        # Weighted risk score
        risk = (overfitting_risk * 0.3 + 
                underfitting_risk * 0.2 + 
                (1 - sufficiency) * 100 * 0.3 + 
                (100 - relevance) * 0.2)
        
        return int(min(100, risk))
    
    def _generate_warnings_and_actions(self, results: Dict):
        """Generate warnings and preventive actions"""
        
        # Overfitting warnings
        if results['overfitting_risk']['risk_level'] in ['critical', 'high']:
            results['critical_warnings'].append({
                'type': 'overfitting',
                'message': 'HIGH OVERFITTING RISK DETECTED',
                'severity': results['overfitting_risk']['risk_level']
            })
            results['preventive_actions'].extend([
                {'action': 'Use regularization (L1/L2)', 'priority': 'critical'},
                {'action': 'Apply cross-validation', 'priority': 'critical'},
                {'action': 'Reduce model complexity', 'priority': 'high'},
                {'action': 'Feature selection/PCA', 'priority': 'high'}
            ])
        
        # Data sufficiency warnings
        if results['data_sufficiency']['status'] == 'insufficient':
            results['critical_warnings'].append({
                'type': 'data_insufficiency',
                'message': 'INSUFFICIENT DATA FOR RELIABLE MODELING',
                'severity': 'critical'
            })
            results['preventive_actions'].extend([
                {'action': 'Collect more data', 'priority': 'critical'},
                {'action': 'Use data augmentation', 'priority': 'high'},
                {'action': 'Apply transfer learning', 'priority': 'medium'}
            ])
        
        # Feature relevance warnings
        if results['feature_relevance']['n_issues'] > 0:
            for issue in results['feature_relevance']['issues']:
                if issue['severity'] == 'critical':
                    results['critical_warnings'].append({
                        'type': issue['type'],
                        'message': issue['message'],
                        'severity': 'critical'
                    })
                    results['preventive_actions'].append({
                        'action': 'Remove leakage features immediately',
                        'priority': 'critical'
                    })
