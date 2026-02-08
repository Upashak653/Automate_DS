"""
Report Generator - ISD
Generates comprehensive ML system design reports
"""

from typing import Dict, Any
from datetime import datetime
import json


class ReportGenerator:
    """
    Generates professional ML architecture reports
    """
    
    def __init__(self, data_intelligence: Dict, problem_understanding: Dict,
                 model_architect: Dict, failure_prediction: Dict):
        self.data_intel = data_intelligence
        self.problem = problem_understanding
        self.model = model_architect
        self.failure = failure_prediction
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        report = {
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(),
            'data_diagnosis': self._format_data_diagnosis(),
            'problem_analysis': self._format_problem_analysis(),
            'model_recommendations': self._format_model_recommendations(),
            'risk_assessment': self._format_risk_assessment(),
            'implementation_roadmap': self._generate_roadmap(),
            'success_metrics': self._define_success_metrics()
        }
        
        return report
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'report_title': 'ML System Architecture Report',
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'system': 'Intelligent System Designer (ISD)'
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        health_score = self.data_intel['health_score']
        problem_type = self.problem['problem_type']
        best_model = self.model['algorithm_recommendations'][0] if self.model['algorithm_recommendations'] else {}
        overall_risk = self.failure['overall_risk_score']
        
        # Determine project viability
        if health_score >= 70 and overall_risk < 50:
            viability = 'HIGH'
            recommendation = 'Project is viable. Proceed with recommended architecture.'
        elif health_score >= 50 and overall_risk < 70:
            viability = 'MEDIUM'
            recommendation = 'Project is viable with data improvements. Address identified issues.'
        else:
            viability = 'LOW'
            recommendation = 'Significant data quality issues. Recommend data collection/cleaning before modeling.'
        
        return {
            'project_viability': viability,
            'data_health_score': health_score,
            'problem_type': problem_type,
            'recommended_model': best_model.get('name', 'N/A'),
            'overall_risk_score': overall_risk,
            'key_recommendation': recommendation,
            'critical_actions': len(self.failure['critical_warnings']),
            'estimated_complexity': self.problem['task_complexity']['complexity_level']
        }
    
    def _format_data_diagnosis(self) -> Dict[str, Any]:
        """Format data diagnosis section"""
        return {
            'health_score': self.data_intel['health_score'],
            'red_flags': self.data_intel['red_flags'],
            'warnings': self.data_intel['warnings'],
            'insights': self.data_intel['insights'],
            'repair_suggestions': self.data_intel['repair_suggestions'],
            'detailed_metrics': {
                'missing_data': self.data_intel['detailed_analysis']['missing_values'],
                'outliers': self.data_intel['detailed_analysis']['outliers'],
                'class_imbalance': self.data_intel['detailed_analysis']['class_imbalance'],
                'data_leakage': self.data_intel['detailed_analysis']['data_leakage']
            }
        }
    
    def _format_problem_analysis(self) -> Dict[str, Any]:
        """Format problem analysis section"""
        return {
            'problem_classification': {
                'type': self.problem['problem_type'],
                'subtype': self.problem['problem_subtype'],
                'complexity': self.problem['task_complexity']['complexity_level']
            },
            'data_characteristics': self.problem['characteristics'],
            'risk_profile': self.problem['risk_profile'],
            'recommendations': self.problem['recommendations']
        }
    
    def _format_model_recommendations(self) -> Dict[str, Any]:
        """Format model recommendations section"""
        return {
            'top_algorithms': self.model['algorithm_recommendations'][:3],
            'loss_function': self.model['loss_function'],
            'evaluation_metrics': self.model['evaluation_metrics'],
            'validation_strategy': self.model['validation_strategy'],
            'data_split': self.model['data_split'],
            'preprocessing_pipeline': self.model['preprocessing_pipeline'],
            'hyperparameter_tuning': self.model['hyperparameter_priorities'],
            'ensemble_recommendation': self.model['ensemble_strategy']
        }
    
    def _format_risk_assessment(self) -> Dict[str, Any]:
        """Format risk assessment section"""
        return {
            'overall_risk_score': self.failure['overall_risk_score'],
            'overfitting_risk': self.failure['overfitting_risk'],
            'underfitting_risk': self.failure['underfitting_risk'],
            'data_sufficiency': self.failure['data_sufficiency'],
            'feature_relevance': self.failure['feature_relevance'],
            'critical_warnings': self.failure['critical_warnings'],
            'preventive_actions': self.failure['preventive_actions']
        }
    
    def _generate_roadmap(self) -> Dict[str, Any]:
        """Generate implementation roadmap"""
        phases = []
        
        # Phase 1: Data Preparation
        phase1_tasks = ['Handle missing values', 'Remove outliers', 'Encode categorical features']
        if self.data_intel['detailed_analysis']['class_imbalance']['severity'] in ['warning', 'critical']:
            phase1_tasks.append('Address class imbalance')
        
        phases.append({
            'phase': 1,
            'name': 'Data Preparation',
            'duration': '1-2 weeks',
            'tasks': phase1_tasks,
            'deliverable': 'Clean, preprocessed dataset'
        })
        
        # Phase 2: Feature Engineering
        phases.append({
            'phase': 2,
            'name': 'Feature Engineering',
            'duration': '1-2 weeks',
            'tasks': ['Feature selection', 'Feature scaling', 'Dimensionality reduction (if needed)'],
            'deliverable': 'Optimized feature set'
        })
        
        # Phase 3: Model Development
        best_models = [m['name'] for m in self.model['algorithm_recommendations'][:3]]
        phases.append({
            'phase': 3,
            'name': 'Model Development',
            'duration': '2-3 weeks',
            'tasks': [f'Train {m}' for m in best_models] + ['Hyperparameter tuning', 'Cross-validation'],
            'deliverable': 'Trained models with performance metrics'
        })
        
        # Phase 4: Evaluation & Deployment
        phases.append({
            'phase': 4,
            'name': 'Evaluation & Deployment',
            'duration': '1-2 weeks',
            'tasks': ['Model comparison', 'Final evaluation', 'Deploy best model', 'Monitor performance'],
            'deliverable': 'Production-ready model'
        })
        
        return {
            'phases': phases,
            'total_estimated_duration': '5-9 weeks',
            'critical_path': 'Data quality issues must be resolved before modeling'
        }
    
    def _define_success_metrics(self) -> Dict[str, Any]:
        """Define project success metrics"""
        problem_type = self.problem['problem_type']
        
        if problem_type == 'classification':
            primary_metric = 'F1-Score'
            target_value = 0.85
            minimum_acceptable = 0.70
        else:
            primary_metric = 'R² Score'
            target_value = 0.80
            minimum_acceptable = 0.60
        
        return {
            'primary_metric': primary_metric,
            'target_value': target_value,
            'minimum_acceptable': minimum_acceptable,
            'secondary_metrics': [m['name'] for m in self.model['evaluation_metrics'][1:3]],
            'business_metrics': ['Model latency < 100ms', 'Prediction accuracy in production'],
            'monitoring_requirements': ['Track metric drift', 'Monitor data quality', 'Log predictions']
        }
    
    def export_json(self) -> str:
        """Export report as JSON"""
        report = self.generate_report()
        return json.dumps(report, indent=2)
    
    def export_summary(self) -> str:
        """Export executive summary as text"""
        report = self.generate_report()
        summary = report['executive_summary']
        
        text = f"""
=== ML SYSTEM ARCHITECTURE REPORT ===
Generated: {report['metadata']['generated_at']}

PROJECT VIABILITY: {summary['project_viability']}
Data Health Score: {summary['data_health_score']}/100
Overall Risk Score: {summary['overall_risk_score']}/100

Problem Type: {summary['problem_type']}
Recommended Model: {summary['recommended_model']}
Estimated Complexity: {summary['estimated_complexity']}

KEY RECOMMENDATION:
{summary['key_recommendation']}

Critical Actions Required: {summary['critical_actions']}
"""
        return text
