"""
AI Overfitting Agent - Powered by OpenAI
Intelligent agent that analyzes model performance and provides
personalized recommendations to reduce overfitting
"""

import os
import json
from typing import Dict, Any, List
from openai import OpenAI


class AIOverfittingAgent:
    """
    AI-powered agent that uses GPT to analyze overfitting and provide solutions
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
    
    def analyze_overfitting(self, model_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze model results and provide AI-powered recommendations
        
        Args:
            model_results: Training results including train/test scores
            dataset_info: Information about the dataset
            
        Returns:
            AI-generated analysis and recommendations
        """
        if not self.api_key:
            return {
                'error': 'OpenAI API key not provided',
                'message': 'Please set OPENAI_API_KEY environment variable or provide API key'
            }
        
        # Calculate overfitting metrics
        overfitting_analysis = self._calculate_overfitting_metrics(model_results)
        
        # Generate AI prompt
        prompt = self._generate_analysis_prompt(model_results, dataset_info, overfitting_analysis)
        
        try:
            # Call OpenAI API with new client
            if not self.client:
                raise Exception("OpenAI client not initialized. Check your API key.")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini (cheaper and faster than gpt-4)
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert Machine Learning engineer specializing in model optimization and overfitting reduction. 
                        Analyze the provided model performance data and give specific, actionable recommendations.
                        Be concise but thorough. Provide code examples when relevant."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                'success': True,
                'overfitting_detected': overfitting_analysis['overfitting_detected'],
                'overfitting_severity': overfitting_analysis['severity'],
                'overfitting_score': overfitting_analysis['overfitting_score'],
                'metrics': overfitting_analysis,
                'ai_analysis': ai_response,
                'recommendations': self._parse_recommendations(ai_response),
                'quick_fixes': self._generate_quick_fixes(overfitting_analysis)
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"OpenAI API Error: {error_trace}")
            
            # Provide more specific error messages
            error_msg = str(e)
            if 'api_key' in error_msg.lower() or 'authentication' in error_msg.lower():
                message = 'Invalid API key. Please check your OpenAI API key.'
            elif 'rate_limit' in error_msg.lower():
                message = 'Rate limit exceeded. Please wait a moment and try again.'
            elif 'insufficient_quota' in error_msg.lower():
                message = 'Insufficient API credits. Please add credits to your OpenAI account.'
            elif 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
                message = 'Connection error. Please check your internet connection.'
            else:
                message = f'Failed to get AI analysis: {error_msg}'
            
            return {
                'error': str(e),
                'message': message,
                'fallback_recommendations': self._get_fallback_recommendations(overfitting_analysis)
            }
    
    def _calculate_overfitting_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overfitting metrics from model results"""
        
        # Get train and test scores
        if 'train_accuracy' in model_results and 'accuracy' in model_results:
            train_score = model_results['train_accuracy']
            test_score = model_results['accuracy']
        elif 'train_r2_score' in model_results and 'r2_score' in model_results:
            train_score = model_results['train_r2_score']
            test_score = model_results['r2_score']
        else:
            return {
                'overfitting_detected': False,
                'severity': 'unknown',
                'overfitting_score': 0,
                'message': 'Insufficient data to calculate overfitting'
            }
        
        # Calculate overfitting gap
        overfitting_gap = train_score - test_score
        
        # Determine severity
        if overfitting_gap > 0.15:
            severity = 'critical'
            overfitting_detected = True
        elif overfitting_gap > 0.10:
            severity = 'high'
            overfitting_detected = True
        elif overfitting_gap > 0.05:
            severity = 'moderate'
            overfitting_detected = True
        elif overfitting_gap > 0.02:
            severity = 'low'
            overfitting_detected = True
        else:
            severity = 'none'
            overfitting_detected = False
        
        # Calculate overfitting score (0-100)
        overfitting_score = min(100, int(overfitting_gap * 500))
        
        return {
            'overfitting_detected': overfitting_detected,
            'severity': severity,
            'overfitting_score': overfitting_score,
            'train_score': float(train_score),
            'test_score': float(test_score),
            'overfitting_gap': float(overfitting_gap),
            'train_test_ratio': float(train_score / test_score) if test_score > 0 else 0
        }
    
    def _generate_analysis_prompt(self, model_results: Dict, dataset_info: Dict, overfitting_analysis: Dict) -> str:
        """Generate prompt for AI analysis"""
        
        prompt = f"""
Analyze this machine learning model for overfitting:

MODEL PERFORMANCE:
- Model Type: {model_results.get('model_type', 'Unknown')}
- Training Score: {overfitting_analysis.get('train_score', 'N/A'):.4f}
- Test Score: {overfitting_analysis.get('test_score', 'N/A'):.4f}
- Overfitting Gap: {overfitting_analysis.get('overfitting_gap', 0):.4f}
- Severity: {overfitting_analysis.get('severity', 'unknown')}

DATASET INFO:
- Training Size: {model_results.get('train_size', 'Unknown')}
- Test Size: {model_results.get('test_size', 'Unknown')}
- Number of Features: {len(model_results.get('features', []))}

CROSS-VALIDATION:
- CV Mean Score: {model_results.get('cv_mean', 'N/A')}
- CV Std Dev: {model_results.get('cv_std', 'N/A')}

Please provide:
1. **Overfitting Analysis**: Is the model overfitting? How severe is it?
2. **Root Causes**: What's licausing the overfitting?
3. **Specific Recommendations**: Give 3-5 actionable step reduce overfitting
4. **Code Examples**: Prlearn code snippets for the top 2 recommens
5. **Expected Impact*te howh recommendation will help

Be specific and practical. Focus on solutions that can be implemented immediately.
"""
        return prompt
    
    def _parse_recommendations(self, ai_response: str) -> List[Dict[str, str]]:
        """Parse AI response into structured recommendations"""
        recommendations = []
        
        # Simple parsing - look for numbered items
        lines = ai_response.split('\n')
        current_rec = None
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {'text': line, 'priority': 'high'}
            elif current_rec and line:
                current_rec['text'] += ' ' + line
        
        if current_rec:
            recommendations.append(current_rec)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_quick_fixes(self, overfitting_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate quick fix suggestions based on overfitting severity"""
        severity = overfitting_analysis.get('severity', 'none')
        
        quick_fixes = []
        
        if severity in ['critical', 'high']:
            quick_fixes.extend([
                {
                    'title': 'Add Regularization',
                    'description': 'Apply L1/L2 regularization to penalize complex models',
                    'code': "model = RandomForestClassifier(max_depth=10, min_samples_split=10)",
                    'impact': 'high',
                    'difficulty': 'easy'
                },
                {
                    'title': 'Increase Training Data',
                    'description': 'Use data augmentation or collect more samples',
                    'code': "# Use SMOTE for synthetic samples\nfrom imblearn.over_sampling import SMOTE\nsmote = SMOTE()\nX_train, y_train = smote.fit_resample(X_train, y_train)",
                    'impact': 'high',
                    'difficulty': 'medium'
                },
                {
                    'title': 'Reduce Model Complexity',
                    'description': 'Use simpler model or reduce parameters',
                    'code': "# Reduce tree depth\nmodel = RandomForestClassifier(max_depth=5, n_estimators=50)",
                    'impact': 'high',
                    'difficulty': 'easy'
                }
            ])
        
        if severity in ['moderate', 'low']:
            quick_fixes.extend([
                {
                    'title': 'Apply Dropout',
                    'description': 'Add dropout layers to neural networks',
                    'code': "model.add(Dropout(0.3))",
                    'impact': 'medium',
                    'difficulty': 'easy'
                },
                {
                    'title': 'Early Stopping',
                    'description': 'Stop training when validation performance plateaus',
                    'code': "from sklearn.model_selection import EarlyStopping\nearly_stop = EarlyStopping(patience=10)",
                    'impact': 'medium',
                    'difficulty': 'easy'
                }
            ])
        
        quick_fixes.append({
            'title': 'Cross-Validation',
            'description': 'Use k-fold cross-validation for robust evaluation',
            'code': "from sklearn.model_selection import cross_val_score\nscores = cross_val_score(model, X, y, cv=5)",
            'impact': 'medium',
            'difficulty': 'easy'
        })
        
        return quick_fixes
    
    def _get_fallback_recommendations(self, overfitting_analysis: Dict) -> List[str]:
        """Provide fallback recommendations if AI fails"""
        severity = overfitting_analysis.get('severity', 'none')
        
        if severity == 'critical':
            return [
                "CRITICAL: Severe overfitting detected. Model is memorizing training data.",
                "1. Reduce model complexity (use simpler model or fewer parameters)",
                "2. Add strong regularization (L1/L2 with high penalty)",
                "3. Collect more training data or use data augmentation",
                "4. Apply feature selection to remove noisy features",
                "5. Use ensemble methods with bagging to reduce variance"
            ]
        elif severity == 'high':
            return [
                "HIGH: Significant overfitting detected.",
                "1. Add regularization to your model",
                "2. Reduce model complexity (max_depth, n_estimators)",
                "3. Use cross-validation for better evaluation",
                "4. Apply dropout if using neural networks",
                "5. Consider using more training data"
            ]
        elif severity == 'moderate':
            return [
                "MODERATE: Some overfitting detected.",
                "1. Fine-tune regularization parameters",
                "2. Use early stopping during training",
                "3. Apply cross-validation",
                "4. Consider ensemble methods"
            ]
        else:
            return [
                "Model performance looks good!",
                "Minor overfitting is normal and acceptable.",
                "Continue monitoring with cross-validation."
            ]
    
    def get_optimization_plan(self, model_results: Dict, dataset_info: Dict) -> Dict[str, Any]:
        """
        Generate a complete optimization plan to reduce overfitting
        """
        analysis = self.analyze_overfitting(model_results, dataset_info)
        
        if 'error' in analysis:
            return analysis
        
        # Create step-by-step plan
        plan = {
            'current_status': {
                'overfitting_detected': analysis['overfitting_detected'],
                'severity': analysis['overfitting_severity'],
                'score': analysis['overfitting_score']
            },
            'immediate_actions': analysis['quick_fixes'][:3],
            'detailed_recommendations': analysis.get('recommendations', []),
            'ai_insights': analysis.get('ai_analysis', ''),
            'implementation_order': [
                'Start with quick fixes (regularization, complexity reduction)',
                'Implement cross-validation for robust evaluation',
                'Consider data augmentation if needed',
                'Fine-tune hyperparameters',
                'Monitor performance on validation set'
            ],
            'expected_improvement': self._estimate_improvement(analysis['overfitting_score'])
        }
        
        return plan
    
    def _estimate_improvement(self, overfitting_score: int) -> str:
        """Estimate expected improvement"""
        if overfitting_score > 70:
            return "High - Expect 10-20% improvement in test performance"
        elif overfitting_score > 40:
            return "Medium - Expect 5-10% improvement in test performance"
        elif overfitting_score > 20:
            return "Low - Expect 2-5% improvement in test performance"
        else:
            return "Minimal - Model is already well-generalized"
