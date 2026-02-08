"""
CrewAI-based Overfitting Agent
Uses CrewAI framework for structured AI agent workflows
"""

import os
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


class CrewAIOverfittingAgent:
    """
    CrewAI-powered agent for overfitting analysis
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Set environment variable for LangChain/CrewAI
        if self.api_key:
            os.environ['OPENAI_API_KEY'] = self.api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=self.api_key  # Use openai_api_key parameter
        )
        
        # Create specialized agents
        self.diagnostic_agent = self._create_diagnostic_agent()
        self.solution_agent = self._create_solution_agent()
        self.code_agent = self._create_code_agent()
    
    def _create_diagnostic_agent(self) -> Agent:
        """Create agent specialized in diagnosing overfitting"""
        return Agent(
            role='ML Diagnostics Expert',
            goal='Analyze model performance metrics and identify overfitting issues',
            backstory="""You are a senior ML engineer with 10+ years of experience 
            in model optimization. You excel at identifying overfitting patterns 
            and understanding their root causes.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_solution_agent(self) -> Agent:
        """Create agent specialized in recommending solutions"""
        return Agent(
            role='ML Solutions Architect',
            goal='Provide actionable recommendations to reduce overfitting',
            backstory="""You are an expert in ML best practices and have successfully 
            optimized hundreds of models. You know exactly which techniques work 
            for different scenarios.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_code_agent(self) -> Agent:
        """Create agent specialized in generating code"""
        return Agent(
            role='ML Code Generator',
            goal='Generate production-ready code examples for implementing solutions',
            backstory="""You are a skilled ML engineer who writes clean, efficient 
            code. You provide practical sklearn and Python examples that can be 
            implemented immediately.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def analyze_overfitting(self, model_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overfitting using CrewAI agents
        """
        try:
            # Calculate metrics
            metrics = self._calculate_metrics(model_results)
            
            # Create tasks for agents
            diagnostic_task = Task(
                description=f"""Analyze this ML model for overfitting:
                
                Model: {model_results.get('model_type', 'Unknown')}
                Training Score: {metrics['train_score']:.4f}
                Test Score: {metrics['test_score']:.4f}
                Overfitting Gap: {metrics['gap']:.4f}
                
                Dataset:
                - Training samples: {model_results.get('train_size', 'Unknown')}
                - Test samples: {model_results.get('test_size', 'Unknown')}
                - Features: {len(model_results.get('features', []))}
                
                Provide:
                1. Severity assessment (none/low/moderate/high/critical)
                2. Root cause analysis
                3. Impact on model performance
                """,
                agent=self.diagnostic_agent,
                expected_output="Detailed diagnostic report with severity and root causes"
            )
            
            solution_task = Task(
                description=f"""Based on the overfitting analysis, provide solutions:
                
                Context: {metrics['severity']} overfitting detected
                Gap: {metrics['gap']:.4f}
                
                Provide:
                1. Top 5 actionable recommendations
                2. Priority order (what to try first)
                3. Expected impact of each solution
                4. Implementation difficulty
                """,
                agent=self.solution_agent,
                expected_output="Prioritized list of solutions with impact estimates"
            )
            
            code_task = Task(
                description=f"""Generate code examples for the top 3 solutions:
                
                Model: {model_results.get('model_type', 'Unknown')}
                Problem: {metrics['severity']} overfitting
                
                Provide:
                1. Code snippets using sklearn
                2. Before/after comparisons
                3. Parameter explanations
                """,
                agent=self.code_agent,
                expected_output="Production-ready code examples with explanations"
            )
            
            # Create crew
            crew = Crew(
                agents=[self.diagnostic_agent, self.solution_agent, self.code_agent],
                tasks=[diagnostic_task, solution_task, code_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute crew
            result = crew.kickoff()
            
            # Parse and structure results
            return {
                'success': True,
                'overfitting_detected': metrics['detected'],
                'overfitting_severity': metrics['severity'],
                'overfitting_score': metrics['score'],
                'metrics': {
                    'train_score': metrics['train_score'],
                    'test_score': metrics['test_score'],
                    'overfitting_gap': metrics['gap']
                },
                'ai_analysis': str(result),
                'recommendations': self._parse_recommendations(str(result)),
                'quick_fixes': self._generate_quick_fixes(metrics)
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"CrewAI Error: {error_trace}")
            
            # Check for specific API key errors
            error_str = str(e).lower()
            if 'api key' in error_str or '401' in error_str or 'authentication' in error_str:
                message = 'Invalid OpenAI API key. Please check your key and try again.'
            elif 'rate_limit' in error_str or '429' in error_str:
                message = 'Rate limit exceeded. Please wait a moment and try again.'
            elif 'quota' in error_str or 'insufficient' in error_str:
                message = 'Insufficient API credits. Please add credits to your OpenAI account.'
            else:
                message = f'CrewAI analysis failed: {str(e)}'
            
            return {
                'error': str(e),
                'message': message,
                'fallback_recommendations': self._get_fallback_recommendations(
                    self._calculate_metrics(model_results)
                )
            }
    
    def _calculate_metrics(self, model_results: Dict) -> Dict:
        """Calculate overfitting metrics"""
        # Get scores
        if 'train_accuracy' in model_results:
            train_score = model_results['train_accuracy']
            test_score = model_results['accuracy']
        elif 'train_r2_score' in model_results:
            train_score = model_results['train_r2_score']
            test_score = model_results['r2_score']
        else:
            train_score = 0.8
            test_score = 0.75
        
        gap = train_score - test_score
        
        # Determine severity
        if gap > 0.15:
            severity = 'critical'
        elif gap > 0.10:
            severity = 'high'
        elif gap > 0.05:
            severity = 'moderate'
        elif gap > 0.02:
            severity = 'low'
        else:
            severity = 'none'
        
        score = min(100, int(gap * 500))
        detected = gap > 0.02
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'gap': gap,
            'severity': severity,
            'score': score,
            'detected': detected
        }
    
    def _parse_recommendations(self, result: str) -> List[Dict]:
        """Parse recommendations from crew result"""
        recommendations = []
        lines = result.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                recommendations.append({
                    'text': line,
                    'priority': 'high' if i < 3 else 'medium'
                })
        
        return recommendations[:5]
    
    def _generate_quick_fixes(self, metrics: Dict) -> List[Dict]:
        """Generate quick fix suggestions"""
        severity = metrics['severity']
        fixes = []
        
        if severity in ['critical', 'high']:
            fixes.extend([
                {
                    'title': 'Add Regularization',
                    'description': 'Apply L1/L2 regularization to penalize complexity',
                    'code': 'model = RandomForestClassifier(max_depth=10, min_samples_split=10)',
                    'impact': 'high',
                    'difficulty': 'easy'
                },
                {
                    'title': 'Reduce Model Complexity',
                    'description': 'Use simpler model or reduce parameters',
                    'code': 'model = RandomForestClassifier(max_depth=5, n_estimators=50)',
                    'impact': 'high',
                    'difficulty': 'easy'
                },
                {
                    'title': 'Increase Training Data',
                    'description': 'Use data augmentation or collect more samples',
                    'code': 'from imblearn.over_sampling import SMOTE\nsmote = SMOTE()\nX_train, y_train = smote.fit_resample(X_train, y_train)',
                    'impact': 'high',
                    'difficulty': 'medium'
                }
            ])
        
        fixes.append({
            'title': 'Cross-Validation',
            'description': 'Use k-fold cross-validation for robust evaluation',
            'code': 'from sklearn.model_selection import cross_val_score\nscores = cross_val_score(model, X, y, cv=5)',
            'impact': 'medium',
            'difficulty': 'easy'
        })
        
        return fixes
    
    def _get_fallback_recommendations(self, metrics: Dict) -> List[str]:
        """Fallback recommendations if CrewAI fails"""
        severity = metrics['severity']
        
        if severity == 'critical':
            return [
                "CRITICAL: Severe overfitting detected",
                "1. Reduce model complexity immediately",
                "2. Add strong regularization",
                "3. Collect more training data",
                "4. Use cross-validation for evaluation"
            ]
        elif severity == 'high':
            return [
                "HIGH: Significant overfitting detected",
                "1. Add regularization to your model",
                "2. Reduce model complexity",
                "3. Use cross-validation",
                "4. Consider ensemble methods"
            ]
        else:
            return [
                "Model performance is acceptable",
                "Minor overfitting is normal",
                "Continue monitoring with cross-validation"
            ]
