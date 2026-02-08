# Script to complete ISD module files

# Complete model_architect.py
model_architect_code = '''
            # Random Forest
            rf_score = 85
            if n_samples > 1000: rf_score += 10
            algorithms.append({
                'name': 'Random Forest', 'family': 'Ensemble', 'score': min(100, rf_score),
                'reasoning': ['Non-linear', 'Robust', 'Feature importance'],
                'when_to_use': 'Complex patterns', 'limitations': ['Less interpretable'],
                'sklearn_class': 'RandomForestClassifier', 'key_hyperparameters': ['n_estimators', 'max_depth']
            })
            
            # XGBoost
            xgb_score = 90
            if n_samples > 5000: xgb_score += 10
            algorithms.append({
                'name': 'XGBoost', 'family': 'Boosting', 'score': min(100, xgb_score),
                'reasoning': ['SOTA performance', 'Regularization', 'Fast'],
                'when_to_use': 'Max accuracy', 'limitations': ['Many hyperparams'],
                'sklearn_class': 'XGBClassifier', 'key_hyperparameters': ['learning_rate', 'max_depth']
            })
        else:
            algorithms.append({'name': 'Linear Regression', 'family': 'Linear', 'score': 70,
                'reasoning': ['Simple', 'Fast'], 'sklearn_class': 'LinearRegression', 'key_hyperparameters': []})
            algorithms.append({'name': 'Random Forest Regressor', 'family': 'Ensemble', 'score': 85,
                'reasoning': ['Non-linear', 'Robust'], 'sklearn_class': 'RandomForestRegressor', 'key_hyperparameters': ['n_estimators']})
        
        algorithms.sort(key=lambda x: x['score'], reverse=True)
        return algorithms
    
    def _recommend_loss_function(self):
        if self.problem['problem_type'] == 'classification':
            return {'name': 'Cross-Entropy', 'reason': 'Standard for classification'}
        return {'name': 'MSE', 'reason': 'Standard for regression'}
    
    def _recommend_metrics(self):
        if self.problem['problem_type'] == 'classification':
            return [{'name': 'Accuracy'}, {'name': 'F1-Score'}, {'name': 'ROC-AUC'}]
        return [{'name': 'R²'}, {'name': 'RMSE'}, {'name': 'MAE'}]
    
    def _recommend_validation_strategy(self):
        n = self.problem['task_complexity']['n_samples']
        if n < 100: return {'strategy': 'LOOCV', 'n_splits': n}
        return {'strategy': 'StratifiedKFold', 'n_splits': 5}
    
    def _recommend_data_split(self):
        return {'train_size': 0.8, 'test_size': 0.2, 'validation_size': 0.1}
    
    def _recommend_preprocessing(self):
        steps = [{'step': 'Scaling', 'method': 'StandardScaler()'}]
        if self.data['detailed_analysis']['missing_values']['overall_missing_rate'] > 0:
            steps.insert(0, {'step': 'Imputation', 'method': 'SimpleImputer()'})
        return steps
    
    def _recommend_hyperparameters(self, algorithms):
        if not algorithms: return []
        return [{'parameter': p, 'priority': 'high'} for p in algorithms[0].get('key_hyperparameters', [])]
    
    def _recommend_ensemble(self):
        if self.problem['task_complexity']['n_samples'] > 5000:
            return {'recommended': True, 'strategy': 'Stacking'}
        return {'recommended': False}
'''

with open('isd_model_architect.py', 'a') as f:
    f.write(model_architect_code)

print("Model Architect module completed")
