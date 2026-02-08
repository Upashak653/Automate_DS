"""
Flask Backend for ML Visualizer
Uses actual sklearn, seaborn, matplotlib for analysis and visualization
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from pandas_profiling import ProfileReport
    PANDAS_PROFILING_AVAILABLE = True
except ImportError:
    try:
        from ydata_profiling import ProfileReport
        PANDAS_PROFILING_AVAILABLE = True
    except ImportError:
        PANDAS_PROFILING_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Import and register ISD routes
try:
    from isd_api import register_isd_routes
    register_isd_routes(app)
    print("✅ ISD routes registered successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not register ISD routes: {e}")

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

def safe_stratify(y, is_classification, min_samples=2):
    """
    Safely determine if stratification should be used
    Returns y for stratification or None if not safe
    """
    if not is_classification:
        return None
    
    # Check if all classes have at least min_samples
    unique, counts = np.unique(y, return_counts=True)
    if np.min(counts) < min_samples:
        print(f"Warning: Some classes have fewer than {min_samples} samples. Disabling stratification.")
        return None
    
    return y

def get_cv_strategy(y, is_classification, n_splits=5):
    """
    Get appropriate cross-validation strategy
    Returns cv object or integer
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    if not is_classification:
        return n_splits
    
    # Check if stratification is safe
    unique, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    
    if min_count < n_splits:
        # Use regular KFold with adjusted splits
        adjusted_splits = max(2, min(n_splits, min_count))
        print(f"Warning: Some classes have fewer than {n_splits} samples. Using {adjusted_splits}-fold KFold instead of StratifiedKFold.")
        return KFold(n_splits=adjusted_splits, shuffle=True, random_state=42)
    
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def read_csv_safe(file):
    """
    Safely read CSV file with multiple fallback strategies
    Handles malformed CSV files with inconsistent columns
    """
    try:
        # Try standard read
        return pd.read_csv(file)
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}. Trying with on_bad_lines='skip'")
        file.seek(0)
        try:
            return pd.read_csv(file, on_bad_lines='skip')
        except:
            pass
    except UnicodeDecodeError:
        print("UnicodeDecodeError. Trying with latin-1 encoding")
        file.seek(0)
        try:
            return pd.read_csv(file, encoding='latin-1')
        except:
            pass
    
    # Last resort: try with multiple options
    file.seek(0)
    try:
        return pd.read_csv(
            file, 
            on_bad_lines='skip',
            encoding='latin-1',
            skipinitialspace=True,
            quoting=1  # QUOTE_ALL
        )
    except Exception as e:
        # If all else fails, try to detect delimiter
        file.seek(0)
        return pd.read_csv(
            file,
            sep=None,
            engine='python',
            on_bad_lines='skip',
            encoding='latin-1'
        )

def encode_plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Main endpoint for data analysis"""
    try:
        # Get uploaded file
        file = request.files['file']
        df = read_csv_safe(file)
        
        # Basic analysis
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'describe': df.describe().to_dict(),
            'head': df.head().to_dict('records')
        }
        
        # Detect problem type
        target_col = df.columns[-1]
        is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 20
        
        analysis['problem_type'] = 'Classification' if is_classification else 'Regression'
        analysis['target_column'] = target_col
        
        # Column analysis
        column_analysis = []
        for col in df.columns:
            col_info = {
                'name': col,
                'type': 'categorical' if df[col].dtype == 'object' else 'numeric',
                'unique_count': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percent': float(df[col].isnull().sum() / len(df) * 100)
            }
            
            if df[col].dtype != 'object':
                col_info['mean'] = float(df[col].mean())
                col_info['median'] = float(df[col].median())
                col_info['std'] = float(df[col].std())
                col_info['min'] = float(df[col].min())
                col_info['max'] = float(df[col].max())
            
            column_analysis.append(col_info)
        
        analysis['column_analysis'] = column_analysis
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualize', methods=['POST'])
def create_visualizations():
    """Generate actual seaborn/matplotlib plots"""
    try:
        file = request.files['file']
        df = read_csv_safe(file)
        
        plots = {}
        
        # 1. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
            plots['correlation_heatmap'] = encode_plot_to_base64(fig)
        
        # 2. Distribution plots for numeric features
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols[:6]) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(numeric_cols[:6]):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='steelblue')
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            # Hide empty subplots
            for idx in range(len(numeric_cols[:6]), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plots['distributions'] = encode_plot_to_base64(fig)
        
        # 3. Box plots for outlier detection
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols[:6]) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(numeric_cols[:6]):
                sns.boxplot(y=df[col].dropna(), ax=axes[idx], color='lightcoral')
                axes[idx].set_title(f'Box Plot - {col}', fontweight='bold')
                axes[idx].set_ylabel(col)
            
            for idx in range(len(numeric_cols[:6]), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plots['boxplots'] = encode_plot_to_base64(fig)
        
        # 4. Pairplot (for small datasets)
        if len(df) < 1000 and len(numeric_cols) <= 5 and len(numeric_cols) >= 2:
            fig = sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
            fig.fig.suptitle('Pairplot of Numeric Features', y=1.02, fontsize=16, fontweight='bold')
            plots['pairplot'] = encode_plot_to_base64(fig.fig)
        
        # 5. Missing data visualization
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            sns.barplot(x=missing_data.values, y=missing_data.index, ax=ax, palette='Reds_r')
            ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Missing Values')
            ax.set_ylabel('Columns')
            plots['missing_data'] = encode_plot_to_base64(fig)
        
        # 6. Target distribution
        target_col = df.columns[-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
            value_counts = df[target_col].value_counts()
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax, palette='viridis')
            ax.set_title(f'Target Distribution - {target_col}', fontsize=16, fontweight='bold')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
        else:
            sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color='green')
            ax.set_title(f'Target Distribution - {target_col}', fontsize=16, fontweight='bold')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plots['target_distribution'] = encode_plot_to_base64(fig)
        
        return jsonify(plots)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train actual sklearn model and return metrics"""
    try:
        file = request.files['file']
        model_type = request.form.get('model_type', 'auto')
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        # Encode target if classification
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Import advanced models
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        # Try to import XGBoost (optional)
        try:
            from xgboost import XGBClassifier, XGBRegressor
            xgboost_available = True
        except ImportError:
            xgboost_available = False
        
        # Try to import LightGBM (optional)
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            lightgbm_available = True
        except ImportError:
            lightgbm_available = False
        
        # Train model based on type
        if is_classification:
            models = {
                'auto': RandomForestClassifier(n_estimators=100, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic': LogisticRegression(max_iter=1000, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', random_state=42, probability=True),
                'knn': KNeighborsClassifier(n_neighbors=5),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(random_state=42)
            }
            
            if xgboost_available:
                models['xgboost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            
            if lightgbm_available:
                models['lightgbm'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        else:
            models = {
                'auto': RandomForestRegressor(n_estimators=100, random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear': LinearRegression(),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'adaboost': AdaBoostRegressor(n_estimators=100, random_state=42),
                'svr': SVR(kernel='rbf'),
                'knn': KNeighborsRegressor(n_neighbors=5),
                'decision_tree': DecisionTreeRegressor(random_state=42)
            }
            
            if xgboost_available:
                models['xgboost'] = XGBRegressor(n_estimators=100, random_state=42)
            
            if lightgbm_available:
                models['lightgbm'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        # Get the selected model
        model = models.get(model_type, models['auto'])
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        results = {
            'model_type': type(model).__name__,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': X.columns.tolist()
        }
        
        if is_classification:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            # Basic metrics
            results['accuracy'] = float(model.score(X_test_scaled, y_test))
            results['train_accuracy'] = float(model.score(X_train_scaled, y_train))
            
            # Detailed metrics
            n_classes = len(np.unique(y))
            average_method = 'binary' if n_classes == 2 else 'weighted'
            
            results['precision'] = float(precision_score(y_test, y_pred, average=average_method, zero_division=0))
            results['recall'] = float(recall_score(y_test, y_pred, average=average_method, zero_division=0))
            results['f1_score'] = float(f1_score(y_test, y_pred, average=average_method, zero_division=0))
            
            # ROC-AUC (for binary classification or if model has predict_proba)
            if n_classes == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    results['roc_auc'] = float(roc_auc_score(y_test, y_proba))
                except:
                    results['roc_auc'] = None
            
            # Classification report
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Evaluation matrix explanation
            results['evaluation_metrics'] = {
                'accuracy': {
                    'value': results['accuracy'],
                    'description': 'Overall correctness of predictions',
                    'interpretation': 'Higher is better (0-1 scale)',
                    'best_for': 'Balanced datasets'
                },
                'precision': {
                    'value': results['precision'],
                    'description': 'Of all positive predictions, how many were correct',
                    'interpretation': 'Higher is better (0-1 scale)',
                    'best_for': 'When false positives are costly'
                },
                'recall': {
                    'value': results['recall'],
                    'description': 'Of all actual positives, how many were found',
                    'interpretation': 'Higher is better (0-1 scale)',
                    'best_for': 'When false negatives are costly'
                },
                'f1_score': {
                    'value': results['f1_score'],
                    'description': 'Harmonic mean of precision and recall',
                    'interpretation': 'Higher is better (0-1 scale)',
                    'best_for': 'Imbalanced datasets'
                }
            }
            
            if results.get('roc_auc'):
                results['evaluation_metrics']['roc_auc'] = {
                    'value': results['roc_auc'],
                    'description': 'Area under ROC curve',
                    'interpretation': 'Higher is better (0.5-1.0 scale)',
                    'best_for': 'Binary classification performance'
                }
            
            # Confusion matrix plot
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            results['confusion_matrix_plot'] = encode_plot_to_base64(fig)
        else:
            from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
            
            # Basic metrics
            results['r2_score'] = float(r2_score(y_test, y_pred))
            results['train_r2_score'] = float(r2_score(y_train, model.predict(X_train_scaled)))
            results['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['mae'] = float(mean_absolute_error(y_test, y_pred))
            results['mse'] = float(mean_squared_error(y_test, y_pred))
            
            # Additional metrics
            try:
                results['mape'] = float(mean_absolute_percentage_error(y_test, y_pred))
            except:
                results['mape'] = None
            
            results['explained_variance'] = float(explained_variance_score(y_test, y_pred))
            
            # Evaluation matrix explanation
            results['evaluation_metrics'] = {
                'r2_score': {
                    'value': results['r2_score'],
                    'description': 'Proportion of variance explained by the model',
                    'interpretation': 'Higher is better (-∞ to 1, perfect=1)',
                    'best_for': 'Overall model performance'
                },
                'rmse': {
                    'value': results['rmse'],
                    'description': 'Root Mean Squared Error',
                    'interpretation': 'Lower is better (same units as target)',
                    'best_for': 'Penalizes large errors more'
                },
                'mae': {
                    'value': results['mae'],
                    'description': 'Mean Absolute Error',
                    'interpretation': 'Lower is better (same units as target)',
                    'best_for': 'Average prediction error'
                },
                'mse': {
                    'value': results['mse'],
                    'description': 'Mean Squared Error',
                    'interpretation': 'Lower is better (squared units)',
                    'best_for': 'Mathematical optimization'
                },
                'explained_variance': {
                    'value': results['explained_variance'],
                    'description': 'Explained variance score',
                    'interpretation': 'Higher is better (0-1 scale)',
                    'best_for': 'Variance captured by model'
                }
            }
            
            if results.get('mape') is not None:
                results['evaluation_metrics']['mape'] = {
                    'value': results['mape'],
                    'description': 'Mean Absolute Percentage Error',
                    'interpretation': 'Lower is better (percentage)',
                    'best_for': 'Relative error measurement'
                }
            
            # Actual vs Predicted plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title('Actual vs Predicted', fontsize=16, fontweight='bold')
            results['prediction_plot'] = encode_plot_to_base64(fig)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title('Top 10 Feature Importances', fontsize=16, fontweight='bold')
            ax.set_xlabel('Importance')
            results['feature_importance_plot'] = encode_plot_to_base64(fig)
            results['feature_importance'] = importance_df.to_dict('records')
        
        # Cross-validation
        cv_strategy = get_cv_strategy(y_train, is_classification, n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy)
        results['cv_scores'] = cv_scores.tolist()
        results['cv_mean'] = float(cv_scores.mean())
        results['cv_std'] = float(cv_scores.std())
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommend-model', methods=['POST'])
def recommend_model():
    """Recommend best model based on dataset characteristics"""
    try:
        file = request.files['file']
        df = read_csv_safe(file)
        
        # Analyze dataset
        n_rows, n_cols = df.shape
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        problem_type = 'Classification' if is_classification else 'Regression'
        
        # Calculate dataset characteristics
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        missing_percent = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100
        feature_ratio = n_cols / n_rows
        categorical_ratio = len(categorical_cols) / n_cols if n_cols > 0 else 0
        
        # Model scoring system
        models = []
        
        if is_classification:
            # Logistic Regression
            lr_score = 70
            if n_rows < 1000: lr_score += 15
            if categorical_ratio > 0.5: lr_score -= 10
            if feature_ratio > 0.1: lr_score -= 15
            if missing_percent > 10: lr_score -= 10
            
            models.append({
                'name': 'Logistic Regression',
                'score': max(0, min(100, lr_score)),
                'reason': 'Fast, interpretable, works well with small datasets',
                'pros': ['Simple', 'Interpretable', 'Fast training', 'Low memory'],
                'cons': ['Linear decision boundary', 'May underfit complex patterns'],
                'best_for': 'Small datasets with linear relationships'
            })
            
            # Random Forest
            rf_score = 80
            if n_rows > 1000: rf_score += 10
            if categorical_ratio > 0.3: rf_score += 10
            if missing_percent > 10: rf_score += 5
            if n_rows < 500: rf_score -= 10
            
            models.append({
                'name': 'Random Forest',
                'score': max(0, min(100, rf_score)),
                'reason': 'Robust, handles missing data, works with categorical features',
                'pros': ['Handles missing values', 'Feature importance', 'Non-linear', 'Robust'],
                'cons': ['Slower training', 'Larger memory', 'Less interpretable'],
                'best_for': 'Medium to large datasets with mixed feature types'
            })
            
            # Gradient Boosting
            gb_score = 75
            if n_rows > 5000: gb_score += 15
            if feature_ratio < 0.05: gb_score += 10
            if n_rows < 1000: gb_score -= 15
            
            models.append({
                'name': 'Gradient Boosting',
                'score': max(0, min(100, gb_score)),
                'reason': 'High accuracy, handles complex patterns',
                'pros': ['High accuracy', 'Handles non-linearity', 'Feature importance'],
                'cons': ['Slow training', 'Requires tuning', 'Can overfit'],
                'best_for': 'Large datasets where accuracy is critical'
            })
            
            # XGBoost
            xgb_score = 85
            if n_rows > 5000: xgb_score += 10
            if missing_percent > 10: xgb_score += 5
            if n_rows < 1000: xgb_score -= 10
            
            models.append({
                'name': 'XGBoost',
                'score': max(0, min(100, xgb_score)),
                'reason': 'State-of-the-art gradient boosting, handles missing values',
                'pros': ['Excellent accuracy', 'Fast training', 'Handles missing data', 'Built-in regularization'],
                'cons': ['Many hyperparameters', 'Can overfit', 'Requires tuning'],
                'best_for': 'Kaggle competitions and production systems'
            })
            
            # LightGBM
            lgbm_score = 83
            if n_rows > 10000: lgbm_score += 12
            if n_cols > 50: lgbm_score += 8
            if n_rows < 1000: lgbm_score -= 15
            
            models.append({
                'name': 'LightGBM',
                'score': max(0, min(100, lgbm_score)),
                'reason': 'Very fast gradient boosting, excellent for large datasets',
                'pros': ['Extremely fast', 'Low memory', 'High accuracy', 'Handles categorical'],
                'cons': ['Can overfit small datasets', 'Requires tuning'],
                'best_for': 'Very large datasets (>10K rows)'
            })
            
            # AdaBoost
            ada_score = 70
            if n_rows > 1000 and n_rows < 10000: ada_score += 10
            if n_cols < 20: ada_score += 5
            if n_rows > 50000: ada_score -= 15
            
            models.append({
                'name': 'AdaBoost',
                'score': max(0, min(100, ada_score)),
                'reason': 'Ensemble method that combines weak learners',
                'pros': ['Less prone to overfitting', 'No hyperparameter tuning needed', 'Works with weak learners'],
                'cons': ['Sensitive to noise', 'Slower than other boosting', 'Can underperform on complex data'],
                'best_for': 'Medium-sized datasets with low noise'
            })
            
            # SVM
            svm_score = 65
            if n_rows < 1000: svm_score += 10
            if n_cols < 20: svm_score += 10
            if n_rows > 10000: svm_score -= 20
            
            models.append({
                'name': 'SVM',
                'score': max(0, min(100, svm_score)),
                'reason': 'Effective in high dimensions with clear margins',
                'pros': ['Works in high dimensions', 'Memory efficient', 'Effective with clear margins'],
                'cons': ['Slow on large datasets', 'Requires scaling', 'Hard to interpret'],
                'best_for': 'Small to medium datasets with clear decision boundaries'
            })
            
        else:  # Regression
            # Linear Regression
            lr_score = 70
            if n_rows < 1000: lr_score += 10
            if feature_ratio > 0.1: lr_score -= 15
            if missing_percent > 10: lr_score -= 10
            
            models.append({
                'name': 'Linear Regression',
                'score': max(0, min(100, lr_score)),
                'reason': 'Simple, interpretable, fast for linear relationships',
                'pros': ['Simple', 'Interpretable', 'Fast', 'Low memory'],
                'cons': ['Assumes linearity', 'Sensitive to outliers'],
                'best_for': 'Linear relationships with few features'
            })
            
            # Random Forest Regressor
            rf_score = 80
            if n_rows > 1000: rf_score += 10
            if categorical_ratio > 0.3: rf_score += 10
            if missing_percent > 10: rf_score += 5
            
            models.append({
                'name': 'Random Forest Regressor',
                'score': max(0, min(100, rf_score)),
                'reason': 'Robust, handles non-linearity and missing data',
                'pros': ['Handles missing values', 'Non-linear', 'Feature importance', 'Robust'],
                'cons': ['Slower training', 'Larger memory'],
                'best_for': 'Medium to large datasets with complex patterns'
            })
            
            # Gradient Boosting Regressor
            gb_score = 75
            if n_rows > 5000: gb_score += 15
            if feature_ratio < 0.05: gb_score += 10
            
            models.append({
                'name': 'Gradient Boosting Regressor',
                'score': max(0, min(100, gb_score)),
                'reason': 'High accuracy for complex non-linear relationships',
                'pros': ['High accuracy', 'Handles non-linearity', 'Feature importance'],
                'cons': ['Slow training', 'Requires tuning', 'Can overfit'],
                'best_for': 'Large datasets where accuracy is critical'
            })
            
            # XGBoost Regressor
            xgb_score = 85
            if n_rows > 5000: xgb_score += 10
            if missing_percent > 10: xgb_score += 5
            if n_rows < 1000: xgb_score -= 10
            
            models.append({
                'name': 'XGBoost Regressor',
                'score': max(0, min(100, xgb_score)),
                'reason': 'State-of-the-art gradient boosting for regression',
                'pros': ['Excellent accuracy', 'Fast training', 'Handles missing data', 'Built-in regularization'],
                'cons': ['Many hyperparameters', 'Can overfit', 'Requires tuning'],
                'best_for': 'Production systems and competitions'
            })
            
            # LightGBM Regressor
            lgbm_score = 83
            if n_rows > 10000: lgbm_score += 12
            if n_cols > 50: lgbm_score += 8
            if n_rows < 1000: lgbm_score -= 15
            
            models.append({
                'name': 'LightGBM Regressor',
                'score': max(0, min(100, lgbm_score)),
                'reason': 'Very fast gradient boosting for large datasets',
                'pros': ['Extremely fast', 'Low memory', 'High accuracy', 'Handles categorical'],
                'cons': ['Can overfit small datasets', 'Requires tuning'],
                'best_for': 'Very large datasets (>10K rows)'
            })
            
            # AdaBoost Regressor
            ada_score = 70
            if n_rows > 1000 and n_rows < 10000: ada_score += 10
            if n_cols < 20: ada_score += 5
            if n_rows > 50000: ada_score -= 15
            
            models.append({
                'name': 'AdaBoost Regressor',
                'score': max(0, min(100, ada_score)),
                'reason': 'Ensemble method for regression tasks',
                'pros': ['Less prone to overfitting', 'Simple to use', 'Works with weak learners'],
                'cons': ['Sensitive to noise', 'Slower training', 'Can underperform on complex data'],
                'best_for': 'Medium-sized datasets with low noise'
            })
            
            # Ridge/Lasso
            ridge_score = 65
            if feature_ratio > 0.1: ridge_score += 15
            if n_cols > 50: ridge_score += 10
            
            models.append({
                'name': 'Ridge/Lasso Regression',
                'score': max(0, min(100, ridge_score)),
                'reason': 'Regularization prevents overfitting with many features',
                'pros': ['Prevents overfitting', 'Feature selection (Lasso)', 'Fast'],
                'cons': ['Assumes linearity', 'Requires scaling'],
                'best_for': 'High-dimensional data with multicollinearity'
            })
        
        # Sort by score
        models.sort(key=lambda x: x['score'], reverse=True)
        
        # Add priority labels
        for i, model in enumerate(models):
            if i == 0:
                model['priority'] = 'Best Match'
            elif model['score'] >= 75:
                model['priority'] = 'High'
            elif model['score'] >= 60:
                model['priority'] = 'Medium'
            else:
                model['priority'] = 'Low'
        
        # Dataset insights
        insights = []
        
        if n_rows < 100:
            insights.append({
                'type': 'warning',
                'message': f'Very small dataset ({n_rows} rows)',
                'recommendation': 'Use simple models and cross-validation'
            })
        elif n_rows > 10000:
            insights.append({
                'type': 'success',
                'message': f'Large dataset ({n_rows:,} rows)',
                'recommendation': 'Can use complex models like Gradient Boosting'
            })
        
        if feature_ratio > 0.1:
            insights.append({
                'type': 'warning',
                'message': f'High feature-to-sample ratio ({feature_ratio:.3f})',
                'recommendation': 'Use regularization or feature selection'
            })
        
        if missing_percent > 20:
            insights.append({
                'type': 'warning',
                'message': f'High missing data ({missing_percent:.1f}%)',
                'recommendation': 'Tree-based models handle missing values well'
            })
        
        if categorical_ratio > 0.5:
            insights.append({
                'type': 'info',
                'message': f'Many categorical features ({len(categorical_cols)}/{n_cols})',
                'recommendation': 'Tree-based models work well without encoding'
            })
        
        return jsonify({
            'problem_type': problem_type,
            'target_column': target_col,
            'dataset_stats': {
                'rows': n_rows,
                'columns': n_cols,
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'missing_percent': round(missing_percent, 2),
                'feature_ratio': round(feature_ratio, 4)
            },
            'recommended_models': models,
            'insights': insights,
            'best_model': models[0]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    Advanced data preprocessing with multiple options
    """
    try:
        file = request.files['file']
        df = read_csv_safe(file)
        
        # Get preprocessing options from request
        options = request.form.to_dict()
        
        # Store original shape
        original_shape = df.shape
        preprocessing_steps = []
        
        # 1. Remove Duplicates
        if options.get('remove_duplicates', 'false') == 'true':
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0:
                preprocessing_steps.append(f"Removed {removed} duplicate rows")
        
        # 2. Handle Missing Values
        missing_strategy = options.get('missing_strategy', 'none')
        if missing_strategy != 'none':
            missing_before = df.isnull().sum().sum()
            if missing_strategy == 'drop_rows':
                df = df.dropna()
                preprocessing_steps.append(f"Dropped rows with missing values")
            elif missing_strategy == 'drop_columns':
                threshold = float(options.get('missing_threshold', 0.5))
                cols_before = len(df.columns)
                df = df.dropna(thresh=int(len(df) * (1 - threshold)), axis=1)
                cols_removed = cols_before - len(df.columns)
                if cols_removed > 0:
                    preprocessing_steps.append(f"Dropped {cols_removed} columns with >{threshold*100}% missing")
            elif missing_strategy == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                preprocessing_steps.append("Filled missing values with mean")
            elif missing_strategy == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                preprocessing_steps.append("Filled missing values with median")
            elif missing_strategy == 'mode':
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col])
                preprocessing_steps.append("Filled missing values with mode")
        
        # 3. Remove Outliers (IQR method)
        if options.get('remove_outliers', 'false') == 'true':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            before = len(df)
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = before - len(df)
            if removed > 0:
                preprocessing_steps.append(f"Removed {removed} outlier rows (IQR method)")
        
        # 4. Encode Categorical Variables
        if options.get('encode_categorical', 'false') == 'true':
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = le.fit_transform(df[col].astype(str))
                preprocessing_steps.append(f"Label encoded {len(categorical_cols)} categorical columns")
        
        # 5. Standardization (Z-score normalization)
        if options.get('standardize', 'false') == 'true':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                preprocessing_steps.append(f"Standardized {len(numeric_cols)} numeric columns (mean=0, std=1)")
        
        # 6. Normalization (Min-Max scaling)
        if options.get('normalize', 'false') == 'true':
            from sklearn.preprocessing import MinMaxScaler
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                preprocessing_steps.append(f"Normalized {len(numeric_cols)} numeric columns (range 0-1)")
        
        # 7. Feature Selection (Remove low variance features)
        if options.get('feature_selection', 'false') == 'true':
            from sklearn.feature_selection import VarianceThreshold
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                threshold = float(options.get('variance_threshold', 0.01))
                selector = VarianceThreshold(threshold=threshold)
                cols_before = len(numeric_cols)
                selected_data = selector.fit_transform(df[numeric_cols])
                selected_cols = numeric_cols[selector.get_support()]
                df = df.drop(columns=numeric_cols)
                df[selected_cols] = selected_data
                removed = cols_before - len(selected_cols)
                if removed > 0:
                    preprocessing_steps.append(f"Removed {removed} low-variance features")
        
        # 8. PCA (Dimensionality Reduction)
        if options.get('apply_pca', 'false') == 'true':
            from sklearn.decomposition import PCA
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                n_components = options.get('pca_components', 'auto')
                if n_components == 'auto':
                    # Keep 95% variance
                    pca = PCA(n_components=0.95)
                else:
                    pca = PCA(n_components=min(int(n_components), len(numeric_cols)))
                
                pca_data = pca.fit_transform(df[numeric_cols])
                
                # Replace numeric columns with PCA components
                df = df.drop(columns=numeric_cols)
                for i in range(pca_data.shape[1]):
                    df[f'PC{i+1}'] = pca_data[:, i]
                
                variance_explained = sum(pca.explained_variance_ratio_) * 100
                preprocessing_steps.append(
                    f"Applied PCA: {len(numeric_cols)} → {pca_data.shape[1]} components "
                    f"({variance_explained:.1f}% variance explained)"
                )
        
        # Generate summary
        summary = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'rows_removed': original_shape[0] - df.shape[0],
            'columns_removed': original_shape[1] - df.shape[1],
            'preprocessing_steps': preprocessing_steps,
            'preview': df.head(10).to_dict('records'),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # Save preprocessed data to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return jsonify({
            'success': True,
            'summary': summary,
            'csv_data': csv_data,
            'message': f'Preprocessing complete: {len(preprocessing_steps)} steps applied'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare multiple models and return performance metrics with plots"""
    try:
        file = request.files['file']
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        # Encode target if classification
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Import models
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        try:
            from xgboost import XGBClassifier, XGBRegressor
            xgboost_available = True
        except ImportError:
            xgboost_available = False
        
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            lightgbm_available = True
        except ImportError:
            lightgbm_available = False
        
        # Define models to compare
        if is_classification:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
            if xgboost_available:
                models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            if lightgbm_available:
                models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf'),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
            if xgboost_available:
                models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
            if lightgbm_available:
                models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        # Train and evaluate all models
        results = []
        for name, model in models.items():
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                if is_classification:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
                    test_score = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results.append({
                        'model': name,
                        'train_score': float(train_score),
                        'test_score': float(test_score),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'overfitting': float(train_score - test_score)
                    })
                else:
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    train_score = r2_score(y_train, model.predict(X_train_scaled))
                    test_score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    results.append({
                        'model': name,
                        'train_score': float(train_score),
                        'test_score': float(test_score),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'overfitting': float(train_score - test_score)
                    })
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Test Score Comparison
        models_list = [r['model'] for r in results]
        test_scores = [r['test_score'] for r in results]
        
        axes[0, 0].barh(models_list, test_scores, color='steelblue')
        axes[0, 0].set_xlabel('Test Score', fontsize=12)
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Train vs Test (Overfitting Detection)
        train_scores = [r['train_score'] for r in results]
        x = np.arange(len(models_list))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, train_scores, width, label='Train', color='lightgreen')
        axes[0, 1].bar(x + width/2, test_scores, width, label='Test', color='lightcoral')
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Train vs Test Score (Overfitting Check)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models_list, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Overfitting Gap
        overfitting = [r['overfitting'] for r in results]
        colors = ['red' if o > 0.1 else 'orange' if o > 0.05 else 'green' for o in overfitting]
        
        axes[1, 0].barh(models_list, overfitting, color=colors)
        axes[1, 0].set_xlabel('Overfitting Gap (Train - Test)', fontsize=12)
        axes[1, 0].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(x=0.1, color='red', linestyle='--', label='High Risk')
        axes[1, 0].axvline(x=0.05, color='orange', linestyle='--', label='Medium Risk')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Plot 4: Additional Metrics
        if is_classification:
            precision_scores = [r['precision'] for r in results]
            recall_scores = [r['recall'] for r in results]
            f1_scores = [r['f1_score'] for r in results]
            
            x = np.arange(len(models_list))
            width = 0.25
            
            axes[1, 1].bar(x - width, precision_scores, width, label='Precision', color='skyblue')
            axes[1, 1].bar(x, recall_scores, width, label='Recall', color='lightgreen')
            axes[1, 1].bar(x + width, f1_scores, width, label='F1-Score', color='salmon')
            axes[1, 1].set_ylabel('Score', fontsize=12)
            axes[1, 1].set_title('Classification Metrics', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(models_list, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            rmse_scores = [r['rmse'] for r in results]
            mae_scores = [r['mae'] for r in results]
            
            x = np.arange(len(models_list))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, rmse_scores, width, label='RMSE', color='skyblue')
            axes[1, 1].bar(x + width/2, mae_scores, width, label='MAE', color='lightgreen')
            axes[1, 1].set_ylabel('Error', fontsize=12)
            axes[1, 1].set_title('Regression Error Metrics', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(models_list, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        comparison_plot = encode_plot_to_base64(fig)
        
        # Find best model
        best_model = max(results, key=lambda x: x['test_score'])
        
        return jsonify({
            'results': results,
            'comparison_plot': comparison_plot,
            'best_model': best_model,
            'problem_type': 'Classification' if is_classification else 'Regression'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/tune-hyperparameters', methods=['POST'])
def tune_hyperparameters():
    """Hyperparameter tuning with RandomizedSearchCV or Optuna"""
    try:
        file = request.files['file']
        model_name = request.form.get('model_name', 'random_forest')
        tuning_method = request.form.get('tuning_method', 'randomized')  # 'randomized' or 'optuna'
        cv_folds = int(request.form.get('cv_folds', 5))
        
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical and missing
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter distributions
        if model_name == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(random_state=42)
                param_dist = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_dist = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        elif model_name == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            if is_classification:
                model = XGBClassifier(random_state=42, eval_metric='logloss')
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                model = XGBRegressor(random_state=42)
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
        else:
            return jsonify({'error': 'Model not supported for tuning'}), 400
        
        if tuning_method == 'randomized':
            # RandomizedSearchCV
            search = RandomizedSearchCV(
                model, param_dist, n_iter=20, cv=cv_folds,
                scoring='accuracy' if is_classification else 'r2',
                random_state=42, n_jobs=-1
            )
            search.fit(X_train_scaled, y_train)
            
            best_params = search.best_params_
            best_score = float(search.best_score_)
            cv_results = search.cv_results_
            
        elif tuning_method == 'optuna' and OPTUNA_AVAILABLE:
            # Optuna optimization
            def objective(trial):
                if model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                    }
                else:  # xgboost
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'subsample': trial.suggest_float('subsample', 0.8, 1.0)
                    }
                
                if is_classification:
                    if model_name == 'random_forest':
                        clf = RandomForestClassifier(**params, random_state=42)
                    else:
                        clf = XGBClassifier(**params, random_state=42, eval_metric='logloss')
                else:
                    if model_name == 'random_forest':
                        clf = RandomForestRegressor(**params, random_state=42)
                    else:
                        clf = XGBRegressor(**params, random_state=42)
                
                cv_strategy = get_cv_strategy(y_train, is_classification, n_splits=cv_folds)
                scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_strategy,
                                       scoring='accuracy' if is_classification else 'r2')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            
            best_params = study.best_params
            best_score = float(study.best_value)
            cv_results = None
        else:
            return jsonify({'error': 'Optuna not available'}), 400
        
        # Train final model with best params
        if is_classification:
            if model_name == 'random_forest':
                final_model = RandomForestClassifier(**best_params, random_state=42)
            else:
                final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        else:
            if model_name == 'random_forest':
                final_model = RandomForestRegressor(**best_params, random_state=42)
            else:
                final_model = XGBRegressor(**best_params, random_state=42)
        
        final_model.fit(X_train_scaled, y_train)
        test_score = float(final_model.score(X_test_scaled, y_test))
        
        # Cross-validation for overfitting check
        cv_strategy = get_cv_strategy(y_train, is_classification, n_splits=cv_folds)
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=cv_strategy)
        
        return jsonify({
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_score': test_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'tuning_method': tuning_method,
            'overfitting_risk': 'High' if best_score - test_score > 0.1 else 'Low'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validate-csv', methods=['POST'])
def validate_csv():
    """Validate CSV file before processing"""
    try:
        file = request.files['file']
        
        # Try to read and get basic info
        df = read_csv_safe(file)
        
        issues = []
        warnings = []
        
        # Check for issues
        if df.shape[0] < 10:
            warnings.append('Very small dataset (< 10 rows)')
        
        if df.shape[1] < 2:
            issues.append('Need at least 2 columns (features + target)')
        
        # Check for missing values
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percent > 50:
            warnings.append(f'High missing data: {missing_percent:.1f}%')
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            issues.append('Duplicate column names detected')
        
        return jsonify({
            'valid': len(issues) == 0,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'issues': issues,
            'warnings': warnings,
            'message': 'CSV is valid and ready for analysis' if len(issues) == 0 else 'CSV has issues that need to be fixed'
        })
    
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e),
            'message': 'Failed to read CSV file. Please check the file format.'
        }), 400

@app.route('/api/reduce-overfitting', methods=['POST'])
def reduce_overfitting():
    """Apply overfitting reduction techniques and compare before/after"""
    try:
        file = request.files['file']
        method = request.form.get('method', 'cross_validation')
        cv_folds = int(request.form.get('cv_folds', 5))
        regularization_strength = float(request.form.get('regularization_strength', 0.1))
        max_depth = int(request.form.get('max_depth', 10))
        min_samples_split = int(request.form.get('min_samples_split', 10))
        ensemble_size = int(request.form.get('ensemble_size', 5))
        
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical and missing
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # BEFORE: Train baseline model (prone to overfitting)
        if is_classification:
            baseline_model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=None,  # No limit - prone to overfitting
                min_samples_split=2,
                random_state=42
            )
        else:
            baseline_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        
        baseline_model.fit(X_train_scaled, y_train)
        before_train_score = baseline_model.score(X_train_scaled, y_train)
        before_test_score = baseline_model.score(X_test_scaled, y_test)
        before_gap = before_train_score - before_test_score
        
        # AFTER: Apply overfitting reduction techniques
        applied_techniques = []
        
        if method == 'cross_validation' or method == 'all':
            # Use cross-validation for better generalization
            if is_classification:
                improved_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42
                )
            else:
                improved_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42
                )
            
            # Use cross-validation
            cv_strategy = get_cv_strategy(y_train, is_classification, n_splits=cv_folds)
            cv_scores = cross_val_score(improved_model, X_train_scaled, y_train, cv=cv_strategy)
            improved_model.fit(X_train_scaled, y_train)
            applied_techniques.append(f'{cv_folds}-fold Cross-Validation (CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})')
        
        elif method == 'regularization':
            # Apply regularization
            from sklearn.linear_model import Ridge, Lasso, RidgeClassifier, LogisticRegression
            
            if is_classification:
                improved_model = LogisticRegression(
                    C=1/regularization_strength,  # Inverse of regularization
                    max_iter=1000,
                    random_state=42
                )
            else:
                improved_model = Ridge(alpha=regularization_strength, random_state=42)
            
            improved_model.fit(X_train_scaled, y_train)
            applied_techniques.append(f'L2 Regularization (alpha={regularization_strength})')
        
        elif method == 'early_stopping':
            # Use gradient boosting with early stopping
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            
            if is_classification:
                improved_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    validation_fraction=0.2,
                    n_iter_no_change=10,
                    random_state=42
                )
            else:
                improved_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    validation_fraction=0.2,
                    n_iter_no_change=10,
                    random_state=42
                )
            
            improved_model.fit(X_train_scaled, y_train)
            applied_techniques.append('Early Stopping (stops when validation score plateaus)')
        
        elif method == 'pruning':
            # Tree pruning with constraints
            if is_classification:
                improved_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=5,
                    random_state=42
                )
            else:
                improved_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=5,
                    random_state=42
                )
            
            improved_model.fit(X_train_scaled, y_train)
            applied_techniques.append(f'Tree Pruning (max_depth={max_depth}, min_samples_split={min_samples_split})')
        
        elif method == 'feature_selection':
            # Feature selection to reduce noise
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            
            k_features = max(5, int(X.shape[1] * 0.7))  # Keep 70% of features
            if is_classification:
                selector = SelectKBest(f_classif, k=k_features)
            else:
                selector = SelectKBest(f_regression, k=k_features)
            
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            if is_classification:
                improved_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                improved_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            improved_model.fit(X_train_selected, y_train)
            after_train_score = improved_model.score(X_train_selected, y_train)
            after_test_score = improved_model.score(X_test_selected, y_test)
            after_gap = after_train_score - after_test_score
            
            applied_techniques.append(f'Feature Selection (kept {k_features}/{X.shape[1]} features)')
        
        elif method == 'ensemble':
            # Ensemble with bagging
            from sklearn.ensemble import BaggingClassifier, BaggingRegressor
            
            if is_classification:
                base_model = RandomForestClassifier(max_depth=10, random_state=42)
                improved_model = BaggingClassifier(
                    base_model,
                    n_estimators=ensemble_size,
                    max_samples=0.8,
                    max_features=0.8,
                    random_state=42
                )
            else:
                base_model = RandomForestRegressor(max_depth=10, random_state=42)
                improved_model = BaggingRegressor(
                    base_model,
                    n_estimators=ensemble_size,
                    max_samples=0.8,
                    max_features=0.8,
                    random_state=42
                )
            
            improved_model.fit(X_train_scaled, y_train)
            applied_techniques.append(f'Ensemble Bagging ({ensemble_size} models)')
        
        # Calculate after scores (if not already done in feature_selection)
        if method != 'feature_selection':
            after_train_score = improved_model.score(X_train_scaled, y_train)
            after_test_score = improved_model.score(X_test_scaled, y_test)
            after_gap = after_train_score - after_test_score
        
        # Generate comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Before
        models = ['Baseline']
        train_scores = [before_train_score]
        test_scores = [before_test_score]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, train_scores, width, label='Train', color='lightgreen', alpha=0.8)
        axes[0].bar(x + width/2, test_scores, width, label='Test', color='lightcoral', alpha=0.8)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Before: Overfitting Present', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].axhline(y=before_test_score, color='red', linestyle='--', alpha=0.5)
        axes[0].text(0, before_train_score + 0.02, f'{before_train_score:.3f}', ha='center', fontweight='bold')
        axes[0].text(0, before_test_score - 0.05, f'{before_test_score:.3f}', ha='center', fontweight='bold')
        axes[0].text(0, (before_train_score + before_test_score) / 2, 
                    f'Gap: {before_gap:.3f}', ha='center', color='red', fontweight='bold')
        
        # Plot 2: After
        models = ['Improved']
        train_scores = [after_train_score]
        test_scores = [after_test_score]
        
        axes[1].bar(x - width/2, train_scores, width, label='Train', color='lightgreen', alpha=0.8)
        axes[1].bar(x + width/2, test_scores, width, label='Test', color='lightcoral', alpha=0.8)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('After: Overfitting Reduced', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].set_ylim([0, 1.1])
        axes[1].axhline(y=after_test_score, color='green', linestyle='--', alpha=0.5)
        axes[1].text(0, after_train_score + 0.02, f'{after_train_score:.3f}', ha='center', fontweight='bold')
        axes[1].text(0, after_test_score - 0.05, f'{after_test_score:.3f}', ha='center', fontweight='bold')
        axes[1].text(0, (after_train_score + after_test_score) / 2, 
                    f'Gap: {after_gap:.3f}', ha='center', color='green', fontweight='bold')
        
        plt.tight_layout()
        plot_base64 = encode_plot_to_base64(fig)
        
        # Determine generalization status
        if after_gap < 0.05:
            generalization_status = 'Excellent - Model generalizes well'
        elif after_gap < 0.1:
            generalization_status = 'Good - Acceptable generalization'
        elif after_gap < 0.15:
            generalization_status = 'Fair - Some overfitting remains'
        else:
            generalization_status = 'Poor - Still overfitting'
        
        # Generate recommendations
        recommendations = []
        if after_gap > 0.1:
            recommendations.append('Consider collecting more training data')
            recommendations.append('Try combining multiple techniques (use method="all")')
        if after_test_score < before_test_score:
            recommendations.append('Model may be underfitting now - try relaxing constraints slightly')
        if len(X.columns) > 20:
            recommendations.append('High number of features - consider feature selection or PCA')
        if len(df) < 1000:
            recommendations.append('Small dataset - use cross-validation and simpler models')
        
        return jsonify({
            'before': {
                'train_score': float(before_train_score),
                'test_score': float(before_test_score),
                'overfitting_gap': float(before_gap)
            },
            'after': {
                'train_score': float(after_train_score),
                'test_score': float(after_test_score),
                'overfitting_gap': float(after_gap)
            },
            'improvement': {
                'gap_reduction': float(before_gap - after_gap),
                'test_score_change': float(after_test_score - before_test_score)
            },
            'applied_techniques': applied_techniques,
            'generalization_status': generalization_status,
            'recommendations': recommendations,
            'plot': plot_base64,
            'method_used': method
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/grid-search', methods=['POST'])
def grid_search_cv():
    """Perform GridSearchCV with custom parameter grid"""
    try:
        import time
        from sklearn.model_selection import GridSearchCV
        
        file = request.files['file']
        model_name = request.form.get('model', 'random_forest')
        cv_folds = int(request.form.get('cv_folds', 5))
        scoring = request.form.get('scoring', 'auto')
        n_jobs = int(request.form.get('n_jobs', -1))
        param_grid_str = request.form.get('param_grid', '{}')
        
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical and missing
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Parse parameter grid
        try:
            param_grid = json.loads(param_grid_str) if isinstance(param_grid_str, str) else param_grid_str
        except:
            param_grid = eval(param_grid_str)
        
        # Select base model
        if is_classification:
            if model_name == 'random_forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'xgboost':
                from xgboost import XGBClassifier
                base_model = XGBClassifier(random_state=42, eval_metric='logloss')
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                base_model = GradientBoostingClassifier(random_state=42)
            elif model_name == 'logistic':
                base_model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == 'svm':
                from sklearn.svm import SVC
                base_model = SVC(random_state=42)
            elif model_name == 'lightgbm':
                from lightgbm import LGBMClassifier
                base_model = LGBMClassifier(random_state=42, verbose=-1)
            else:
                base_model = RandomForestClassifier(random_state=42)
        else:
            if model_name == 'random_forest':
                base_model = RandomForestRegressor(random_state=42)
            elif model_name == 'xgboost':
                from xgboost import XGBRegressor
                base_model = XGBRegressor(random_state=42)
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                base_model = GradientBoostingRegressor(random_state=42)
            elif model_name == 'linear':
                base_model = LinearRegression()
            elif model_name == 'svm':
                from sklearn.svm import SVR
                base_model = SVR()
            elif model_name == 'lightgbm':
                from lightgbm import LGBMRegressor
                base_model = LGBMRegressor(random_state=42, verbose=-1)
            else:
                base_model = RandomForestRegressor(random_state=42)
        
        # Determine scoring metric
        if scoring == 'auto':
            scoring = 'accuracy' if is_classification else 'r2'
        
        # Perform Grid Search
        start_time = time.time()
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train_scaled, y_train)
        search_time = time.time() - start_time
        
        # Get results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Test on hold-out set
        test_score = grid_search.score(X_test_scaled, y_test)
        
        # Get top 5 combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        top_5 = results_df.head(5)
        
        top_combinations = []
        for idx, row in top_5.iterrows():
            top_combinations.append({
                'params': row['params'],
                'mean_score': float(row['mean_test_score']),
                'std_score': float(row['std_test_score'])
            })
        
        # Calculate total fits
        total_fits = len(grid_search.cv_results_['params']) * cv_folds
        
        # Generate visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Score distribution
        scores = results_df['mean_test_score'].values
        axes[0].hist(scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(best_score, color='red', linestyle='--', linewidth=2, label=f'Best: {best_score:.3f}')
        axes[0].set_xlabel('CV Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of CV Scores', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Top parameters (if applicable)
        if len(param_grid) > 0:
            param_name = list(param_grid.keys())[0]
            if param_name in results_df['params'].iloc[0]:
                param_values = [p[param_name] for p in results_df['params']]
                param_scores = results_df['mean_test_score'].values
                
                # Group by parameter value
                unique_vals = list(set(param_values))
                if len(unique_vals) <= 10 and all(isinstance(v, (int, float)) for v in unique_vals if v is not None):
                    grouped_scores = {}
                    for val, score in zip(param_values, param_scores):
                        if val not in grouped_scores:
                            grouped_scores[val] = []
                        grouped_scores[val].append(score)
                    
                    vals = sorted([v for v in grouped_scores.keys() if v is not None])
                    means = [np.mean(grouped_scores[v]) for v in vals]
                    
                    axes[1].plot(vals, means, marker='o', linewidth=2, markersize=8, color='green')
                    axes[1].set_xlabel(param_name, fontsize=12)
                    axes[1].set_ylabel('Mean CV Score', fontsize=12)
                    axes[1].set_title(f'Impact of {param_name}', fontsize=14, fontweight='bold')
                    axes[1].grid(alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'Parameter visualization\nnot available for\ncategorical parameters',
                               ha='center', va='center', fontsize=12)
                    axes[1].axis('off')
            else:
                axes[1].axis('off')
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        plot_base64 = encode_plot_to_base64(fig)
        
        # Generate model code
        model_code = f"""# Best model from GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Your best parameters
best_params = {best_params}

# Create model with best parameters
model = {type(base_model).__name__}(**best_params)

# Train on your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Train Score: {{train_score:.4f}}")
print(f"Test Score: {{test_score:.4f}}")
"""
        
        return jsonify({
            'best_params': best_params,
            'best_score': float(best_score),
            'test_score': float(test_score),
            'total_fits': int(total_fits),
            'search_time': float(search_time),
            'top_combinations': top_combinations,
            'plot': plot_base64,
            'model_code': model_code,
            'cv_folds': cv_folds,
            'scoring_metric': scoring
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/train-full-model', methods=['POST'])
def train_full_model():
    """Train a complete model with full performance metrics - no need for Colab!"""
    try:
        file = request.files['file']
        model_name = request.form.get('model', 'random_forest')
        use_best_params = request.form.get('use_best_params', 'false') == 'true'
        best_params_str = request.form.get('best_params', '{}')
        
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical and missing
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Parse best params if provided
        if use_best_params:
            try:
                best_params = json.loads(best_params_str) if isinstance(best_params_str, str) else best_params_str
            except:
                best_params = {}
        else:
            best_params = {}
        
        # Create model
        if is_classification:
            if model_name == 'random_forest':
                model = RandomForestClassifier(random_state=42, **best_params)
            elif model_name == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=42, eval_metric='logloss', **best_params)
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(random_state=42, **best_params)
            elif model_name == 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42, **best_params)
            elif model_name == 'svm':
                from sklearn.svm import SVC
                model = SVC(random_state=42, probability=True, **best_params)
            elif model_name == 'lightgbm':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(random_state=42, verbose=-1, **best_params)
            else:
                model = RandomForestClassifier(random_state=42, **best_params)
        else:
            if model_name == 'random_forest':
                model = RandomForestRegressor(random_state=42, **best_params)
            elif model_name == 'xgboost':
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=42, **best_params)
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42, **best_params)
            elif model_name == 'linear':
                model = LinearRegression(**best_params)
            elif model_name == 'svm':
                from sklearn.svm import SVR
                model = SVR(**best_params)
            elif model_name == 'lightgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(random_state=42, verbose=-1, **best_params)
            else:
                model = RandomForestRegressor(random_state=42, **best_params)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Get comprehensive metrics
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        results = {
            'model_name': type(model).__name__,
            'parameters': best_params if use_best_params else model.get_params(),
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.columns.tolist(),
                'n_features': len(X.columns)
            }
        }
        
        if is_classification:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                        f1_score, roc_auc_score, classification_report,
                                        confusion_matrix)
            
            results['metrics'] = {
                'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
                'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
                'precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
            }
            
            # ROC AUC if binary and has predict_proba
            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    results['metrics']['roc_auc'] = float(roc_auc_score(y_test, y_proba))
                except:
                    pass
            
            # Classification report
            results['classification_report'] = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Confusion matrix plot
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            results['confusion_matrix_plot'] = encode_plot_to_base64(fig)
            
        else:
            from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                                        mean_absolute_percentage_error, explained_variance_score)
            
            results['metrics'] = {
                'train_r2': float(r2_score(y_train, y_train_pred)),
                'test_r2': float(r2_score(y_test, y_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                'mae': float(mean_absolute_error(y_test, y_test_pred)),
                'mse': float(mean_squared_error(y_test, y_test_pred)),
                'explained_variance': float(explained_variance_score(y_test, y_test_pred))
            }
            
            try:
                results['metrics']['mape'] = float(mean_absolute_percentage_error(y_test, y_test_pred))
            except:
                pass
            
            # Actual vs Predicted plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title('Actual vs Predicted', fontsize=16, fontweight='bold')
            results['prediction_plot'] = encode_plot_to_base64(fig)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
            ax.set_xlabel('Importance')
            results['feature_importance_plot'] = encode_plot_to_base64(fig)
            results['feature_importance'] = importance_df.to_dict('records')
        
        # Cross-validation scores
        cv_strategy = get_cv_strategy(y_train, is_classification, n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy)
        results['cross_validation'] = {
            'scores': cv_scores.tolist(),
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std())
        }
        
        # Overfitting check
        train_score = results['metrics'].get('train_accuracy') or results['metrics'].get('train_r2')
        test_score = results['metrics'].get('test_accuracy') or results['metrics'].get('test_r2')
        overfitting_gap = train_score - test_score
        
        results['overfitting_analysis'] = {
            'gap': float(overfitting_gap),
            'status': 'Good' if overfitting_gap < 0.1 else 'Warning' if overfitting_gap < 0.15 else 'High',
            'recommendation': 'Model generalizes well' if overfitting_gap < 0.1 else 'Consider regularization or more data'
        }
        
        # Generate deployment code
        deployment_code = f"""# Production-Ready Model Code
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('{target_col}', axis=1)
y = df['{target_col}']

# Preprocessing
{chr(10).join([f"X['{col}'] = LabelEncoder().fit_transform(X['{col}'].astype(str))" for col in categorical_cols])}
X = X.fillna(X.mean())

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = {type(model).__name__}({', '.join([f'{k}={repr(v)}' for k, v in (best_params if use_best_params else {}).items()])})
model.fit(X_train_scaled, y_train)

# Evaluate
print(f"Test Score: {{model.score(X_test_scaled, y_test):.4f}}")

# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# To use later:
# model = joblib.load('model.pkl')
# scaler = joblib.load('scaler.pkl')
# predictions = model.predict(scaler.transform(new_data))
"""
        
        results['deployment_code'] = deployment_code
        results['problem_type'] = 'Classification' if is_classification else 'Regression'
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/train-deep-learning', methods=['POST'])
def train_deep_learning():
    """Train deep learning model with TensorFlow/Keras"""
    try:
        import time
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models, callbacks
        from io import StringIO
        
        file = request.files['file']
        layers_config = json.loads(request.form.get('layers', '[]'))
        config = json.loads(request.form.get('config', '{}'))
        
        df = read_csv_safe(file)
        
        # Prepare data
        target_col = df.columns[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical and missing
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Detect problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        n_classes = y.nunique() if is_classification else 1
        
        # Encode target
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Convert to numpy
        X = X.values.astype(np.float32)
        y = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=safe_stratify(y, is_classification)
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Determine if we need to reshape for RNN/CNN
        has_rnn = any(lc.get('type') in ['lstm', 'gru', 'simplernn', 'bidirectional'] for lc in layers_config[:-1])
        has_cnn = any(lc.get('type') in ['conv1d', 'conv2d'] for lc in layers_config[:-1])
        
        # Reshape data if needed
        if has_rnn:
            # Reshape to (samples, timesteps, features) for RNN
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            input_shape = (1, X_train_scaled.shape[2])
        elif has_cnn and 'conv1d' in str(layers_config):
            # Reshape for Conv1D: (samples, timesteps, features)
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            input_shape = (X_train_scaled.shape[1], 1)
        else:
            input_shape = (X_train_scaled.shape[1],)
        
        # Build model
        model = models.Sequential()
        
        # Add hidden layers
        first_layer = True
        for i, layer_config in enumerate(layers_config[:-1]):  # Exclude output layer
            layer_type = layer_config.get('type', 'dense')
            
            if layer_type == 'dense':
                if first_layer:
                    model.add(layers.Dense(
                        layer_config['units'],
                        activation=layer_config.get('activation', 'relu'),
                        input_shape=input_shape,
                        name=f'dense_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.Dense(
                        layer_config['units'],
                        activation=layer_config.get('activation', 'relu'),
                        name=f'dense_{i+1}'
                    ))
            
            elif layer_type == 'conv1d':
                if first_layer:
                    model.add(layers.Conv1D(
                        filters=layer_config.get('filters', 32),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu'),
                        padding=layer_config.get('padding', 'same'),
                        input_shape=input_shape,
                        name=f'conv1d_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.Conv1D(
                        filters=layer_config.get('filters', 32),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu'),
                        padding=layer_config.get('padding', 'same'),
                        name=f'conv1d_{i+1}'
                    ))
            
            elif layer_type == 'conv2d':
                if first_layer:
                    model.add(layers.Conv2D(
                        filters=layer_config.get('filters', 32),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu'),
                        padding=layer_config.get('padding', 'same'),
                        input_shape=input_shape,
                        name=f'conv2d_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.Conv2D(
                        filters=layer_config.get('filters', 32),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu'),
                        padding=layer_config.get('padding', 'same'),
                        name=f'conv2d_{i+1}'
                    ))
            
            elif layer_type == 'maxpool1d':
                model.add(layers.MaxPooling1D(
                    pool_size=layer_config.get('pool_size', 2),
                    name=f'maxpool1d_{i+1}'
                ))
            
            elif layer_type == 'maxpool2d':
                model.add(layers.MaxPooling2D(
                    pool_size=layer_config.get('pool_size', 2),
                    name=f'maxpool2d_{i+1}'
                ))
            
            elif layer_type == 'lstm':
                if first_layer:
                    model.add(layers.LSTM(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False),
                        dropout=layer_config.get('dropout', 0),
                        input_shape=input_shape,
                        name=f'lstm_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.LSTM(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False),
                        dropout=layer_config.get('dropout', 0),
                        name=f'lstm_{i+1}'
                    ))
            
            elif layer_type == 'gru':
                if first_layer:
                    model.add(layers.GRU(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False),
                        dropout=layer_config.get('dropout', 0),
                        input_shape=input_shape,
                        name=f'gru_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.GRU(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False),
                        dropout=layer_config.get('dropout', 0),
                        name=f'gru_{i+1}'
                    ))
            
            elif layer_type == 'simplernn':
                if first_layer:
                    model.add(layers.SimpleRNN(
                        units=layer_config.get('units', 32),
                        return_sequences=layer_config.get('return_sequences', False),
                        input_shape=input_shape,
                        name=f'simplernn_{i+1}'
                    ))
                    first_layer = False
                else:
                    model.add(layers.SimpleRNN(
                        units=layer_config.get('units', 32),
                        return_sequences=layer_config.get('return_sequences', False),
                        name=f'simplernn_{i+1}'
                    ))
            
            elif layer_type == 'bidirectional':
                wrapped_layer_type = layer_config.get('layer_type', 'lstm')
                if wrapped_layer_type == 'lstm':
                    wrapped_layer = layers.LSTM(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False)
                    )
                elif wrapped_layer_type == 'gru':
                    wrapped_layer = layers.GRU(
                        units=layer_config.get('units', 64),
                        return_sequences=layer_config.get('return_sequences', False)
                    )
                else:
                    wrapped_layer = layers.SimpleRNN(
                        units=layer_config.get('units', 32),
                        return_sequences=layer_config.get('return_sequences', False)
                    )
                
                if first_layer:
                    model.add(layers.Bidirectional(wrapped_layer, input_shape=input_shape, name=f'bidirectional_{i+1}'))
                    first_layer = False
                else:
                    model.add(layers.Bidirectional(wrapped_layer, name=f'bidirectional_{i+1}'))
            
            elif layer_type == 'flatten':
                model.add(layers.Flatten(name=f'flatten_{i+1}'))
            
            elif layer_type == 'dropout':
                model.add(layers.Dropout(layer_config.get('rate', 0.3), name=f'dropout_{i+1}'))
            
            elif layer_type == 'batchnorm':
                model.add(layers.BatchNormalization(name=f'batchnorm_{i+1}'))
        
        # Add output layer
        if is_classification:
            if n_classes == 2:
                # Binary classification
                model.add(layers.Dense(1, activation='sigmoid', name='output'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                # Multi-class classification
                model.add(layers.Dense(n_classes, activation='softmax', name='output'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regression
            model.add(layers.Dense(1, activation='linear', name='output'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        optimizer_name = config.get('optimizer', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == 'adamax':
            optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
        else:
            optimizer = 'adam'
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Prepare callbacks
        callback_list = []
        
        if config.get('early_stopping', True):
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Train model
        start_time = time.time()
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            validation_split=config.get('validation_split', 0.2),
            callbacks=callback_list,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled, verbose=0)
        
        if is_classification:
            if n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate plots
        plots = {}
        
        # 1. Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plots['loss_plot'] = encode_plot_to_base64(fig)
        
        # 2. Accuracy/MAE curves
        metric_key = 'accuracy' if is_classification else 'mae'
        if metric_key in history.history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history.history[metric_key], label=f'Training {metric_key.upper()}', linewidth=2)
            ax.plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_key.upper()}', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_key.upper(), fontsize=12)
            ax.set_title(f'Training and Validation {metric_key.upper()}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plots['accuracy_plot'] = encode_plot_to_base64(fig)
        
        # 3. Confusion matrix or predictions plot
        if is_classification:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plots['confusion_matrix_plot'] = encode_plot_to_base64(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title('Predictions vs Actual', fontsize=16, fontweight='bold')
            plots['predictions_plot'] = encode_plot_to_base64(fig)
        
        # 4. Model architecture visualization
        try:
            # Create architecture diagram
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            # Draw layers
            layer_names = [layer.name for layer in model.layers]
            layer_outputs = [str(layer.output_shape) for layer in model.layers]
            
            y_pos = len(layer_names)
            for i, (name, shape) in enumerate(zip(layer_names, layer_outputs)):
                y = y_pos - i
                ax.text(0.5, y, f'{name}\n{shape}', 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'),
                       fontsize=10, fontweight='bold')
                
                if i < len(layer_names) - 1:
                    ax.arrow(0.5, y-0.3, 0, -0.4, head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_pos + 1)
            ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold')
            plots['architecture_plot'] = encode_plot_to_base64(fig)
        except:
            pass
        
        # Get model summary
        summary_buffer = StringIO()
        model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        model_summary = summary_buffer.getvalue()
        
        # Find best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        early_stopped = len(history.history['loss']) < config.get('epochs', 50)
        
        # Generate deployment code
        activation_imports = set()
        for layer_config in layers_config[:-1]:
            act = layer_config['activation']
            if act not in ['relu', 'sigmoid', 'tanh', 'linear']:
                activation_imports.add(act)
        
        model_code = f"""# Deep Learning Model with TensorFlow/Keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Load and preprocess data
df = pd.read_csv('your_data.csv')
X = df.drop('{target_col}', axis=1)
y = df['{target_col}']

# Encode categorical features
{chr(10).join([f"X['{col}'] = LabelEncoder().fit_transform(X['{col}'].astype(str))" for col in categorical_cols])}
X = X.fillna(X.mean())

# Encode target (if classification)
{'y = LabelEncoder().fit_transform(y)' if is_classification and y.dtype == 'object' else ''}

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = models.Sequential([
    layers.Dense({layers_config[0]['units']}, activation='{layers_config[0]['activation']}', input_shape=({X_train_scaled.shape[1]},)),
    {chr(10).join([f"layers.Dropout({lc['dropout']})," if lc.get('dropout', 0) > 0 else '' for lc in layers_config[:-1]])}
    {chr(10).join([f"layers.Dense({lc['units']}, activation='{lc['activation']}')," for lc in layers_config[1:-1]])}
    layers.Dense({1 if n_classes == 2 or not is_classification else n_classes}, activation='{'sigmoid' if n_classes == 2 else 'softmax' if is_classification else 'linear'}')
])

# Compile
model.compile(
    optimizer=keras.optimizers.{optimizer_name.capitalize()}(learning_rate={learning_rate}),
    loss='{loss}',
    metrics=['{metrics[0]}']
)

# Train
history = model.fit(
    X_train_scaled, y_train,
    epochs={config.get('epochs', 50)},
    batch_size={config.get('batch_size', 32)},
    validation_split={config.get('validation_split', 0.2)},
    verbose=1
)

# Evaluate
test_loss, test_metric = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {{test_loss:.4f}}')
print(f'Test {metrics[0].upper()}: {{test_metric:.4f}}')

# Save model
model.save('my_model.h5')

# To load later:
# model = keras.models.load_model('my_model.h5')
# predictions = model.predict(scaler.transform(new_data))
"""
        
        # Prepare results
        results = {
            'model_summary': model_summary,
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.size(w).numpy() for w in model.trainable_weights])),
            'epochs_trained': len(history.history['loss']),
            'best_epoch': int(best_epoch),
            'early_stopped': early_stopped,
            'training_time': float(training_time),
            'final_train_metric': float(history.history[metric_key][-1]),
            'final_val_metric': float(history.history[f'val_{metric_key}'][-1]),
            'test_metric': float(test_results[1]),
            'metric_name': metric_key.upper(),
            'model_code': model_code,
            **plots
        }
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'ML Backend is running'})

# ============================================================================
# INTELLIGENT SYSTEM DESIGNER (ISD) - Industry-Grade ML Architecture System
# ============================================================================

# Import and register ISD routes
try:
    from isd_api import register_isd_routes
    register_isd_routes(app)
    print("✓ ISD modules loaded successfully")
except Exception as e:
    print(f"⚠ ISD modules not available: {e}")

# ============================================================================
# AI OVERFITTING AGENT - OpenAI Powered
# ============================================================================

@app.route('/api/ai-agent/analyze-overfitting', methods=['POST'])
def ai_analyze_overfitting():
    """
    AI-powered overfitting analysis using OpenAI GPT
    """
    try:
        print("🤖 AI Agent: Received analysis request")
        # Try CrewAI first, fallback to simple OpenAI
        try:
            from crewai_overfitting_agent import CrewAIOverfittingAgent as AIOverfittingAgent
            print("✅ Using CrewAI agent")
        except ImportError as e:
            print(f"⚠️ CrewAI not available ({e}), using simple OpenAI agent")
            from ai_overfitting_agent import AIOverfittingAgent
        
        # Get API key from request or environment
        api_key = request.form.get('api_key') or request.headers.get('X-OpenAI-Key')
        print(f"🔑 API Key present: {bool(api_key)}")
        if api_key:
            print(f"🔑 API Key starts with: {api_key[:10]}...")
            print(f"🔑 API Key length: {len(api_key)}")
        
        if not api_key:
            return jsonify({
                'error': 'OpenAI API key required',
                'message': 'Please provide your OpenAI API key in the request'
            }), 400
        
        # Get model results from request
        model_results_str = request.form.get('model_results', '{}')
        dataset_info_str = request.form.get('dataset_info', '{}')
        
        print(f"📊 Model results length: {len(model_results_str)}")
        print(f"📊 Dataset info length: {len(dataset_info_str)}")
        
        model_results = json.loads(model_results_str)
        dataset_info = json.loads(dataset_info_str)
        
        print(f"✅ Parsed data successfully")
        
        # Initialize AI agent
        agent = AIOverfittingAgent(api_key=api_key)
        print(f"✅ AI Agent initialized")
        
        # Analyze overfitting
        print(f"🔍 Starting analysis...")
        analysis = agent.analyze_overfitting(model_results, dataset_info)
        print(f"✅ Analysis complete")
        
        return jsonify(analysis)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in AI agent: {error_trace}")
        return jsonify({
            'error': str(e),
            'message': f'Backend error: {str(e)}',
            'traceback': error_trace
        }), 400

@app.route('/api/ai-agent/optimization-plan', methods=['POST'])
def ai_optimization_plan():
    """
    Get complete optimization plan from AI agent
    """
    try:
        from ai_overfitting_agent import AIOverfittingAgent
        
        api_key = request.form.get('api_key') or request.headers.get('X-OpenAI-Key')
        
        if not api_key:
            return jsonify({
                'error': 'OpenAI API key required',
                'message': 'Please provide your OpenAI API key'
            }), 400
        
        model_results = json.loads(request.form.get('model_results', '{}'))
        dataset_info = json.loads(request.form.get('dataset_info', '{}'))
        
        agent = AIOverfittingAgent(api_key=api_key)
        plan = agent.get_optimization_plan(model_results, dataset_info)
        
        return jsonify(plan)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ML Visualizer Backend Starting...")
    print("="*60)
    print("✅ Flask app initialized")
    print("✅ ISD modules loaded")
    print("✅ AI Agent endpoints registered")
    print("="*60)
    print("📡 Server running on http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
