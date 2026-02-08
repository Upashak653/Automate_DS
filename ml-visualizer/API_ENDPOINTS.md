# 🔌 API Endpoints Reference

Base URL: `http://localhost:5000`

## Table of Contents
1. [Health Check](#1-health-check)
2. [Data Analysis](#2-data-analysis)
3. [Visualizations](#3-visualizations)
4. [CSV Validation](#4-csv-validation)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Recommendations](#6-model-recommendations)
7. [Train Model](#7-train-model)
8. [Train Full Model](#8-train-full-model)
9. [Compare Models](#9-compare-models)
10. [Hyperparameter Tuning](#10-hyperparameter-tuning)
11. [GridSearchCV](#11-gridsearchcv)
12. [Reduce Overfitting](#12-reduce-overfitting)
13. [Deep Learning](#13-deep-learning)

---

## 1. Health Check

**Endpoint**: `GET /api/health`

**Description**: Check if backend is running

**Request**: None

**Response**:
```json
{
  "status": "healthy",
  "message": "ML Backend is running"
}
```

**Example**:
```bash
curl http://localhost:5000/api/health
```

---

## 2. Data Analysis

**Endpoint**: `POST /api/analyze`

**Description**: Analyze dataset and get statistics

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file

**Response**:
```json
{
  "shape": [1000, 10],
  "columns": ["col1", "col2", ...],
  "dtypes": {"col1": "int64", "col2": "float64", ...},
  "missing": {"col1": 5, "col2": 0, ...},
  "describe": {...},
  "head": [{...}, {...}, ...],
  "problem_type": "Classification",
  "target_column": "target",
  "column_analysis": [
    {
      "name": "col1",
      "type": "numeric",
      "unique_count": 100,
      "missing_count": 5,
      "missing_percent": 0.5,
      "mean": 50.5,
      "median": 50.0,
      "std": 15.2,
      "min": 0,
      "max": 100
    }
  ]
}
```

**Example**:
```python
import requests

with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/api/analyze', 
                           files={'file': f})
    analysis = response.json()
```

---

## 3. Visualizations

**Endpoint**: `POST /api/visualize`

**Description**: Generate seaborn/matplotlib plots

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file

**Response**:
```json
{
  "correlation_heatmap": "data:image/png;base64,...",
  "distributions": "data:image/png;base64,...",
  "boxplots": "data:image/png;base64,...",
  "pairplot": "data:image/png;base64,...",
  "missing_data": "data:image/png;base64,...",
  "target_distribution": "data:image/png;base64,..."
}
```

**Plots Generated**:
- Correlation heatmap
- Distribution plots (histograms with KDE)
- Box plots (outlier detection)
- Pairplot (for small datasets)
- Missing data visualization
- Target distribution

**Example**:
```python
response = requests.post('http://localhost:5000/api/visualize',
                        files={'file': open('data.csv', 'rb')})
plots = response.json()
```

---

## 4. CSV Validation

**Endpoint**: `POST /api/validate-csv`

**Description**: Validate CSV file before processing

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file

**Response**:
```json
{
  "valid": true,
  "shape": [1000, 10],
  "columns": ["col1", "col2", ...],
  "issues": [],
  "warnings": ["High missing data: 25.5%"],
  "message": "CSV is valid and ready for analysis"
}
```

**Example**:
```python
response = requests.post('http://localhost:5000/api/validate-csv',
                        files={'file': open('data.csv', 'rb')})
validation = response.json()
```

---

## 5. Data Preprocessing

**Endpoint**: `POST /api/preprocess`

**Description**: Apply preprocessing techniques to data

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `remove_duplicates`: `true` or `false`
  - `missing_strategy`: `none`, `drop_rows`, `drop_columns`, `mean`, `median`, `mode`
  - `missing_threshold`: `0.5` (for drop_columns)
  - `remove_outliers`: `true` or `false`
  - `encode_categorical`: `true` or `false`
  - `standardize`: `true` or `false`
  - `normalize`: `true` or `false`
  - `feature_selection`: `true` or `false`
  - `variance_threshold`: `0.01`
  - `apply_pca`: `true` or `false`
  - `pca_components`: `auto` or number

**Response**:
```json
{
  "success": true,
  "summary": {
    "original_shape": [1000, 10],
    "final_shape": [950, 8],
    "rows_removed": 50,
    "columns_removed": 2,
    "preprocessing_steps": [
      "Removed 50 duplicate rows",
      "Filled missing values with mean",
      "Label encoded 3 categorical columns",
      "Standardized 5 numeric columns"
    ],
    "preview": [{...}, {...}, ...],
    "columns": ["col1", "col2", ...],
    "dtypes": {...}
  },
  "csv_data": "col1,col2,...\n1,2,...",
  "message": "Preprocessing complete: 4 steps applied"
}
```

**Example**:
```python
data = {
    'remove_duplicates': 'true',
    'missing_strategy': 'mean',
    'standardize': 'true'
}
response = requests.post('http://localhost:5000/api/preprocess',
                        files={'file': open('data.csv', 'rb')},
                        data=data)
```

---

## 6. Model Recommendations

**Endpoint**: `POST /api/recommend-model`

**Description**: Get ranked model recommendations based on dataset

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file

**Response**:
```json
{
  "problem_type": "Classification",
  "target_column": "target",
  "dataset_stats": {
    "rows": 1000,
    "columns": 10,
    "numeric_features": 7,
    "categorical_features": 3,
    "missing_percent": 5.2,
    "feature_ratio": 0.01
  },
  "recommended_models": [
    {
      "name": "XGBoost",
      "score": 92,
      "priority": "Best Match",
      "reason": "State-of-the-art gradient boosting",
      "pros": ["Excellent accuracy", "Fast training", "Handles missing data"],
      "cons": ["Many hyperparameters", "Can overfit"],
      "best_for": "Kaggle competitions and production systems"
    }
  ],
  "insights": [
    {
      "type": "success",
      "message": "Large dataset (1,000 rows)",
      "recommendation": "Can use complex models like Gradient Boosting"
    }
  ],
  "best_model": {...}
}
```

**Example**:
```python
response = requests.post('http://localhost:5000/api/recommend-model',
                        files={'file': open('data.csv', 'rb')})
recommendations = response.json()
```

---

## 7. Train Model

**Endpoint**: `POST /api/train-model`

**Description**: Train a single ML model

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `model_type`: `auto`, `random_forest`, `xgboost`, `logistic`, `linear`, `gradient_boosting`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `lightgbm`, `adaboost`

**Response**:
```json
{
  "model_type": "RandomForestClassifier",
  "train_size": 800,
  "test_size": 200,
  "features": ["col1", "col2", ...],
  "accuracy": 0.87,
  "train_accuracy": 0.92,
  "precision": 0.85,
  "recall": 0.83,
  "f1_score": 0.84,
  "roc_auc": 0.91,
  "classification_report": {...},
  "evaluation_metrics": {
    "accuracy": {
      "value": 0.87,
      "description": "Overall correctness of predictions",
      "interpretation": "Higher is better (0-1 scale)",
      "best_for": "Balanced datasets"
    }
  },
  "confusion_matrix_plot": "data:image/png;base64,...",
  "feature_importance_plot": "data:image/png;base64,...",
  "feature_importance": [{...}],
  "cv_scores": [0.85, 0.87, 0.86, 0.88, 0.84],
  "cv_mean": 0.86,
  "cv_std": 0.015
}
```

**Example**:
```python
response = requests.post('http://localhost:5000/api/train-model',
                        files={'file': open('data.csv', 'rb')},
                        data={'model_type': 'random_forest'})
results = response.json()
```

---

## 8. Train Full Model

**Endpoint**: `POST /api/train-full-model`

**Description**: Train model with comprehensive metrics (production-ready)

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `model`: Model name
  - `use_best_params`: `true` or `false`
  - `best_params`: JSON string of parameters (if use_best_params=true)

**Response**:
```json
{
  "model_name": "RandomForestClassifier",
  "parameters": {...},
  "dataset_info": {
    "total_samples": 1000,
    "train_samples": 800,
    "test_samples": 200,
    "features": [...],
    "n_features": 10
  },
  "metrics": {
    "train_accuracy": 0.92,
    "test_accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.83,
    "f1_score": 0.84
  },
  "classification_report": {...},
  "confusion_matrix_plot": "data:image/png;base64,...",
  "feature_importance_plot": "data:image/png;base64,...",
  "feature_importance": [...],
  "cross_validation": {
    "scores": [0.85, 0.87, 0.86, 0.88, 0.84],
    "mean": 0.86,
    "std": 0.015
  },
  "overfitting_analysis": {
    "gap": 0.05,
    "status": "Good",
    "recommendation": "Model generalizes well"
  },
  "deployment_code": "# Production-Ready Model Code\n...",
  "problem_type": "Classification"
}
```

**Example**:
```python
response = requests.post('http://localhost:5000/api/train-full-model',
                        files={'file': open('data.csv', 'rb')},
                        data={'model': 'random_forest'})
```

---

## 9. Compare Models

**Endpoint**: `POST /api/compare-models`

**Description**: Compare multiple models side-by-side

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file

**Response**:
```json
{
  "results": [
    {
      "model": "Random Forest",
      "train_score": 0.92,
      "test_score": 0.87,
      "precision": 0.85,
      "recall": 0.83,
      "f1_score": 0.84,
      "overfitting": 0.05
    },
    {
      "model": "XGBoost",
      "train_score": 0.94,
      "test_score": 0.89,
      "precision": 0.87,
      "recall": 0.85,
      "f1_score": 0.86,
      "overfitting": 0.05
    }
  ],
  "comparison_plot": "data:image/png;base64,...",
  "best_model": {
    "model": "XGBoost",
    "test_score": 0.89
  },
  "problem_type": "Classification"
}
```

**Example**:
```python
response = requests.post('http://localhost:5000/api/compare-models',
                        files={'file': open('data.csv', 'rb')})
comparison = response.json()
```

---

## 10. Hyperparameter Tuning

**Endpoint**: `POST /api/tune-hyperparameters`

**Description**: Tune hyperparameters using RandomizedSearchCV or Optuna

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `model_name`: `random_forest` or `xgboost`
  - `tuning_method`: `randomized` or `optuna`
  - `cv_folds`: Number (default: 5)

**Response**:
```json
{
  "best_params": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5
  },
  "best_cv_score": 0.87,
  "test_score": 0.86,
  "cv_scores": [0.85, 0.87, 0.86, 0.88, 0.84],
  "cv_mean": 0.86,
  "cv_std": 0.015,
  "tuning_method": "randomized",
  "overfitting_risk": "Low"
}
```

**Example**:
```python
data = {
    'model_name': 'random_forest',
    'tuning_method': 'randomized',
    'cv_folds': 5
}
response = requests.post('http://localhost:5000/api/tune-hyperparameters',
                        files={'file': open('data.csv', 'rb')},
                        data=data)
```

---

## 11. GridSearchCV

**Endpoint**: `POST /api/grid-search`

**Description**: Exhaustive hyperparameter search with GridSearchCV

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `model`: Model name
  - `cv_folds`: Number (default: 5)
  - `scoring`: Metric (default: `auto`)
  - `n_jobs`: Number of parallel jobs (default: -1)
  - `param_grid`: JSON string of parameter grid

**Response**:
```json
{
  "best_params": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5
  },
  "best_score": 0.87,
  "test_score": 0.86,
  "total_fits": 240,
  "search_time": 45.2,
  "top_combinations": [
    {
      "params": {...},
      "mean_score": 0.87,
      "std_score": 0.015
    }
  ],
  "plot": "data:image/png;base64,...",
  "model_code": "# Best model from GridSearchCV\n...",
  "cv_folds": 5,
  "scoring_metric": "accuracy"
}
```

**Example**:
```python
import json

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}

data = {
    'model': 'random_forest',
    'cv_folds': 5,
    'param_grid': json.dumps(param_grid)
}

response = requests.post('http://localhost:5000/api/grid-search',
                        files={'file': open('data.csv', 'rb')},
                        data=data)
```

---

## 12. Reduce Overfitting

**Endpoint**: `POST /api/reduce-overfitting`

**Description**: Apply overfitting reduction techniques

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `method`: `cross_validation`, `regularization`, `early_stopping`, `pruning`, `feature_selection`, `ensemble`, `all`
  - `cv_folds`: Number (default: 5)
  - `regularization_strength`: Float (default: 0.1)
  - `max_depth`: Number (default: 10)
  - `min_samples_split`: Number (default: 10)
  - `ensemble_size`: Number (default: 5)

**Response**:
```json
{
  "before": {
    "train_score": 0.95,
    "test_score": 0.75,
    "overfitting_gap": 0.20
  },
  "after": {
    "train_score": 0.88,
    "test_score": 0.82,
    "overfitting_gap": 0.06
  },
  "improvement": {
    "gap_reduction": 0.14,
    "test_score_change": 0.07
  },
  "applied_techniques": [
    "5-fold Cross-Validation (CV Score: 0.850 ± 0.023)"
  ],
  "generalization_status": "Good - Acceptable generalization",
  "recommendations": [
    "Consider collecting more training data"
  ],
  "plot": "data:image/png;base64,...",
  "method_used": "cross_validation"
}
```

**Example**:
```python
data = {
    'method': 'cross_validation',
    'cv_folds': 5
}
response = requests.post('http://localhost:5000/api/reduce-overfitting',
                        files={'file': open('data.csv', 'rb')},
                        data=data)
```

---

## 13. Deep Learning

**Endpoint**: `POST /api/train-deep-learning`

**Description**: Train custom neural network with TensorFlow/Keras

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: CSV file
  - `layers`: JSON string of layer configurations
  - `config`: JSON string of training configuration

**Layer Configuration Example**:
```json
[
  {"id": 1, "type": "dense", "units": 64, "activation": "relu", "dropout": 0},
  {"id": 2, "type": "lstm", "units": 128, "return_sequences": true, "dropout": 0.2},
  {"id": 3, "type": "conv1d", "filters": 32, "kernel_size": 3, "activation": "relu"},
  {"id": 4, "type": "maxpool1d", "pool_size": 2},
  {"id": 5, "type": "flatten"},
  {"id": 6, "type": "dropout", "rate": 0.3},
  {"id": 7, "type": "batchnorm"},
  {"id": 8, "type": "output", "units": "auto", "activation": "auto"}
]
```

**Config Example**:
```json
{
  "epochs": 50,
  "batch_size": 32,
  "validation_split": 0.2,
  "optimizer": "adam",
  "learning_rate": 0.001,
  "early_stopping": true,
  "patience": 10
}
```

**Response**:
```json
{
  "model_summary": "Layer-by-layer summary...",
  "total_params": 12345,
  "trainable_params": 12345,
  "epochs_trained": 35,
  "best_epoch": 28,
  "early_stopped": true,
  "training_time": 45.2,
  "final_train_metric": 0.92,
  "final_val_metric": 0.89,
  "test_metric": 0.88,
  "metric_name": "ACCURACY",
  "architecture_plot": "data:image/png;base64,...",
  "loss_plot": "data:image/png;base64,...",
  "accuracy_plot": "data:image/png;base64,...",
  "confusion_matrix_plot": "data:image/png;base64,...",
  "predictions_plot": "data:image/png;base64,...",
  "model_code": "# TensorFlow/Keras code..."
}
```

**Supported Layer Types**:
- `dense`: Fully connected layer
- `conv1d`: 1D convolutional layer
- `conv2d`: 2D convolutional layer
- `maxpool1d`: 1D max pooling
- `maxpool2d`: 2D max pooling
- `lstm`: LSTM layer
- `gru`: GRU layer
- `simplernn`: Simple RNN layer
- `bidirectional`: Bidirectional wrapper
- `flatten`: Flatten layer
- `dropout`: Dropout layer
- `batchnorm`: Batch normalization

**Example**:
```python
import json

layers = [
    {"id": 1, "type": "dense", "units": 64, "activation": "relu", "dropout": 0},
    {"id": 2, "type": "dense", "units": 32, "activation": "relu", "dropout": 0.2},
    {"id": 3, "type": "output", "units": "auto", "activation": "auto", "dropout": 0}
]

config = {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001
}

data = {
    'layers': json.dumps(layers),
    'config': json.dumps(config)
}

response = requests.post('http://localhost:5000/api/train-deep-learning',
                        files={'file': open('data.csv', 'rb')},
                        data=data)
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message description"
}
```

**Common HTTP Status Codes**:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

---

## Complete Example Workflow

```python
import requests
import json

BASE_URL = 'http://localhost:5000'

# 1. Validate CSV
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/validate-csv', files={'file': f})
    print(response.json())

# 2. Analyze data
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/analyze', files={'file': f})
    analysis = response.json()
    print(f"Problem type: {analysis['problem_type']}")

# 3. Get visualizations
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/visualize', files={'file': f})
    plots = response.json()

# 4. Get model recommendations
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/recommend-model', files={'file': f})
    recommendations = response.json()
    best_model = recommendations['best_model']['name']
    print(f"Best model: {best_model}")

# 5. GridSearch for best parameters
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/grid-search',
                            files={'file': f},
                            data={
                                'model': 'random_forest',
                                'param_grid': json.dumps(param_grid)
                            })
    grid_results = response.json()
    best_params = grid_results['best_params']

# 6. Train final model with best parameters
with open('data.csv', 'rb') as f:
    response = requests.post(f'{BASE_URL}/api/train-full-model',
                            files={'file': f},
                            data={
                                'model': 'random_forest',
                                'use_best_params': 'true',
                                'best_params': json.dumps(best_params)
                            })
    final_results = response.json()
    print(f"Test accuracy: {final_results['metrics']['test_accuracy']}")
    print(final_results['deployment_code'])
```

---

## Summary

**Total Endpoints**: 13

**Categories**:
- Health & Validation: 2 endpoints
- Data Analysis: 2 endpoints
- Preprocessing: 1 endpoint
- Model Training: 5 endpoints
- Hyperparameter Tuning: 2 endpoints
- Deep Learning: 1 endpoint

**All endpoints use**:
- Base URL: `http://localhost:5000`
- File upload: `multipart/form-data`
- Response format: JSON
- Image format: Base64-encoded PNG

---

**Need help?** Check the [USER_GUIDE.md](./USER_GUIDE.md) for detailed usage instructions.
