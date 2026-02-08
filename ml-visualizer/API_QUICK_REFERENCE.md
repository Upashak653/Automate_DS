# 🚀 API Quick Reference

Base URL: `http://localhost:5000`

## All Endpoints (13 Total)

| # | Endpoint | Method | Purpose | Key Parameters |
|---|----------|--------|---------|----------------|
| 1 | `/api/health` | GET | Health check | None |
| 2 | `/api/validate-csv` | POST | Validate CSV file | `file` |
| 3 | `/api/analyze` | POST | Dataset analysis | `file` |
| 4 | `/api/visualize` | POST | Generate plots | `file` |
| 5 | `/api/preprocess` | POST | Data preprocessing | `file`, preprocessing options |
| 6 | `/api/recommend-model` | POST | Model recommendations | `file` |
| 7 | `/api/train-model` | POST | Train single model | `file`, `model_type` |
| 8 | `/api/train-full-model` | POST | Full model training | `file`, `model`, `best_params` |
| 9 | `/api/compare-models` | POST | Compare all models | `file` |
| 10 | `/api/tune-hyperparameters` | POST | RandomizedSearch/Optuna | `file`, `model_name`, `tuning_method` |
| 11 | `/api/grid-search` | POST | GridSearchCV | `file`, `model`, `param_grid` |
| 12 | `/api/reduce-overfitting` | POST | Overfitting reduction | `file`, `method` |
| 13 | `/api/train-deep-learning` | POST | Neural networks | `file`, `layers`, `config` |

## By Category

### 📊 Data Operations (4)
- `GET /api/health` - Health check
- `POST /api/validate-csv` - Validate CSV
- `POST /api/analyze` - Analyze data
- `POST /api/visualize` - Generate plots

### 🔧 Preprocessing (1)
- `POST /api/preprocess` - Data preprocessing

### 🤖 Traditional ML (5)
- `POST /api/recommend-model` - Get recommendations
- `POST /api/train-model` - Train model
- `POST /api/train-full-model` - Full training
- `POST /api/compare-models` - Compare models
- `POST /api/reduce-overfitting` - Reduce overfitting

### ⚙️ Hyperparameter Tuning (2)
- `POST /api/tune-hyperparameters` - RandomizedSearch/Optuna
- `POST /api/grid-search` - GridSearchCV

### 🧠 Deep Learning (1)
- `POST /api/train-deep-learning` - Neural networks

## Quick Examples

### Python
```python
import requests

BASE = 'http://localhost:5000'

# Health check
requests.get(f'{BASE}/api/health')

# Analyze data
with open('data.csv', 'rb') as f:
    requests.post(f'{BASE}/api/analyze', files={'file': f})

# Train model
with open('data.csv', 'rb') as f:
    requests.post(f'{BASE}/api/train-model', 
                 files={'file': f},
                 data={'model_type': 'random_forest'})
```

### cURL
```bash
# Health check
curl http://localhost:5000/api/health

# Analyze data
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@data.csv"

# Train model
curl -X POST http://localhost:5000/api/train-model \
  -F "file=@data.csv" \
  -F "model_type=random_forest"
```

### JavaScript
```javascript
const BASE = 'http://localhost:5000';

// Health check
fetch(`${BASE}/api/health`);

// Analyze data
const formData = new FormData();
formData.append('file', fileInput.files[0]);
fetch(`${BASE}/api/analyze`, {
  method: 'POST',
  body: formData
});

// Train model
formData.append('model_type', 'random_forest');
fetch(`${BASE}/api/train-model`, {
  method: 'POST',
  body: formData
});
```

## Response Format

All endpoints return JSON:

**Success**:
```json
{
  "data": "...",
  "results": "..."
}
```

**Error**:
```json
{
  "error": "Error message"
}
```

## Common Parameters

### File Upload
- **Parameter**: `file`
- **Type**: CSV file
- **Required**: Yes (for all POST endpoints except health)

### Model Types
- `auto`, `random_forest`, `xgboost`, `lightgbm`
- `gradient_boosting`, `adaboost`
- `logistic`, `linear`, `svm`, `knn`
- `naive_bayes`, `decision_tree`

### Preprocessing Options
- `remove_duplicates`: boolean
- `missing_strategy`: `none`, `drop_rows`, `drop_columns`, `mean`, `median`, `mode`
- `remove_outliers`: boolean
- `encode_categorical`: boolean
- `standardize`: boolean
- `normalize`: boolean
- `feature_selection`: boolean
- `apply_pca`: boolean

### Overfitting Methods
- `cross_validation`, `regularization`, `early_stopping`
- `pruning`, `feature_selection`, `ensemble`, `all`

### Deep Learning Layer Types
- `dense`, `conv1d`, `conv2d`, `maxpool1d`, `maxpool2d`
- `lstm`, `gru`, `simplernn`, `bidirectional`
- `flatten`, `dropout`, `batchnorm`

## Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `500` - Internal Server Error

## Full Documentation

See [API_ENDPOINTS.md](./API_ENDPOINTS.md) for complete details with examples.

---

**Total**: 13 endpoints covering data analysis, preprocessing, ML training, hyperparameter tuning, and deep learning.
