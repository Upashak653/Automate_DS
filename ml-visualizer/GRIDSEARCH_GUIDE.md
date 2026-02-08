# 🔍 GridSearchCV Guide - Automated Hyperparameter Tuning

## What is GridSearchCV?

GridSearchCV is an exhaustive search method that:
- Tests **every combination** of hyperparameters you specify
- Uses **cross-validation** to evaluate each combination
- Finds the **best parameters** automatically
- Saves you hours of manual tuning

**No more Google Colab needed!** Everything runs right here in your browser.

## Quick Start

1. **Upload your CSV file**
2. **Click "🔍 GridSearchCV" tab**
3. **Select your model** (Random Forest, XGBoost, etc.)
4. **Configure CV settings** (folds, scoring metric)
5. **Edit parameter grid** (or use defaults)
6. **Click "Start Grid Search"**
7. **Get best parameters + ready-to-use code!**

## Step-by-Step Guide

### Step 1: Select Model

Choose from 6 powerful models:
- 🌲 **Random Forest**: Great all-rounder, handles non-linearity
- ⚡ **XGBoost**: State-of-the-art gradient boosting
- 📈 **Gradient Boosting**: High accuracy, slower training
- 📊 **Logistic Regression**: Fast, interpretable (classification)
- 🎯 **SVM**: Effective in high dimensions
- 💡 **LightGBM**: Extremely fast for large datasets

### Step 2: Configure Cross-Validation

**CV Folds** (2-10, default: 5)
- More folds = more robust but slower
- 5 is standard for most datasets
- Use 10 for small datasets

**Scoring Metric**
- **Auto**: Accuracy (classification) or R² (regression)
- **Classification**: accuracy, f1, precision, recall, roc_auc
- **Regression**: r2, neg_mean_squared_error, neg_mean_absolute_error

**Parallel Jobs**
- **-1**: Use all CPU cores (fastest)
- **1**: Single core (slower but uses less resources)

### Step 3: Define Parameter Grid

#### Option A: Use Predefined Grid (Recommended)

We provide optimized grids for each model:

**Random Forest:**
```json
{
  "n_estimators": [50, 100, 200],
  "max_depth": [10, 20, 30, null],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4],
  "max_features": ["sqrt", "log2"]
}
```

**XGBoost:**
```json
{
  "n_estimators": [50, 100, 200],
  "learning_rate": [0.01, 0.1, 0.3],
  "max_depth": [3, 5, 7],
  "subsample": [0.8, 0.9, 1.0],
  "colsample_bytree": [0.8, 0.9, 1.0]
}
```

**Logistic Regression:**
```json
{
  "C": [0.001, 0.01, 0.1, 1, 10, 100],
  "penalty": ["l1", "l2"],
  "solver": ["liblinear", "saga"]
}
```

You can edit these directly in the UI!

#### Option B: Custom Grid (Advanced)

Click "Custom Grid (JSON)" and enter your own:

```json
{
  "n_estimators": [100, 200, 300, 500],
  "max_depth": [5, 10, 15, 20, 25],
  "min_samples_split": [2, 5, 10, 20],
  "learning_rate": [0.01, 0.05, 0.1, 0.2]
}
```

**Tips for Custom Grids:**
- Start with fewer values to test quickly
- Expand ranges based on initial results
- Use logarithmic scales for learning rates: [0.001, 0.01, 0.1]
- Include `null` for parameters like `max_depth` (no limit)

### Step 4: Start Search

Click "🚀 Start Grid Search"

**What happens:**
1. Data is preprocessed automatically
2. Every parameter combination is tested
3. Each combination is evaluated with CV
4. Best parameters are identified
5. Model is tested on hold-out set
6. Results and code are generated

**Time estimate:**
- Small grid (< 50 combinations): 10-30 seconds
- Medium grid (50-200 combinations): 1-3 minutes
- Large grid (> 200 combinations): 3-10 minutes

## Understanding Results

### Best Parameters Found

Shows the optimal hyperparameters:
```
n_estimators: 200
max_depth: 20
min_samples_split: 5
learning_rate: 0.1
```

**These are production-ready!** Use them directly in your models.

### Performance Metrics

**Best CV Score**: Average score across all folds
- Higher is better
- Shows how well model generalizes

**Test Score**: Performance on hold-out test set
- Should be close to CV score
- If much lower, might be overfitting

**Total Fits**: Number of models trained
- = (combinations × CV folds)
- Example: 48 combinations × 5 folds = 240 fits

**Search Time**: Total time taken
- Depends on dataset size and grid size

### Top 5 Combinations

See the best 5 parameter sets:
- Compare scores
- Check standard deviations
- Understand parameter impact

**Low std dev** = consistent performance (good!)
**High std dev** = unstable across folds (warning!)

### Visualizations

**Score Distribution**: 
- Shows range of all scores
- Best score marked with red line
- Tight distribution = stable parameters

**Parameter Impact**:
- Shows how one parameter affects score
- Helps understand parameter importance

### Ready-to-Use Code

Copy-paste Python code with:
- ✅ Best parameters already filled in
- ✅ Complete training pipeline
- ✅ Model saving/loading
- ✅ Production-ready

## Advanced Tips

### 1. Iterative Refinement

**First Pass** (Coarse Grid):
```json
{
  "n_estimators": [50, 100, 200],
  "max_depth": [5, 10, 20]
}
```

**Second Pass** (Fine Grid):
If best was `n_estimators=100, max_depth=10`, try:
```json
{
  "n_estimators": [80, 90, 100, 110, 120],
  "max_depth": [8, 9, 10, 11, 12]
}
```

### 2. Parameter Priorities

**Most Important** (search first):
- `n_estimators`, `max_depth`, `learning_rate`

**Secondary**:
- `min_samples_split`, `subsample`

**Fine-tuning**:
- `min_samples_leaf`, `max_features`

### 3. Computational Efficiency

**To speed up search:**
- Use fewer parameter values
- Reduce CV folds (but not below 3)
- Use `n_jobs=-1` for parallel processing
- Start with smaller dataset sample

**To improve accuracy:**
- More parameter values
- More CV folds (5-10)
- Larger parameter ranges

### 4. Avoiding Overfitting

**Warning signs:**
- CV score much higher than test score
- Very complex parameters (high n_estimators, deep trees)

**Solutions:**
- Add regularization parameters
- Limit max_depth
- Increase min_samples_split
- Use more training data

## Common Scenarios

### Scenario 1: First Time Tuning
```
Action: Use predefined grid
Reason: Optimized ranges, good starting point
Time: 1-3 minutes
```

### Scenario 2: Model Underperforming
```
Action: Expand parameter ranges
Example: Try deeper trees, more estimators
Time: 3-5 minutes
```

### Scenario 3: Overfitting Issues
```
Action: Add constraints
Example: Lower max_depth, higher min_samples_split
Time: 1-2 minutes
```

### Scenario 4: Large Dataset (> 100K rows)
```
Action: Use LightGBM with coarse grid
Reason: Faster training
Time: 5-10 minutes
```

### Scenario 5: Small Dataset (< 1000 rows)
```
Action: Use 10-fold CV, simpler models
Reason: Better generalization estimates
Time: 1-2 minutes
```

## Model-Specific Guides

### Random Forest

**Key Parameters:**
- `n_estimators`: More trees = better (but slower)
- `max_depth`: Deeper = more complex (risk overfitting)
- `min_samples_split`: Higher = more conservative

**Recommended Grid:**
```json
{
  "n_estimators": [100, 200, 300],
  "max_depth": [10, 20, 30, null],
  "min_samples_split": [2, 5, 10],
  "max_features": ["sqrt", "log2"]
}
```

### XGBoost

**Key Parameters:**
- `learning_rate`: Lower = better (but needs more estimators)
- `max_depth`: Usually 3-7 is optimal
- `subsample`: 0.8-1.0 prevents overfitting

**Recommended Grid:**
```json
{
  "n_estimators": [100, 200, 300],
  "learning_rate": [0.01, 0.1, 0.3],
  "max_depth": [3, 5, 7],
  "subsample": [0.8, 0.9, 1.0]
}
```

### Logistic Regression

**Key Parameters:**
- `C`: Inverse regularization (higher = less regularization)
- `penalty`: L1 for feature selection, L2 for general use

**Recommended Grid:**
```json
{
  "C": [0.001, 0.01, 0.1, 1, 10, 100],
  "penalty": ["l1", "l2"],
  "solver": ["liblinear", "saga"]
}
```

## Troubleshooting

### "Search taking too long"
- Reduce parameter grid size
- Use fewer CV folds
- Sample your dataset
- Use `n_jobs=-1`

### "All scores are similar"
- Parameters might not matter much
- Try wider ranges
- Check if data preprocessing needed

### "Best score is poor"
- Try different model
- Check data quality
- Add feature engineering
- Get more training data

### "Test score much lower than CV score"
- Overfitting detected
- Use simpler parameters
- Add regularization
- Get more training data

## API Usage

For programmatic access:

```python
import requests
import json

url = 'http://localhost:5000/api/grid-search'

files = {'file': open('data.csv', 'rb')}
data = {
    'model': 'random_forest',
    'cv_folds': 5,
    'scoring': 'accuracy',
    'n_jobs': -1,
    'param_grid': json.dumps({
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30]
    })
}

response = requests.post(url, files=files, data=data)
results = response.json()

print(f"Best Parameters: {results['best_params']}")
print(f"Best Score: {results['best_score']:.4f}")
print(f"Test Score: {results['test_score']:.4f}")
```

## Best Practices

1. **Start Simple**: Use predefined grids first
2. **Iterate**: Refine based on initial results
3. **Monitor Time**: Balance thoroughness vs speed
4. **Check Overfitting**: Compare CV and test scores
5. **Save Results**: Copy the generated code
6. **Document**: Note which parameters worked best
7. **Validate**: Test on completely new data

## Next Steps After GridSearch

1. **Use Best Parameters**: Copy the generated code
2. **Train Full Model**: Use all your data
3. **Validate**: Test on new data
4. **Deploy**: Save model with joblib
5. **Monitor**: Track performance in production

## Comparison: GridSearchCV vs Manual Tuning

| Aspect | GridSearchCV | Manual Tuning |
|--------|--------------|---------------|
| **Time** | 1-10 minutes | Hours/Days |
| **Coverage** | Exhaustive | Limited |
| **Reproducibility** | Perfect | Variable |
| **Expertise Needed** | Low | High |
| **Best For** | Most cases | Very specific needs |

## Why Use This Instead of Google Colab?

✅ **Faster**: No notebook setup, instant results
✅ **Easier**: Visual interface, no coding needed
✅ **Complete**: Get code + visualizations + metrics
✅ **Integrated**: Works with your uploaded data
✅ **No Setup**: No environment configuration
✅ **Reproducible**: Same results every time

---

**Pro Tip**: Run GridSearchCV first, then use the best parameters in the "Train Model" tab for full performance metrics!
