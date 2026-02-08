# Troubleshooting Guide

## "Failed to fetch" Error

### Cause
The frontend cannot connect to the Python backend server.

### Solutions

#### 1. Check if Backend is Running

**Windows:**
```bash
netstat -ano | findstr :5000
```

**Mac/Linux:**
```bash
lsof -i :5000
```

If nothing shows up, the backend is not running.

#### 2. Start the Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

#### 3. Check Backend Status in Browser

Open: http://localhost:5000/api/health

You should see:
```json
{"status": "healthy", "message": "ML Backend is running"}
```

#### 4. Check CORS Settings

If backend is running but still getting errors, check:
- Backend is on port 5000
- Frontend is on port 5173
- No firewall blocking connections

#### 5. Check for Port Conflicts

If port 5000 is already in use:

**Windows:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
lsof -ti:5000 | xargs kill -9
```

## Other Common Errors

### "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

### "UnicodeDecodeError" or CSV Parsing Errors

**Solution:** The backend now handles this automatically. If issues persist:
1. Save CSV as UTF-8 encoding
2. Remove special characters from column names
3. Ensure all rows have same number of columns

### Backend Crashes During Training

**Cause:** Large dataset or insufficient memory

**Solution:**
1. Reduce dataset size
2. Use sampling in preprocessing
3. Choose simpler models (Logistic Regression, Linear Regression)

### Slow Performance

**Cause:** Large dataset or complex model

**Solution:**
1. Use LightGBM instead of XGBoost for large datasets
2. Reduce n_estimators for tree-based models
3. Sample data in preprocessing tab

## Quick Checklist

- [ ] Backend server is running (check http://localhost:5000/api/health)
- [ ] Frontend shows "Backend: connected" (green)
- [ ] No firewall blocking port 5000
- [ ] All Python dependencies installed
- [ ] CSV file is properly formatted
- [ ] No port conflicts

## Still Having Issues?

1. Check backend terminal for error messages
2. Check browser console (F12) for errors
3. Restart both backend and frontend
4. Try with a small sample dataset first
5. Check Python version (requires 3.8+)

## Test with Sample Data

Generate test data:
```bash
node generate-test-data.js
```

Upload `test-data-1k.csv` to verify everything works.


---

## Class Imbalance Errors

### "The least populated class in y has only 1 member"

**Error Message:**
```
Failed to train model: The least populated class in y has only 1 member, 
which is too few. The minimum number of groups for any class cannot be less than 2.
```

**Cause:**
- Your dataset has a class with only 1 sample
- Stratified train-test splitting requires at least 2 samples per class
- This happens with very small or highly imbalanced datasets

**Solution (Automatic):**
✅ **Fixed in latest version!** The backend now automatically detects this and disables stratification.

**Manual Solutions:**

1. **Remove rare classes:**
```python
# Remove classes with < 2 samples
value_counts = df['target'].value_counts()
valid_classes = value_counts[value_counts >= 2].index
df = df[df['target'].isin(valid_classes)]
```

2. **Collect more data:**
- Ensure each class has at least 10 samples
- Ideally 30+ samples per class for reliable training

3. **Combine rare classes:**
```python
# Combine rare classes into "Other"
rare_classes = value_counts[value_counts < 10].index
df.loc[df['target'].isin(rare_classes), 'target'] = 'Other'
```

**Prevention:**
- Check class distribution before training
- Use `df['target'].value_counts()` to see class counts
- Aim for at least 10 samples per class

---

## Imbalanced Dataset Warnings

### "Dataset is highly imbalanced"

**Symptoms:**
- One class has 90%+ of samples
- Model predicts only the majority class
- High accuracy but poor performance on minority class

**Solutions:**

1. **Use appropriate metrics:**
- Don't rely on accuracy alone
- Use F1-score, precision, recall
- Check confusion matrix

2. **Apply class weights:**
```python
# In model training, use class_weight='balanced'
model = RandomForestClassifier(class_weight='balanced')
```

3. **Resample data:**
- **Oversample minority**: Duplicate minority class samples
- **Undersample majority**: Remove some majority class samples
- **SMOTE**: Synthetic Minority Over-sampling Technique

4. **Collect more minority class data:**
- Best solution if possible
- Ensures model learns minority patterns

**Check class balance:**
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))
```

---

## Small Dataset Issues

### "Not enough data to train"

**Minimum Requirements:**
- **Classification**: 50+ samples, 10+ per class
- **Regression**: 100+ samples
- **Deep Learning**: 1000+ samples

**Solutions for Small Datasets:**

1. **Use simpler models:**
- Logistic Regression instead of Random Forest
- Linear Regression instead of Neural Networks

2. **Use cross-validation:**
- 5-fold or 10-fold CV
- Better use of limited data

3. **Reduce features:**
- Feature selection
- Remove correlated features
- Use PCA

4. **Collect more data:**
- Best solution
- Even 2x more data helps significantly

---

## Data Quality Issues

### "High missing values detected"

**Solutions:**
1. Use Data Preprocessing tab
2. Choose missing value strategy:
   - Drop rows (if < 5% missing)
   - Drop columns (if > 50% missing)
   - Fill with mean/median/mode

### "Duplicate rows detected"

**Solution:**
- Enable "Remove Duplicates" in preprocessing
- Or manually: `df.drop_duplicates()`

### "Outliers detected"

**Solutions:**
1. Use "Remove Outliers" in preprocessing (IQR method)
2. Or manually cap outliers:
```python
# Cap at 1st and 99th percentile
df['column'] = df['column'].clip(
    lower=df['column'].quantile(0.01),
    upper=df['column'].quantile(0.99)
)
```

---

## Quick Fixes Summary

| Error | Quick Fix |
|-------|-----------|
| "Only 1 member in class" | ✅ Auto-fixed, or remove rare classes |
| "Failed to fetch" | Start backend: `python app.py` |
| "High missing values" | Use preprocessing tab |
| "Imbalanced dataset" | Use F1-score, class weights |
| "Not enough data" | Use simpler models, CV |
| "Overfitting" | Use overfitting reduction tab |
| "Model not learning" | Check data quality, try different model |

---

**Still having issues?** Check the [USER_GUIDE.md](./USER_GUIDE.md) or [CSV_TROUBLESHOOTING.md](./CSV_TROUBLESHOOTING.md)
