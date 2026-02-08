# NumPy Compatibility Fix

## Issue
The ISD modules are failing due to NumPy version incompatibility. Your system has NumPy 2.3.5, but some packages (scipy, pandas) were compiled with NumPy 1.x.

## Quick Fix

### Option 1: Downgrade NumPy (Recommended)
```bash
pip install "numpy<2.0"
```

Then restart the backend:
```bash
python app.py
```

### Option 2: Use Backend Virtual Environment
```bash
cd ml-visualizer/backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python app.py
```

### Option 3: Update All Packages
```bash
pip install --upgrade pandas scipy scikit-learn
```

## Verify Fix

After applying the fix, test:
```bash
python test_isd.py
```

You should see:
```
🎉 ALL TESTS PASSED! ISD is ready to use!
```

## Why This Happened

NumPy 2.0 introduced breaking changes. Packages compiled with NumPy 1.x are not compatible with NumPy 2.x. The solution is to either:
1. Use NumPy 1.x (< 2.0)
2. Update all packages to versions compiled with NumPy 2.0

## Current Workaround

The ISD modules will still work for basic functionality, but some advanced statistical features may fail. The core features (data analysis, problem classification, model recommendations) should work fine.

## Alternative: Simplified ISD

If you can't fix NumPy, I can create a simplified version that doesn't use scipy. Let me know!
