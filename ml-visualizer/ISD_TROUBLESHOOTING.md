# ISD Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: NumPy Version Incompatibility

**Symptoms:**
- Error: "numpy.dtype size changed"
- Error: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"
- ISD endpoints return 400 errors

**Solution:**
```bash
# Downgrade NumPy to 1.x
pip install "numpy<2.0"

# Or upgrade all packages
pip install --upgrade pandas scipy scikit-learn

# Restart backend
python app.py
```

### Issue 2: scipy Import Error

**Symptoms:**
- Error: "cannot import name 'stats' from 'scipy'"
- Warning: "scipy not available"

**Solution:**
The code now includes fallback methods. If you see the warning, scipy features will use simplified calculations. To get full functionality:

```bash
pip install scipy>=1.10.0
```

### Issue 3: ISD Modules Not Loading

**Symptoms:**
- Backend starts but no "✓ ISD modules loaded successfully" message
- Or shows "⚠ ISD modules not available"

**Solution:**
```bash
# Check if files exist
ls backend/isd_*.py

# Try importing manually
python -c "from isd_data_intelligence import DataIntelligenceEngine; print('OK')"

# If import fails, check error message
```

### Issue 4: 400 Error on /api/isd/analyze-complete

**Symptoms:**
- Frontend shows "Analysis failed"
- Backend logs show 400 status

**Possible Causes:**
1. **File upload issue** - Check file is valid CSV
2. **Import error** - Check backend logs for import errors
3. **Data processing error** - Check CSV has valid data

**Debug Steps:**
```bash
# 1. Check backend logs
# Look for error messages in terminal

# 2. Test with curl
curl -X POST http://localhost:5000/api/isd/health-check

# Should return:
# {"status": "operational", "modules": [...]}

# 3. Test with sample data
curl -X POST http://localhost:5000/api/isd/analyze-complete \
  -F "file=@public/sample-classification.csv"
```

### Issue 5: Slow Analysis (> 10 seconds)

**Symptoms:**
- Analysis takes very long
- Browser times out

**Solutions:**
1. **Large dataset** - ISD samples datasets > 100K rows
2. **Many features** - Reduce features or use feature selection first
3. **Complex calculations** - This is normal for very large/complex data

**Optimization:**
```python
# In isd_data_intelligence.py, reduce sample size
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)
```

### Issue 6: Missing Dependencies

**Symptoms:**
- ModuleNotFoundError
- ImportError

**Solution:**
```bash
# Install all dependencies
cd backend
pip install -r requirements.txt

# Or install individually
pip install flask flask-cors pandas numpy scipy matplotlib seaborn scikit-learn
```

### Issue 7: Frontend Not Showing ISD Tab

**Symptoms:**
- No "🎯 ISD - System Designer" tab visible

**Solution:**
1. **Check if tab is added** - Look for ISD tab button in App.jsx
2. **Rebuild frontend**:
```bash
npm run build
npm run dev
```

3. **Clear browser cache** - Hard refresh (Ctrl+Shift+R)

### Issue 8: Report Download Not Working

**Symptoms:**
- Download button doesn't work
- No file downloaded

**Solution:**
```javascript
// Check browser console for errors
// The download uses Blob API

// Manual download:
const report = /* your report data */;
const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'ml-report.json';
a.click();
```

## Quick Diagnostics

### Test Backend Health
```bash
curl http://localhost:5000/api/health
# Should return: {"status": "healthy", ...}

curl http://localhost:5000/api/isd/health-check
# Should return: {"status": "operational", "modules": [...]}
```

### Test ISD Modules
```bash
cd backend
python test_isd.py
```

Expected output:
```
🎉 ALL TESTS PASSED! ISD is ready to use!
```

### Check Python Environment
```bash
python --version
# Should be Python 3.9+

pip list | grep numpy
# Should show numpy < 2.0

pip list | grep scipy
# Should show scipy >= 1.10

pip list | grep pandas
# Should show pandas >= 2.0
```

## Environment Setup (Clean Install)

If nothing works, try a clean install:

```bash
# 1. Create virtual environment
cd ml-visualizer/backend
python -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify
python test_isd.py

# 6. Run backend
python app.py
```

## Still Having Issues?

### Check Logs
1. **Backend logs** - Terminal where `python app.py` is running
2. **Browser console** - F12 → Console tab
3. **Network tab** - F12 → Network tab → Look for failed requests

### Common Error Messages

**"Cannot connect to backend"**
- Backend not running
- Wrong port (should be 5000)
- Firewall blocking

**"Analysis failed. Please try again."**
- Check backend logs for actual error
- Try with sample CSV first
- Check CSV format (no special characters, valid encoding)

**"Module not found"**
- Missing dependency
- Run: `pip install -r requirements.txt`

**"Memory error"**
- Dataset too large
- Reduce sample size in code
- Use more powerful machine

## Performance Tips

1. **Use smaller datasets for testing** (< 10K rows)
2. **Remove unnecessary columns** before upload
3. **Use CSV with UTF-8 encoding**
4. **Close other applications** to free memory
5. **Use virtual environment** to avoid conflicts

## Getting Help

If you're still stuck:

1. **Check documentation**:
   - ISD_DOCUMENTATION.md
   - ISD_QUICK_START.md
   - ISD_IMPLEMENTATION_SUMMARY.md

2. **Run diagnostics**:
   ```bash
   python test_isd.py
   ```

3. **Check GitHub issues** (if applicable)

4. **Provide error details**:
   - Python version
   - NumPy version
   - Full error message
   - Backend logs
   - Browser console errors

## Success Checklist

- ✅ Python 3.9+ installed
- ✅ NumPy < 2.0 installed
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Backend starts without errors
- ✅ "✓ ISD modules loaded successfully" message appears
- ✅ Health check returns operational: `curl http://localhost:5000/api/isd/health-check`
- ✅ Test suite passes: `python test_isd.py`
- ✅ Frontend shows ISD tab
- ✅ Can upload CSV and get analysis

If all checkboxes are ✅, ISD is working correctly!

---

**Need more help?** Check the main documentation or create an issue with full error details.
