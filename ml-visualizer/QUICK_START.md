# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Start Python Backend ⚡

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

✅ You should see: `Running on http://127.0.0.1:5000`

### Step 2: Start Frontend 🎨

Open a **new terminal**:

```bash
npm install
npm run dev
```

✅ You should see: `Local: http://localhost:5173`

### Step 3: Analyze Data 📊

1. Open `http://localhost:5173` in your browser
2. Check that backend status shows **"Backend: connected"** (green)
3. Click "Click to upload CSV file"
4. Select your dataset
5. Click one of the buttons:
   - **Analyze Data** - Get sklearn statistics + **Best Model Recommendation** 🎯
   - **Generate Plots** - Create seaborn visualizations
   - **Train Model** - Train actual sklearn models

## 🎯 What You Get

### Analyze Data (sklearn + pandas)
- Dataset shape and statistics
- Column types (numeric/categorical)
- Missing value analysis
- Problem type detection (classification/regression)
- Target column identification
- **🎯 Best Model Recommendation** - Intelligent model selection based on your data!

### Generate Plots (seaborn + matplotlib)
- **Correlation Heatmap** - Feature relationships
- **Distribution Plots** - Histograms with KDE curves
- **Box Plots** - Outlier detection
- **Pairplot** - Feature scatter matrix
- **Missing Data** - Visualization of missing values
- **Target Distribution** - Target variable analysis

### Train Model (sklearn)
- **Actual Model Training** - RandomForest or LogisticRegression
- **Metrics** - Accuracy, R², RMSE, etc.
- **Confusion Matrix** - For classification
- **Actual vs Predicted** - For regression
- **Feature Importance** - From trained model
- **Cross-Validation** - 5-fold CV scores

## 📝 Example Workflow

```bash
# Terminal 1: Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Terminal 2: Frontend
npm install
npm run dev

# Browser: http://localhost:5173
# 1. Upload iris.csv
# 2. Click "Analyze Data" → See sklearn analysis
# 3. Click "Generate Plots" → See seaborn plots
# 4. Click "Train Model" → Train RandomForest
```

## ⚠️ Troubleshooting

### Backend shows "disconnected"
```bash
# Check if backend is running
curl http://localhost:5000/api/health

# If not, start it:
cd backend
python app.py
```

### "ModuleNotFoundError"
```bash
# Install dependencies
pip install -r requirements.txt
```

### Port already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:5000 | xargs kill -9
```

## 📚 Next Steps

- Read [PYTHON_BACKEND_GUIDE.md](./PYTHON_BACKEND_GUIDE.md) for detailed API docs
- Read [MIGRATION_TO_BACKEND.md](./MIGRATION_TO_BACKEND.md) for architecture details
- Check [backend/README.md](./backend/README.md) for API endpoints

## 🎉 You're Ready!

Your ML Visualizer is now running with:
- ✅ Real sklearn models
- ✅ Professional seaborn plots
- ✅ Accurate calculations
- ✅ Production-ready analysis

Upload your dataset and start analyzing! 🚀
