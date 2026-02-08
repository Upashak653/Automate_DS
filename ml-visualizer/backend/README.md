# ML Visualizer Backend

Flask backend with sklearn, TensorFlow, and matplotlib for ML analysis.

## Quick Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server
python app.py
```

Server runs on `http://localhost:5000`

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Dataset analysis
- `POST /api/visualize` - Generate plots
- `POST /api/train-model` - Train ML model
- `POST /api/recommend-model` - Get model recommendations
- `POST /api/preprocess` - Data preprocessing
- `POST /api/compare-models` - Compare multiple models
- `POST /api/grid-search` - Hyperparameter tuning
- `POST /api/reduce-overfitting` - Apply overfitting techniques
- `POST /api/train-deep-learning` - Train neural network

## Technologies

- Flask, pandas, numpy, scikit-learn
- TensorFlow/Keras for deep learning
- matplotlib, seaborn for visualization
- XGBoost, LightGBM for gradient boosting

## Notes

- Plots returned as base64-encoded PNG
- Auto-detects classification vs regression
- Handles missing values and categorical encoding
- Features auto-scaled before training
