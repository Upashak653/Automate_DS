# 📚 ML Visualizer - Complete User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Features Overview](#features-overview)
3. [Data Analysis](#data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [GridSearchCV](#gridsearchcv)
7. [Overfitting Reduction](#overfitting-reduction)
8. [Deep Learning](#deep-learning)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python app.py
```

### 2. Setup Frontend
```bash
npm install
npm run dev
```

### 3. Use the App
1. Open http://localhost:5173
2. Upload CSV file
3. Explore tabs: Analysis, Preprocessing, Models, GridSearch, Deep Learning

---

## Features Overview

### ✅ What This Tool Does
- **Automatic ML**: No coding required
- **Real sklearn/TensorFlow**: Actual ML libraries, not approximations
- **Complete Pipeline**: Data → Preprocessing → Training → Deployment
- **Visual Interface**: Build models visually
- **Production Code**: Get copy-paste ready code

### 🎯 Use Cases
- Quick ML prototyping
- Learning machine learning
- Model comparison
- Hyperparameter tuning
- Deep learning experiments

---

## Data Analysis

**Tab**: Analysis

**What it does**: Analyzes your dataset and provides insights

**Features**:
- Dataset statistics (rows, columns, types)
- Missing value detection
- Correlation analysis
- Distribution plots
- Problem type detection (classification/regression)

**How to use**:
1. Upload CSV file
2. View automatic analysis
3. Check visualizations
4. Note problem type

---

## Data Preprocessing

**Tab**: Data Preprocessing

**Options**:
- **Remove Duplicates**: Clean duplicate rows
- **Handle Missing Values**: Drop, fill with mean/median/mode
- **Remove Outliers**: IQR method
- **Encode Categorical**: Label encoding
- **Standardize**: Z-score normalization (mean=0, std=1)
- **Normalize**: Min-max scaling (0-1 range)
- **Feature Selection**: Remove low-variance features
- **PCA**: Dimensionality reduction

**Best Practices**:
- Always check for missing values
- Standardize for neural networks
- Normalize for distance-based algorithms
- Use PCA for high-dimensional data

---

## Model Training

**Tab**: Train Model

**Available Models**:
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Logistic Regression
- Linear Regression
- SVM
- KNN

**What you get**:
- Performance metrics (accuracy, precision, recall, F1, R²)
- Confusion matrix / predictions plot
- Feature importance
- Cross-validation scores
- Overfitting analysis
- Production-ready code

**Tips**:
- Check model recommendations first
- Compare multiple models
- Watch for overfitting (train vs test gap)

---

## GridSearchCV

**Tab**: GridSearchCV

**What it does**: Automatically finds best hyperparameters

**How to use**:
1. Select model (Random Forest, XGBoost, etc.)
2. Configure CV settings (folds, scoring)
3. Edit parameter grid or use defaults
4. Click "Start Grid Search"
5. Get best parameters + code

**Parameter Grids**:
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, subsample
- **Logistic**: C, penalty, solver

**Results**:
- Best parameters found
- CV score and test score
- Top 5 combinations
- Parameter importance plot
- Ready-to-use code

**Time Estimates**:
- Small grid (< 50 combinations): 10-30 seconds
- Medium grid (50-200): 1-3 minutes
- Large grid (> 200): 3-10 minutes

---

## Overfitting Reduction

**Tab**: Reduce Overfitting

**When to use**: When train score >> test score

**6 Techniques**:

1. **Cross-Validation** 🔄
   - Use k-fold CV for better generalization
   - Best for: All datasets

2. **Regularization** ⚖️
   - L1/L2 penalties
   - Best for: Linear models, high dimensions

3. **Early Stopping** ⏹️
   - Stop when validation plateaus
   - Best for: Gradient boosting

4. **Tree Pruning** ✂️
   - Limit depth and samples
   - Best for: Tree-based models

5. **Feature Selection** 🎯
   - Remove noisy features
   - Best for: High-dimensional data

6. **Ensemble Methods** 🤝
   - Bagging multiple models
   - Best for: Medium/large datasets

**Interpreting Results**:
- Gap < 0.05: ✅ Excellent
- Gap 0.05-0.10: ✅ Good
- Gap 0.10-0.15: ⚠️ Fair
- Gap > 0.15: ❌ Poor

---

## Deep Learning

**Tab**: Deep Learning

### Architecture Templates

1. **Feedforward** 🔷
   - Standard neural network
   - For: Tabular data

2. **CNN** 🖼️
   - Convolutional layers
   - For: Images, spatial data, time series

3. **RNN** ↩️
   - Simple recurrent
   - For: Short sequences

4. **LSTM** 🔄
   - Long Short-Term Memory
   - For: Long sequences, complex patterns

5. **GRU** 🌀
   - Gated Recurrent Unit
   - For: Sequences, faster than LSTM

6. **Encoder-Decoder** 🔀
   - Sequence-to-sequence
   - For: Translation, seq2seq tasks

### 12 Layer Types

**Basic**:
- Dense: Fully connected
- Dropout: Regularization
- BatchNorm: Normalize activations

**Convolutional**:
- Conv1D: For sequences
- Conv2D: For images
- MaxPool1D/2D: Downsampling
- Flatten: Convert to 1D

**Recurrent**:
- LSTM: Long-term memory
- GRU: Efficient recurrent
- SimpleRNN: Basic recurrent
- Bidirectional: Both directions

### Activation Functions

- **ReLU**: Default for hidden layers
- **Sigmoid**: Binary classification output
- **Tanh**: Zero-centered alternative
- **Softmax**: Multi-class output
- **Linear**: Regression output
- **Leaky ReLU**: Fixes dying ReLU
- **ELU**: Smooth negatives
- **SELU**: Self-normalizing

### Training Configuration

- **Epochs**: 1-500 (default: 50)
- **Batch Size**: 1-512 (default: 32)
- **Optimizers**: Adam, SGD, RMSprop, Adagrad, Adamax
- **Learning Rate**: 0.0001-0.1 (default: 0.001)
- **Early Stopping**: Auto-stop when validation plateaus

### Common Patterns

**Simple Classification**:
```
Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Output
```

**CNN for Time Series**:
```
Conv1D(64) → MaxPool → Conv1D(32) → MaxPool → Flatten → Dense(64) → Output
```

**LSTM for Sequences**:
```
LSTM(128, return_seq=True) → LSTM(64) → Dense(32) → Output
```

**Bidirectional LSTM**:
```
Bidirectional(LSTM(64)) → Dense(32) → Output
```

---

## Troubleshooting

### CSV Upload Issues

**"Failed to parse CSV"**
- Check file encoding (try UTF-8)
- Verify CSV format
- Remove special characters
- Check for consistent columns

**"Too many missing values"**
- Use preprocessing to handle missing data
- Consider dropping columns with >50% missing

### Model Training Issues

**"Model not learning"**
- Check data preprocessing
- Try different model
- Adjust hyperparameters
- Verify target variable

**"Overfitting detected"**
- Use overfitting reduction tab
- Add regularization
- Get more training data
- Simplify model

**"Training too slow"**
- Reduce dataset size (sample)
- Use simpler model
- Reduce parameter grid
- Use fewer CV folds

### Deep Learning Issues

**"Loss is NaN"**
- Reduce learning rate (0.0001)
- Check data normalization
- Verify no extreme values

**"Not learning"**
- Increase learning rate
- Add more layers
- Check activation functions
- Verify data preprocessing

**"Shape mismatch"**
- Add Flatten after Conv/Pool layers
- Set return_sequences=True for stacked RNNs
- Check input shape

---

## Best Practices

### 1. Data Preparation
- Always analyze data first
- Handle missing values
- Encode categorical variables
- Scale/normalize features

### 2. Model Selection
- Check recommendations
- Try top 3 models
- Compare performance
- Use GridSearch for best model

### 3. Hyperparameter Tuning
- Start with defaults
- Use GridSearch
- Refine based on results
- Balance time vs accuracy

### 4. Overfitting Prevention
- Monitor train vs test scores
- Use cross-validation
- Apply regularization
- Get more data if possible

### 5. Deep Learning
- Start simple (2-3 layers)
- Use early stopping
- Monitor validation loss
- Experiment systematically

---

## Keyboard Shortcuts

- **Ctrl+U**: Upload file
- **Tab**: Navigate between tabs
- **Enter**: Submit forms
- **Esc**: Close modals

---

## Tips & Tricks

### Quick Workflow
1. Upload → 2. Analyze → 3. Preprocess → 4. Train → 5. Deploy

### Model Comparison
1. Train multiple models
2. Compare metrics
3. Use GridSearch on best
4. Check overfitting
5. Deploy winner

### Deep Learning Workflow
1. Choose template
2. Adjust layers
3. Configure training
4. Train and evaluate
5. Iterate if needed

---

## Getting Help

- **README.md**: Project overview
- **TROUBLESHOOTING.md**: Common issues
- **LAYER_TYPES_GUIDE.md**: Deep learning layers
- **GitHub Issues**: Report bugs

---

## API Reference

All endpoints available at `http://localhost:5000/api/`

- `/analyze`: Dataset analysis
- `/visualize`: Generate plots
- `/train-model`: Train ML model
- `/recommend-model`: Get recommendations
- `/preprocess`: Data preprocessing
- `/compare-models`: Compare multiple models
- `/grid-search`: Hyperparameter tuning
- `/reduce-overfitting`: Apply overfitting techniques
- `/train-deep-learning`: Train neural network

---

**Remember**: Start simple, iterate, and have fun! 🚀
