# ML Data Analysis & Model Predictor

A production-ready web application that uses **real sklearn, seaborn, and matplotlib** for ML analysis and visualization.

**📚 [User Guide](./USER_GUIDE.md)** | **🚀 [Quick Start](./QUICK_START.md)** | **🔌 [API Reference](./API_ENDPOINTS.md)** | **🔧 [Troubleshooting](./TROUBLESHOOTING.md)**

## 🎯 NEW: Intelligent System Designer (ISD)

**Your AI ML Architect** - Analyzes datasets and recommends complete ML system architectures like a senior engineer!

- **🏥 Data Health Score**: 20+ quality checks with actionable repair suggestions
- **🎯 Problem Classification**: Automatic detection of problem type, complexity, and risk
- **🤖 Model Recommendations**: Scored algorithm rankings with detailed reasoning
- **⚠️ Failure Prediction**: Overfitting/underfitting risk before training
- **📋 Professional Reports**: Downloadable architecture reports with implementation roadmap
- **🚀 Production Ready**: Industry-grade code suitable for FAANG interviews

**[📖 ISD Documentation](./ISD_DOCUMENTATION.md)** | **[🚀 ISD Quick Start](./ISD_QUICK_START.md)** | **[📊 Implementation Summary](./ISD_IMPLEMENTATION_SUMMARY.md)**

## Features

### 🐍 Python Backend (NEW!)
- **Real sklearn Models**: Train actual RandomForest, LogisticRegression, LinearRegression
- **Seaborn Visualizations**: Professional plots using seaborn and matplotlib
- **Actual Calculations**: Real sklearn metrics, cross-validation, feature importance
- **Base64 Plots**: High-quality matplotlib plots sent to frontend
- **Automatic Detection**: Problem type, missing values, categorical encoding
- **Full ML Pipeline**: Data preprocessing, model training, evaluation, visualization

### 🌙 Dark Mode
- Toggle between light and dark themes
- Preference saved locally
- Easy on the eyes for long analysis sessions

### 🤖 AI Assistant
- **Intelligent ML Guidance**: Get real-time advice from an AI assistant that can see your data
- **Context-Aware**: The AI understands your dataset, visualizations, and analysis results
- **OpenAI Integration**: Uses GPT-4 to provide expert ML recommendations
- **Quick Questions**: Pre-built prompts for common ML scenarios
- **Privacy First**: API key stored locally in your browser
- **Screen Vision**: Toggle to let AI see your current analysis or ask general questions

### 📊 Comprehensive ML Visualizations

#### Basic EDA
- **Histograms**: Distribution analysis for numeric features with mean, median, range
- **Bar Charts**: Frequency analysis for categorical features (top 10 values)
- **Scatter Plots**: Pairwise relationship visualization between numeric features

#### Advanced ML Analysis
- **Feature Importance**: Variance-based ranking to identify most predictive features
- **Correlation Heatmap**: Detect multicollinearity with color-coded correlation matrix
- **Box Plots**: Outlier detection with quartiles, IQR, and outlier counts
- **Class Balance Analysis**: Identify imbalanced classes with severity indicators
- **Missing Data Pattern**: Visualize missing data severity across features
- **Distribution Statistics**: Skewness & kurtosis analysis with transformation recommendations
- **Feature Radar Chart**: Normalized feature comparison for understanding scales
- **ML Preprocessing Recommendations**: Automated suggestions for data cleaning, feature engineering, scaling, and model selection

### 🎯 Automatic Model Prediction
- Intelligently analyzes your dataset and predicts the best ML model
- Scoring system evaluates multiple factors: dataset size, feature types, missing data, dimensionality
- Provides confidence scores for each model recommendation

### 🛡️ Overfitting Reduction (NEW!)
- **Before/After Comparison**: See the impact of overfitting reduction techniques
- **6 Powerful Techniques**: Cross-validation, regularization, early stopping, tree pruning, feature selection, ensemble methods
- **Visual Analysis**: Compare train vs test scores with detailed plots
- **Smart Recommendations**: Get personalized suggestions based on your results
- **Generalization Metrics**: Track overfitting gap and model generalization
- **[Full Guide](./OVERFITTING_GUIDE.md)**: Comprehensive documentation on reducing overfitting

### 🔍 GridSearchCV - Automated Hyperparameter Tuning (NEW!)
- **No Google Colab Needed**: Complete hyperparameter optimization in your browser
- **Exhaustive Search**: Tests every parameter combination automatically
- **6 Pre-configured Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, Logistic Regression
- **Custom Parameter Grids**: Define your own ranges or use optimized defaults
- **Visual Results**: See parameter impact and score distributions
- **Production-Ready Code**: Get copy-paste Python code with best parameters
- **Top 5 Combinations**: Compare best parameter sets with scores
- **[Full Guide](./GRIDSEARCH_GUIDE.md)**: Complete GridSearchCV documentation

### 🚀 Full Model Training (NEW!)
- **Complete Performance Metrics**: Train any model and get comprehensive evaluation
- **No Manual Coding**: Everything automated - no need for Google Colab
- **Use GridSearch Results**: Apply best parameters directly
- **Deployment Code**: Get production-ready code with model saving/loading
- **Cross-Validation**: Built-in CV for robust performance estimates
- **Overfitting Analysis**: Automatic detection and recommendations

### 🧠 Deep Learning Builder (NEW!)
- **Visual Neural Network Designer**: Build networks layer by layer with drag-and-drop simplicity
- **12 Layer Types**: Dense, Conv1D, Conv2D, MaxPooling, LSTM, GRU, SimpleRNN, Bidirectional, Flatten, Dropout, BatchNorm
- **6 Architecture Templates**: Feedforward, CNN, RNN, LSTM, GRU, Encoder-Decoder
- **8 Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, Linear, Leaky ReLU, ELU, SELU
- **Real TensorFlow/Keras**: Train actual deep learning models, not approximations
- **CNN Support**: Convolutional layers for images and spatial data
- **RNN/LSTM/GRU**: Recurrent layers for sequences and time series
- **Bidirectional RNNs**: Process sequences in both directions
- **Architecture Visualization**: See your network structure visually
- **Training Curves**: Loss and accuracy plots in real-time
- **Dropout & Regularization**: Built-in overfitting prevention
- **Multiple Optimizers**: Adam, SGD, RMSprop, Adagrad, Adamax
- **Early Stopping**: Automatic training termination when validation plateaus
- **Production Code**: Get TensorFlow/Keras code to deploy your model
- **[Full Guide](./DEEP_LEARNING_GUIDE.md)**: Complete deep learning documentation
- **[Layer Types Guide](./LAYER_TYPES_GUIDE.md)**: Complete reference for all 12 layer types

### 🧠 AI Recommendation System
- **Intelligent Analysis**: Automatically analyzes your dataset statistics and provides personalized recommendations
- **Dataset Score**: Overall quality score (0-100) based on size, quality, and feature ratio
- **Key Insights**: Identifies data size issues, dimensionality problems, missing values, and class imbalance
- **Model Strategy**: Detailed advice on primary model, alternatives, ensemble methods, and hyperparameter focus
- **Preprocessing Roadmap**: Step-by-step preprocessing plan with priority levels and code examples
- **Feature Engineering**: Suggested techniques with impact assessment and complexity ratings
- **Evaluation Strategy**: Recommended cross-validation methods and metrics based on your dataset
- **Potential Issues**: Identifies overfitting risks, data leakage, and computational complexity with solutions

### 📊 Data Analysis
- **Data Validation**: Checks dataset quality and identifies issues
- **Exploratory Data Analysis**: Analyzes each column with statistics
- **Problem Type Detection**: Automatically identifies classification vs regression tasks
- **Target Variable Identification**: Suggests the most likely target column

### 🧠 Smart Model Recommendations
Recommends from a variety of models including:
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Neural Networks, Naive Bayes
- **Regression**: Linear Regression, Ridge/Lasso, Random Forest, Gradient Boosting, SVR, Neural Networks

Each recommendation includes:
- Priority level (Best Match, High, Medium, Low)
- Confidence score (0-100%)
- Detailed reasoning
- Pros and cons
- When to use it

### 🎓 Key Capabilities

**AI Assistant**:
- Analyzes your specific dataset
- Explains model recommendations
- Provides preprocessing strategies
- Generates code examples
- Answers ML questions in real-time

**Visualizations**:
- 10+ chart types for comprehensive EDA
- Outlier detection with severity levels
- Multicollinearity identification
- Class imbalance analysis
- Missing data pattern visualization

**Smart Recommendations**:
- Automated preprocessing checklist
- Feature engineering suggestions
- Scaling strategy selection
- Model-specific hyperparameter guidance

### 💡 Data Insights
Provides actionable insights about your data:
- Dataset size recommendations
- Feature engineering suggestions
- Missing data handling tips
- Dimensionality considerations
- Categorical feature handling

## How It Works

1. **Upload CSV**: Drop your CSV file with headers
2. **Automatic Analysis**: The app analyzes:
   - Dataset size and shape
   - Feature types (numeric vs categorical)
   - Missing values
   - Data distributions
   - Feature-to-sample ratio
   - Cardinality of categorical features
3. **Model Scoring**: Each model is scored based on:
   - Dataset size compatibility
   - Feature type suitability
   - Handling of missing data
   - Computational efficiency
   - Expected performance
4. **Get Recommendations**: Receive ranked model suggestions with the best match highlighted

## Getting Started

### ⚠️ IMPORTANT: Backend Required

**This application REQUIRES the Python backend to function!**

The frontend is a lightweight UI that displays results from sklearn-based analysis. Without the backend running, you will see "Failed to fetch" errors.

**Quick Check:** Open http://localhost:5000/api/health - if it works, backend is running!

### 📁 CSV File Support

The backend automatically handles:
- ✅ Inconsistent columns (skips bad rows)
- ✅ Different encodings (UTF-8, latin-1)
- ✅ Various delimiters (comma, semicolon, tab)
- ✅ Malformed CSV files

See [CSV_TROUBLESHOOTING.md](./CSV_TROUBLESHOOTING.md) if you encounter issues.

### Step 1: Start Python Backend (Required)

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

Backend will run on `http://localhost:5000`

### Step 2: Start Frontend

```bash
# In a new terminal, navigate to project root
cd ml-visualizer

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run on `http://localhost:5173`

### Step 3: Upload CSV and Analyze

1. Open `http://localhost:5173` in your browser
2. Verify backend status shows "connected"
3. Upload a CSV file
4. Click "Analyze Data", "Generate Plots", or "Train Model"

See [backend/README.md](./backend/README.md) for detailed API documentation.

## Performance & Large Datasets

This app is optimized to handle datasets with **10,000+ rows** efficiently:

- **Smart Sampling**: Automatically samples large datasets (10k for analysis, 5k for visualizations)
- **File Size Limit**: Maximum 100MB file size
- **Memory Optimized**: Efficient data processing prevents browser crashes
- **Fast Loading**: Optimized algorithms for quick analysis

### Performance Metrics

| Dataset Size | Load Time | Status |
|-------------|-----------|--------|
| < 10,000 rows | ~1s | Full analysis |
| 10,000-50,000 rows | ~2-3s | Sampled |
| 50,000-100,000 rows | ~3-4s | Sampled |

When data is sampled, you'll see an indicator showing "Analyzed: 10,000" in the UI.

### Testing with Large Datasets

Generate test CSV files:

```bash
node generate-test-data.js
```

This creates test files with 1k, 10k, 50k, and 100k rows for performance testing.

See [PERFORMANCE_OPTIMIZATIONS.md](./PERFORMANCE_OPTIMIZATIONS.md) for technical details.

## Using the AI Assistant

1. **Get OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Click AI Assistant**: Purple floating button in bottom-right corner
3. **Enter API Key**: Click settings (⚙️) and paste your key
4. **Start Chatting**: Ask questions about your data and get expert ML advice!

The AI can see your data analysis and provide context-aware recommendations. See [AI Assistant Guide](./AI_ASSISTANT_GUIDE.md) for detailed usage.

## Sample Data

Two sample CSV files are included in the `public` folder:
- `sample-classification.csv` - Loan approval prediction
- `sample-regression.csv` - House price prediction

## Tech Stack

### Frontend
- React 19
- Vite
- Tailwind CSS
- Lucide React (icons)
- OpenAI API (AI Assistant)

### Backend (Python)
- Python 3.8+
- Flask (web framework)
- **scikit-learn** (machine learning)
- **XGBoost** (gradient boosting)
- **LightGBM** (fast gradient boosting)
- pandas (data manipulation)
- seaborn (visualization)
- matplotlib (plotting)
- numpy (numerical computing)

## Architecture

This application uses a **Python Flask backend** for all ML calculations and visualizations:

- ✅ **Real sklearn models** (RandomForest, LogisticRegression, etc.)
- ✅ **Professional seaborn plots** (publication-quality)
- ✅ **Accurate sklearn calculations** (real metrics, cross-validation)
- ✅ **Production-ready analysis**
- ✅ **Actual feature importance** from trained models
- ⚠️ **Requires Python backend** to be running

**Note**: The frontend is now a lightweight React UI that displays results from the Python backend. All calculations are done server-side using actual sklearn, pandas, seaborn, and matplotlib libraries.

See [PYTHON_BACKEND_GUIDE.md](./PYTHON_BACKEND_GUIDE.md) for detailed setup and usage.

## New Features Added

### 🔧 Data Preprocessing
- Handle missing data (mean/median/mode imputation)
- Remove outliers (IQR method)
- Feature scaling (StandardScaler/MinMaxScaler)
- Preview and download processed data

### ✨ Feature Engineering
- Automatic suggestions based on data
- Polynomial features, binning, log transforms
- Ratio features, encoding strategies
- Copy-paste Python code

### ⚙️ Hyperparameter Tuning
- Model-specific parameter ranges
- GridSearchCV/RandomizedSearchCV code
- Optimized for your dataset size
- Top 3 models with complete tuning scripts

### 🕐 Time Series Detection
- Auto-detect date/time columns
- Feature extraction (year, month, day, etc.)
- Lag features and rolling statistics
- ARIMA, Prophet, LSTM recommendations

### 📝 Text Feature Detection
- Auto-detect text columns
- NLP techniques (TF-IDF, embeddings, BERT)
- Text feature engineering
- Ready-to-use code examples

### 🌙 Dark Mode
- Toggle light/dark theme
- Saved preference
- All components support dark mode

## Documentation

- **[AI Assistant Guide](./AI_ASSISTANT_GUIDE.md)** - How to use the AI assistant
- **[ML Visualizations Guide](./ML_VISUALIZATIONS_GUIDE.md)** - Interpret charts and make decisions

## Model Selection Algorithm

The app uses a sophisticated scoring system that considers:
- **Dataset Size**: Small (<1000), Medium (1000-5000), Large (>5000)
- **Feature Ratio**: Number of features relative to samples
- **Categorical Ratio**: Proportion of categorical features
- **Missing Data**: Presence and percentage of missing values
- **High Cardinality**: Features with many unique values
- **Dimensionality**: Total number of features

Each model receives a base score adjusted by these factors, ensuring the recommendation matches your specific dataset characteristics.
