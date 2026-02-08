# Automate_DS - ML Data Analysis & Model Predictor

A production-ready web application that automates machine learning workflows with real sklearn, seaborn, and matplotlib integration. Features an AI-powered Intelligent System Designer (ISD) that acts as your ML architect.

## 🌟 Key Features

### 🤖 Intelligent System Designer (ISD)
Your AI ML Architect that analyzes datasets and recommends complete ML system architectures:
- **Data Health Score**: 20+ quality checks with actionable repair suggestions
- **Problem Classification**: Automatic detection of problem type, complexity, and risk
- **Model Recommendations**: Scored algorithm rankings with detailed reasoning
- **Failure Prediction**: Overfitting/underfitting risk assessment before training
- **Professional Reports**: Downloadable architecture reports with implementation roadmap

### 🐍 Real Python ML Backend
- Train actual sklearn models (RandomForest, LogisticRegression, XGBoost, etc.)
- Professional seaborn and matplotlib visualizations
- Real sklearn metrics, cross-validation, and feature importance
- Complete ML pipeline: preprocessing, training, evaluation, visualization

### 🧠 AI Assistant
- Context-aware ML guidance using GPT-4
- Analyzes your specific dataset and provides recommendations
- Generates code examples and preprocessing strategies
- Screen vision to see your current analysis

### 📊 Comprehensive Visualizations
- 10+ chart types for exploratory data analysis
- Feature importance, correlation heatmaps, box plots
- Class balance analysis and missing data patterns
- Distribution statistics with transformation recommendations

### 🛡️ Advanced ML Tools
- **Overfitting Reduction**: 6 powerful techniques with before/after comparison
- **GridSearchCV**: Automated hyperparameter tuning with visual results
- **Deep Learning Builder**: Visual neural network designer with 12 layer types
- **Full Model Training**: Complete performance metrics and deployment code

### 🎯 Smart Recommendations
- Automatic model selection based on dataset characteristics
- Preprocessing roadmap with priority levels
- Feature engineering suggestions with impact assessment
- Evaluation strategy recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key (for AI Assistant)

### 1. Start Python Backend

```bash
cd ml-visualizer/backend

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

Backend runs on `http://localhost:5000`

### 2. Start Frontend

```bash
cd ml-visualizer

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on `http://localhost:5173`

### 3. Upload and Analyze

1. Open `http://localhost:5173` in your browser
2. Verify backend status shows "connected"
3. Upload a CSV file
4. Click "Analyze Data" or use the ISD for comprehensive analysis

## 📚 Documentation

- **[User Guide](ml-visualizer/USER_GUIDE.md)** - Complete usage guide
- **[Quick Start](ml-visualizer/QUICK_START.md)** - Get started quickly
- **[ISD Documentation](ml-visualizer/ISD_DOCUMENTATION.md)** - Intelligent System Designer guide
- **[API Reference](ml-visualizer/API_ENDPOINTS.md)** - Backend API documentation
- **[Troubleshooting](ml-visualizer/TROUBLESHOOTING.md)** - Common issues and solutions

## 🛠️ Tech Stack

### Frontend
- React 19
- Vite
- Tailwind CSS
- Lucide React (icons)
- Recharts (visualizations)
- OpenAI API (AI Assistant)

### Backend
- Python 3.8+
- Flask
- scikit-learn
- XGBoost & LightGBM
- pandas, numpy
- seaborn, matplotlib
- TensorFlow/Keras (Deep Learning)

## 📊 Performance

Optimized for large datasets:
- Handles 10,000+ rows efficiently
- Smart sampling for large datasets
- Maximum 100MB file size
- Memory-optimized processing

## 🎯 Use Cases

- **Data Scientists**: Rapid prototyping and model selection
- **ML Engineers**: Production-ready code generation
- **Students**: Learn ML best practices with guided recommendations
- **Researchers**: Quick exploratory data analysis and visualization
- **Business Analysts**: Automated insights from data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Repository**: [Upashak653/Automate_DS](https://github.com/Upashak653/Automate_DS)
- **Issues**: [Report a bug](https://github.com/Upashak653/Automate_DS/issues)

## 🙏 Acknowledgments

Built with modern ML tools and frameworks to automate the data science workflow.