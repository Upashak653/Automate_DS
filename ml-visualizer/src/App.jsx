import React, { useState, useEffect } from 'react';
import { Upload, TrendingUp, AlertCircle, CheckCircle, Image, Play, RefreshCw, BarChart3 } from 'lucide-react';
import AIAssistant from './components/AIAssistant';
import DarkModeToggle from './components/DarkModeToggle';
import DataPreprocessor from './components/DataPreprocessor';
import OverfittingReducer from './components/OverfittingReducer';
import GridSearchCV from './components/GridSearchCV';
import DeepLearningBuilder from './components/DeepLearningBuilder';
import IntelligentSystemDesigner from './components/IntelligentSystemDesigner';
import AIOverfittingAgent from './components/AIOverfittingAgent';

const BACKEND_URL = 'http://localhost:5000';

export default function AppWithBackend() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [analysis, setAnalysis] = useState(null);
  const [plots, setPlots] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [selectedModel, setSelectedModel] = useState('random_forest');
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/health`);
      if (response.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('error');
      }
    } catch {
      setBackendStatus('disconnected');
    }
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    if (!uploadedFile.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    setFile(uploadedFile);
    setError(null);
    
    // Automatically analyze when file is uploaded
    if (backendStatus === 'connected') {
      await analyzeData(uploadedFile);
    }
  };

  const analyzeData = async (fileToAnalyze = file) => {
    if (!fileToAnalyze) return;
    
    setLoading(true);
    setError(null);
    setActiveTab('analysis');
    
    try {
      const formData = new FormData();
      formData.append('file', fileToAnalyze);
      
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }
      
      const data = await response.json();
      setAnalysis(data);
      
      // Also get model recommendations
      await getRecommendations(fileToAnalyze);
    } catch (err) {
      let errorMessage = 'Failed to analyze data: ' + err.message;
      
      // Add helpful hints for common errors
      if (err.message.includes('tokenizing') || err.message.includes('fields')) {
        errorMessage += '\n\n💡 Tip: Your CSV file may have inconsistent columns. The backend will try to skip problematic rows automatically.';
      } else if (err.message.includes('encoding')) {
        errorMessage += '\n\n💡 Tip: Try saving your CSV with UTF-8 encoding.';
      }
      
      setError(errorMessage);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getRecommendations = async (fileToAnalyze = file) => {
    if (!fileToAnalyze) return;
    
    try {
      const formData = new FormData();
      formData.append('file', fileToAnalyze);
      
      const response = await fetch(`${BACKEND_URL}/api/recommend-model`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Recommendation failed');
      }
      
      const data = await response.json();
      setRecommendations(data);
    } catch (err) {
      console.error('Failed to get recommendations:', err);
    }
  };

  const generatePlots = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setActiveTab('plots');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${BACKEND_URL}/api/visualize`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Visualization failed');
      }
      
      const data = await response.json();
      setPlots(data);
    } catch (err) {
      let errorMessage = 'Failed to generate plots: ' + err.message;
      
      if (err.message === 'Failed to fetch') {
        errorMessage = '❌ Cannot connect to backend!\n\nMake sure backend is running: cd backend && python app.py';
      }
      
      setError(errorMessage);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setActiveTab('model');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_type', selectedModel);
      
      const response = await fetch(`${BACKEND_URL}/api/train-model`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Model training failed');
      }
      
      const data = await response.json();
      setModelResults(data);
    } catch (err) {
      let errorMessage = 'Failed to train model: ' + err.message;
      
      // Add helpful hints for common errors
      if (err.message === 'Failed to fetch') {
        errorMessage = '❌ Cannot connect to backend server!\n\n' +
                      '🔧 Solutions:\n' +
                      '1. Start the backend: cd backend && python app.py\n' +
                      '2. Check http://localhost:5000/api/health\n' +
                      '3. Ensure no firewall blocking port 5000\n\n' +
                      'See TROUBLESHOOTING.md for detailed help.';
      }
      
      setError(errorMessage);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const compareModels = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setActiveTab('comparison');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${BACKEND_URL}/api/compare-models`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Model comparison failed');
      }
      
      const data = await response.json();
      setModelComparison(data);
    } catch (err) {
      let errorMessage = 'Failed to compare models: ' + err.message;
      
      if (err.message === 'Failed to fetch') {
        errorMessage = '❌ Cannot connect to backend!\n\nMake sure backend is running: cd backend && python app.py';
      }
      
      setError(errorMessage);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black transition-colors duration-300">
      <DarkModeToggle />
      <AIAssistant />
      
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl shadow-2xl p-6 mb-6 transition-colors duration-300">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
                ML Data Analysis with Python Backend
              </h1>
              <p className="text-gray-400">
                Real sklearn, seaborn, and matplotlib for production-grade ML analysis
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={checkBackendHealth}
                className="p-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg hover:bg-[#2a2a2a] transition-colors"
                title="Refresh backend status"
              >
                <RefreshCw className="w-5 h-5 text-gray-400" />
              </button>
              <div className={`px-4 py-2 rounded-lg font-semibold border ${
                backendStatus === 'connected' ? 'bg-green-900/20 text-green-400 border-green-600/50' :
                backendStatus === 'disconnected' ? 'bg-red-900/20 text-red-400 border-red-600/50' :
                'bg-yellow-900/20 text-yellow-400 border-yellow-600/50'
              }`}>
                Backend: {backendStatus}
              </div>
            </div>
          </div>
        </div>

        {/* Backend Not Running Warning */}
        {backendStatus === 'disconnected' && (
          <div className="bg-red-900/20 border-2 border-red-600/50 rounded-xl p-6 mb-6 shadow-2xl">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-8 h-8 text-red-400 mt-0.5 flex-shrink-0 animate-pulse" />
              <div className="flex-1">
                <p className="font-bold text-red-400 text-xl mb-2">
                  ⚠️ Python Backend Not Running
                </p>
                <p className="text-sm text-red-700 dark:text-red-400 mb-3">
                  This app requires the Python backend for sklearn-based analysis. Please start the server:
                </p>
                
                {/* Quick Start Steps */}
                <div className="space-y-3 mb-4">
                  <div className="bg-red-100 dark:bg-red-950 p-4 rounded-lg">
                    <p className="text-xs font-semibold text-red-900 dark:text-red-200 mb-2">
                      📝 Step 1: Navigate to backend folder
                    </p>
                    <pre className="text-xs text-red-900 dark:text-red-200">cd backend</pre>
                  </div>
                  
                  <div className="bg-red-100 dark:bg-red-950 p-4 rounded-lg">
                    <p className="text-xs font-semibold text-red-900 dark:text-red-200 mb-2">
                      📝 Step 2: Create virtual environment (first time only)
                    </p>
                    <pre className="text-xs text-red-900 dark:text-red-200">python -m venv venv</pre>
                  </div>
                  
                  <div className="bg-red-100 dark:bg-red-950 p-4 rounded-lg">
                    <p className="text-xs font-semibold text-red-900 dark:text-red-200 mb-2">
                      📝 Step 3: Activate virtual environment
                    </p>
                    <pre className="text-xs text-red-900 dark:text-red-200">
{`# Windows:
venv\\Scripts\\activate

# Mac/Linux:
source venv/bin/activate`}
                    </pre>
                  </div>
                  
                  <div className="bg-red-100 dark:bg-red-950 p-4 rounded-lg">
                    <p className="text-xs font-semibold text-red-900 dark:text-red-200 mb-2">
                      📝 Step 4: Install dependencies (first time only)
                    </p>
                    <pre className="text-xs text-red-900 dark:text-red-200">pip install -r requirements.txt</pre>
                  </div>
                  
                  <div className="bg-red-100 dark:bg-red-950 p-4 rounded-lg">
                    <p className="text-xs font-semibold text-red-900 dark:text-red-200 mb-2">
                      📝 Step 5: Start the server
                    </p>
                    <pre className="text-xs text-red-900 dark:text-red-200">python app.py</pre>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
                  <span className="text-2xl">💡</span>
                  <div className="text-xs text-yellow-800 dark:text-yellow-200">
                    <strong>Tip:</strong> After starting, you should see "Running on http://127.0.0.1:5000"
                    <br />
                    Test it: <a href="http://localhost:5000/api/health" target="_blank" rel="noopener noreferrer" className="underline">http://localhost:5000/api/health</a>
                  </div>
                </div>
                
                <div className="mt-3 text-xs text-red-600 dark:text-red-400">
                  📚 Need more help? Check <strong>TROUBLESHOOTING.md</strong> or <strong>QUICK_START.md</strong>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* File Upload */}
        <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl shadow-2xl p-6 mb-6 transition-colors duration-300">
          <div className="border-2 border-dashed border-purple-600/30 rounded-lg p-8 text-center bg-purple-900/10 hover:bg-purple-900/20 hover:border-purple-600/50 transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
              disabled={backendStatus !== 'connected'}
            />
            <label htmlFor="file-upload" className={backendStatus === 'connected' ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}>
              <Upload className="w-12 h-12 text-purple-400 mx-auto mb-3" />
              <p className="text-lg font-semibold text-white mb-2">
                {file ? file.name : 'Click to upload CSV file'}
              </p>
              <p className="text-sm text-gray-400">
                {backendStatus === 'connected' 
                  ? 'Upload your dataset for Python-based sklearn analysis'
                  : 'Please start the backend server first'}
              </p>
            </label>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-900/20 border border-red-600/50 rounded-lg">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {file && backendStatus === 'connected' && !loading && (
            <div className="mt-6 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <button
                  onClick={() => analyzeData()}
                  className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white py-3 px-6 rounded-lg transition-all flex items-center justify-center gap-2 font-semibold shadow-lg shadow-blue-600/30"
                >
                  <TrendingUp className="w-5 h-5" />
                  Analyze Data
                </button>
                <button
                  onClick={generatePlots}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-3 px-6 rounded-lg transition-all flex items-center justify-center gap-2 font-semibold shadow-lg shadow-purple-600/30"
                >
                  <Image className="w-5 h-5" />
                  Generate Plots
                </button>
                <button
                  onClick={compareModels}
                  className="bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white py-3 px-6 rounded-lg transition-all flex items-center justify-center gap-2 font-semibold shadow-lg shadow-orange-600/30"
                >
                  <BarChart3 className="w-5 h-5" />
                  Compare Models
                </button>
                <button
                  onClick={trainModel}
                  disabled={!selectedModel}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white py-3 px-6 rounded-lg transition-all flex items-center justify-center gap-2 font-semibold disabled:opacity-50 disabled:from-gray-700 disabled:to-gray-800 shadow-lg shadow-green-600/30"
                >
                  <Play className="w-5 h-5" />
                  Train Selected
                </button>
              </div>
              
              {/* Model Selection */}
              <div className="bg-black border border-[#1a1a1a] p-4 rounded-lg">
                <label className="block text-sm font-semibold text-gray-300 mb-2">
                  Select Model for Training:
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-3 border border-[#2a2a2a] rounded-lg bg-[#0a0a0a] text-white focus:border-purple-600/50 focus:outline-none transition-all"
                >
                  <option value="">-- Select a Model --</option>
                  <optgroup label="Tree-Based Models">
                    <option value="random_forest">Random Forest</option>
                    <option value="gradient_boosting">Gradient Boosting</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="lightgbm">LightGBM</option>
                    <option value="adaboost">AdaBoost</option>
                    <option value="decision_tree">Decision Tree</option>
                  </optgroup>
                  <optgroup label="Linear Models">
                    <option value="logistic">Logistic Regression</option>
                    <option value="linear">Linear Regression</option>
                  </optgroup>
                  <optgroup label="Other Models">
                    <option value="svm">SVM / SVR</option>
                    <option value="knn">K-Nearest Neighbors</option>
                    <option value="naive_bayes">Naive Bayes</option>
                  </optgroup>
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  💡 Use "Best Model" tab to see recommendations, or compare all models
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-[#0a0a0a] border border-purple-600/30 rounded-xl shadow-2xl p-8 text-center mb-6 transition-colors duration-300">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
            <p className="text-white font-semibold">
              Processing with Python backend (sklearn + seaborn)...
            </p>
            <p className="text-sm text-gray-400 mt-2">
              This may take a few seconds for large datasets
            </p>
          </div>
        )}

        {/* Tabs */}
        {(analysis || plots || modelResults || recommendations || modelComparison) && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg mb-6 transition-colors duration-300">
            <div className="flex border-b border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setActiveTab('analysis')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'analysis'
                    ? 'border-b-2 border-blue-600 text-blue-600 dark:text-blue-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Analysis
              </button>
              <button
                onClick={() => setActiveTab('preprocess')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'preprocess'
                    ? 'border-b-2 border-indigo-600 text-indigo-600 dark:text-indigo-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Preprocess
              </button>
              <button
                onClick={() => setActiveTab('recommendations')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'recommendations'
                    ? 'border-b-2 border-orange-600 text-orange-600 dark:text-orange-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Best Model
              </button>
              <button
                onClick={() => setActiveTab('plots')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'plots'
                    ? 'border-b-2 border-purple-600 text-purple-600 dark:text-purple-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Visualizations
              </button>
              <button
                onClick={() => setActiveTab('comparison')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'comparison'
                    ? 'border-b-2 border-orange-600 text-orange-600 dark:text-orange-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Model Comparison
              </button>
              <button
                onClick={() => setActiveTab('model')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'model'
                    ? 'border-b-2 border-green-600 text-green-600 dark:text-green-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                Train Model
              </button>
              <button
                onClick={() => setActiveTab('overfitting')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'overfitting'
                    ? 'border-b-2 border-red-600 text-red-600 dark:text-red-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                🛡️ Reduce Overfitting
              </button>
              <button
                onClick={() => setActiveTab('gridsearch')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'gridsearch'
                    ? 'border-b-2 border-purple-600 text-purple-600 dark:text-purple-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                🔍 GridSearchCV
              </button>
              <button
                onClick={() => setActiveTab('deeplearning')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'deeplearning'
                    ? 'border-b-2 border-pink-600 text-pink-600 dark:text-pink-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                🧠 Deep Learning
              </button>
              <button
                onClick={() => setActiveTab('isd')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'isd'
                    ? 'border-b-2 border-purple-600 text-purple-600 dark:text-purple-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                🎯 ISD - System Designer
              </button>
              <button
                onClick={() => setActiveTab('ai-agent')}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === 'ai-agent'
                    ? 'border-b-2 border-pink-600 text-pink-600 dark:text-pink-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                🤖 AI Agent - Overfitting
              </button>
            </div>
          </div>
        )}

        {/* Analysis Results */}
        {activeTab === 'analysis' && analysis && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
                  Dataset Analysis (sklearn + pandas)
                </h2>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Rows</p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {analysis.shape[0].toLocaleString()}
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Columns</p>
                  <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {analysis.shape[1]}
                  </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Problem Type</p>
                  <p className="text-lg font-bold text-green-600 dark:text-green-400">
                    {analysis.problem_type}
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Target Column</p>
                  <p className="text-sm font-bold text-orange-600 dark:text-orange-400">
                    {analysis.target_column}
                  </p>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="font-semibold text-gray-800 dark:text-white mb-3 text-lg">
                  Column Analysis
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-700">
                    <thead className="bg-gray-100 dark:bg-gray-700">
                      <tr>
                        <th className="border border-gray-300 dark:border-gray-600 p-3 text-left text-sm font-semibold">Column</th>
                        <th className="border border-gray-300 dark:border-gray-600 p-3 text-left text-sm font-semibold">Type</th>
                        <th className="border border-gray-300 dark:border-gray-600 p-3 text-left text-sm font-semibold">Unique</th>
                        <th className="border border-gray-300 dark:border-gray-600 p-3 text-left text-sm font-semibold">Missing</th>
                        <th className="border border-gray-300 dark:border-gray-600 p-3 text-left text-sm font-semibold">Missing %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.column_analysis.map((col, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50 dark:bg-gray-900'}>
                          <td className="border border-gray-300 dark:border-gray-600 p-3 font-medium">{col.name}</td>
                          <td className="border border-gray-300 dark:border-gray-600 p-3">
                            <span className={`px-2 py-1 rounded text-xs font-semibold ${
                              col.type === 'numeric' 
                                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300' 
                                : 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300'
                            }`}>
                              {col.type}
                            </span>
                          </td>
                          <td className="border border-gray-300 dark:border-gray-600 p-3">{col.unique_count}</td>
                          <td className="border border-gray-300 dark:border-gray-600 p-3">{col.missing_count}</td>
                          <td className="border border-gray-300 dark:border-gray-600 p-3">
                            <span className={`font-semibold ${
                              col.missing_percent > 20 ? 'text-red-600 dark:text-red-400' :
                              col.missing_percent > 5 ? 'text-yellow-600 dark:text-yellow-400' :
                              'text-green-600 dark:text-green-400'
                            }`}>
                              {col.missing_percent.toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Data Preprocessing */}
        {activeTab === 'preprocess' && file && (
          <DataPreprocessor file={file} />
        )}

        {/* Model Recommendations */}
        {activeTab === 'recommendations' && recommendations && (
          <div className="space-y-6">
            {/* Best Model Highlight */}
            <div className="bg-gradient-to-r from-orange-500 to-red-500 rounded-xl shadow-lg p-6 text-white">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-8 h-8" />
                <h2 className="text-2xl font-bold">Best Model for Your Dataset</h2>
              </div>
              <div className="bg-white/10 backdrop-blur rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-3xl font-bold">{recommendations.best_model.name}</h3>
                  <div className="text-right">
                    <div className="text-4xl font-bold">{recommendations.best_model.score}%</div>
                    <div className="text-sm opacity-90">Confidence</div>
                  </div>
                </div>
                <p className="text-lg mb-4 opacity-90">{recommendations.best_model.reason}</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-semibold mb-2">✅ Pros:</p>
                    <ul className="space-y-1 text-sm opacity-90">
                      {recommendations.best_model.pros.map((pro, idx) => (
                        <li key={idx}>• {pro}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <p className="font-semibold mb-2">⚠️ Cons:</p>
                    <ul className="space-y-1 text-sm opacity-90">
                      {recommendations.best_model.cons.map((con, idx) => (
                        <li key={idx}>• {con}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-white/20">
                  <p className="text-sm opacity-90">
                    <strong>Best for:</strong> {recommendations.best_model.best_for}
                  </p>
                </div>
              </div>
            </div>

            {/* Dataset Insights */}
            {recommendations.insights && recommendations.insights.length > 0 && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
                <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                  Dataset Insights
                </h3>
                <div className="space-y-3">
                  {recommendations.insights.map((insight, idx) => (
                    <div key={idx} className={`p-4 rounded-lg border-l-4 ${
                      insight.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500' :
                      insight.type === 'success' ? 'bg-green-50 dark:bg-green-900/20 border-green-500' :
                      'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
                    }`}>
                      <p className="font-semibold text-gray-800 dark:text-white mb-1">
                        {insight.message}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        💡 {insight.recommendation}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* All Model Recommendations */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                All Model Recommendations
              </h3>
              <div className="space-y-4">
                {recommendations.recommended_models.map((model, idx) => (
                  <div key={idx} className={`border-2 rounded-lg p-5 transition-all ${
                    model.priority === 'Best Match'
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : model.priority === 'High'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : model.priority === 'Medium'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900'
                  }`}>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <h4 className="text-lg font-bold text-gray-800 dark:text-white">
                          {model.name}
                        </h4>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          model.priority === 'Best Match' ? 'bg-orange-500 text-white' :
                          model.priority === 'High' ? 'bg-green-500 text-white' :
                          model.priority === 'Medium' ? 'bg-blue-500 text-white' :
                          'bg-gray-400 text-white'
                        }`}>
                          {model.priority}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-gray-800 dark:text-white">
                          {model.score}%
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">Score</div>
                      </div>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 italic">
                      {model.reason}
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="font-semibold text-gray-800 dark:text-white mb-1">Pros:</p>
                        <ul className="text-gray-600 dark:text-gray-400 space-y-1">
                          {model.pros.map((pro, i) => (
                            <li key={i}>✓ {pro}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800 dark:text-white mb-1">Cons:</p>
                        <ul className="text-gray-600 dark:text-gray-400 space-y-1">
                          {model.cons.map((con, i) => (
                            <li key={i}>✗ {con}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        <strong>Best for:</strong> {model.best_for}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Dataset Statistics */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                Dataset Statistics
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Problem Type</p>
                  <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {recommendations.problem_type}
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Rows</p>
                  <p className="text-xl font-bold text-purple-600 dark:text-purple-400">
                    {recommendations.dataset_stats.rows.toLocaleString()}
                  </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Features</p>
                  <p className="text-xl font-bold text-green-600 dark:text-green-400">
                    {recommendations.dataset_stats.columns}
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Numeric</p>
                  <p className="text-xl font-bold text-orange-600 dark:text-orange-400">
                    {recommendations.dataset_stats.numeric_features}
                  </p>
                </div>
                <div className="bg-pink-50 dark:bg-pink-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Categorical</p>
                  <p className="text-xl font-bold text-pink-600 dark:text-pink-400">
                    {recommendations.dataset_stats.categorical_features}
                  </p>
                </div>
                <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Missing %</p>
                  <p className="text-xl font-bold text-red-600 dark:text-red-400">
                    {recommendations.dataset_stats.missing_percent}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Model Comparison */}
        {activeTab === 'comparison' && modelComparison && (
          <div className="space-y-6">
            {/* Comparison Plot */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                Model Performance Comparison
              </h2>
              <img src={modelComparison.comparison_plot} alt="Model Comparison" className="w-full rounded shadow-md" />
            </div>

            {/* Best Model from Comparison */}
            <div className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl shadow-lg p-6 text-white">
              <h3 className="text-2xl font-bold mb-2">🏆 Best Performing Model</h3>
              <div className="bg-white/10 backdrop-blur rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-3xl font-bold">{modelComparison.best_model.model}</h4>
                  <div className="text-right">
                    <div className="text-4xl font-bold">{(modelComparison.best_model.test_score * 100).toFixed(2)}%</div>
                    <div className="text-sm opacity-90">Test Score</div>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="opacity-75">Train Score:</p>
                    <p className="text-xl font-bold">{(modelComparison.best_model.train_score * 100).toFixed(2)}%</p>
                  </div>
                  <div>
                    <p className="opacity-75">Overfitting Gap:</p>
                    <p className={`text-xl font-bold ${
                      modelComparison.best_model.overfitting > 0.1 ? 'text-red-300' :
                      modelComparison.best_model.overfitting > 0.05 ? 'text-yellow-300' :
                      'text-green-300'
                    }`}>
                      {(modelComparison.best_model.overfitting * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Results Table */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                Detailed Model Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-700">
                  <thead className="bg-gray-100 dark:bg-gray-700">
                    <tr>
                      <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Model</th>
                      <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Train Score</th>
                      <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Test Score</th>
                      <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Overfitting</th>
                      {modelComparison.problem_type === 'Classification' && (
                        <>
                          <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Precision</th>
                          <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">Recall</th>
                          <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">F1-Score</th>
                        </>
                      )}
                      {modelComparison.problem_type === 'Regression' && (
                        <>
                          <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">RMSE</th>
                          <th className="border border-gray-300 dark:border-gray-600 p-3 text-left">MAE</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {modelComparison.results.map((result, idx) => (
                      <tr key={idx} className={`${
                        result.model === modelComparison.best_model.model 
                          ? 'bg-green-50 dark:bg-green-900/20 font-semibold' 
                          : idx % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50 dark:bg-gray-900'
                      }`}>
                        <td className="border border-gray-300 dark:border-gray-600 p-3">
                          {result.model}
                          {result.model === modelComparison.best_model.model && ' 🏆'}
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-3">
                          {(result.train_score * 100).toFixed(2)}%
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-3">
                          {(result.test_score * 100).toFixed(2)}%
                        </td>
                        <td className={`border border-gray-300 dark:border-gray-600 p-3 ${
                          result.overfitting > 0.1 ? 'text-red-600 dark:text-red-400 font-bold' :
                          result.overfitting > 0.05 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-green-600 dark:text-green-400'
                        }`}>
                          {(result.overfitting * 100).toFixed(2)}%
                        </td>
                        {modelComparison.problem_type === 'Classification' && (
                          <>
                            <td className="border border-gray-300 dark:border-gray-600 p-3">
                              {(result.precision * 100).toFixed(2)}%
                            </td>
                            <td className="border border-gray-300 dark:border-gray-600 p-3">
                              {(result.recall * 100).toFixed(2)}%
                            </td>
                            <td className="border border-gray-300 dark:border-gray-600 p-3">
                              {(result.f1_score * 100).toFixed(2)}%
                            </td>
                          </>
                        )}
                        {modelComparison.problem_type === 'Regression' && (
                          <>
                            <td className="border border-gray-300 dark:border-gray-600 p-3">
                              {result.rmse.toFixed(4)}
                            </td>
                            <td className="border border-gray-300 dark:border-gray-600 p-3">
                              {result.mae.toFixed(4)}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>💡 Overfitting Guide:</strong> Gap &lt; 5% = Good | 5-10% = Moderate | &gt; 10% = High Risk
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Plots */}
        {activeTab === 'plots' && plots && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
              Visualizations (seaborn + matplotlib)
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
              Professional publication-quality plots generated with Python libraries
            </p>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {Object.entries(plots).map(([key, value]) => (
                <div key={key} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900">
                  <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3 capitalize text-lg">
                    {key.replace(/_/g, ' ')}
                  </h3>
                  <img src={value} alt={key} className="w-full rounded shadow-md" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model Results */}
        {activeTab === 'model' && modelResults && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                Model Training Results (sklearn)
              </h2>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Model</p>
                  <p className="text-lg font-bold text-green-600 dark:text-green-400">
                    {modelResults.model_type}
                  </p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Train Size</p>
                  <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {modelResults.train_size.toLocaleString()}
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Test Size</p>
                  <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                    {modelResults.test_size.toLocaleString()}
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/30 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {modelResults.accuracy ? 'Accuracy' : 'R² Score'}
                  </p>
                  <p className="text-lg font-bold text-orange-600 dark:text-orange-400">
                    {modelResults.accuracy 
                      ? (modelResults.accuracy * 100).toFixed(2) + '%' 
                      : modelResults.r2_score?.toFixed(4)}
                  </p>
                </div>
              </div>

              {/* Evaluation Metrics Matrix */}
              {modelResults.evaluation_metrics && (
                <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-lg border-2 border-blue-200 dark:border-blue-700">
                  <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
                    📊 Evaluation Metrics Matrix
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(modelResults.evaluation_metrics).map(([key, metric]) => (
                      <div key={key} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold text-gray-800 dark:text-white capitalize">
                            {key.replace(/_/g, ' ')}
                          </h4>
                          <span className={`text-2xl font-bold ${
                            metric.value >= 0.8 ? 'text-green-600 dark:text-green-400' :
                            metric.value >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' :
                            'text-red-600 dark:text-red-400'
                          }`}>
                            {typeof metric.value === 'number' ? metric.value.toFixed(4) : 'N/A'}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                          <strong>What:</strong> {metric.description}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                          <strong>How to read:</strong> {metric.interpretation}
                        </p>
                        <p className="text-xs text-blue-600 dark:text-blue-400">
                          <strong>Best for:</strong> {metric.best_for}
                        </p>
                      </div>
                    ))}
                  </div>
                  
                  {/* Performance Indicator */}
                  <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Overall Performance:
                      </span>
                      <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                        (modelResults.accuracy || modelResults.r2_score) >= 0.9 ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' :
                        (modelResults.accuracy || modelResults.r2_score) >= 0.7 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' :
                        'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                      }`}>
                        {(modelResults.accuracy || modelResults.r2_score) >= 0.9 ? '🎉 Excellent' :
                         (modelResults.accuracy || modelResults.r2_score) >= 0.7 ? '👍 Good' :
                         '⚠️ Needs Improvement'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Cross-Validation */}
              {modelResults.cv_scores && (
                <div className="mb-6 p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
                  <h3 className="font-semibold text-gray-800 dark:text-white mb-2">
                    Cross-Validation (5-fold)
                  </h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    Mean Score: <span className="font-bold text-purple-600 dark:text-purple-400">
                      {modelResults.cv_mean.toFixed(4)}
                    </span>
                    {' '}(± {modelResults.cv_std.toFixed(4)})
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    💡 Cross-validation helps detect overfitting by testing on multiple data splits
                  </p>
                </div>
              )}

              {/* Train vs Test Performance */}
              {(modelResults.train_accuracy || modelResults.train_r2_score) && (
                <div className="mb-6 p-4 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
                  <h3 className="font-semibold text-gray-800 dark:text-white mb-3">
                    Train vs Test Performance
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Training Score</p>
                      <p className="text-2xl font-bold text-gray-800 dark:text-white">
                        {(modelResults.train_accuracy || modelResults.train_r2_score).toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Test Score</p>
                      <p className="text-2xl font-bold text-gray-800 dark:text-white">
                        {(modelResults.accuracy || modelResults.r2_score).toFixed(4)}
                      </p>
                    </div>
                  </div>
                  {Math.abs((modelResults.train_accuracy || modelResults.train_r2_score) - 
                            (modelResults.accuracy || modelResults.r2_score)) > 0.1 && (
                    <div className="mt-3 p-2 bg-red-100 dark:bg-red-900/30 rounded">
                      <p className="text-xs text-red-700 dark:text-red-300">
                        ⚠️ Large gap between train and test scores suggests overfitting. 
                        Consider regularization or more data.
                      </p>
                    </div>
                  )}
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {modelResults.confusion_matrix_plot && (
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900">
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3 text-lg">
                      Confusion Matrix
                    </h3>
                    <img src={modelResults.confusion_matrix_plot} alt="Confusion Matrix" className="w-full rounded shadow-md" />
                  </div>
                )}
                {modelResults.prediction_plot && (
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900">
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3 text-lg">
                      Actual vs Predicted
                    </h3>
                    <img src={modelResults.prediction_plot} alt="Predictions" className="w-full rounded shadow-md" />
                  </div>
                )}
                {modelResults.feature_importance_plot && (
                  <div className="lg:col-span-2 border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900">
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3 text-lg">
                      Feature Importance (from trained model)
                    </h3>
                    <img src={modelResults.feature_importance_plot} alt="Feature Importance" className="w-full rounded shadow-md" />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Overfitting Reducer */}
        {activeTab === 'overfitting' && file && (
          <OverfittingReducer file={file} />
        )}

        {/* GridSearchCV */}
        {activeTab === 'gridsearch' && file && (
          <GridSearchCV file={file} />
        )}

        {/* Deep Learning */}
        {activeTab === 'deeplearning' && file && (
          <DeepLearningBuilder file={file} />
        )}

        {/* Intelligent System Designer */}
        {activeTab === 'isd' && (
          <IntelligentSystemDesigner />
        )}

        {/* AI Overfitting Agent */}
        {activeTab === 'ai-agent' && (
          <div>
            {/* Debug info - remove in production */}
            {!modelResults && (
              <div className="bg-yellow-900/20 border border-yellow-600/50 rounded-lg p-4 mb-4 text-yellow-400">
                ⚠️ No model results available. Please train a model first in the "Train Model" tab.
              </div>
            )}
            <AIOverfittingAgent 
              modelResults={modelResults} 
              datasetInfo={{
                shape: analysis?.shape,
                columns: analysis?.columns,
                problem_type: analysis?.problem_type
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
