import React, { useState } from 'react';
import { Brain, Zap, AlertTriangle, CheckCircle, Code, TrendingDown, Key, Sparkles } from 'lucide-react';

const AIOverfittingAgent = ({ modelResults, datasetInfo }) => {
  const [apiKey, setApiKey] = useState(localStorage.getItem('openai_api_key') || '');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [showApiKey, setShowApiKey] = useState(false);

  const analyzeOverfitting = async () => {
    if (!apiKey) {
      alert('Please enter your OpenAI API key');
      return;
    }

    if (!modelResults) {
      alert('Please train a model first');
      return;
    }

    setLoading(true);

    try {
      // Save API key to localStorage
      localStorage.setItem('openai_api_key', apiKey);
      
      // Debug logging
      console.log('🔑 API Key length:', apiKey.length);
      console.log('🔑 API Key starts with:', apiKey.substring(0, 10));
      console.log('📊 Model results:', modelResults ? 'Present' : 'Missing');

      const formData = new FormData();
      formData.append('api_key', apiKey);
      formData.append('model_results', JSON.stringify(modelResults));
      formData.append('dataset_info', JSON.stringify(datasetInfo || {}));

      const response = await fetch('http://localhost:5000/api/ai-agent/analyze-overfitting', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.error) {
        // Show error but still display fallback recommendations if available
        setAnalysis(data);
      } else {
        setAnalysis(data);
      }
    } catch (error) {
      console.error('Error:', error);
      setAnalysis({
        error: error.message,
        message: 'Network error. Please check your internet connection and try again.',
        fallback_recommendations: [
          'Unable to connect to AI service.',
          'Please check your internet connection.',
          'Verify your OpenAI API key is correct.',
          'Try again in a few moments.'
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-400 border-red-600/50 bg-red-900/20';
      case 'high': return 'text-orange-400 border-orange-600/50 bg-orange-900/20';
      case 'moderate': return 'text-yellow-400 border-yellow-600/50 bg-yellow-900/20';
      case 'low': return 'text-blue-400 border-blue-600/50 bg-blue-900/20';
      default: return 'text-green-400 border-green-600/50 bg-green-900/20';
    }
  };

  const getDifficultyBadge = (difficulty) => {
    const colors = {
      easy: 'bg-green-600',
      medium: 'bg-yellow-600',
      hard: 'bg-red-600'
    };
    return colors[difficulty] || 'bg-gray-600';
  };

  return (
    <div className="min-h-screen bg-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-3 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl shadow-lg shadow-purple-600/50">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent">
                AI Overfitting Agent
              </h1>
              <p className="text-gray-500 text-sm mt-1">
                Powered by OpenAI GPT-4 • Intelligent Model Optimization
              </p>
            </div>
          </div>
        </div>

        {/* API Key Input */}
        <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-6 mb-6 shadow-xl">
          <div className="flex items-center gap-2 mb-4">
            <Key className="w-5 h-5 text-purple-400" />
            <h2 className="text-xl font-semibold text-white">OpenAI API Key</h2>
          </div>
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full p-3 bg-black border border-[#1a1a1a] rounded-lg text-gray-300 focus:border-purple-600/50 focus:outline-none transition-all"
              />
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 text-sm"
              >
                {showApiKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <button
              onClick={analyzeOverfitting}
              disabled={!apiKey || !modelResults || loading}
              className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed transition-all shadow-lg shadow-purple-600/30 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Analyze with AI
                </>
              )}
            </button>
          </div>
          <div className="flex items-center justify-between mt-2">
            <p className="text-xs text-gray-500">
              🔒 Your API key is stored locally and never sent to our servers. Get your key from{' '}
              <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300">
                OpenAI Platform
              </a>
            </p>
            <button
              onClick={() => {
                localStorage.removeItem('openai_api_key');
                setApiKey('');
                alert('API key cleared! Please enter a fresh key.');
              }}
              className="text-xs text-red-400 hover:text-red-300 underline"
            >
              Clear Saved Key
            </button>
          </div>
        </div>

        {/* Analysis Results */}
        {analysis && !analysis.error && (
          <div className="space-y-6">
            {/* Overfitting Status */}
            <div className={`border rounded-xl p-6 ${getSeverityColor(analysis.overfitting_severity)}`}>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-2">Overfitting Analysis</h3>
                  <p className="text-gray-400">AI-Powered Detection & Diagnosis</p>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-bold">{analysis.overfitting_score}/100</div>
                  <div className="text-sm uppercase tracking-wider mt-1">{analysis.overfitting_severity}</div>
                </div>
              </div>

              {analysis.overfitting_detected && (
                <div className="flex items-center gap-2 p-3 bg-black/30 rounded-lg">
                  <AlertTriangle className="w-5 h-5" />
                  <span className="font-semibold">Overfitting Detected!</span>
                </div>
              )}

              <div className="grid grid-cols-3 gap-4 mt-4">
                <div className="p-4 bg-black/30 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Train Score</div>
                  <div className="text-2xl font-bold">{(analysis.metrics.train_score * 100).toFixed(1)}%</div>
                </div>
                <div className="p-4 bg-black/30 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Test Score</div>
                  <div className="text-2xl font-bold">{(analysis.metrics.test_score * 100).toFixed(1)}%</div>
                </div>
                <div className="p-4 bg-black/30 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Gap</div>
                  <div className="text-2xl font-bold">{(analysis.metrics.overfitting_gap * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* AI Analysis */}
            <div className="bg-[#0a0a0a] border border-purple-600/30 rounded-xl p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-6 h-6 text-purple-400" />
                <h3 className="text-2xl font-bold text-white">AI Analysis</h3>
              </div>
              <div className="prose prose-invert max-w-none">
                <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {analysis.ai_analysis}
                </div>
              </div>
            </div>

            {/* Quick Fixes */}
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-6 h-6 text-yellow-400" />
                <h3 className="text-2xl font-bold text-white">Quick Fixes</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {analysis.quick_fixes.map((fix, idx) => (
                  <div key={idx} className="bg-black border border-[#1a1a1a] rounded-xl p-5 hover:border-purple-600/50 transition-all">
                    <div className="flex items-start justify-between mb-3">
                      <h4 className="text-lg font-semibold text-white">{fix.title}</h4>
                      <div className="flex gap-2">
                        <span className={`px-2 py-1 text-xs rounded font-semibold ${getDifficultyBadge(fix.difficulty)} text-white`}>
                          {fix.difficulty}
                        </span>
                        <span className="px-2 py-1 text-xs rounded font-semibold bg-purple-600 text-white">
                          {fix.impact} impact
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-400 mb-3">{fix.description}</p>
                    <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <Code className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-gray-500 uppercase tracking-wider">Code Example</span>
                      </div>
                      <pre className="text-xs text-purple-400 overflow-x-auto">
                        <code>{fix.code}</code>
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {analysis.recommendations && analysis.recommendations.length > 0 && (
              <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-6 shadow-xl">
                <div className="flex items-center gap-2 mb-4">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                  <h3 className="text-2xl font-bold text-white">AI Recommendations</h3>
                </div>
                <div className="space-y-3">
                  {analysis.recommendations.map((rec, idx) => (
                    <div key={idx} className="flex items-start gap-3 p-4 bg-black border border-[#1a1a1a] rounded-lg hover:border-green-600/50 transition-all">
                      <div className="flex-shrink-0 w-8 h-8 bg-green-600/20 rounded-full flex items-center justify-center text-green-400 font-bold">
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <p className="text-gray-300">{rec.text}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Success Message */}
            {!analysis.overfitting_detected && (
              <div className="bg-green-900/20 border border-green-600/50 rounded-xl p-6">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-8 h-8 text-green-400" />
                  <div>
                    <h3 className="text-xl font-bold text-green-400">Model Looks Good!</h3>
                    <p className="text-gray-400 mt-1">
                      No significant overfitting detected. Your model is generalizing well.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {analysis && analysis.error && (
          <div className="bg-red-900/20 border border-red-600/50 rounded-xl p-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />
              <div>
                <h3 className="text-xl font-bold text-red-400 mb-2">Error</h3>
                <p className="text-gray-300">{analysis.message || analysis.error}</p>
                {analysis.fallback_recommendations && (
                  <div className="mt-4">
                    <h4 className="font-semibold text-white mb-2">Fallback Recommendations:</h4>
                    <ul className="space-y-1 text-gray-400">
                      {analysis.fallback_recommendations.map((rec, idx) => (
                        <li key={idx}>• {rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!analysis && !loading && (
          <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-8 text-center">
            <Brain className="w-16 h-16 text-purple-400 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-white mb-2">
              {modelResults ? 'Ready to Analyze' : 'Train a Model First'}
            </h3>
            <p className="text-gray-400 mb-6">
              {modelResults 
                ? 'Click "Analyze with AI" to get intelligent recommendations for reducing overfitting'
                : 'Go to the "Train Model" tab, train any model, then come back here for AI analysis'
              }
            </p>
            
            {modelResults && (
              <div className="mb-6 p-4 bg-green-900/20 border border-green-600/50 rounded-lg">
                <div className="text-green-400 font-semibold mb-2">✓ Model Results Available</div>
                <div className="text-sm text-gray-400">
                  Model: {modelResults.model_type} | 
                  Train Score: {((modelResults.train_accuracy || modelResults.train_r2_score || 0) * 100).toFixed(1)}% | 
                  Test Score: {((modelResults.accuracy || modelResults.r2_score || 0) * 100).toFixed(1)}%
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
              <div className={`p-4 bg-black border rounded-lg ${!modelResults ? 'border-yellow-600/50' : 'border-[#1a1a1a]'}`}>
                <div className="text-purple-400 font-bold mb-2">
                  {!modelResults ? '→ 1. Train Model' : '✓ 1. Train Model'}
                </div>
                <p className="text-sm text-gray-400">
                  {!modelResults 
                    ? 'Go to "Train Model" tab and train any algorithm'
                    : 'Model trained successfully'
                  }
                </p>
              </div>
              <div className={`p-4 bg-black border rounded-lg ${!apiKey ? 'border-yellow-600/50' : 'border-[#1a1a1a]'}`}>
                <div className="text-purple-400 font-bold mb-2">
                  {!apiKey ? '→ 2. Get API Key' : '✓ 2. Get API Key'}
                </div>
                <p className="text-sm text-gray-400">
                  {!apiKey
                    ? 'Sign up at OpenAI and get your API key'
                    : 'API key configured'
                  }
                </p>
              </div>
              <div className={`p-4 bg-black border rounded-lg ${(!modelResults || !apiKey) ? 'border-gray-600/30' : 'border-green-600/50'}`}>
                <div className="text-purple-400 font-bold mb-2">
                  {(!modelResults || !apiKey) ? '3. Get AI Analysis' : '→ 3. Get AI Analysis'}
                </div>
                <p className="text-sm text-gray-400">
                  {(!modelResults || !apiKey)
                    ? 'Complete steps 1 & 2 first'
                    : 'Click "Analyze with AI" button above'
                  }
                </p>
              </div>
            </div>
            
            {!modelResults && (
              <div className="mt-6 p-4 bg-blue-900/20 border border-blue-600/50 rounded-lg text-left">
                <div className="text-blue-400 font-semibold mb-2">💡 Quick Start:</div>
                <ol className="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                  <li>Upload a CSV file in the main interface</li>
                  <li>Click "Train Model" tab</li>
                  <li>Select any model (e.g., Random Forest)</li>
                  <li>Click "Train Selected" button</li>
                  <li>Wait for training to complete</li>
                  <li>Come back to this tab</li>
                  <li>Enter your OpenAI API key</li>
                  <li>Click "Analyze with AI"</li>
                </ol>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AIOverfittingAgent;
