import React, { useState } from 'react';

const OverfittingReducer = ({ file, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  // Overfitting reduction options
  const [options, setOptions] = useState({
    method: 'cross_validation',
    cv_folds: 5,
    regularization_strength: 0.1,
    dropout_rate: 0.3,
    early_stopping: true,
    max_depth: 10,
    min_samples_split: 10,
    feature_selection: false,
    ensemble_size: 5
  });

  const handleReduce = async () => {
    if (!file) {
      setError('Please upload a CSV file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    Object.keys(options).forEach(key => {
      formData.append(key, options[key]);
    });

    try {
      const response = await fetch('http://localhost:5000/api/reduce-overfitting', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to reduce overfitting');
      }

      const data = await response.json();
      setResults(data);
      if (onSuccess) onSuccess(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const techniques = [
    {
      id: 'cross_validation',
      name: 'Cross-Validation',
      description: 'Use k-fold cross-validation to ensure model generalizes well',
      icon: '🔄'
    },
    {
      id: 'regularization',
      name: 'Regularization (L1/L2)',
      description: 'Add penalty to model complexity to prevent overfitting',
      icon: '⚖️'
    },
    {
      id: 'early_stopping',
      name: 'Early Stopping',
      description: 'Stop training when validation performance stops improving',
      icon: '⏹️'
    },
    {
      id: 'pruning',
      name: 'Tree Pruning',
      description: 'Limit tree depth and minimum samples per leaf',
      icon: '✂️'
    },
    {
      id: 'dropout',
      name: 'Dropout',
      description: 'Randomly drop features during training (for neural networks)',
      icon: '🎲'
    },
    {
      id: 'data_augmentation',
      name: 'More Training Data',
      description: 'Collect more data or use data augmentation techniques',
      icon: '📊'
    },
    {
      id: 'feature_selection',
      name: 'Feature Selection',
      description: 'Remove irrelevant features that cause noise',
      icon: '🎯'
    },
    {
      id: 'ensemble',
      name: 'Ensemble Methods',
      description: 'Combine multiple models to reduce variance',
      icon: '🤝'
    }
  ];

  return (
    <div className="overfitting-reducer">
      <h2>🛡️ Reduce Overfitting</h2>
      <p className="description">
        Apply techniques to prevent your model from memorizing training data and improve generalization.
      </p>

      {/* Technique Selection */}
      <div className="technique-selector">
        <h3>Select Reduction Method</h3>
        <select 
          value={options.method} 
          onChange={(e) => setOptions({...options, method: e.target.value})}
          className="method-select"
        >
          <option value="cross_validation">Cross-Validation</option>
          <option value="regularization">Regularization</option>
          <option value="early_stopping">Early Stopping</option>
          <option value="pruning">Tree Pruning</option>
          <option value="feature_selection">Feature Selection</option>
          <option value="ensemble">Ensemble Methods</option>
          <option value="all">Apply All Techniques</option>
        </select>
      </div>

      {/* Method-specific Options */}
      <div className="options-panel">
        {options.method === 'cross_validation' && (
          <div className="option-group">
            <label>
              Number of Folds:
              <input 
                type="number" 
                min="2" 
                max="10" 
                value={options.cv_folds}
                onChange={(e) => setOptions({...options, cv_folds: parseInt(e.target.value)})}
              />
            </label>
          </div>
        )}

        {options.method === 'regularization' && (
          <div className="option-group">
            <label>
              Regularization Strength (alpha):
              <input 
                type="number" 
                step="0.01" 
                min="0.001" 
                max="10" 
                value={options.regularization_strength}
                onChange={(e) => setOptions({...options, regularization_strength: parseFloat(e.target.value)})}
              />
            </label>
            <small>Higher values = stronger regularization</small>
          </div>
        )}

        {options.method === 'pruning' && (
          <div className="option-group">
            <label>
              Max Tree Depth:
              <input 
                type="number" 
                min="1" 
                max="50" 
                value={options.max_depth}
                onChange={(e) => setOptions({...options, max_depth: parseInt(e.target.value)})}
              />
            </label>
            <label>
              Min Samples per Split:
              <input 
                type="number" 
                min="2" 
                max="100" 
                value={options.min_samples_split}
                onChange={(e) => setOptions({...options, min_samples_split: parseInt(e.target.value)})}
              />
            </label>
          </div>
        )}

        {options.method === 'ensemble' && (
          <div className="option-group">
            <label>
              Number of Models:
              <input 
                type="number" 
                min="3" 
                max="10" 
                value={options.ensemble_size}
                onChange={(e) => setOptions({...options, ensemble_size: parseInt(e.target.value)})}
              />
            </label>
          </div>
        )}
      </div>

      {/* Techniques Info Grid */}
      <div className="techniques-grid">
        {techniques.map(tech => (
          <div 
            key={tech.id} 
            className={`technique-card ${options.method === tech.id ? 'selected' : ''}`}
            onClick={() => setOptions({...options, method: tech.id})}
          >
            <div className="technique-icon">{tech.icon}</div>
            <h4>{tech.name}</h4>
            <p>{tech.description}</p>
          </div>
        ))}
      </div>

      {/* Action Button */}
      <button 
        onClick={handleReduce} 
        disabled={loading || !file}
        className="reduce-button"
      >
        {loading ? 'Applying Techniques...' : 'Reduce Overfitting'}
      </button>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="results-section">
          <h3>✅ Overfitting Reduction Results</h3>
          
          {/* Before/After Comparison */}
          <div className="comparison-grid">
            <div className="metric-card">
              <h4>Before</h4>
              <div className="metric">
                <span className="label">Train Score:</span>
                <span className="value">{(results.before.train_score * 100).toFixed(2)}%</span>
              </div>
              <div className="metric">
                <span className="label">Test Score:</span>
                <span className="value">{(results.before.test_score * 100).toFixed(2)}%</span>
              </div>
              <div className="metric overfitting-high">
                <span className="label">Overfitting Gap:</span>
                <span className="value">{(results.before.overfitting_gap * 100).toFixed(2)}%</span>
              </div>
            </div>

            <div className="arrow">→</div>

            <div className="metric-card improved">
              <h4>After</h4>
              <div className="metric">
                <span className="label">Train Score:</span>
                <span className="value">{(results.after.train_score * 100).toFixed(2)}%</span>
              </div>
              <div className="metric">
                <span className="label">Test Score:</span>
                <span className="value">{(results.after.test_score * 100).toFixed(2)}%</span>
              </div>
              <div className="metric overfitting-low">
                <span className="label">Overfitting Gap:</span>
                <span className="value">{(results.after.overfitting_gap * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>

          {/* Improvement Summary */}
          <div className="improvement-summary">
            <h4>📈 Improvements</h4>
            <ul>
              <li>
                Overfitting reduced by: <strong>{((results.before.overfitting_gap - results.after.overfitting_gap) * 100).toFixed(2)}%</strong>
              </li>
              <li>
                Test score change: <strong>{((results.after.test_score - results.before.test_score) * 100).toFixed(2)}%</strong>
              </li>
              <li>
                Generalization: <strong>{results.generalization_status}</strong>
              </li>
            </ul>
          </div>

          {/* Applied Techniques */}
          <div className="applied-techniques">
            <h4>🔧 Applied Techniques</h4>
            <ul>
              {results.applied_techniques.map((tech, idx) => (
                <li key={idx}>{tech}</li>
              ))}
            </ul>
          </div>

          {/* Recommendations */}
          {results.recommendations && results.recommendations.length > 0 && (
            <div className="recommendations">
              <h4>💡 Additional Recommendations</h4>
              <ul>
                {results.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Visualization */}
          {results.plot && (
            <div className="plot-container">
              <h4>Overfitting Comparison Plot</h4>
              <img src={results.plot} alt="Overfitting comparison" />
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .overfitting-reducer {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .description {
          color: #666;
          margin-bottom: 30px;
        }

        .technique-selector {
          margin-bottom: 30px;
        }

        .method-select {
          width: 100%;
          padding: 12px;
          font-size: 16px;
          border: 2px solid #ddd;
          border-radius: 8px;
          margin-top: 10px;
        }

        .options-panel {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 30px;
        }

        .option-group {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .option-group label {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-weight: 500;
        }

        .option-group input {
          padding: 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .option-group small {
          color: #666;
          font-size: 12px;
        }

        .techniques-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 15px;
          margin-bottom: 30px;
        }

        .technique-card {
          background: white;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          padding: 20px;
          cursor: pointer;
          transition: all 0.3s;
        }

        .technique-card:hover {
          border-color: #4CAF50;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .technique-card.selected {
          border-color: #4CAF50;
          background: #f1f8f4;
        }

        .technique-icon {
          font-size: 32px;
          margin-bottom: 10px;
        }

        .technique-card h4 {
          margin: 10px 0;
          color: #333;
        }

        .technique-card p {
          font-size: 14px;
          color: #666;
          margin: 0;
        }

        .reduce-button {
          width: 100%;
          padding: 15px;
          font-size: 18px;
          font-weight: bold;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.3s;
        }

        .reduce-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .reduce-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error-message {
          background: #fee;
          border: 1px solid #fcc;
          color: #c33;
          padding: 15px;
          border-radius: 8px;
          margin-top: 20px;
        }

        .results-section {
          margin-top: 30px;
          padding: 20px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .comparison-grid {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          gap: 20px;
          align-items: center;
          margin: 20px 0;
        }

        .metric-card {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          border: 2px solid #ddd;
        }

        .metric-card.improved {
          background: #f1f8f4;
          border-color: #4CAF50;
        }

        .metric-card h4 {
          margin-top: 0;
          color: #333;
        }

        .metric {
          display: flex;
          justify-content: space-between;
          padding: 10px 0;
          border-bottom: 1px solid #e0e0e0;
        }

        .metric:last-child {
          border-bottom: none;
        }

        .metric .label {
          font-weight: 500;
          color: #666;
        }

        .metric .value {
          font-weight: bold;
          color: #333;
        }

        .metric.overfitting-high .value {
          color: #f44336;
        }

        .metric.overfitting-low .value {
          color: #4CAF50;
        }

        .arrow {
          font-size: 32px;
          color: #4CAF50;
          font-weight: bold;
        }

        .improvement-summary,
        .applied-techniques,
        .recommendations {
          margin-top: 20px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .improvement-summary h4,
        .applied-techniques h4,
        .recommendations h4 {
          margin-top: 0;
          color: #333;
        }

        .improvement-summary ul,
        .applied-techniques ul,
        .recommendations ul {
          margin: 10px 0;
          padding-left: 20px;
        }

        .improvement-summary li,
        .applied-techniques li,
        .recommendations li {
          margin: 8px 0;
          color: #555;
        }

        .plot-container {
          margin-top: 20px;
        }

        .plot-container img {
          width: 100%;
          max-width: 800px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        @media (max-width: 768px) {
          .comparison-grid {
            grid-template-columns: 1fr;
          }

          .arrow {
            transform: rotate(90deg);
          }

          .techniques-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default OverfittingReducer;
