import React, { useState } from 'react';

const GridSearchCV = ({ file, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const [config, setConfig] = useState({
    model: 'random_forest',
    cv_folds: 5,
    scoring: 'auto',
    n_jobs: -1,
    verbose: 1
  });

  // Predefined parameter grids
  const [paramGrids, setParamGrids] = useState({
    random_forest: {
      n_estimators: [50, 100, 200],
      max_depth: [10, 20, 30, null],
      min_samples_split: [2, 5, 10],
      min_samples_leaf: [1, 2, 4],
      max_features: ['sqrt', 'log2']
    },
    xgboost: {
      n_estimators: [50, 100, 200],
      learning_rate: [0.01, 0.1, 0.3],
      max_depth: [3, 5, 7],
      subsample: [0.8, 0.9, 1.0],
      colsample_bytree: [0.8, 0.9, 1.0]
    },
    logistic: {
      C: [0.001, 0.01, 0.1, 1, 10, 100],
      penalty: ['l1', 'l2'],
      solver: ['liblinear', 'saga']
    },
    svm: {
      C: [0.1, 1, 10, 100],
      kernel: ['rbf', 'poly', 'sigmoid'],
      gamma: ['scale', 'auto', 0.001, 0.01]
    },
    gradient_boosting: {
      n_estimators: [50, 100, 200],
      learning_rate: [0.01, 0.1, 0.3],
      max_depth: [3, 5, 7],
      subsample: [0.8, 0.9, 1.0]
    }
  });

  const [customGrid, setCustomGrid] = useState('');
  const [useCustomGrid, setUseCustomGrid] = useState(false);

  const modelOptions = [
    { value: 'random_forest', label: 'Random Forest', icon: '🌲' },
    { value: 'xgboost', label: 'XGBoost', icon: '⚡' },
    { value: 'gradient_boosting', label: 'Gradient Boosting', icon: '📈' },
    { value: 'logistic', label: 'Logistic Regression', icon: '📊' },
    { value: 'svm', label: 'Support Vector Machine', icon: '🎯' },
    { value: 'lightgbm', label: 'LightGBM', icon: '💡' }
  ];

  const handleGridSearch = async () => {
    if (!file) {
      setError('Please upload a CSV file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', config.model);
    formData.append('cv_folds', config.cv_folds);
    formData.append('scoring', config.scoring);
    formData.append('n_jobs', config.n_jobs);
    formData.append('verbose', config.verbose);

    // Add parameter grid
    if (useCustomGrid && customGrid) {
      formData.append('param_grid', customGrid);
    } else {
      formData.append('param_grid', JSON.stringify(paramGrids[config.model]));
    }

    try {
      const response = await fetch('http://localhost:5000/api/grid-search', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to perform grid search');
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

  const updateParamGrid = (param, value) => {
    setParamGrids({
      ...paramGrids,
      [config.model]: {
        ...paramGrids[config.model],
        [param]: value
      }
    });
  };

  return (
    <div className="grid-search-cv">
      <h2>🔍 GridSearchCV - Hyperparameter Optimization</h2>
      <p className="description">
        Automatically find the best hyperparameters for your model using exhaustive grid search with cross-validation.
      </p>

      {/* Model Selection */}
      <div className="config-section">
        <h3>1. Select Model</h3>
        <div className="model-grid">
          {modelOptions.map(model => (
            <div
              key={model.value}
              className={`model-card ${config.model === model.value ? 'selected' : ''}`}
              onClick={() => setConfig({...config, model: model.value})}
            >
              <div className="model-icon">{model.icon}</div>
              <div className="model-name">{model.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* CV Configuration */}
      <div className="config-section">
        <h3>2. Cross-Validation Settings</h3>
        <div className="cv-settings">
          <label>
            CV Folds:
            <input
              type="number"
              min="2"
              max="10"
              value={config.cv_folds}
              onChange={(e) => setConfig({...config, cv_folds: parseInt(e.target.value)})}
            />
          </label>
          <label>
            Scoring Metric:
            <select
              value={config.scoring}
              onChange={(e) => setConfig({...config, scoring: e.target.value})}
            >
              <option value="auto">Auto (accuracy/r2)</option>
              <option value="accuracy">Accuracy</option>
              <option value="f1">F1 Score</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="roc_auc">ROC AUC</option>
              <option value="r2">R² Score</option>
              <option value="neg_mean_squared_error">MSE</option>
              <option value="neg_mean_absolute_error">MAE</option>
            </select>
          </label>
          <label>
            Parallel Jobs:
            <select
              value={config.n_jobs}
              onChange={(e) => setConfig({...config, n_jobs: parseInt(e.target.value)})}
            >
              <option value="-1">All CPUs (-1)</option>
              <option value="1">Single CPU</option>
              <option value="2">2 CPUs</option>
              <option value="4">4 CPUs</option>
            </select>
          </label>
        </div>
      </div>

      {/* Parameter Grid */}
      <div className="config-section">
        <h3>3. Parameter Grid</h3>
        <div className="grid-toggle">
          <button
            className={!useCustomGrid ? 'active' : ''}
            onClick={() => setUseCustomGrid(false)}
          >
            Use Predefined Grid
          </button>
          <button
            className={useCustomGrid ? 'active' : ''}
            onClick={() => setUseCustomGrid(true)}
          >
            Custom Grid (JSON)
          </button>
        </div>

        {!useCustomGrid ? (
          <div className="param-grid-editor">
            <p className="info">Edit the parameter ranges to search:</p>
            {Object.entries(paramGrids[config.model] || {}).map(([param, values]) => (
              <div key={param} className="param-row">
                <label>{param}:</label>
                <input
                  type="text"
                  value={JSON.stringify(values)}
                  onChange={(e) => {
                    try {
                      const parsed = JSON.parse(e.target.value);
                      updateParamGrid(param, parsed);
                    } catch {
                      // Invalid JSON, ignore
                    }
                  }}
                  placeholder={`e.g., [1, 2, 3] or ["auto", "sqrt"]`}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="custom-grid-editor">
            <p className="info">Enter custom parameter grid as JSON:</p>
            <textarea
              value={customGrid}
              onChange={(e) => setCustomGrid(e.target.value)}
              placeholder={`{\n  "n_estimators": [50, 100, 200],\n  "max_depth": [10, 20, 30],\n  "learning_rate": [0.01, 0.1]\n}`}
              rows="10"
            />
          </div>
        )}
      </div>

      {/* Search Button */}
      <button
        onClick={handleGridSearch}
        disabled={loading || !file}
        className="search-button"
      >
        {loading ? '🔄 Searching... This may take a few minutes' : '🚀 Start Grid Search'}
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
          <h3>✅ Grid Search Complete!</h3>

          {/* Best Parameters */}
          <div className="best-params-card">
            <h4>🏆 Best Parameters Found</h4>
            <div className="params-grid">
              {Object.entries(results.best_params).map(([param, value]) => (
                <div key={param} className="param-item">
                  <span className="param-name">{param}:</span>
                  <span className="param-value">{JSON.stringify(value)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Best CV Score</div>
              <div className="metric-value">{(results.best_score * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Test Score</div>
              <div className="metric-value">{(results.test_score * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Total Fits</div>
              <div className="metric-value">{results.total_fits}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Search Time</div>
              <div className="metric-value">{results.search_time.toFixed(1)}s</div>
            </div>
          </div>

          {/* Top 5 Combinations */}
          {results.top_combinations && (
            <div className="top-combinations">
              <h4>📊 Top 5 Parameter Combinations</h4>
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Mean Score</th>
                    <th>Std Dev</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {results.top_combinations.map((combo, idx) => (
                    <tr key={idx}>
                      <td>{idx + 1}</td>
                      <td>{(combo.mean_score * 100).toFixed(2)}%</td>
                      <td>±{(combo.std_score * 100).toFixed(2)}%</td>
                      <td className="params-cell">
                        {Object.entries(combo.params).map(([k, v]) => (
                          <span key={k} className="param-badge">
                            {k}={JSON.stringify(v)}
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Visualization */}
          {results.plot && (
            <div className="plot-container">
              <h4>Parameter Importance</h4>
              <img src={results.plot} alt="Grid Search Results" />
            </div>
          )}

          {/* Model Code */}
          <div className="code-section">
            <h4>💻 Use This Model in Your Code</h4>
            <pre>
              <code>{results.model_code}</code>
            </pre>
            <button onClick={() => navigator.clipboard.writeText(results.model_code)}>
              📋 Copy Code
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        .grid-search-cv {
          padding: 20px;
          max-width: 1400px;
          margin: 0 auto;
        }

        .description {
          color: #666;
          margin-bottom: 30px;
        }

        .config-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .config-section h3 {
          margin-top: 0;
          color: #333;
        }

        .model-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .model-card {
          background: #f8f9fa;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          padding: 20px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s;
        }

        .model-card:hover {
          border-color: #4CAF50;
          transform: translateY(-2px);
        }

        .model-card.selected {
          border-color: #4CAF50;
          background: #f1f8f4;
        }

        .model-icon {
          font-size: 32px;
          margin-bottom: 10px;
        }

        .model-name {
          font-weight: 500;
          color: #333;
        }

        .cv-settings {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .cv-settings label {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-weight: 500;
        }

        .cv-settings input,
        .cv-settings select {
          padding: 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .grid-toggle {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
        }

        .grid-toggle button {
          flex: 1;
          padding: 10px;
          border: 2px solid #ddd;
          background: white;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.3s;
        }

        .grid-toggle button.active {
          border-color: #4CAF50;
          background: #f1f8f4;
          font-weight: bold;
        }

        .param-grid-editor {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 4px;
        }

        .param-row {
          display: grid;
          grid-template-columns: 200px 1fr;
          gap: 10px;
          margin-bottom: 10px;
          align-items: center;
        }

        .param-row label {
          font-weight: 500;
          color: #555;
        }

        .param-row input {
          padding: 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: monospace;
        }

        .custom-grid-editor textarea {
          width: 100%;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: monospace;
          font-size: 14px;
        }

        .info {
          color: #666;
          font-size: 14px;
          margin-bottom: 10px;
        }

        .search-button {
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
          margin-top: 20px;
        }

        .search-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .search-button:disabled {
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
        }

        .best-params-card {
          background: #f1f8f4;
          border: 2px solid #4CAF50;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .best-params-card h4 {
          margin-top: 0;
          color: #2e7d32;
        }

        .params-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 10px;
        }

        .param-item {
          background: white;
          padding: 10px;
          border-radius: 4px;
          display: flex;
          justify-content: space-between;
        }

        .param-name {
          font-weight: 500;
          color: #555;
        }

        .param-value {
          font-family: monospace;
          color: #2e7d32;
          font-weight: bold;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-bottom: 20px;
        }

        .metric-card {
          background: white;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 20px;
          text-align: center;
        }

        .metric-label {
          font-size: 14px;
          color: #666;
          margin-bottom: 10px;
        }

        .metric-value {
          font-size: 24px;
          font-weight: bold;
          color: #333;
        }

        .top-combinations {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .top-combinations table {
          width: 100%;
          border-collapse: collapse;
        }

        .top-combinations th,
        .top-combinations td {
          padding: 12px;
          text-align: left;
          border-bottom: 1px solid #ddd;
        }

        .top-combinations th {
          background: #f8f9fa;
          font-weight: bold;
          color: #333;
        }

        .params-cell {
          display: flex;
          flex-wrap: wrap;
          gap: 5px;
        }

        .param-badge {
          background: #e3f2fd;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-family: monospace;
        }

        .plot-container {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .plot-container img {
          width: 100%;
          max-width: 800px;
          border-radius: 4px;
        }

        .code-section {
          background: #1e1e1e;
          color: #d4d4d4;
          padding: 20px;
          border-radius: 8px;
          position: relative;
        }

        .code-section h4 {
          color: #d4d4d4;
          margin-top: 0;
        }

        .code-section pre {
          margin: 15px 0;
          overflow-x: auto;
        }

        .code-section code {
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 14px;
          line-height: 1.6;
        }

        .code-section button {
          background: #4CAF50;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 4px;
          cursor: pointer;
          font-weight: bold;
        }

        .code-section button:hover {
          background: #45a049;
        }
      `}</style>
    </div>
  );
};

export default GridSearchCV;
