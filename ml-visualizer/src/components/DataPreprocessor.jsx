import React, { useState } from 'react';
import { Settings, Download, Play, CheckCircle } from 'lucide-react';

const BACKEND_URL = 'http://localhost:5000';

export default function DataPreprocessor({ file, onPreprocessed }) {
  const [options, setOptions] = useState({
    remove_duplicates: false,
    missing_strategy: 'none',
    missing_threshold: 0.5,
    remove_outliers: false,
    encode_categorical: false,
    standardize: false,
    normalize: false,
    feature_selection: false,
    variance_threshold: 0.01,
    apply_pca: false,
    pca_components: 'auto'
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePreprocess = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Add all options to form data
      Object.entries(options).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });
      
      const response = await fetch(`${BACKEND_URL}/api/preprocess`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Preprocessing failed');
      }
      
      const data = await response.json();
      setResult(data);
      
      if (onPreprocessed) {
        onPreprocessed(data);
      }
    } catch (err) {
      setError('Failed to preprocess: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const downloadProcessed = () => {
    if (!result || !result.csv_data) return;
    
    const blob = new Blob([result.csv_data], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'preprocessed_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
      <div className="flex items-center gap-3 mb-4">
        <Settings className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
          Data Preprocessing
        </h2>
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Apply professional data preprocessing techniques using sklearn
      </p>

      {/* Options Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Basic Cleaning */}
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-800 dark:text-white">Basic Cleaning</h3>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.remove_duplicates}
              onChange={(e) => setOptions({...options, remove_duplicates: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Remove Duplicate Rows</span>
          </label>

          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Handle Missing Values
            </label>
            <select
              value={options.missing_strategy}
              onChange={(e) => setOptions({...options, missing_strategy: e.target.value})}
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-white text-sm"
            >
              <option value="none">Keep Missing Values</option>
              <option value="drop_rows">Drop Rows with Missing</option>
              <option value="drop_columns">Drop Columns with Missing</option>
              <option value="mean">Fill with Mean</option>
              <option value="median">Fill with Median</option>
              <option value="mode">Fill with Mode</option>
            </select>
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.remove_outliers}
              onChange={(e) => setOptions({...options, remove_outliers: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Remove Outliers (IQR Method)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.encode_categorical}
              onChange={(e) => setOptions({...options, encode_categorical: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Encode Categorical Variables</span>
          </label>
        </div>

        {/* Scaling & Transformation */}
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-800 dark:text-white">Scaling & Transformation</h3>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.standardize}
              onChange={(e) => setOptions({...options, standardize: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Standardize (Z-score: mean=0, std=1)
            </span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.normalize}
              onChange={(e) => setOptions({...options, normalize: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Normalize (Min-Max: range 0-1)
            </span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={options.feature_selection}
              onChange={(e) => setOptions({...options, feature_selection: e.target.checked})}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Feature Selection (Remove Low Variance)
            </span>
          </label>

          <div>
            <label className="flex items-center gap-2 cursor-pointer mb-2">
              <input
                type="checkbox"
                checked={options.apply_pca}
                onChange={(e) => setOptions({...options, apply_pca: e.target.checked})}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Apply PCA (Dimensionality Reduction)
              </span>
            </label>
            {options.apply_pca && (
              <input
                type="text"
                value={options.pca_components}
                onChange={(e) => setOptions({...options, pca_components: e.target.value})}
                placeholder="auto or number"
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-white text-sm"
              />
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={handlePreprocess}
          disabled={loading || !file}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white py-3 px-6 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2 font-semibold"
        >
          <Play className="w-5 h-5" />
          {loading ? 'Processing...' : 'Apply Preprocessing'}
        </button>
        
        {result && (
          <button
            onClick={downloadProcessed}
            className="bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg transition-colors flex items-center gap-2 font-semibold"
          >
            <Download className="w-5 h-5" />
            Download
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mt-6 space-y-4">
          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
              <h3 className="font-semibold text-green-800 dark:text-green-300">
                {result.message}
              </h3>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs text-gray-600 dark:text-gray-400">Original Shape</p>
                <p className="text-lg font-bold text-gray-800 dark:text-white">
                  {result.summary.original_shape[0]} × {result.summary.original_shape[1]}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs text-gray-600 dark:text-gray-400">Final Shape</p>
                <p className="text-lg font-bold text-gray-800 dark:text-white">
                  {result.summary.final_shape[0]} × {result.summary.final_shape[1]}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs text-gray-600 dark:text-gray-400">Rows Removed</p>
                <p className="text-lg font-bold text-red-600 dark:text-red-400">
                  {result.summary.rows_removed}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs text-gray-600 dark:text-gray-400">Columns Removed</p>
                <p className="text-lg font-bold text-red-600 dark:text-red-400">
                  {result.summary.columns_removed}
                </p>
              </div>
            </div>
          </div>

          {/* Steps Applied */}
          {result.summary.preprocessing_steps.length > 0 && (
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                Steps Applied:
              </h4>
              <ul className="space-y-1">
                {result.summary.preprocessing_steps.map((step, idx) => (
                  <li key={idx} className="text-sm text-blue-800 dark:text-blue-200">
                    ✓ {step}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Preview */}
          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <h4 className="font-semibold text-gray-800 dark:text-white mb-2">
              Preview (first 5 rows):
            </h4>
            <div className="overflow-x-auto">
              <pre className="text-xs text-gray-800 dark:text-gray-200">
                {JSON.stringify(result.summary.preview.slice(0, 5), null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
