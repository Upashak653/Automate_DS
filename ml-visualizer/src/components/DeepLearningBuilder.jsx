import React, { useState } from 'react';

const DeepLearningBuilder = ({ file, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  // Network architecture
  const [layers, setLayers] = useState([
    { id: 1, type: 'dense', units: 64, activation: 'relu', dropout: 0 },
    { id: 2, type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
    { id: 3, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
  ]);

  const [architectureType, setArchitectureType] = useState('feedforward'); // feedforward, cnn, rnn, lstm, gru, encoder_decoder

  // Training configuration
  const [config, setConfig] = useState({
    epochs: 50,
    batch_size: 32,
    validation_split: 0.2,
    optimizer: 'adam',
    learning_rate: 0.001,
    loss: 'auto',
    early_stopping: true,
    patience: 10
  });

  const activationFunctions = [
    { value: 'relu', label: 'ReLU', description: 'Most common, good for hidden layers', icon: '📈' },
    { value: 'sigmoid', label: 'Sigmoid', description: 'Output 0-1, good for binary classification', icon: '〰️' },
    { value: 'tanh', label: 'Tanh', description: 'Output -1 to 1, centered around 0', icon: '🌊' },
    { value: 'softmax', label: 'Softmax', description: 'Multi-class classification output', icon: '🎯' },
    { value: 'linear', label: 'Linear', description: 'No activation, for regression output', icon: '📏' },
    { value: 'leaky_relu', label: 'Leaky ReLU', description: 'ReLU with small negative slope', icon: '📉' },
    { value: 'elu', label: 'ELU', description: 'Exponential Linear Unit', icon: '🔄' },
    { value: 'selu', label: 'SELU', description: 'Scaled ELU, self-normalizing', icon: '⚡' }
  ];

  const optimizers = [
    { value: 'adam', label: 'Adam', description: 'Adaptive learning rate, most popular' },
    { value: 'sgd', label: 'SGD', description: 'Stochastic Gradient Descent' },
    { value: 'rmsprop', label: 'RMSprop', description: 'Good for RNNs' },
    { value: 'adagrad', label: 'Adagrad', description: 'Adaptive gradient' },
    { value: 'adamax', label: 'Adamax', description: 'Adam with infinity norm' }
  ];

  const layerTypes = [
    { value: 'dense', label: 'Dense (Fully Connected)', icon: '🔷', description: 'Standard neural network layer' },
    { value: 'conv1d', label: 'Conv1D', icon: '📊', description: 'Convolutional layer for sequences' },
    { value: 'conv2d', label: 'Conv2D', icon: '🖼️', description: 'Convolutional layer for images' },
    { value: 'maxpool1d', label: 'MaxPooling1D', icon: '⬇️', description: 'Downsampling for sequences' },
    { value: 'maxpool2d', label: 'MaxPooling2D', icon: '⬇️', description: 'Downsampling for images' },
    { value: 'lstm', label: 'LSTM', icon: '🔄', description: 'Long Short-Term Memory' },
    { value: 'gru', label: 'GRU', icon: '🌀', description: 'Gated Recurrent Unit' },
    { value: 'simplernn', label: 'SimpleRNN', icon: '↩️', description: 'Basic recurrent layer' },
    { value: 'bidirectional', label: 'Bidirectional', icon: '↔️', description: 'Bidirectional RNN wrapper' },
    { value: 'flatten', label: 'Flatten', icon: '📏', description: 'Flatten multi-dimensional input' },
    { value: 'dropout', label: 'Dropout', icon: '🎲', description: 'Regularization layer' },
    { value: 'batchnorm', label: 'Batch Normalization', icon: '⚖️', description: 'Normalize activations' }
  ];

  const architectureTemplates = [
    { value: 'feedforward', label: 'Feedforward (Dense)', icon: '🔷', description: 'Standard neural network' },
    { value: 'cnn', label: 'CNN (Convolutional)', icon: '🖼️', description: 'For images and spatial data' },
    { value: 'rnn', label: 'RNN (Recurrent)', icon: '↩️', description: 'For sequences and time series' },
    { value: 'lstm', label: 'LSTM Network', icon: '🔄', description: 'For long sequences' },
    { value: 'gru', label: 'GRU Network', icon: '🌀', description: 'Efficient alternative to LSTM' },
    { value: 'encoder_decoder', label: 'Encoder-Decoder', icon: '🔀', description: 'For seq2seq tasks' },
    { value: 'custom', label: 'Custom Architecture', icon: '🎨', description: 'Build from scratch' }
  ];

  const addLayer = (layerType = 'dense') => {
    const newId = Math.max(...layers.map(l => l.id)) + 1;
    const newLayers = [...layers];
    
    let newLayer = { id: newId, type: layerType };
    
    // Set default parameters based on layer type
    switch(layerType) {
      case 'dense':
        newLayer = { ...newLayer, units: 32, activation: 'relu', dropout: 0 };
        break;
      case 'conv1d':
        newLayer = { ...newLayer, filters: 32, kernel_size: 3, activation: 'relu', padding: 'same' };
        break;
      case 'conv2d':
        newLayer = { ...newLayer, filters: 32, kernel_size: 3, activation: 'relu', padding: 'same' };
        break;
      case 'maxpool1d':
        newLayer = { ...newLayer, pool_size: 2 };
        break;
      case 'maxpool2d':
        newLayer = { ...newLayer, pool_size: 2 };
        break;
      case 'lstm':
        newLayer = { ...newLayer, units: 64, return_sequences: false, dropout: 0 };
        break;
      case 'gru':
        newLayer = { ...newLayer, units: 64, return_sequences: false, dropout: 0 };
        break;
      case 'simplernn':
        newLayer = { ...newLayer, units: 32, return_sequences: false };
        break;
      case 'bidirectional':
        newLayer = { ...newLayer, layer_type: 'lstm', units: 64, return_sequences: false };
        break;
      case 'flatten':
        newLayer = { ...newLayer };
        break;
      case 'dropout':
        newLayer = { ...newLayer, rate: 0.3 };
        break;
      case 'batchnorm':
        newLayer = { ...newLayer };
        break;
      default:
        newLayer = { ...newLayer, units: 32, activation: 'relu', dropout: 0 };
    }
    
    // Insert before output layer
    newLayers.splice(layers.length - 1, 0, newLayer);
    setLayers(newLayers);
  };

  const loadTemplate = (template) => {
    setArchitectureType(template);
    
    switch(template) {
      case 'feedforward':
        setLayers([
          { id: 1, type: 'dense', units: 64, activation: 'relu', dropout: 0 },
          { id: 2, type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
          { id: 3, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'cnn':
        setLayers([
          { id: 1, type: 'conv1d', filters: 64, kernel_size: 3, activation: 'relu', padding: 'same' },
          { id: 2, type: 'maxpool1d', pool_size: 2 },
          { id: 3, type: 'conv1d', filters: 32, kernel_size: 3, activation: 'relu', padding: 'same' },
          { id: 4, type: 'maxpool1d', pool_size: 2 },
          { id: 5, type: 'flatten' },
          { id: 6, type: 'dense', units: 64, activation: 'relu', dropout: 0.3 },
          { id: 7, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'rnn':
        setLayers([
          { id: 1, type: 'simplernn', units: 64, return_sequences: true },
          { id: 2, type: 'simplernn', units: 32, return_sequences: false },
          { id: 3, type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
          { id: 4, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'lstm':
        setLayers([
          { id: 1, type: 'lstm', units: 128, return_sequences: true, dropout: 0.2 },
          { id: 2, type: 'lstm', units: 64, return_sequences: false, dropout: 0.2 },
          { id: 3, type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
          { id: 4, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'gru':
        setLayers([
          { id: 1, type: 'gru', units: 128, return_sequences: true, dropout: 0.2 },
          { id: 2, type: 'gru', units: 64, return_sequences: false, dropout: 0.2 },
          { id: 3, type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
          { id: 4, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'encoder_decoder':
        setLayers([
          // Encoder
          { id: 1, type: 'lstm', units: 128, return_sequences: true, dropout: 0.2 },
          { id: 2, type: 'lstm', units: 64, return_sequences: false, dropout: 0.2 },
          // Decoder
          { id: 3, type: 'dense', units: 64, activation: 'relu', dropout: 0.2 },
          { id: 4, type: 'dense', units: 128, activation: 'relu', dropout: 0.2 },
          { id: 5, type: 'output', units: 'auto', activation: 'auto', dropout: 0 }
        ]);
        break;
      
      case 'custom':
        // Keep current layers
        break;
    }
  };

  const removeLayer = (id) => {
    if (layers.length <= 2) return; // Keep at least input and output
    setLayers(layers.filter(l => l.id !== id));
  };

  const updateLayer = (id, field, value) => {
    setLayers(layers.map(l => 
      l.id === id ? { ...l, [field]: value } : l
    ));
  };

  const handleTrain = async () => {
    if (!file) {
      setError('Please upload a CSV file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('layers', JSON.stringify(layers));
    formData.append('config', JSON.stringify(config));

    try {
      const response = await fetch('http://localhost:5000/api/train-deep-learning', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to train neural network');
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

  return (
    <div className="deep-learning-builder">
      <h2>🧠 Deep Learning Model Builder</h2>
      <p className="description">
        Build and train custom neural networks with TensorFlow/Keras. Design your architecture layer by layer!
      </p>

      {/* Architecture Templates */}
      <div className="templates-section">
        <h3>🎯 Architecture Templates</h3>
        <p className="info">Choose a pre-built architecture or build custom</p>
        <div className="templates-grid">
          {architectureTemplates.map(template => (
            <div
              key={template.value}
              className={`template-card ${architectureType === template.value ? 'selected' : ''}`}
              onClick={() => loadTemplate(template.value)}
            >
              <div className="template-icon">{template.icon}</div>
              <div className="template-name">{template.label}</div>
              <div className="template-desc">{template.description}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Builder */}
      <div className="architecture-section">
        <h3>🏗️ Network Architecture</h3>
        
        <div className="layers-container">
          {/* Input Layer (Auto) */}
          <div className="layer-card input-layer">
            <div className="layer-header">
              <span className="layer-icon">📥</span>
              <span className="layer-title">Input Layer</span>
              <span className="layer-badge">Auto-detected</span>
            </div>
            <div className="layer-info">
              Shape will be determined from your data
            </div>
          </div>

          {/* Hidden Layers */}
          {layers.slice(0, -1).map((layer, idx) => (
            <div key={layer.id}>
              <div className="layer-connector">↓</div>
              <div className="layer-card hidden-layer">
                <div className="layer-header">
                  <span className="layer-icon">
                    {layerTypes.find(lt => lt.value === layer.type)?.icon || '🔷'}
                  </span>
                  <span className="layer-title">
                    {layerTypes.find(lt => lt.value === layer.type)?.label || 'Layer'} {idx + 1}
                  </span>
                  <button 
                    className="remove-btn"
                    onClick={() => removeLayer(layer.id)}
                    disabled={layers.length <= 2}
                  >
                    ✕
                  </button>
                </div>
                
                <div className="layer-controls">
                  {/* Layer Type Selector */}
                  <label>
                    Layer Type:
                    <select
                      value={layer.type}
                      onChange={(e) => updateLayer(layer.id, 'type', e.target.value)}
                    >
                      {layerTypes.map(lt => (
                        <option key={lt.value} value={lt.value}>
                          {lt.icon} {lt.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  {/* Dense Layer Controls */}
                  {layer.type === 'dense' && (
                    <>
                      <label>
                        Units (Neurons):
                        <input
                          type="number"
                          min="1"
                          max="512"
                          value={layer.units}
                          onChange={(e) => updateLayer(layer.id, 'units', parseInt(e.target.value))}
                        />
                      </label>
                      <label>
                        Activation Function:
                        <select
                          value={layer.activation}
                          onChange={(e) => updateLayer(layer.id, 'activation', e.target.value)}
                        >
                          {activationFunctions.filter(a => a.value !== 'softmax').map(act => (
                            <option key={act.value} value={act.value}>
                              {act.icon} {act.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </>
                  )}

                  {/* Conv1D/Conv2D Controls */}
                  {(layer.type === 'conv1d' || layer.type === 'conv2d') && (
                    <>
                      <label>
                        Filters:
                        <input
                          type="number"
                          min="1"
                          max="512"
                          value={layer.filters || 32}
                          onChange={(e) => updateLayer(layer.id, 'filters', parseInt(e.target.value))}
                        />
                      </label>
                      <label>
                        Kernel Size:
                        <input
                          type="number"
                          min="1"
                          max="11"
                          value={layer.kernel_size || 3}
                          onChange={(e) => updateLayer(layer.id, 'kernel_size', parseInt(e.target.value))}
                        />
                      </label>
                      <label>
                        Activation:
                        <select
                          value={layer.activation || 'relu'}
                          onChange={(e) => updateLayer(layer.id, 'activation', e.target.value)}
                        >
                          {activationFunctions.filter(a => a.value !== 'softmax').map(act => (
                            <option key={act.value} value={act.value}>
                              {act.icon} {act.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Padding:
                        <select
                          value={layer.padding || 'same'}
                          onChange={(e) => updateLayer(layer.id, 'padding', e.target.value)}
                        >
                          <option value="same">Same</option>
                          <option value="valid">Valid</option>
                        </select>
                      </label>
                    </>
                  )}

                  {/* MaxPooling Controls */}
                  {(layer.type === 'maxpool1d' || layer.type === 'maxpool2d') && (
                    <label>
                      Pool Size:
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={layer.pool_size || 2}
                        onChange={(e) => updateLayer(layer.id, 'pool_size', parseInt(e.target.value))}
                      />
                    </label>
                  )}

                  {/* LSTM/GRU/RNN Controls */}
                  {(layer.type === 'lstm' || layer.type === 'gru' || layer.type === 'simplernn') && (
                    <>
                      <label>
                        Units:
                        <input
                          type="number"
                          min="1"
                          max="512"
                          value={layer.units || 64}
                          onChange={(e) => updateLayer(layer.id, 'units', parseInt(e.target.value))}
                        />
                      </label>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={layer.return_sequences || false}
                          onChange={(e) => updateLayer(layer.id, 'return_sequences', e.target.checked)}
                        />
                        Return Sequences
                        <small>Enable for stacked RNN layers</small>
                      </label>
                      <label>
                        Dropout:
                        <input
                          type="number"
                          min="0"
                          max="0.9"
                          step="0.1"
                          value={layer.dropout || 0}
                          onChange={(e) => updateLayer(layer.id, 'dropout', parseFloat(e.target.value))}
                        />
                      </label>
                    </>
                  )}

                  {/* Bidirectional Controls */}
                  {layer.type === 'bidirectional' && (
                    <>
                      <label>
                        Wrapped Layer:
                        <select
                          value={layer.layer_type || 'lstm'}
                          onChange={(e) => updateLayer(layer.id, 'layer_type', e.target.value)}
                        >
                          <option value="lstm">LSTM</option>
                          <option value="gru">GRU</option>
                          <option value="simplernn">SimpleRNN</option>
                        </select>
                      </label>
                      <label>
                        Units:
                        <input
                          type="number"
                          min="1"
                          max="512"
                          value={layer.units || 64}
                          onChange={(e) => updateLayer(layer.id, 'units', parseInt(e.target.value))}
                        />
                      </label>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={layer.return_sequences || false}
                          onChange={(e) => updateLayer(layer.id, 'return_sequences', e.target.checked)}
                        />
                        Return Sequences
                      </label>
                    </>
                  )}

                  {/* Dropout Layer Controls */}
                  {layer.type === 'dropout' && (
                    <label>
                      Dropout Rate:
                      <input
                        type="number"
                        min="0"
                        max="0.9"
                        step="0.1"
                        value={layer.rate || 0.3}
                        onChange={(e) => updateLayer(layer.id, 'rate', parseFloat(e.target.value))}
                      />
                      <small>{((layer.rate || 0.3) * 100).toFixed(0)}% neurons dropped</small>
                    </label>
                  )}

                  {/* Flatten and BatchNorm have no parameters */}
                  {(layer.type === 'flatten' || layer.type === 'batchnorm') && (
                    <div className="layer-info">
                      No parameters to configure
                    </div>
                  )}
                </div>

                {/* Layer Description */}
                <div className="activation-info">
                  {layerTypes.find(lt => lt.value === layer.type)?.description}
                </div>
              </div>
            </div>
          ))}

          {/* Add Layer Button */}
          <div className="layer-connector">↓</div>
          <div className="add-layer-section">
            <select 
              className="layer-type-select"
              onChange={(e) => { if(e.target.value) { addLayer(e.target.value); e.target.value = ''; } }}
              defaultValue=""
            >
              <option value="" disabled>➕ Add Layer...</option>
              <optgroup label="Basic Layers">
                <option value="dense">🔷 Dense (Fully Connected)</option>
                <option value="dropout">🎲 Dropout</option>
                <option value="batchnorm">⚖️ Batch Normalization</option>
              </optgroup>
              <optgroup label="Convolutional Layers">
                <option value="conv1d">📊 Conv1D</option>
                <option value="conv2d">🖼️ Conv2D</option>
                <option value="maxpool1d">⬇️ MaxPooling1D</option>
                <option value="maxpool2d">⬇️ MaxPooling2D</option>
                <option value="flatten">📏 Flatten</option>
              </optgroup>
              <optgroup label="Recurrent Layers">
                <option value="lstm">🔄 LSTM</option>
                <option value="gru">🌀 GRU</option>
                <option value="simplernn">↩️ SimpleRNN</option>
                <option value="bidirectional">↔️ Bidirectional</option>
              </optgroup>
            </select>
          </div>

          {/* Output Layer */}
          <div className="layer-connector">↓</div>
          <div className="layer-card output-layer">
            <div className="layer-header">
              <span className="layer-icon">📤</span>
              <span className="layer-title">Output Layer</span>
              <span className="layer-badge">Auto-configured</span>
            </div>
            <div className="layer-info">
              Units and activation will be set based on your target variable
              <ul>
                <li>Binary Classification → 1 unit, sigmoid</li>
                <li>Multi-class → N units, softmax</li>
                <li>Regression → 1 unit, linear</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="config-section">
        <h3>⚙️ Training Configuration</h3>
        
        <div className="config-grid">
          <label>
            Epochs:
            <input
              type="number"
              min="1"
              max="500"
              value={config.epochs}
              onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
            />
            <small>Number of complete passes through the dataset</small>
          </label>

          <label>
            Batch Size:
            <input
              type="number"
              min="1"
              max="512"
              value={config.batch_size}
              onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
            />
            <small>Samples per gradient update</small>
          </label>

          <label>
            Validation Split:
            <input
              type="number"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.validation_split}
              onChange={(e) => setConfig({...config, validation_split: parseFloat(e.target.value)})}
            />
            <small>{(config.validation_split * 100).toFixed(0)}% of data for validation</small>
          </label>

          <label>
            Optimizer:
            <select
              value={config.optimizer}
              onChange={(e) => setConfig({...config, optimizer: e.target.value})}
            >
              {optimizers.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.description}
                </option>
              ))}
            </select>
          </label>

          <label>
            Learning Rate:
            <input
              type="number"
              min="0.0001"
              max="0.1"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
            />
            <small>Step size for weight updates</small>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.early_stopping}
              onChange={(e) => setConfig({...config, early_stopping: e.target.checked})}
            />
            Early Stopping
            <small>Stop training when validation loss stops improving</small>
          </label>

          {config.early_stopping && (
            <label>
              Patience:
              <input
                type="number"
                min="1"
                max="50"
                value={config.patience}
                onChange={(e) => setConfig({...config, patience: parseInt(e.target.value)})}
              />
              <small>Epochs to wait before stopping</small>
            </label>
          )}
        </div>
      </div>

      {/* Activation Functions Reference */}
      <div className="reference-section">
        <h3>📚 Activation Functions Reference</h3>
        <div className="activation-grid">
          {activationFunctions.map(act => (
            <div key={act.value} className="activation-card">
              <div className="act-header">
                <span className="act-icon">{act.icon}</span>
                <span className="act-name">{act.label}</span>
              </div>
              <p className="act-desc">{act.description}</p>
              <div className="act-formula">
                {act.value === 'relu' && 'f(x) = max(0, x)'}
                {act.value === 'sigmoid' && 'f(x) = 1 / (1 + e^(-x))'}
                {act.value === 'tanh' && 'f(x) = (e^x - e^(-x)) / (e^x + e^(-x))'}
                {act.value === 'softmax' && 'f(x)ᵢ = e^xᵢ / Σe^xⱼ'}
                {act.value === 'linear' && 'f(x) = x'}
                {act.value === 'leaky_relu' && 'f(x) = max(0.01x, x)'}
                {act.value === 'elu' && 'f(x) = x if x>0 else α(e^x - 1)'}
                {act.value === 'selu' && 'f(x) = λx if x>0 else λα(e^x - 1)'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Train Button */}
      <button
        onClick={handleTrain}
        disabled={loading || !file}
        className="train-button"
      >
        {loading ? '🔄 Training Neural Network...' : '🚀 Train Deep Learning Model'}
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
          <h3>✅ Training Complete!</h3>

          {/* Model Architecture Visualization */}
          {results.architecture_plot && (
            <div className="architecture-viz">
              <h4>🏗️ Model Architecture</h4>
              <img src={results.architecture_plot} alt="Model Architecture" />
            </div>
          )}

          {/* Performance Metrics */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Final Train {results.metric_name}</div>
              <div className="metric-value">{(results.final_train_metric * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Final Val {results.metric_name}</div>
              <div className="metric-value">{(results.final_val_metric * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Test {results.metric_name}</div>
              <div className="metric-value">{(results.test_metric * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Total Epochs</div>
              <div className="metric-value">{results.epochs_trained}</div>
            </div>
          </div>

          {/* Training History Plots */}
          <div className="plots-grid">
            {results.loss_plot && (
              <div className="plot-card">
                <h4>📉 Loss Curves</h4>
                <img src={results.loss_plot} alt="Loss curves" />
              </div>
            )}
            {results.accuracy_plot && (
              <div className="plot-card">
                <h4>📈 Accuracy Curves</h4>
                <img src={results.accuracy_plot} alt="Accuracy curves" />
              </div>
            )}
          </div>

          {/* Confusion Matrix / Predictions */}
          {results.confusion_matrix_plot && (
            <div className="plot-card">
              <h4>🎯 Confusion Matrix</h4>
              <img src={results.confusion_matrix_plot} alt="Confusion Matrix" />
            </div>
          )}

          {results.predictions_plot && (
            <div className="plot-card">
              <h4>📊 Predictions vs Actual</h4>
              <img src={results.predictions_plot} alt="Predictions" />
            </div>
          )}

          {/* Model Summary */}
          <div className="model-summary">
            <h4>📋 Model Summary</h4>
            <pre>{results.model_summary}</pre>
          </div>

          {/* Training Info */}
          <div className="training-info">
            <h4>ℹ️ Training Information</h4>
            <ul>
              <li>Total Parameters: <strong>{results.total_params?.toLocaleString()}</strong></li>
              <li>Trainable Parameters: <strong>{results.trainable_params?.toLocaleString()}</strong></li>
              <li>Training Time: <strong>{results.training_time?.toFixed(1)}s</strong></li>
              <li>Best Epoch: <strong>{results.best_epoch}</strong></li>
              {results.early_stopped && (
                <li className="warning">⚠️ Training stopped early (no improvement)</li>
              )}
            </ul>
          </div>

          {/* Deployment Code */}
          <div className="code-section">
            <h4>💻 TensorFlow/Keras Code</h4>
            <pre><code>{results.model_code}</code></pre>
            <button onClick={() => navigator.clipboard.writeText(results.model_code)}>
              📋 Copy Code
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        .deep-learning-builder {
          padding: 20px;
          max-width: 1400px;
          margin: 0 auto;
        }

        .description {
          color: #666;
          margin-bottom: 30px;
        }

        .architecture-section,
        .config-section,
        .reference-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .layers-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .layer-card {
          width: 100%;
          max-width: 600px;
          background: white;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .input-layer {
          border-left: 4px solid #4CAF50;
        }

        .hidden-layer {
          border-left: 4px solid #2196F3;
        }

        .output-layer {
          border-left: 4px solid #FF9800;
        }

        .layer-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 15px;
        }

        .layer-icon {
          font-size: 24px;
        }

        .layer-title {
          font-weight: bold;
          font-size: 18px;
          flex: 1;
        }

        .layer-badge {
          background: #e3f2fd;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          color: #1976d2;
        }

        .remove-btn {
          background: #f44336;
          color: white;
          border: none;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          cursor: pointer;
          font-size: 14px;
        }

        .remove-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .layer-controls {
          display: grid;
          gap: 15px;
        }

        .layer-controls label {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-weight: 500;
        }

        .layer-controls input,
        .layer-controls select {
          padding: 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .layer-controls small {
          color: #666;
          font-size: 12px;
          font-weight: normal;
        }

        .activation-info {
          margin-top: 10px;
          padding: 10px;
          background: #f1f8f4;
          border-radius: 4px;
          font-size: 13px;
          color: #2e7d32;
        }

        .layer-connector {
          font-size: 24px;
          color: #999;
          margin: 10px 0;
        }

        .templates-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .templates-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .template-card {
          background: #f8f9fa;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          padding: 15px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s;
        }

        .template-card:hover {
          border-color: #667eea;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .template-card.selected {
          border-color: #667eea;
          background: #f0f4ff;
        }

        .template-icon {
          font-size: 32px;
          margin-bottom: 10px;
        }

        .template-name {
          font-weight: bold;
          color: #333;
          margin-bottom: 5px;
        }

        .template-desc {
          font-size: 12px;
          color: #666;
        }

        .add-layer-section {
          width: 100%;
          max-width: 600px;
        }

        .layer-type-select {
          width: 100%;
          padding: 12px 24px;
          background: #4CAF50;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-size: 16px;
          font-weight: bold;
          transition: all 0.3s;
        }

        .layer-type-select:hover {
          background: #45a049;
        }

        .layer-type-select option {
          background: white;
          color: #333;
        }

        .layer-type-select optgroup {
          background: #f8f9fa;
          font-weight: bold;
        }

        .layer-info {
          color: #666;
          font-size: 14px;
        }

        .layer-info ul {
          margin: 10px 0 0 20px;
          padding: 0;
        }

        .layer-info li {
          margin: 5px 0;
        }

        .config-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
        }

        .config-grid label {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-weight: 500;
        }

        .config-grid input,
        .config-grid select {
          padding: 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .config-grid small {
          color: #666;
          font-size: 12px;
          font-weight: normal;
        }

        .checkbox-label {
          flex-direction: row !important;
          align-items: center;
          gap: 10px !important;
        }

        .checkbox-label input[type="checkbox"] {
          width: auto;
        }

        .activation-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
        }

        .activation-card {
          background: #f8f9fa;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          padding: 15px;
          transition: all 0.3s;
        }

        .activation-card:hover {
          border-color: #2196F3;
          transform: translateY(-2px);
        }

        .act-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }

        .act-icon {
          font-size: 24px;
        }

        .act-name {
          font-weight: bold;
          color: #333;
        }

        .act-desc {
          font-size: 13px;
          color: #666;
          margin: 10px 0;
        }

        .act-formula {
          font-family: monospace;
          font-size: 12px;
          background: white;
          padding: 8px;
          border-radius: 4px;
          color: #2196F3;
        }

        .train-button {
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

        .train-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .train-button:disabled {
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

        .architecture-viz {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .architecture-viz img {
          width: 100%;
          max-width: 800px;
          border-radius: 4px;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin: 20px 0;
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

        .plots-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 20px;
          margin: 20px 0;
        }

        .plot-card {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .plot-card img {
          width: 100%;
          border-radius: 4px;
        }

        .model-summary {
          background: #1e1e1e;
          color: #d4d4d4;
          padding: 20px;
          border-radius: 8px;
          margin: 20px 0;
        }

        .model-summary pre {
          margin: 0;
          overflow-x: auto;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 13px;
          line-height: 1.6;
        }

        .training-info {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          margin: 20px 0;
        }

        .training-info ul {
          list-style: none;
          padding: 0;
          margin: 10px 0;
        }

        .training-info li {
          padding: 8px 0;
          border-bottom: 1px solid #e0e0e0;
        }

        .training-info li:last-child {
          border-bottom: none;
        }

        .training-info .warning {
          color: #ff9800;
        }

        .code-section {
          background: #1e1e1e;
          color: #d4d4d4;
          padding: 20px;
          border-radius: 8px;
          position: relative;
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

        @media (max-width: 768px) {
          .plots-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default DeepLearningBuilder;
