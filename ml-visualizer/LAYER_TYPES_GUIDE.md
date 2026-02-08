# 🧱 Layer Types Guide - Complete Reference

## Overview

The Deep Learning Builder now supports **12 different layer types** for building custom neural networks including CNNs, RNNs, LSTMs, GRUs, and Encoder-Decoder architectures.

## Architecture Templates

### 🔷 Feedforward (Dense)
**Best for**: Tabular data, standard classification/regression
```
Input → Dense(64, ReLU) → Dense(32, ReLU) → Output
```

### 🖼️ CNN (Convolutional)
**Best for**: Images, spatial data, pattern recognition
```
Input → Conv1D(64) → MaxPool → Conv1D(32) → MaxPool → Flatten → Dense(64) → Output
```

### ↩️ RNN (Recurrent)
**Best for**: Simple sequences, time series
```
Input → SimpleRNN(64) → SimpleRNN(32) → Dense(32) → Output
```

### 🔄 LSTM Network
**Best for**: Long sequences, complex temporal patterns
```
Input → LSTM(128) → LSTM(64) → Dense(32) → Output
```

### 🌀 GRU Network
**Best for**: Sequences, faster alternative to LSTM
```
Input → GRU(128) → GRU(64) → Dense(32) → Output
```

### 🔀 Encoder-Decoder
**Best for**: Sequence-to-sequence tasks, translation
```
Input → LSTM(128) → LSTM(64) → Dense(64) → Dense(128) → Output
```

## Layer Types

### 1. 🔷 Dense (Fully Connected)

**What it does**: Standard neural network layer where every neuron connects to every neuron in the next layer.

**Parameters**:
- **Units**: Number of neurons (1-512)
- **Activation**: relu, sigmoid, tanh, etc.

**When to use**:
- Hidden layers in feedforward networks
- After flattening in CNNs
- Final processing before output

**Example**:
```python
Dense(64, activation='relu')
```

---

### 2. 📊 Conv1D (1D Convolution)

**What it does**: Applies convolution filters over 1D sequences (time series, text).

**Parameters**:
- **Filters**: Number of filters (1-512)
- **Kernel Size**: Size of convolution window (1-11)
- **Activation**: relu, tanh, etc.
- **Padding**: 'same' or 'valid'

**When to use**:
- Time series data
- Text sequences
- Audio signals
- Sensor data

**Example**:
```python
Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
```

**Best practices**:
- Start with 32-64 filters
- Kernel size 3-5 for most tasks
- Use 'same' padding to maintain dimensions
- Stack multiple Conv1D layers

---

### 3. 🖼️ Conv2D (2D Convolution)

**What it does**: Applies convolution filters over 2D images.

**Parameters**:
- **Filters**: Number of filters (1-512)
- **Kernel Size**: Size of convolution window (1-11)
- **Activation**: relu, tanh, etc.
- **Padding**: 'same' or 'valid'

**When to use**:
- Image classification
- Object detection
- Image segmentation
- Computer vision tasks

**Example**:
```python
Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')
```

**Best practices**:
- Start with 32 filters, increase in deeper layers
- Kernel size 3x3 is most common
- Use 'same' padding for consistent dimensions
- Follow with MaxPooling2D

---

### 4. ⬇️ MaxPooling1D

**What it does**: Downsamples 1D sequences by taking maximum value in each window.

**Parameters**:
- **Pool Size**: Size of pooling window (1-10)

**When to use**:
- After Conv1D layers
- To reduce sequence length
- To extract dominant features

**Example**:
```python
MaxPooling1D(pool_size=2)
```

**Best practices**:
- Use pool_size=2 most commonly
- Place after Conv1D layers
- Reduces computational cost
- Helps prevent overfitting

---

### 5. ⬇️ MaxPooling2D

**What it does**: Downsamples 2D images by taking maximum value in each window.

**Parameters**:
- **Pool Size**: Size of pooling window (1-10)

**When to use**:
- After Conv2D layers
- To reduce image dimensions
- To extract dominant features

**Example**:
```python
MaxPooling2D(pool_size=2)
```

**Best practices**:
- Use pool_size=2 (2x2 window)
- Place after Conv2D layers
- Reduces spatial dimensions by half
- Makes network translation-invariant

---

### 6. 🔄 LSTM (Long Short-Term Memory)

**What it does**: Recurrent layer that can learn long-term dependencies in sequences.

**Parameters**:
- **Units**: Number of LSTM cells (1-512)
- **Return Sequences**: True for stacked LSTMs, False for last layer
- **Dropout**: Dropout rate (0-0.9)

**When to use**:
- Long sequences (> 100 timesteps)
- Text generation
- Speech recognition
- Time series forecasting
- When you need to remember long-term patterns

**Example**:
```python
LSTM(units=128, return_sequences=True, dropout=0.2)
```

**Best practices**:
- Use 64-256 units
- Set return_sequences=True for stacked LSTMs
- Add dropout (0.2-0.3) to prevent overfitting
- Use for sequences longer than 50 timesteps

**Stacking LSTMs**:
```python
LSTM(128, return_sequences=True)  # First layer
LSTM(64, return_sequences=False)  # Last layer
```

---

### 7. 🌀 GRU (Gated Recurrent Unit)

**What it does**: Simplified version of LSTM, faster training, similar performance.

**Parameters**:
- **Units**: Number of GRU cells (1-512)
- **Return Sequences**: True for stacked GRUs, False for last layer
- **Dropout**: Dropout rate (0-0.9)

**When to use**:
- Alternative to LSTM
- When training time is important
- Sequences of medium length
- When LSTM is too slow

**Example**:
```python
GRU(units=128, return_sequences=True, dropout=0.2)
```

**Best practices**:
- Use 64-256 units
- Faster than LSTM, try first
- Set return_sequences=True for stacking
- Good default choice for RNNs

**GRU vs LSTM**:
- GRU: Faster, fewer parameters, good for most tasks
- LSTM: Better for very long sequences, more expressive

---

### 8. ↩️ SimpleRNN

**What it does**: Basic recurrent layer, simplest form of RNN.

**Parameters**:
- **Units**: Number of RNN cells (1-512)
- **Return Sequences**: True for stacked RNNs, False for last layer

**When to use**:
- Short sequences (< 50 timesteps)
- Simple temporal patterns
- When LSTM/GRU is overkill
- Learning RNNs (educational)

**Example**:
```python
SimpleRNN(units=64, return_sequences=True)
```

**Best practices**:
- Use for simple tasks only
- Suffers from vanishing gradient
- Prefer LSTM/GRU for real applications
- Good for learning concepts

---

### 9. ↔️ Bidirectional

**What it does**: Wraps RNN/LSTM/GRU to process sequences in both directions.

**Parameters**:
- **Wrapped Layer**: lstm, gru, or simplernn
- **Units**: Number of cells (1-512)
- **Return Sequences**: True for stacking, False for last layer

**When to use**:
- When future context matters
- Text classification
- Named entity recognition
- When you have full sequence available

**Example**:
```python
Bidirectional(LSTM(64, return_sequences=True))
```

**Best practices**:
- Doubles the number of parameters
- Use when full sequence is available
- Not for real-time prediction
- Excellent for text tasks

**Note**: Cannot use for real-time streaming data (needs full sequence).

---

### 10. 📏 Flatten

**What it does**: Converts multi-dimensional input to 1D vector.

**Parameters**: None

**When to use**:
- After Conv layers before Dense layers
- To connect CNN to Dense layers
- Required transition layer

**Example**:
```python
Flatten()
```

**Best practices**:
- Always use after Conv/Pooling before Dense
- No parameters to configure
- Essential for CNN architectures

**Typical pattern**:
```python
Conv1D(64) → MaxPooling1D() → Flatten() → Dense(64)
```

---

### 11. 🎲 Dropout

**What it does**: Randomly drops neurons during training to prevent overfitting.

**Parameters**:
- **Rate**: Dropout rate (0-0.9)

**When to use**:
- After Dense layers
- To prevent overfitting
- When validation loss increases

**Example**:
```python
Dropout(rate=0.3)
```

**Best practices**:
- Use 0.2-0.5 for most cases
- Place after Dense layers
- Don't use before output layer
- Higher rate = more regularization

**Dropout rates**:
- 0.2: Light regularization
- 0.3-0.4: Standard
- 0.5: Heavy regularization

---

### 12. ⚖️ Batch Normalization

**What it does**: Normalizes layer inputs to stabilize and speed up training.

**Parameters**: None (auto-configured)

**When to use**:
- After Dense/Conv layers
- Before activation
- To speed up training
- To allow higher learning rates

**Example**:
```python
Dense(64) → BatchNormalization() → Activation('relu')
```

**Best practices**:
- Place after Dense/Conv, before activation
- Helps with deep networks
- Reduces need for dropout
- Allows higher learning rates

---

## Common Architecture Patterns

### Pattern 1: Simple Feedforward
```
Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Output
```
**Use for**: Tabular data, simple classification

### Pattern 2: CNN for Sequences
```
Conv1D(64) → MaxPool1D → Conv1D(32) → MaxPool1D → Flatten → Dense(64) → Output
```
**Use for**: Time series, text classification

### Pattern 3: LSTM for Sequences
```
LSTM(128, return_seq=True) → LSTM(64) → Dense(32) → Output
```
**Use for**: Long sequences, time series forecasting

### Pattern 4: Bidirectional LSTM
```
Bidirectional(LSTM(64, return_seq=True)) → LSTM(32) → Dense(32) → Output
```
**Use for**: Text classification, NER

### Pattern 5: CNN + LSTM
```
Conv1D(64) → MaxPool1D → LSTM(64) → Dense(32) → Output
```
**Use for**: Complex time series, audio

### Pattern 6: Encoder-Decoder
```
# Encoder
LSTM(128, return_seq=True) → LSTM(64)
# Decoder  
Dense(64) → Dense(128) → Output
```
**Use for**: Sequence-to-sequence, translation

### Pattern 7: Deep CNN
```
Conv1D(64) → Conv1D(64) → MaxPool1D →
Conv1D(32) → Conv1D(32) → MaxPool1D →
Flatten → Dense(128) → Dropout(0.3) → Output
```
**Use for**: Complex pattern recognition

### Pattern 8: Residual-like
```
Dense(64) → BatchNorm → Dense(64) → BatchNorm → Dense(32) → Output
```
**Use for**: Very deep networks

## Quick Decision Guide

### For Tabular Data:
```
Use: Dense layers
Pattern: Dense → Dropout → Dense → Output
```

### For Time Series:
```
Short sequences (< 50): SimpleRNN or Conv1D
Medium sequences (50-200): GRU or LSTM
Long sequences (> 200): LSTM or Bidirectional LSTM
```

### For Images:
```
Use: Conv2D → MaxPool2D → Flatten → Dense
Pattern: Multiple Conv2D blocks, then Dense layers
```

### For Text:
```
Classification: Bidirectional LSTM or Conv1D
Generation: LSTM with return_sequences=True
```

### For Audio:
```
Use: Conv1D → LSTM → Dense
Pattern: Extract features with Conv1D, model with LSTM
```

## Tips & Tricks

### 1. Return Sequences
- **True**: Pass full sequence to next layer (for stacking RNNs)
- **False**: Pass only last output (for final RNN layer)

### 2. Stacking RNNs
```python
LSTM(128, return_sequences=True)  # Must be True
LSTM(64, return_sequences=True)   # Must be True  
LSTM(32, return_sequences=False)  # Last one False
```

### 3. CNN Best Practices
- Start with small filters (32-64)
- Increase filters in deeper layers
- Use MaxPooling after Conv layers
- Always Flatten before Dense layers

### 4. Regularization
- Use Dropout (0.2-0.5) for Dense layers
- Use Dropout in LSTM/GRU (0.2-0.3)
- Use BatchNormalization for deep networks
- Combine techniques for best results

### 5. Common Mistakes
- ❌ Forgetting Flatten after Conv layers
- ❌ Not setting return_sequences=True for stacked RNNs
- ❌ Using Dropout before output layer
- ❌ Too many parameters (overfitting)
- ❌ Wrong input shape for RNN/CNN

## Troubleshooting

### "Shape mismatch error"
- Check if you need Flatten after Conv/Pool layers
- Verify return_sequences for stacked RNNs
- Ensure input shape matches layer type

### "Model not learning"
- Try different layer types
- Adjust number of units
- Check activation functions
- Verify data preprocessing

### "Overfitting"
- Add Dropout layers
- Use BatchNormalization
- Reduce model complexity
- Get more training data

### "Training too slow"
- Use GRU instead of LSTM
- Reduce number of units
- Use fewer layers
- Increase batch size

---

**Remember**: Start simple, then add complexity as needed!
