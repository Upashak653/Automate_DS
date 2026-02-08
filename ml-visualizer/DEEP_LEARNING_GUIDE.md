# 🧠 Deep Learning Guide - Build Custom Neural Networks

## What is Deep Learning?

Deep Learning uses artificial neural networks with multiple layers to learn complex patterns in data. Unlike traditional ML, deep learning can:
- **Automatically learn features** from raw data
- **Handle complex non-linear relationships**
- **Scale with more data** (more data = better performance)
- **Work with images, text, and tabular data**

**No coding required!** Build and train neural networks visually.

## Quick Start (4 Steps)

1. **Upload CSV** → Your dataset
2. **Design Architecture** → Add layers, select activations
3. **Configure Training** → Set epochs, batch size, optimizer
4. **Train & Visualize** → See loss curves, accuracy, architecture

## Building Your Neural Network

### Step 1: Network Architecture

#### Input Layer (Automatic)
- Automatically sized based on your features
- No configuration needed

#### Hidden Layers
Click "➕ Add Hidden Layer" to add layers between input and output.

**For each hidden layer, configure:**

**Units (Neurons):**
- More units = more capacity to learn
- Typical ranges: 16, 32, 64, 128, 256, 512
- Start with 64 and adjust

**Activation Function:**
Choose based on your needs (see Activation Functions section)

**Dropout Rate:**
- 0.0 = No dropout
- 0.2-0.5 = Recommended for regularization
- Prevents overfitting by randomly dropping neurons

#### Output Layer (Automatic)
- Configured based on your problem:
  - **Binary Classification**: 1 unit, sigmoid
  - **Multi-class**: N units, softmax
  - **Regression**: 1 unit, linear

### Step 2: Activation Functions

#### 📈 ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
**When to use:**
- Default choice for hidden layers
- Fast and effective
- Most popular activation

**Pros:**
- Computationally efficient
- Helps avoid vanishing gradient
- Works well in practice

**Cons:**
- Can cause "dying ReLU" (neurons stop learning)

---

#### 〰️ Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
**When to use:**
- Binary classification output layer
- When you need probabilities (0-1)

**Pros:**
- Smooth gradient
- Output bounded between 0 and 1
- Interpretable as probability

**Cons:**
- Vanishing gradient problem
- Not zero-centered
- Slow for hidden layers

---

#### 🌊 Tanh (Hyperbolic Tangent)
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
**When to use:**
- Hidden layers (alternative to ReLU)
- When you want zero-centered outputs

**Pros:**
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Output between -1 and 1

**Cons:**
- Still suffers from vanishing gradient
- Slower than ReLU

---

#### 🎯 Softmax
```
f(x)ᵢ = e^xᵢ / Σe^xⱼ
```
**When to use:**
- Multi-class classification output layer ONLY
- Never use in hidden layers

**Pros:**
- Outputs sum to 1 (probabilities)
- Perfect for multi-class problems

**Cons:**
- Only for output layer
- Computationally expensive

---

#### 📏 Linear
```
f(x) = x
```
**When to use:**
- Regression output layer
- When you need unbounded outputs

**Pros:**
- Simple, no transformation
- Good for regression

**Cons:**
- No non-linearity
- Not suitable for hidden layers

---

#### 📉 Leaky ReLU
```
f(x) = max(0.01x, x)
```
**When to use:**
- Alternative to ReLU
- When experiencing dying ReLU problem

**Pros:**
- Fixes dying ReLU problem
- Small negative slope allows gradient flow

**Cons:**
- Slightly more complex than ReLU

---

#### 🔄 ELU (Exponential Linear Unit)
```
f(x) = x if x>0 else α(e^x - 1)
```
**When to use:**
- Hidden layers, alternative to ReLU
- When you want smooth negative values

**Pros:**
- Smooth everywhere
- Can produce negative outputs
- Reduces bias shift

**Cons:**
- Computationally expensive (exponential)

---

#### ⚡ SELU (Scaled ELU)
```
f(x) = λx if x>0 else λα(e^x - 1)
```
**When to use:**
- Deep networks (many layers)
- Self-normalizing networks

**Pros:**
- Self-normalizing property
- Good for very deep networks
- Maintains mean and variance

**Cons:**
- Requires specific initialization
- Less common, less tested

---

## Choosing Activation Functions

### For Hidden Layers:
```
1st Choice: ReLU
2nd Choice: Leaky ReLU
3rd Choice: ELU
For Deep Networks: SELU
```

### For Output Layer:
```
Binary Classification: Sigmoid
Multi-class Classification: Softmax
Regression: Linear
```

### Common Patterns:

**Simple Network (2-3 layers):**
```
Input → Dense(64, ReLU) → Dense(32, ReLU) → Output
```

**Deep Network (4+ layers):**
```
Input → Dense(128, ReLU) → Dropout(0.3) 
      → Dense(64, ReLU) → Dropout(0.3)
      → Dense(32, ReLU) → Output
```

**Self-Normalizing Network:**
```
Input → Dense(128, SELU) → Dense(64, SELU) 
      → Dense(32, SELU) → Output
```

## Training Configuration

### Epochs
- **What**: Complete passes through the dataset
- **Typical**: 50-100 for small data, 10-50 for large data
- **Tip**: Use early stopping, don't worry about exact number

### Batch Size
- **What**: Samples processed before updating weights
- **Small (16-32)**: More updates, noisier, better generalization
- **Large (128-512)**: Fewer updates, smoother, faster training
- **Typical**: 32 is a good default

### Validation Split
- **What**: Portion of training data used for validation
- **Typical**: 0.2 (20%)
- **Purpose**: Monitor overfitting during training

### Optimizer

#### Adam (Default - Recommended)
- Adaptive learning rate
- Works well out of the box
- Best for most cases

#### SGD (Stochastic Gradient Descent)
- Simple, classic optimizer
- Requires careful learning rate tuning
- Can achieve better final performance

#### RMSprop
- Good for RNNs
- Adaptive learning rate
- Alternative to Adam

#### Adagrad
- Adapts learning rate per parameter
- Good for sparse data
- Learning rate decreases over time

#### Adamax
- Variant of Adam
- Uses infinity norm
- More stable for some problems

### Learning Rate
- **What**: Step size for weight updates
- **Small (0.0001-0.001)**: Slow but stable
- **Medium (0.001-0.01)**: Good default
- **Large (0.01-0.1)**: Fast but unstable
- **Typical**: 0.001 with Adam

### Early Stopping
- **What**: Stop training when validation loss stops improving
- **Patience**: How many epochs to wait
- **Typical**: Patience of 10
- **Benefit**: Prevents overfitting automatically

## Architecture Design Tips

### How Many Layers?

**Simple Problems (linear-ish):**
```
1-2 hidden layers
```

**Medium Complexity:**
```
2-3 hidden layers
```

**Complex Problems:**
```
3-5 hidden layers
```

**Very Complex (images, text):**
```
5+ hidden layers (consider CNNs/RNNs)
```

### How Many Units Per Layer?

**Funnel Pattern (Recommended):**
```
Input(100) → 64 → 32 → 16 → Output
Gradually decrease units
```

**Constant Pattern:**
```
Input(100) → 64 → 64 → 64 → Output
Same units in each layer
```

**Inverted Funnel:**
```
Input(100) → 128 → 256 → 128 → Output
Expand then contract (less common)
```

### Dropout Strategy

**No Dropout:**
- Small datasets (< 1000 samples)
- Simple problems

**Light Dropout (0.1-0.2):**
- Medium datasets
- Slight overfitting

**Medium Dropout (0.3-0.4):**
- Large datasets
- Clear overfitting

**Heavy Dropout (0.5):**
- Very large datasets
- Severe overfitting

**Pattern:**
```
Input → Dense → Dropout(0.3) → Dense → Dropout(0.3) → Output
Apply after each hidden layer
```

## Common Architectures

### Binary Classification
```
Input → Dense(64, ReLU) → Dropout(0.2)
      → Dense(32, ReLU) → Dropout(0.2)
      → Dense(1, Sigmoid)

Loss: binary_crossentropy
Optimizer: Adam
Metric: Accuracy
```

### Multi-class Classification
```
Input → Dense(128, ReLU) → Dropout(0.3)
      → Dense(64, ReLU) → Dropout(0.3)
      → Dense(N_classes, Softmax)

Loss: sparse_categorical_crossentropy
Optimizer: Adam
Metric: Accuracy
```

### Regression
```
Input → Dense(64, ReLU) → Dropout(0.2)
      → Dense(32, ReLU) → Dropout(0.2)
      → Dense(1, Linear)

Loss: MSE
Optimizer: Adam
Metric: MAE
```

### Deep Network (Complex Data)
```
Input → Dense(256, ReLU) → Dropout(0.4)
      → Dense(128, ReLU) → Dropout(0.4)
      → Dense(64, ReLU) → Dropout(0.3)
      → Dense(32, ReLU) → Dropout(0.2)
      → Output

More layers for complex patterns
Gradually decrease dropout
```

## Understanding Results

### Loss Curves
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease and stabilize
- **Gap**: Small gap = good, large gap = overfitting

**Good Training:**
```
Both curves decrease
Small gap between them
Validation loss stabilizes
```

**Overfitting:**
```
Training loss keeps decreasing
Validation loss increases or plateaus
Large gap between curves
→ Solution: Add dropout, reduce complexity
```

**Underfitting:**
```
Both losses high
Both curves plateau early
→ Solution: Add layers, more units, train longer
```

### Accuracy Curves
- **Training Accuracy**: Should increase
- **Validation Accuracy**: Should increase and stabilize
- **Test Accuracy**: Should be close to validation

### Model Summary
Shows:
- Layer types and shapes
- Number of parameters
- Total model size

**Parameter Count:**
- More parameters = more capacity
- Too many = overfitting risk
- Rule of thumb: params < training samples

## Troubleshooting

### "Loss is NaN"
**Causes:**
- Learning rate too high
- Exploding gradients

**Solutions:**
- Reduce learning rate (try 0.0001)
- Add gradient clipping
- Check for data issues

### "Not Learning (Loss Not Decreasing)"
**Causes:**
- Learning rate too low
- Bad initialization
- Data not normalized

**Solutions:**
- Increase learning rate
- Check data preprocessing
- Try different optimizer

### "Overfitting"
**Signs:**
- Training accuracy high, validation low
- Large gap in loss curves

**Solutions:**
- Add dropout (0.3-0.5)
- Reduce model complexity
- Get more training data
- Use early stopping

### "Underfitting"
**Signs:**
- Both accuracies low
- Loss plateaus early

**Solutions:**
- Add more layers
- Increase units per layer
- Train longer
- Reduce dropout

### "Training Too Slow"
**Solutions:**
- Increase batch size
- Reduce model size
- Use fewer epochs
- Sample your data

## Best Practices

### 1. Start Simple
```
Begin with: 2 layers, 64 units, ReLU
Then: Add complexity if needed
```

### 2. Use Early Stopping
```
Always enable early stopping
Patience: 10-20 epochs
Saves time and prevents overfitting
```

### 3. Monitor Validation
```
Watch validation loss, not training loss
Validation tells you about generalization
```

### 4. Experiment Systematically
```
Change one thing at a time:
1. Try different architectures
2. Adjust learning rate
3. Add/remove dropout
4. Change batch size
```

### 5. Save Your Best Model
```
Early stopping automatically saves best weights
Use the generated code to recreate
```

## Comparison: Deep Learning vs Traditional ML

| Aspect | Deep Learning | Traditional ML |
|--------|---------------|----------------|
| **Data Needed** | More (1000+) | Less (100+) |
| **Training Time** | Longer | Faster |
| **Feature Engineering** | Automatic | Manual |
| **Interpretability** | Lower | Higher |
| **Performance (Large Data)** | Better | Good |
| **Performance (Small Data)** | Good | Better |

### When to Use Deep Learning:
✅ Large datasets (> 10,000 samples)
✅ Complex non-linear patterns
✅ Images, text, sequences
✅ When you have time to train
✅ When you need state-of-the-art performance

### When to Use Traditional ML:
✅ Small datasets (< 1,000 samples)
✅ Need interpretability
✅ Limited computational resources
✅ Tabular data with clear features
✅ Need fast training

## Example Workflows

### Workflow 1: Binary Classification
```
1. Upload data (e.g., customer churn)
2. Build architecture:
   - Hidden: 64 units, ReLU, dropout 0.2
   - Hidden: 32 units, ReLU, dropout 0.2
   - Output: Auto (1 unit, sigmoid)
3. Configure:
   - Epochs: 50
   - Batch size: 32
   - Optimizer: Adam
   - Early stopping: Yes
4. Train and evaluate
5. Check loss curves for overfitting
6. Adjust if needed
```

### Workflow 2: Multi-class Classification
```
1. Upload data (e.g., iris species)
2. Build architecture:
   - Hidden: 128 units, ReLU, dropout 0.3
   - Hidden: 64 units, ReLU, dropout 0.3
   - Output: Auto (N units, softmax)
3. Configure:
   - Epochs: 100
   - Batch size: 16 (small dataset)
   - Optimizer: Adam
   - Early stopping: Yes, patience 15
4. Train and evaluate
5. Check confusion matrix
6. Iterate if needed
```

### Workflow 3: Regression
```
1. Upload data (e.g., house prices)
2. Build architecture:
   - Hidden: 64 units, ReLU, dropout 0.2
   - Hidden: 32 units, ReLU, dropout 0.2
   - Output: Auto (1 unit, linear)
3. Configure:
   - Epochs: 50
   - Batch size: 32
   - Optimizer: Adam
   - Learning rate: 0.001
4. Train and evaluate
5. Check predictions vs actual plot
6. Adjust architecture if needed
```

## Advanced Tips

### 1. Learning Rate Scheduling
Start with default (0.001), if not learning well:
- Try 0.01 (higher)
- Try 0.0001 (lower)

### 2. Batch Normalization
Not available in UI, but in code:
```python
model.add(layers.Dense(64))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
```

### 3. Different Optimizers for Different Problems
- **Adam**: Default, works for most
- **SGD**: Better final performance, needs tuning
- **RMSprop**: Good for RNNs

### 4. Regularization Techniques
- **Dropout**: Available in UI
- **L1/L2**: Add in code with `kernel_regularizer`
- **Early Stopping**: Available in UI

## Resources

- **TensorFlow Docs**: https://www.tensorflow.org/
- **Keras Guide**: https://keras.io/guides/
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **CS231n**: http://cs231n.stanford.edu/

---

**Remember**: Deep learning is iterative. Start simple, train, evaluate, adjust, repeat!
