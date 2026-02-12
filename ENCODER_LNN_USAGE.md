# Encoder-Based Liquid Neural Networks for NILM

## Overview

This document explains the three new **Encoder-Based Liquid Neural Network** architectures that combine powerful feature extractors (encoders) with liquid dynamics for improved NILM performance.

## Why Encoder-Based LNN?

Traditional LNN processes raw input sequences directly. Encoder-based architectures **first extract meaningful features** from the input, then process these features with liquid dynamics. This two-stage approach can:

1. **Better Feature Extraction**: Encoders specialize in capturing patterns (local, global, or bidirectional)
2. **Improved Generalization**: Pre-processing reduces noise before liquid dynamics
3. **Higher Performance**: Combining strengths of different architectures
4. **More Interpretable**: Separates feature learning from temporal dynamics

## Three New Architectures

### 1. CNN Encoder + Liquid Decoder 🔷

**Architecture:**
```
Input → CNN Layers (1D convolutions) → Liquid ODE Cell → Output
```

**How it works:**
- **CNN Encoder**: Extracts local temporal patterns using 1D convolutions
- **Multi-layer**: Stack of Conv → BatchNorm → ReLU → Dropout
- **Liquid Decoder**: Processes extracted features with ODE dynamics

**Best for:**
- Appliances with **short, sharp events** (microwave, kettle)
- **Local temporal patterns** (quick on/off cycles)
- **Computational efficiency** (fewer parameters than Transformer)

**Parameters:**
- `num_conv_layers`: Number of CNN layers (default: 3)
- `kernel_size`: Convolution kernel size (default: 5)
- `hidden_size`: Feature dimension (default: 256)

**Example:**
```python
model = CNNEncoderLiquidNetworkModel(
    input_size=1,
    hidden_size=256,
    output_size=1,
    dt=0.1,
    num_conv_layers=3,
    kernel_size=5,
    dropout=0.1
)
```

---

### 2. Transformer Encoder + Liquid Decoder 🔶

**Architecture:**
```
Input → Transformer Encoder (Multi-head Self-Attention) → Liquid ODE Cell → Output
```

**How it works:**
- **Transformer Encoder**: Captures long-range dependencies using self-attention
- **Multi-layer**: Stack of Self-Attention → Feed-Forward → LayerNorm
- **Liquid Decoder**: Refines global features with continuous-time dynamics

**Best for:**
- Appliances with **long cycles** (fridge, washing machine)
- **Complex temporal dependencies** (multi-phase operations)
- **Global context** (understanding entire sequence)

**Parameters:**
- `num_encoder_layers`: Number of Transformer layers (default: 2)
- `num_heads`: Number of attention heads (default: 4)
- `hidden_size`: Feature dimension (default: 256)

**Example:**
```python
model = TransformerEncoderLiquidNetworkModel(
    input_size=1,
    hidden_size=256,
    output_size=1,
    dt=0.1,
    num_encoder_layers=2,
    num_heads=4,
    dropout=0.1
)
```

---

### 3. Bidirectional Encoder + Liquid Decoder 🔵

**Architecture:**
```
Input → Forward Liquid Layer
      ↓
      Backward Liquid Layer → Concatenate → Output
```

**How it works:**
- **Forward Pass**: Processes sequence from start to end
- **Backward Pass**: Processes sequence from end to start
- **Concatenation**: Combines both directions for complete context
- **Best of Both Worlds**: Captures past and future information

**Best for:**
- Appliances with **context-dependent behavior** (dishwasher phases)
- **State transitions** (understanding both lead-up and aftermath)
- **Ambiguous patterns** (needs both past and future context)

**Parameters:**
- `hidden_size`: Feature dimension per direction (default: 256)
  - Note: Output is `hidden_size * 2` due to concatenation
- `dt`: Time step for ODE integration (default: 0.1)

**Example:**
```python
model = BidirectionalEncoderLiquidNetworkModel(
    input_size=1,
    hidden_size=256,
    output_size=1,
    dt=0.1,
    dropout=0.1
)
```

---

## Quick Start

### Run All Three Models

```bash
python test_encoder_liquidnn.py
```

This will:
1. Train CNN Encoder + Liquid on all 4 appliances
2. Train Transformer Encoder + Liquid on all 4 appliances
3. Train Bidirectional Encoder + Liquid on all 4 appliances
4. Compare all results automatically

**Expected Runtime:** ~90-120 minutes total (30-40 min per model type)

### Run Individual Model Type

Edit `test_encoder_liquidnn.py` and comment out the models you don't want to test:

```python
# Only test CNN Encoder
results_cnn = test_encoder_liquidnn_on_all_appliances(
    model_type='cnn_encoder',
    window_size=100,
    hidden_size=256,
    epochs=20
)
```

---

## Configuration Parameters

### Model Architecture

| Parameter | CNN Encoder | Transformer Encoder | Bidirectional |
|-----------|-------------|---------------------|---------------|
| `hidden_size` | 256 | 256 | 256 |
| `dt` | 0.1 | 0.1 | 0.1 |
| `num_conv_layers` | 3 | N/A | N/A |
| `kernel_size` | 5 | N/A | N/A |
| `num_encoder_layers` | N/A | 2 | N/A |
| `num_heads` | N/A | 4 | N/A |
| `dropout` | 0.1 | 0.1 | 0.1 |

### Training

```python
window_size=100      # Input sequence length
epochs=20            # Maximum training epochs
lr=0.001            # Learning rate (Adam optimizer)
patience=10          # Early stopping patience
batch_size=128       # Batch size
```

---

## Expected Performance

### Baseline Comparison

| Model Type | Parameters | Speed | Expected F1 | Best For |
|------------|------------|-------|-------------|----------|
| Standard LNN | ~65k | Fast | 0.45-0.49 | Baseline |
| Attention LNN | ~200k | Medium | 0.48-0.52 | Temporal attention |
| **CNN Encoder + Liquid** | ~180k | Fast | **0.50-0.54** | Local patterns |
| **Transformer Encoder + Liquid** | ~350k | Slow | **0.52-0.56** | Long dependencies |
| **Bidirectional Encoder + Liquid** | ~270k | Medium | **0.51-0.55** | Context-aware |

### Per-Appliance Expectations

| Appliance | CNN Encoder | Transformer Encoder | Bidirectional | Why? |
|-----------|-------------|---------------------|---------------|------|
| **Microwave** | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐⭐ Good | Short, sharp events favor CNN |
| **Fridge** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐⭐ Best | Long cycles favor global context |
| **Dishwasher** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐⭐ Best | Multi-phase operation |
| **Washer Dryer** | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐⭐ Best | Complex temporal patterns |

---

## Output Structure

```
models/encoder_liquidnn_test_YYYYMMDD_HHMMSS/
├── dish_washer/
│   ├── cnn_encoder_liquid_dish_washer_best.pth
│   ├── cnn_encoder_liquid_dish_washer_history.json
│   ├── cnn_encoder_liquid_dish_washer_history.png
│   ├── cnn_encoder_liquid_dish_washer_predictions.png
│   ├── transformer_encoder_liquid_dish_washer_best.pth
│   ├── transformer_encoder_liquid_dish_washer_history.json
│   ├── ...
├── fridge/
│   └── (same structure)
├── microwave/
│   └── (same structure)
├── washer_dryer/
│   └── (same structure)
└── summary.json
```

---

## Advanced Usage

### Hyperparameter Tuning

#### CNN Encoder
```python
# More layers for deeper feature extraction
num_conv_layers=5
kernel_size=7  # Larger receptive field
hidden_size=512  # More capacity

# Less layers for speed
num_conv_layers=2
kernel_size=3
hidden_size=128
```

#### Transformer Encoder
```python
# Deeper network for complex patterns
num_encoder_layers=4
num_heads=8
hidden_size=512

# Lighter network for speed
num_encoder_layers=1
num_heads=2
hidden_size=128
```

#### Bidirectional Encoder
```python
# More capacity
hidden_size=512  # Output will be 1024

# Faster training
hidden_size=128  # Output will be 256
```

### Combining with Data Augmentation

You can integrate MatNILM-exact augmentation from `test_attention_liquidnn.py`:

```python
# In training loop
if augmentation != 'none' and np.random.rand() < aug_probability:
    inputs_np = inputs.cpu().numpy()
    inputs_aug = apply_augmentation_to_batch(inputs_np, mode='mixed')
    inputs = torch.FloatTensor(inputs_aug)
```

Expected improvement: **+5-10% F1 score** across all encoder types

---

## Model Comparison

### Strengths and Weaknesses

#### CNN Encoder + Liquid ✅
**Strengths:**
- Fast training and inference
- Excellent for local patterns
- Lower memory footprint
- Good regularization with BatchNorm

**Weaknesses:**
- Limited global context
- May miss long-range dependencies
- Fixed receptive field

#### Transformer Encoder + Liquid ✅
**Strengths:**
- Captures long-range dependencies
- Global attention mechanism
- Best for complex sequences
- Highly expressive

**Weaknesses:**
- Higher computational cost
- More parameters (slower training)
- Risk of overfitting on small data
- Requires more memory

#### Bidirectional Encoder + Liquid ✅
**Strengths:**
- Complete temporal context
- Understands past and future
- Good for state transitions
- Balanced performance

**Weaknesses:**
- Cannot be used for real-time prediction (needs full sequence)
- Double the computational cost of unidirectional
- More parameters than single-direction

---

## When to Use Which Model?

### Decision Tree 🌳

```
Do you need real-time inference?
├─ YES → Use CNN Encoder (fastest)
└─ NO
   └─ Is your data large (>10k samples)?
      ├─ YES
      │  └─ Do appliances have long cycles?
      │     ├─ YES → Use Transformer Encoder
      │     └─ NO → Use CNN Encoder
      └─ NO
         └─ Use Bidirectional Encoder (best generalization)
```

### Use Case Recommendations

| Use Case | Recommended Model | Why? |
|----------|------------------|-------|
| **Real-time monitoring** | CNN Encoder | Fastest inference |
| **Offline analysis** | Transformer Encoder | Best accuracy |
| **Small dataset** | Bidirectional Encoder | Best generalization |
| **Multi-phase appliances** | Transformer Encoder | Captures phases |
| **Short events** | CNN Encoder | Local pattern specialist |
| **Limited compute** | CNN Encoder | Fewer parameters |
| **Research/benchmark** | All three | Comprehensive comparison |

---

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```python
batch_size=64  # Instead of 128
```

**Solution 2: Reduce model size**
```python
hidden_size=128  # Instead of 256
num_encoder_layers=1  # For Transformer
num_conv_layers=2  # For CNN
```

### Slow Training (Transformer)

**Solution: Use mixed precision training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Poor Performance

**Try:**
1. Increase `hidden_size` to 512
2. Train for more epochs: `epochs=30`
3. Adjust learning rate: `lr=0.0005` or `lr=0.002`
4. Add data augmentation (see above)
5. Use learning rate scheduling

---

## Integration with Existing Code

### Import Models
```python
from models import (
    CNNEncoderLiquidNetworkModel,
    TransformerEncoderLiquidNetworkModel,
    BidirectionalEncoderLiquidNetworkModel
)
```

### Use in Your Training Script
```python
# Replace your existing model creation
model = CNNEncoderLiquidNetworkModel(
    input_size=1,
    hidden_size=256,
    output_size=1,
    dt=0.1
)

# Training loop stays the same
for epoch in range(epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # ... rest of training
```

---

## Future Improvements

### 1. Hybrid Architectures
Combine multiple encoders:
```python
# CNN for local + Transformer for global
CNN → Transformer → Liquid → Output
```

### 2. Attention + Encoder
Add attention after encoder:
```python
Encoder → Self-Attention → Liquid → Output
```

### 3. Multi-Task Learning
Predict power + state simultaneously:
```python
Encoder → Liquid → [Power Head, State Head]
```

### 4. Ensemble Methods
Average predictions from all three encoders:
```python
prediction = (cnn_pred + transformer_pred + bidirectional_pred) / 3
```

---

## References

### Original Papers

1. **Liquid Time-Constant Networks**
   - Hasani et al., "Liquid Time-constant Networks" (NeurIPS 2020)

2. **CNN for Time Series**
   - Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)

3. **Transformer**
   - Vaswani et al., "Attention Is All You Need" (NIPS 2017)

4. **Bidirectional RNN**
   - Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (1997)

### NILM Applications

1. **MatNILM**: Ye et al., "MatNILM: A Multi-Appliance Multi-Task NILM Model" (2021)
2. **Sequence-to-Point**: Zhang et al., "Sequence-to-Point Learning with Neural Networks for NILM" (2018)

---

## Summary

### Key Files

| File | Purpose |
|------|---------|
| `Source Code/models.py` | Contains all three encoder-based model classes |
| `test_encoder_liquidnn.py` | Training and testing script |
| `ENCODER_LNN_USAGE.md` | This documentation file |

### Quick Commands

```bash
# Test all three encoders
python test_encoder_liquidnn.py

# Expected output directories:
# models/encoder_liquidnn_test_YYYYMMDD_HHMMSS/
```

### Expected Results

**Average F1 Scores:**
- CNN Encoder + Liquid: **0.50-0.54**
- Transformer Encoder + Liquid: **0.52-0.56**
- Bidirectional Encoder + Liquid: **0.51-0.55**

**Training Time per Appliance:**
- CNN Encoder: ~6-8 minutes
- Transformer Encoder: ~10-12 minutes
- Bidirectional Encoder: ~8-10 minutes

**Total Time (All 3 models × 4 appliances):** ~90-120 minutes

---

**Created**: February 2026
**Models**: `CNNEncoderLiquidNetworkModel`, `TransformerEncoderLiquidNetworkModel`, `BidirectionalEncoderLiquidNetworkModel`
**Script**: `test_encoder_liquidnn.py`
**Status**: ✅ Ready for Testing
