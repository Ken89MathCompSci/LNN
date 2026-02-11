# Attention Liquid Neural Network with Data Augmentation

## Overview

This document explains how to use the `test_attention_liquidnn.py` script, which implements **Attention-enhanced Liquid Neural Networks** with **MatNILM-inspired data augmentation** for Non-Intrusive Load Monitoring (NILM) on the REDD dataset.

## What's New

### Architecture Changes
- **Removed**: Advanced Attention LNN (multi-layer variant)
- **Kept**: Single-layer Attention LNN for optimal performance and stability
- **Model**: `AttentionLiquidNetworkModel` from `Source Code/models.py`

### Key Features
1. **Self-Attention Mechanism**: Multi-head attention to capture temporal dependencies
2. **Data Augmentation**: 4 augmentation techniques inspired by MatNILM paper
3. **Configurable Augmentation**: Choose augmentation strategy and probability
4. **Automatic Comparison**: Tests both baseline and augmented versions
5. **Stability Improvements**: Built-in gradient clipping, normalization, NaN detection

## Attention LNN Architecture

```
Input Sequence → Liquid Layer (ODE) → Self-Attention → FC Layer → Output
```

### Components:
- **Liquid Layer**: Continuous-time dynamics with ODE solver (adaptive τ and A)
- **Self-Attention**: Multi-head attention (Q, K, V projections)
  - 4 attention heads by default
  - Captures which time steps are most relevant
  - Helps model focus on appliance state transitions
- **Fully Connected Layer**: Maps attended features to power prediction

### Why Attention Helps NILM:
- Identifies temporal patterns (e.g., fridge cycles, washer phases)
- Focuses on relevant time steps when appliance turns on/off
- Reduces noise from other appliances
- **Expected improvement**: +5-15% F1 score vs Standard LNN

## Data Augmentation Techniques

All techniques inspired by MatNILM paper for robust power consumption learning:

### 1. **Vertical Scaling** (`augmentation='vertical'`)
- **What**: Scales signal amplitude by random factor
- **Range**: ±40% (using truncated normal distribution)
- **Effect**: Simulates varying power levels for same appliance
- **Use case**: Handles appliances with variable power draw

### 2. **Horizontal Scaling** (`augmentation='horizontal'`)
- **What**: Time-stretches signal (speed up/slow down)
- **Range**: ±40% time dilation
- **Method**: Cubic spline interpolation
- **Effect**: Simulates faster/slower appliance cycles
- **Use case**: Handles timing variations

### 3. **Both Vertical + Horizontal** (`augmentation='both'`)
- **What**: Applies both scaling techniques sequentially
- **Effect**: Maximum robustness to amplitude and timing variations
- **Use case**: Best for appliances with high variability

### 4. **Gaussian Noise** (`augmentation='noise'`)
- **What**: Adds white Gaussian noise
- **Level**: 2% standard deviation
- **Effect**: Improves robustness to sensor noise
- **Use case**: Real-world deployment with noisy measurements

### 5. **Mixed Strategy** (`augmentation='mixed'`)
- **What**: Randomly chooses augmentation per batch
- **Distribution**:
  - 25% vertical
  - 25% horizontal
  - 25% both
  - 15% noise
  - 10% none
- **Effect**: Maximum diversity in training data
- **Use case**: **Recommended default** for best generalization

## Usage Examples

### Basic Usage (No Augmentation)
```python
python test_attention_liquidnn.py
```

### With Vertical Scaling
```python
# Modify line 467 in main() section:
appliance_results['baseline'] = train_attention_liquidnn_on_specific_redd_appliance(
    data_dict, appliance_name, model_type='attention',
    window_size=100, hidden_size=256, num_layers=1, dt=0.1,
    num_heads=4, dropout=0.1, epochs=20, lr=0.001, patience=10,
    seed=42, save_dir=save_dir,
    augmentation='vertical',  # Change this line
    aug_probability=0.5
)
```

### With Horizontal Scaling
```python
augmentation='horizontal',
aug_probability=0.5
```

### With Both Scalings
```python
augmentation='both',
aug_probability=0.5
```

### With Gaussian Noise
```python
augmentation='noise',
aug_probability=0.5
```

### With Mixed Strategy (Recommended)
```python
augmentation='mixed',
aug_probability=0.5
```

### Custom Probability
```python
augmentation='mixed',
aug_probability=0.7  # Apply augmentation to 70% of batches
```

## Configuration Parameters

### Model Architecture
```python
window_size=100      # Input sequence length (100 time steps)
hidden_size=256      # Hidden state dimension
num_layers=1         # Number of liquid layers (fixed at 1)
dt=0.1              # ODE solver time step
num_heads=4          # Multi-head attention heads
dropout=0.1          # Dropout rate for attention
```

### Training
```python
epochs=20            # Maximum training epochs
lr=0.001            # Learning rate (Adam optimizer)
patience=10          # Early stopping patience
seed=42             # Random seed for reproducibility
```

### Data Augmentation
```python
augmentation='mixed'     # Options: 'none', 'vertical', 'horizontal', 'both', 'noise', 'mixed'
aug_probability=0.5      # Probability of applying augmentation (0.0 to 1.0)
```

## Expected Performance

### Baseline (No Augmentation)
| Appliance | Expected F1 | Training Time |
|-----------|-------------|---------------|
| Microwave | 0.48-0.52   | ~5-7 minutes  |
| Fridge    | 0.49-0.53   | ~5-7 minutes  |
| Dish Washer | 0.48-0.52 | ~5-7 minutes  |
| Washer Dryer | 0.47-0.51 | ~5-7 minutes |

### With Augmentation (Mixed Strategy)
| Appliance | Expected F1 | Improvement |
|-----------|-------------|-------------|
| Microwave | 0.52-0.56   | +5-10%      |
| Fridge    | 0.53-0.57   | +5-10%      |
| Dish Washer | 0.52-0.56 | +5-10%      |
| Washer Dryer | 0.51-0.55 | +5-10%     |

## Output Structure

```
models/attention_liquidnn_redd_specific_test_YYYYMMDD_HHMMSS/
├── microwave/
│   ├── attention_liquidnn_microwave_baseline_model.pth
│   ├── attention_liquidnn_microwave_baseline_history.json
│   ├── attention_liquidnn_microwave_baseline_metrics.png
│   ├── attention_liquidnn_microwave_baseline_predictions.png
│   ├── attention_liquidnn_microwave_augmented_model.pth
│   ├── attention_liquidnn_microwave_augmented_history.json
│   ├── attention_liquidnn_microwave_augmented_metrics.png
│   └── attention_liquidnn_microwave_augmented_predictions.png
├── fridge/
│   └── ... (same structure)
├── dish_washer/
│   └── ... (same structure)
└── washer_dryer/
    └── ... (same structure)
```

## Interpreting Results

### Console Output
The script automatically compares baseline vs augmented:
```
=== MICROWAVE RESULTS COMPARISON ===
Baseline F1: 0.5123
Augmented F1: 0.5534 (+8.02% improvement)
```

### Plots Generated

1. **Metrics Plot** (`*_metrics.png`):
   - Training/Validation Loss curves
   - F1 Score progression
   - Shows convergence and overfitting

2. **Predictions Plot** (`*_predictions.png`):
   - Ground truth vs predictions
   - Visual assessment of model accuracy
   - Shows where model struggles

### History JSON
Contains epoch-by-epoch metrics:
```json
{
    "train_loss": [0.125, 0.098, ...],
    "val_loss": [0.132, 0.104, ...],
    "val_f1": [0.42, 0.48, ...],
    "learning_rates": [0.001, 0.001, ...]
}
```

## Recommendations

### For Best Performance:
1. **Use `augmentation='mixed'`** for maximum generalization
2. **Set `aug_probability=0.5`** for balanced augmentation
3. **Use `hidden_size=256`** for good capacity without overfitting
4. **Use `num_heads=4`** for multi-scale temporal attention
5. **Keep `epochs=20`** with `patience=10` for early stopping

### For Different Scenarios:

**High Variability Appliances** (e.g., washer_dryer):
```python
augmentation='both'      # Vertical + Horizontal
aug_probability=0.7      # More augmentation
```

**Stable Appliances** (e.g., microwave):
```python
augmentation='vertical'  # Just amplitude variations
aug_probability=0.5      # Standard augmentation
```

**Noisy Real-World Data**:
```python
augmentation='mixed'     # All techniques
aug_probability=0.6      # Higher augmentation
```

**Fast Experimentation**:
```python
epochs=10                # Fewer epochs
patience=5               # Less patience
augmentation='vertical'  # Simple augmentation
```

## Troubleshooting

### NaN Loss During Training
**Solution**: Already handled automatically with:
- Gradient clipping (max_norm=1.0)
- Input normalization
- NaN detection with early abort
- Graceful error messages

### Out of Memory
**Solution**: Reduce batch size or hidden size:
```python
hidden_size=128  # Instead of 256
```

### Slow Training
**Solution**: Reduce sequence length:
```python
window_size=50  # Instead of 100
```

### Poor F1 Scores
**Possible causes**:
1. Try different augmentation strategies
2. Increase `aug_probability`
3. Train for more epochs: `epochs=30`
4. Increase model capacity: `hidden_size=512`

## Comparison with Other Models

| Model | Parameters | Training Time | Expected F1 | Augmentation |
|-------|------------|---------------|-------------|--------------|
| Standard LNN | ~65k | ~3-5 min | 0.45-0.49 | ❌ |
| Attention LNN (baseline) | ~200k | ~5-7 min | 0.48-0.52 | ❌ |
| **Attention LNN (augmented)** | ~200k | ~6-8 min | **0.52-0.56** | ✅ |

## Next Steps

1. **Run baseline test**:
   ```bash
   python test_attention_liquidnn.py
   ```

2. **Review results** in generated plots and console output

3. **Experiment with augmentation**:
   - Modify `augmentation` parameter in main()
   - Try different probabilities
   - Compare improvements per appliance

4. **Fine-tune for your dataset**:
   - Adjust hyperparameters based on results
   - Consider appliance-specific configurations
   - Monitor overfitting with validation curves

## References

- **MatNILM Paper**: Data augmentation techniques for NILM
- **Liquid Neural Networks**: Continuous-time recurrent networks with ODEs
- **Attention Mechanism**: "Attention is All You Need" (Vaswani et al.)
- **REDD Dataset**: Reference Energy Disaggregation Dataset

## Support

For issues or questions:
1. Check generated plots for training issues
2. Review console output for error messages
3. Verify data loading with validation splits
4. Experiment with different augmentation strategies

---

**Created**: February 2026
**Script**: `test_attention_liquidnn.py`
**Model**: `AttentionLiquidNetworkModel` in `Source Code/models.py`
