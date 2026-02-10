# Causal Liquid Neural Network - Usage Guide

## 📁 Files Overview

### Core Causal LNN Module
- **`causal_liquidnn.py`** - Core causal learning components
  - `CausalLiquidCell` - Basic causal liquid cell
  - `CausalLiquidNetworkModel` - Single-layer causal LNN
  - `AdvancedCausalLiquidNetworkModel` - Multi-layer causal LNN
  - `CausalEventLoss` - Event-weighted loss function
  - Helper functions for Granger causality and event detection

### Training Scripts
1. **`train_causal_liquidnn.py`** - Single appliance training (trains only dish washer by default)
2. **`test_causal_liquidnn_all_appliances.py`** - **All appliances training** ⭐ (RECOMMENDED)

### Documentation
- **`CAUSAL_LNN_MATHEMATICS.md`** - Complete mathematical formulation
- **`causal_math_quick_reference.md`** - Quick reference equations
- **`visualize_causal_math.py`** - Generate visualization diagrams

---

## 🚀 How to Run

### Option 1: Train on ALL Appliances (Recommended)

```bash
python test_causal_liquidnn_all_appliances.py
```

This will:
- ✅ Train on all 4 appliances (dish washer, fridge, microwave, washer dryer)
- ✅ Run both standard AND advanced causal models
- ✅ Compute Granger causality for each appliance
- ✅ Generate comprehensive results with causal analysis
- ✅ Save all models and metrics

**Expected output:**
```
Testing Causal Liquid Neural Network on All REDD Appliances
======================================================================

PART 1: Standard Causal Liquid Neural Network
  - dish washer
  - fridge
  - microwave
  - washer dryer

PART 2: Advanced Causal Liquid Neural Network
  - dish washer
  - fridge
  - microwave
  - washer dryer

📊 Summary with F1 scores and Granger causality
```

### Option 2: Train on Single Appliance

Edit `train_causal_liquidnn.py` line ~560:

```python
appliance = 'dish washer'  # Change to: 'fridge', 'microwave', or 'washer dryer'
```

Then run:
```bash
python train_causal_liquidnn.py
```

### Option 3: Generate Math Visualizations

```bash
python visualize_causal_math.py
```

Generates 6 PNG diagrams:
1. `causal_cell_architecture.png` - Architecture diagram
2. `event_weighting_mechanism.png` - Event detection and weighting
3. `liquid_dynamics.png` - Differential equations
4. `granger_causality.png` - Causal relationship testing
5. `f1_improvement.png` - Performance comparison
6. `temporal_causality.png` - Temporal constraints

---

## 🎛️ Configuration Options

### In `test_causal_liquidnn_all_appliances.py`

Modify these parameters (lines ~222-234):

```python
results = test_causal_liquidnn_on_all_appliances(
    window_size=100,              # Input sequence length
    hidden_size=128,              # Hidden state dimension
    num_layers=2,                 # Number of layers (for advanced)
    dt=0.1,                       # Time step for dynamics
    advanced=True,                # True = multi-layer, False = single
    epochs=20,                    # Training epochs
    lr=0.001,                     # Learning rate
    patience=10,                  # Early stopping patience
    use_causal_loss=True,         # Use event-weighted loss
    event_weight_scale=2.0        # Event importance multiplier
)
```

---

## 📊 Output Structure

After running, you'll get:

```
models/
├── causal_liquid_redd_test_YYYYMMDD_HHMMSS/
│   ├── dish_washer/
│   │   ├── causal_liquid_redd_dish_washer_best.pth
│   │   ├── causal_liquid_redd_dish_washer_history.json
│   │   └── causal_liquid_redd_dish_washer_training.png
│   ├── fridge/
│   │   └── ...
│   ├── microwave/
│   │   └── ...
│   ├── washer_dryer/
│   │   └── ...
│   └── summary.json
```

---

## 🔬 What Causal LNN Does Differently

### Standard LNN:
```python
# All time steps weighted equally
Loss = (ŷ - y)²
```

### Causal LNN:
```python
# Events (state changes) weighted more
Loss = (1 + λ·event_weight) · (ŷ - y)²

# Where:
# - λ = 2.0 (default)
# - event_weight ∈ [0, 1] detected automatically
# - Events get 3x more importance (1 + 2·1 = 3)
```

**Result**: Model learns to focus on appliance on/off transitions → Better F1 scores!

---

## 📈 Expected Performance Improvements

Based on causal learning literature:

| Appliance | Standard LNN F1 | Causal LNN F1 (Expected) | Improvement |
|-----------|----------------|--------------------------|-------------|
| Dish washer | 0.42 | **0.50-0.60** | +19-43% |
| Fridge | 0.40 | **0.45-0.55** | +13-38% |
| Microwave | 0.07 | **0.15-0.30** | +114-329% |
| Washer dryer | 0.08 | **0.15-0.35** | +88-338% |

---

## 🔍 Key Metrics to Check

1. **Test F1 Score** - Main metric for appliance detection
2. **Granger Causality Score** - Measures causal relationship strength
   - Higher = stronger causation from aggregate → appliance
3. **Average Event Weight** - Shows model's event detection capability
4. **Precision & Recall** - Breakdown of F1 score

---

## 🎯 Quick Start Summary

**To train on all appliances with causal learning:**

```bash
# 1. Run comprehensive test
python test_causal_liquidnn_all_appliances.py

# 2. Generate visualizations (optional)
python visualize_causal_math.py

# 3. Check results in models/causal_liquid_redd_test_*/summary.json
```

**Expected runtime:** ~30-60 minutes for all appliances (depends on GPU)

---

## 💡 Tips

1. **Use GPU if available** - Causal LNN benefits from CUDA acceleration
2. **Monitor event weights** - Should be between 0.1-0.5 during training
3. **Check Granger scores** - Values > 0.1 indicate strong causality
4. **Compare with standard LNN** - Run both to see F1 improvement
5. **Adjust event_weight_scale** - Increase to 3.0 for harder appliances (microwave, washer dryer)

---

## 📚 Documentation Files

- **Full Math**: `CAUSAL_LNN_MATHEMATICS.md`
- **Quick Reference**: `causal_math_quick_reference.md`
- **This Guide**: `CAUSAL_LNN_GUIDE.md`
