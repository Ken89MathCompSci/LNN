# Improved Liquid Neural Network - Usage Guide

## 🎯 What This Does

Applies **proven NILM techniques** to improve the standard Liquid Neural Network without complex causal learning mechanisms.

**Goal**: Increase F1 scores from ~0.42 to ~0.50+ using established deep learning best practices.

---

## ✨ Key Improvements

### 1. **Advanced Data Augmentation**
Increases training data diversity and model robustness using sophisticated techniques:
- **Vertical Scaling**: Amplitude scaling using **truncated normal distribution** (μ=1, σ=0.2, range=[0.6, 1.4])
  - More realistic than uniform distribution
  - Simulates calibration variations and sensor differences
- **Horizontal Scaling**: Time stretching/compression using **interpolation**
  - Warps signal in time dimension
  - Simulates appliance usage pattern variations
- **Gaussian Noise**: Adds noise proportional to signal std (5%)
  - Simulates measurement noise and electrical interference
- **Random Mode Selection**: Chooses augmentation randomly (25% each):
  - Mode 0: Original (no augmentation)
  - Mode 1: Vertical scaling only
  - Mode 2: Horizontal scaling only
  - Mode 3: Both vertical and horizontal scaling

### 2. **Better Normalization**
- Uses **dataset-specific statistics** (mean/std from training set)
- Applies same normalization to validation and test sets
- Prevents data leakage and improves convergence

### 3. **Increased Model Capacity**
- **Hidden size**: 128 → **256** (2x neurons)
- **Num layers**: 2 → **3** (deeper architecture)
- More parameters to capture complex appliance patterns

### 4. **Learning Rate Scheduling**
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- Helps escape local minima and find better solutions
- Factor: 0.5, Patience: 5 epochs

### 5. **Gradient Clipping**
- Clips gradients to max norm of **1.0**
- Prevents exploding gradients
- Stabilizes training, especially in deeper networks

### 6. **Longer Training**
- **Epochs**: 20 → **50** (2.5x more training)
- **Patience**: 10 → **15** (more tolerance for convergence)
- Allows model to fully optimize

### 7. **Regularization**
- **Weight decay**: 1e-5 (L2 regularization)
- Prevents overfitting
- Improves generalization to test set

---

## 🚀 How to Run

### Train All Appliances (Recommended)

```bash
python test_improved_liquidnn_all_appliances.py
```

**Expected runtime**: ~60-90 minutes for all 4 appliances (with GPU)

### Train Single Appliance

Edit `train_improved_liquidnn.py` line ~556:

```python
appliance = 'dish washer'  # Change to: 'fridge', 'microwave', or 'washer dryer'
```

Then run:
```bash
python train_improved_liquidnn.py
```

---

## 📊 Configuration Options

In `test_improved_liquidnn_all_appliances.py` (lines ~107-117):

```python
results = test_improved_liquidnn_on_all_appliances(
    window_size=100,
    hidden_size=256,         # Model capacity (128, 256, 512)
    num_layers=3,            # Network depth (2, 3, 4)
    dt=0.1,
    advanced=True,           # Use multi-layer model
    epochs=50,               # Training epochs (20, 50, 100)
    lr=0.001,                # Initial learning rate
    patience=15,             # Early stopping patience
    use_augmentation=True,   # Enable data augmentation
    use_lr_scheduler=True,   # Enable LR scheduling
    gradient_clip=1.0        # Gradient clipping (0.5, 1.0, 5.0)
)
```

---

## 📈 Expected Performance

| Appliance | Standard LNN | Improved LNN (Expected) | Improvement |
|-----------|--------------|------------------------|-------------|
| Dish washer | 0.42 | **0.48-0.55** | +14-31% |
| Fridge | 0.40 | **0.45-0.52** | +13-30% |
| Microwave | 0.07 | **0.12-0.20** | +71-186% |
| Washer dryer | 0.08 | **0.12-0.22** | +50-175% |

### Why These Improvements Work

1. **Data augmentation** → Model sees more diverse patterns → Better generalization
2. **Increased capacity** → More parameters → Can learn complex appliance signatures
3. **LR scheduling** → Escapes local minima → Finds better optima
4. **Longer training** → More optimization steps → Converges to better solution
5. **Better normalization** → Faster convergence → Reaches better performance

---

## 📁 Output Structure

After running, you'll get:

```
models/
├── advanced_improved_liquid_redd_test_YYYYMMDD_HHMMSS/
│   ├── dish_washer/
│   │   ├── advanced_improved_liquid_redd_dish_washer_best.pth
│   │   ├── advanced_improved_liquid_redd_dish_washer_history.json
│   │   └── advanced_improved_liquid_redd_dish_washer_training.png
│   ├── fridge/
│   ├── microwave/
│   ├── washer_dryer/
│   └── summary.json  ← Overall results
```

---

## 🔍 Monitoring Training

The training plots (`*_training.png`) show:

1. **Loss curves** (top-left): Train vs validation loss
2. **F1 progress** (top-right): F1 score over epochs
3. **Learning rate** (bottom-left): LR schedule (if enabled)
4. **MAE progress** (bottom-right): Mean absolute error

**Good signs**:
- Validation loss decreasing steadily
- F1 score increasing
- Learning rate reducing when plateau detected
- No large gap between train and validation (no overfitting)

**Warning signs**:
- Validation loss increasing while train loss decreasing → Overfitting
- F1 score stuck at low value → Try higher learning rate
- Large oscillations → Reduce learning rate or increase gradient clipping

---

## 💡 Tuning Tips

### If F1 is still low:

1. **Increase model capacity**:
   ```python
   hidden_size=512  # Up from 256
   num_layers=4     # Up from 3
   ```

2. **More aggressive augmentation**:
   ```python
   # In train_improved_liquidnn.py, modify vertScale and horiScale
   # Change sigma from 0.2 to 0.3 for more variation
   # This widens the scaling range from [0.6, 1.4] to [0.4, 1.6]
   mu, sigma = 1, 0.3  # Increased from 0.2
   ```

3. **Longer training**:
   ```python
   epochs=100     # Up from 50
   patience=20    # Up from 15
   ```

### If overfitting (large train/val gap):

1. **Stronger regularization**:
   ```python
   # In train_improved_liquidnn.py, line ~256
   optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Up from 1e-5
   ```

2. **More augmentation** (see above)

3. **Reduce model capacity**:
   ```python
   hidden_size=128  # Down from 256
   num_layers=2     # Down from 3
   ```

---

## 🆚 Comparison: Standard vs Improved LNN

| Feature | Standard LNN | Improved LNN |
|---------|-------------|--------------|
| Hidden size | 128 | **256** |
| Num layers | 2 | **3** |
| Epochs | 20 | **50** |
| Patience | 10 | **15** |
| Data augmentation | ❌ No | ✅ Yes (noise, scale, shift) |
| Normalization | Basic | ✅ Dataset statistics |
| LR scheduling | ❌ No | ✅ ReduceLROnPlateau |
| Gradient clipping | ❌ No | ✅ Yes (1.0) |
| Weight decay | ❌ No | ✅ Yes (1e-5) |
| **Expected F1 (dish washer)** | 0.42 | **0.48-0.55** |

---

## 📚 Why We Abandoned Causal Learning

The causal approach had fundamental issues:

1. **No supervision for events**: Model tried to learn event detection without explicit labels
2. **Chicken-and-egg problem**: Poor event detection → Poor training → Poor event detection
3. **Added complexity**: More parameters, harder to train, slower convergence
4. **Worse results**: F1 dropped from 0.42 to 0.10

**Better approach**: Focus on proven techniques that work consistently across domains.

---

## 🎯 Quick Start

```bash
# 1. Run improved training on all appliances
python test_improved_liquidnn_all_appliances.py

# 2. Check results
cat models/advanced_improved_liquid_redd_test_*/summary.json

# 3. View training plots
# Look at models/advanced_improved_liquid_redd_test_*/*/training.png
```

**Expected runtime**: ~60-90 minutes with GPU, ~4-6 hours with CPU

---

## 📖 Key Takeaways

✅ **Simple improvements often work better than complex theory**

✅ **Data augmentation is crucial for NILM (limited training data)**

✅ **Model capacity matters** (more neurons → better pattern recognition)

✅ **Training details matter** (LR scheduling, gradient clipping, regularization)

✅ **Longer training helps** (more epochs with early stopping)

---

## 🔗 Related Files

- **Training script**: `train_improved_liquidnn.py`
- **All appliances test**: `test_improved_liquidnn_all_appliances.py`
- **This guide**: `IMPROVED_LNN_GUIDE.md`
- **Original LNN**: `test_liquidnn_redd_specific_splits.py` (for comparison)
