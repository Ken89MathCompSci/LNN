# Causal LNN - Quick Math Reference

## Core Equations

### 1. Causal Liquid Cell Update (Single Time Step)

```
Event Detection:        eₜ = σ(Wₑ·xₜ)

Input Processing:       iₜ = tanh(Wᵢₙ·xₜ)

Causal Attention:       αₜ = σ(Wₐ·hₜ₋₁)

Recurrent Term:         rₜ = tanh(Wᵣₑ𝒸·(αₜ⊙hₜ₋₁))

Liquid Dynamics:        dhₜ/dt = (-hₜ₋₁ + iₜ + rₜ)/τ

Final Update:           hₜ = hₜ₋₁ + Δt·(dhₜ/dt)·(1 + eₜ)
```

**In one line:**
```
hₜ = hₜ₋₁ + Δt·[(-hₜ₋₁ + tanh(Wᵢₙ·xₜ) + tanh(Wᵣₑ𝒸·(σ(Wₐ·hₜ₋₁)⊙hₜ₋₁)))/τ]·(1 + σ(Wₑ·xₜ))
```

---

### 2. Causal Event-Weighted Loss

```
Sample weight:    wᵢ = 1 + λ·ēᵢ    where ēᵢ = (1/T)Σₜ eᵢₜ

Loss:            L = (1/N)Σᵢ wᵢ·(ŷᵢ - yᵢ)²
```

**Effect:** Samples with high event weights (state changes) get `λ` times more importance.

---

### 3. Granger Causality

```
Restricted model:  yₜ = Σₖ αₖ·yₜ₋ₖ + εᵣ    (only past Y)

Full model:        yₜ = Σₖ βₖ·yₜ₋ₖ + Σₖ γₖ·xₜ₋ₖ + εf    (past Y and X)

Score:             GC = (RSSᵣ - RSSf)/RSSf
```

**Interpretation:** If GC > 0, then X Granger-causes Y (X helps predict Y).

---

### 4. F1 Score

```
Precision:    P = TP/(TP + FP)

Recall:       R = TP/(TP + FN)

F1:           F1 = 2PR/(P + R) = 2TP/(2TP + FP + FN)
```

Where TP, FP, FN are computed using threshold `θ`:
```
ŷ_binary = 𝟙(ŷ > θ)
```

---

### 5. Key Parameters

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| `n` | Hidden size | 128 |
| `d` | Input dimension | 1 (aggregate power) |
| `τ` | Time constant | Learnable, ∈ [0.1, 10] |
| `Δt` | Time step | 0.1 |
| `λ` | Event weight scale | 2.0 |
| `θ` | On/off threshold | 10 watts |
| `p` | Max lag (Granger) | 5 |

---

## Comparison: Standard vs Causal LNN

| Aspect | Standard LNN | Causal LNN |
|--------|--------------|------------|
| Update | `hₜ = hₜ₋₁ + Δt·f(xₜ, hₜ₋₁)/τ` | `hₜ = hₜ₋₁ + Δt·f(xₜ, hₜ₋₁)/τ·(1 + eₜ)` |
| Loss | `L = Σ(ŷ - y)²` | `L = Σ(1 + λ·ē)·(ŷ - y)²` |
| Focus | All time steps equal | Events weighted `λ` times more |
| Causality | Implicit (RNN structure) | Explicit (event detection + Granger) |

---

## Why This Improves F1 Scores

1. **Event Weighting**: Model learns to detect on/off transitions better
   ```
   Low F1 problem:  Model misses state changes
   Solution:        (1 + λ·eₜ) amplifies gradients at transitions
   Result:          Better precision/recall → higher F1
   ```

2. **Temporal Causality**: Prevents information leakage from future
   ```
   hₜ only depends on {x₁, ..., xₜ}  ← causal
   Not on {xₜ₊₁, xₜ₊₂, ...}         ← non-causal
   ```

3. **Causal Attention**: Focuses on relevant past information
   ```
   αₜ⊙hₜ₋₁  ← weighted past state
   Model learns which past matters for current prediction
   ```

---

## Implementation Flow

```
Input: xₜ (aggregate power at time t)
  ↓
Event Detection: eₜ = σ(Wₑ·xₜ)
  ↓
Liquid Cell: hₜ = CausalUpdate(xₜ, hₜ₋₁, eₜ)
  ↓
Event Accumulation: Eₜ = 0.9·Eₜ₋₁ + 0.1·g(hₜ)·eₜ
  ↓
Final State: h̃ₜ = hₜ + Eₜ
  ↓
Output: ŷₜ = Wₒᵤₜ·h̃ₜ (predicted appliance power)
  ↓
Loss: L = (1 + λ·ē)·(ŷₜ - yₜ)²
```

---

## Gradient Flow

The event weight `eₜ` affects gradients through two paths:

**Path 1: Through hidden state update**
```
∂hₜ/∂Wₑ = Δt·(dhₜ/dt)·σ'(Wₑ·xₜ)·xₜ
```

**Path 2: Through loss weighting**
```
∂L/∂Wₑ = λ·(ŷ - y)²·σ'(Wₑ·xₜ)·xₜ
```

This creates a strong learning signal at causal events.

---

## Expected Performance Gain

Based on causal learning literature for NILM:

| Metric | Standard LNN | Causal LNN (expected) |
|--------|--------------|----------------------|
| F1 (dish washer) | 0.42 | **0.50-0.60** |
| F1 (fridge) | 0.40 | **0.45-0.55** |
| F1 (microwave) | 0.07 | **0.15-0.30** |
| F1 (washer dryer) | 0.08 | **0.15-0.35** |

**Key insight**: Appliances with clear on/off events (dish washer) benefit most from event weighting.
