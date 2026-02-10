# Mathematical Formulation of Causal Liquid Neural Networks

## 1. Causal Liquid Cell Dynamics

### Basic Liquid Neuron Equation

The core liquid neuron dynamics follows a continuous-time differential equation:

```
dx/dt = (-x + f(inputs)) / τ
```

Where:
- `x ∈ ℝⁿ` is the hidden state (n = hidden_size)
- `τ ∈ ℝ⁺` is the time constant (learnable parameter)
- `f(·)` is a nonlinear activation function
- `t` is continuous time

### Causal Liquid Cell Forward Pass

At each discrete time step `t`, given input `xₜ ∈ ℝᵈ` and previous hidden state `hₜ₋₁ ∈ ℝⁿ`:

**Step 1: Event Detection**
```
eₜ = σ(Wₑ · xₜ + bₑ) ∈ [0,1]
```
Where:
- `Wₑ ∈ ℝ¹ˣᵈ` is the event detector weight matrix
- `bₑ ∈ ℝ` is the bias
- `σ(·)` is the sigmoid function: `σ(z) = 1/(1 + e⁻ᶻ)`

**Step 2: Input Contribution**
```
iₜ = tanh(Wᵢₙ · xₜ + bᵢₙ) ∈ ℝⁿ
```
Where:
- `Wᵢₙ ∈ ℝⁿˣᵈ` is the input weight matrix
- `bᵢₙ ∈ ℝⁿ` is the input bias
- `tanh(·)` is the hyperbolic tangent: `tanh(z) = (e^z - e⁻ᶻ)/(e^z + e⁻ᶻ)`

**Step 3: Causal Attention**
```
αₜ = σ(Wₐ · hₜ₋₁ + bₐ) ∈ [0,1]ⁿ
```
Where:
- `Wₐ ∈ ℝⁿˣⁿ` is the attention weight matrix
- `bₐ ∈ ℝⁿ` is the attention bias
- Element-wise sigmoid for attention weights

**Step 4: Recurrent Contribution with Attention**
```
rₜ = tanh(Wᵣₑ𝒸 · (αₜ ⊙ hₜ₋₁) + bᵣₑ𝒸) ∈ ℝⁿ
```
Where:
- `Wᵣₑ𝒸 ∈ ℝⁿˣⁿ` is the recurrent weight matrix
- `bᵣₑ𝒸 ∈ ℝⁿ` is the recurrent bias
- `⊙` denotes element-wise multiplication (Hadamard product)

**Step 5: Liquid Dynamics with Time Constants**
```
dhₜ/dt = (-hₜ₋₁ + iₜ + rₜ) / τ̃
```
Where:
- `τ̃ = clamp(τ, 0.1, 10.0)` ensures numerical stability
- `τ ∈ ℝⁿ₊` is a learnable per-neuron time constant

**Step 6: Euler Integration with Event Weighting**
```
hₜ = hₜ₋₁ + Δt · (dhₜ/dt) · (1 + eₜ)
```
Where:
- `Δt` is the discrete time step (dt parameter, e.g., 0.1)
- `(1 + eₜ)` amplifies updates during causal events

**Complete Update Equation:**
```
hₜ = hₜ₋₁ + Δt · [(-hₜ₋₁ + tanh(Wᵢₙ·xₜ) + tanh(Wᵣₑ𝒸·(αₜ⊙hₜ₋₁))) / τ̃] · (1 + σ(Wₑ·xₜ))
```

---

## 2. Causal Event Accumulation

To track the cumulative causal influence over time:

```
Eₜ = β·Eₜ₋₁ + (1-β)·tanh(Wₐ𝒸𝒸 · hₜ + bₐ𝒸𝒸) · eₜ
```

Where:
- `Eₜ ∈ ℝⁿ` is the event accumulator state
- `β = 0.9` is the decay factor (momentum)
- `Wₐ𝒸𝒸 ∈ ℝⁿˣⁿ` is the accumulator weight matrix
- `eₜ` is the event weight from Step 1

This implements an exponential moving average of causal events.

---

## 3. Multi-Layer Causal Processing

For `L` layers, the forward pass at time `t`:

```
For layer ℓ = 1, ..., L:
    if ℓ = 1:
        input⁽ˡ⁾ₜ = xₜ
    else:
        input⁽ˡ⁾ₜ = h⁽ˡ⁻¹⁾ₜ

    h⁽ˡ⁾ₜ, e⁽ˡ⁾ₜ = CausalLiquidCell⁽ˡ⁾(input⁽ˡ⁾ₜ, h⁽ˡ⁾ₜ₋₁)

    h⁽ˡ⁾ₜ = LayerNorm(h⁽ˡ⁾ₜ)

    if ℓ > 1:
        skip⁽ˡ⁾ = Wₛₖᵢₚ⁽ˡ⁾ · h⁽ˡ⁻¹⁾ₜ
        h⁽ˡ⁾ₜ = h⁽ˡ⁾ₜ + skip⁽ˡ⁾
```

Where:
- `h⁽ˡ⁾ₜ` is the hidden state of layer `ℓ` at time `t`
- Skip connections enable direct causal paths between layers

---

## 4. Output Prediction

Given a sequence `X = {x₁, x₂, ..., xₜ}`, process causally (past → present):

```
For t = 1 to T:
    hₜ, eₜ = CausalLiquidCell(xₜ, hₜ₋₁)
    Eₜ = 0.9·Eₜ₋₁ + 0.1·tanh(Wₐ𝒸𝒸·hₜ)·eₜ

Final state: h̃ₜ = hₜ + Eₜ

Output: ŷ = Wₒᵤₜ · h̃ₜ + bₒᵤₜ
```

Where:
- `Wₒᵤₜ ∈ ℝᵐˣⁿ` is the output projection matrix
- `bₒᵤₜ ∈ ℝᵐ` is the output bias
- `m` is the output dimension (1 for appliance power)

---

## 5. Causal Event Detection

### Ground Truth Event Detection

Detect state changes (appliance on/off) in ground truth signal `y`:

```
Event at time t: δₜ = 𝟙(|yₜ - yₜ₋₁| > θ)
```

Where:
- `𝟙(·)` is the indicator function (1 if true, 0 if false)
- `θ` is the threshold (e.g., 10 watts for most appliances)
- `yₜ ∈ ℝ` is the appliance power at time `t`

**Binary event mask:**
```
δ ∈ {0,1}ᵀ where δₜ = {
    1  if |yₜ - yₜ₋₁| > θ  (causal event occurred)
    0  otherwise            (no event)
}
```

---

## 6. Causal Event-Weighted Loss

### Standard Loss Functions

**Mean Squared Error (MSE):**
```
L_MSE(ŷ, y) = (1/N) Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²
```

**Mean Absolute Error (MAE):**
```
L_MAE(ŷ, y) = (1/N) Σᵢ₌₁ᴺ |ŷᵢ - yᵢ|
```

### Causal Event-Weighted Loss

For each sample `i` with predicted event weight `ēᵢ` (averaged over sequence):

```
ēᵢ = (1/T) Σₜ₌₁ᵀ eᵢₜ
```

The **causal weight** for sample `i`:
```
wᵢ = 1 + λ · ēᵢ
```

Where:
- `λ > 0` is the event weight scale (e.g., λ = 2.0)
- Higher `ēᵢ` → higher weight → model focuses more on this sample

**Weighted Loss:**
```
L_causal(ŷ, y, e) = (1/N) Σᵢ₌₁ᴺ wᵢ · L_base(ŷᵢ, yᵢ)
```

Where `L_base` is either MSE or MAE.

**Complete formulation:**
```
L_causal(ŷ, y, e) = (1/N) Σᵢ₌₁ᴺ [1 + λ·(1/T)Σₜeᵢₜ] · (ŷᵢ - yᵢ)²
```

This loss function:
- **Increases weight** when model detects events (high `eᵢₜ`)
- **Focuses learning** on causal state transitions
- **Reduces influence** of steady-state periods

---

## 7. Granger Causality

### Definition

Time series `X` **Granger-causes** `Y` if past values of `X` provide statistically significant information about future values of `Y` beyond the information already in past values of `Y`.

### Mathematical Formulation

**Restricted Model (Y only):**
```
yₜ = α₀ + Σₖ₌₁ᵖ αₖ·yₜ₋ₖ + εₜ⁽ʳ⁾
```

**Full Model (Y and X):**
```
yₜ = β₀ + Σₖ₌₁ᵖ βₖ·yₜ₋ₖ + Σₖ₌₁ᵖ γₖ·xₜ₋ₖ + εₜ⁽ᶠ⁾
```

Where:
- `p` is the maximum lag
- `εₜ⁽ʳ⁾` and `εₜ⁽ᶠ⁾` are residuals
- `α, β, γ` are coefficients

### Residual Sum of Squares (RSS)

```
RSS_restricted = Σₜ (εₜ⁽ʳ⁾)²
RSS_full = Σₜ (εₜ⁽ᶠ⁾)²
```

### Granger Causality Score

```
GC(X → Y) = (RSS_restricted - RSS_full) / RSS_full
```

**Interpretation:**
- `GC > 0`: X helps predict Y (evidence of causality)
- `GC = 0`: X doesn't help predict Y
- Higher GC → stronger causal relationship

**F-statistic for significance testing:**
```
F = [(RSS_restricted - RSS_full) / p] / [RSS_full / (n - 2p - 1)]
```

Where `n` is the number of observations.

---

## 8. Temporal Causality Constraint

### Causal Masking

In sequence processing, ensure causality by masking future information:

```
For attention at time t:
    Attention mask M ∈ {0,1}ᵀˣᵀ where Mᵢⱼ = {
        1  if i ≥ j  (attending to past or present)
        0  if i < j  (masking future)
    }
```

This creates a **lower triangular matrix** ensuring:
- Position `t` can only attend to positions `≤ t`
- No information leakage from future to past

### Causal RNN Property

In the liquid cell, causality is inherently maintained:

```
hₜ = f(xₜ, hₜ₋₁)
```

Where:
- `hₜ` depends **only** on current input `xₜ` and past state `hₜ₋₁`
- **Never** depends on `xₜ₊₁, xₜ₊₂, ...` (future inputs)

---

## 9. NILM-Specific Metrics

### Classification Metrics

For binary classification (on/off), using threshold `θ`:

```
ŷ_binary = 𝟙(ŷ > θ)
y_binary = 𝟙(y > θ)
```

**True Positives (TP):**
```
TP = Σᵢ 𝟙(ŷᵢ > θ ∧ yᵢ > θ)
```

**False Positives (FP):**
```
FP = Σᵢ 𝟙(ŷᵢ > θ ∧ yᵢ ≤ θ)
```

**False Negatives (FN):**
```
FN = Σᵢ 𝟙(ŷᵢ ≤ θ ∧ yᵢ > θ)
```

**Precision:**
```
P = TP / (TP + FP)
```

**Recall:**
```
R = TP / (TP + FN)
```

**F1 Score:**
```
F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)
```

### Regression Metrics

**Mean Absolute Error (MAE):**
```
MAE = (1/N) Σᵢ₌₁ᴺ |ŷᵢ - yᵢ|
```

**Signal Aggregate Error (SAE):**
```
SAE = |Σᵢ₌₁ᴺ ŷᵢ - Σᵢ₌₁ᴺ yᵢ| / Σᵢ₌₁ᴺ yᵢ
```

Measures the relative error in total energy estimation.

---

## 10. Layer Normalization

Applied after each liquid cell update for stability:

```
LayerNorm(h) = γ ⊙ (h - μ)/σ + β
```

Where:
- `μ = (1/n)Σᵢ hᵢ` is the mean
- `σ² = (1/n)Σᵢ(hᵢ - μ)²` is the variance
- `γ, β ∈ ℝⁿ` are learnable scale and shift parameters

---

## 11. Optimization

### Loss Gradient

For parameter `θ`:

```
∂L_causal/∂θ = (1/N) Σᵢ₌₁ᴺ wᵢ · ∂L_base(ŷᵢ, yᵢ)/∂θ
```

### Adam Optimizer

Update rule:

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²

m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ = θₜ₋₁ - η · m̂ₜ / (√v̂ₜ + ε)
```

Where:
- `gₜ = ∂L/∂θ` is the gradient
- `β₁ = 0.9, β₂ = 0.999` (default)
- `η` is the learning rate (e.g., 0.001)
- `ε = 10⁻⁸` for numerical stability

---

## 12. Summary of Key Innovations

### 1. Causal Liquid Dynamics
```
hₜ = hₜ₋₁ + Δt · [f(xₜ, hₜ₋₁)/τ] · (1 + eₜ)
       ↑           ↑              ↑          ↑
    previous    liquid       time    event
     state     dynamics   constant  weight
```

### 2. Event-Weighted Learning
```
L = Σᵢ (1 + λ·ēᵢ) · (ŷᵢ - yᵢ)²
         ↑        ↑
    event weight  base loss
```

### 3. Granger Causality
```
GC(aggregate → appliance) = (RSS_y_only - RSS_y_with_agg) / RSS_y_with_agg
```

These innovations enable the model to:
- ✓ Respect temporal causality
- ✓ Focus on causal events (state transitions)
- ✓ Learn causal relationships between aggregate and appliance power
- ✓ Improve F1 scores for appliance detection

---

## References

1. **Liquid Neural Networks**: Hasani et al., "Liquid Time-constant Networks" (2021)
2. **Causal Inference**: Pearl, J., "Causality: Models, Reasoning, and Inference" (2009)
3. **Granger Causality**: Granger, C.W.J., "Investigating Causal Relations by Econometric Models" (1969)
4. **NILM**: Kelly & Knottenbelt, "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation" (2015)
