"""
KAN-PINN (Kolmogorov-Arnold Network + Physics-Informed) — REDD specific splits
===============================================================================
Key ideas adapted from GraphKANLoc for NILM:

  1. KAN layers replace the LNN encoder.
     y_i = Σ_j ϕ_{i,j}(x_j)
     where ϕ_{i,j} is a learnable cubic B-spline + SiLU residual.
     B-splines are C², so higher-order derivative terms in the physics loss
     have well-defined, stable gradients — unlike ReLU whose 'kinks' make
     acceleration/jerk gradients chaotic.

  2. Multi-point output: the model predicts K_OUT power values that span
     the entire input window, not just the midpoint.  The supervised signal
     is still the midpoint label; the remaining K-1 predictions are shaped
     only by the physics loss — unsupervised trajectory regularisation.

  3. Physics-Informed Loss  L_phys = λ_v·L_vel + λ_a·L_acc + λ_j·L_jerk
       Velocity:     L_vel  = mean(Δp²)     — prevents large power jumps
       Acceleration: L_acc  = mean(Δ²p²)    — limits rate-of-change
       Jerk:         L_jerk = mean(Δ³p²)    — suppresses prediction jitter
     Finite differences Δ are taken along the K_OUT axis.

Architecture:
    Mains window (batch, WIN=100, 1)
         ↓
    Feature extraction
        AdaptiveAvgPool1d(N_POOL)  — local segment means
        [mean, std, min, max]      — global statistics
        → feat (batch, N_POOL+4)  ∈ [0,1]
         ↓
    KAN encoder (3 layers)
        KAN(N_POOL+4 → hidden) + sigmoid
        KAN(hidden   → hidden) + sigmoid
        KAN(hidden   → K_OUT)            — K_OUT power predictions
         ↓
    Supervised loss  : MSE + BCE on prediction at K_OUT//2 (midpoint)
    Physics loss     : velocity / acceleration / jerk over all K_OUT preds

Loss schedule (mirrors base script):
    Epochs 0 … WARMUP-1  : MSE + λ_phys · L_phys
    Epochs WARMUP … end  : MSE + λ_phys · L_phys + λ_bce · BCE

CLI:
    python test_kan_pinn_redd_specific_splits.py --appliance fridge
    python test_kan_pinn_redd_specific_splits.py --plot
"""

import sys
import os
import time
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 128
WIN           = 100
STRIDE        = 5
WARMUP_EPOCHS = 15

# KAN architecture
K_OUT        = 10   # output points spanning the window; midpoint at K_OUT//2
N_POOL       = 8    # adaptive-pool segments for local feature extraction
GRID_SIZE    = 5    # B-spline grid intervals on [0,1]
SPLINE_ORDER = 3    # cubic B-splines → C² smooth, stable 3rd-order derivatives
KAN_HIDDEN   = 64

# Physics loss
LAMBDA_PHYS = 0.10   # overall physics weight relative to MSE
LAMBDA_V    = 1.0    # velocity    (Δp)  weight inside L_phys
LAMBDA_A    = 0.5    # acceleration (Δ²p) weight inside L_phys
LAMBDA_J    = 0.1    # jerk         (Δ³p) weight inside L_phys

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
APP_LABELS = ['Dish Washer', 'Fridge', 'Microwave', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer': 10.0,
}

BCE_LAMBDA = {
    'dish washer':  0.3,
    'fridge':       0.3,
    'microwave':    0.3,
    'washer dryer': 0.5,
}

BCE_ALPHA = {
    'dish washer':  1.5,
    'fridge':       1.5,
    'microwave':    4.0,
    'washer dryer': 5.0,
}

SAVE_DIR = os.path.join('results', 'kan_pinn_redd')
COLOR    = '#2CA02C'


# ── B-spline basis ─────────────────────────────────────────────────────────────

def b_splines(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate B-spline basis functions of order k via de Boor recursion.

    x:    (B, D)         — D input features, expected in [0, 1]
    grid: (D, G+2k+1)   — extended knot grid per feature
    Returns: (B, D, G+k) — G+k basis values per feature

    Recursion (Cox–de Boor):
        B_{i,0}(x) = 1  if  t_i ≤ x < t_{i+1},  else 0
        B_{i,j}(x) = (x-t_i)/(t_{i+j}-t_i) · B_{i,j-1}(x)
                   + (t_{i+j+1}-x)/(t_{i+j+1}-t_{i+1}) · B_{i+1,j-1}(x)

    After k recursive steps the (G+2k) order-0 bases collapse to G+k
    cubic-spline bases, each with support over k+1 grid intervals.
    """
    x = x.unsqueeze(-1)   # (B, D, 1)  — broadcasts with (D, G+2k+1)

    # Order-0: indicator functions over each grid interval
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # (B, D, G+2k)

    for j in range(1, k + 1):
        # Left-side numerator / denominator
        denom_l = (grid[:, j:-1] - grid[:, :-(j + 1)]).clamp(min=1e-8)
        left    = (x - grid[:, :-(j + 1)]) / denom_l * bases[..., :-1]

        # Right-side numerator / denominator
        denom_r = (grid[:, (j + 1):] - grid[:, 1:-j]).clamp(min=1e-8)
        right   = (grid[:, (j + 1):] - x) / denom_r * bases[..., 1:]

        bases = left + right   # (B, D, G+2k-j) — one fewer per step

    return bases.contiguous()  # (B, D, G+k)


# ── KAN Layer ──────────────────────────────────────────────────────────────────

class KANLayer(nn.Module):
    """
    Single Kolmogorov-Arnold Network layer.

    y_i = Σ_j  [ w^base_{i,j} · silu(x_j)  +  Σ_k c_{i,j,k} · B_k(x_j) ]
         residual (base) branch         spline branch

    Parameters
    ----------
    spline_weight : (out, in, G+k)  — B-spline coefficients
    base_weight   : (out, in)       — residual SiLU weights
    grid          : (in, G+2k+1)    — fixed extended knot grid  [buffer]
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = GRID_SIZE, spline_order: int = SPLINE_ORDER):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k            = spline_order
        self.n_basis      = grid_size + spline_order   # G + k

        # Extended knot grid on [0, 1]: k extra knots on each side
        h    = 1.0 / grid_size
        grid = torch.linspace(
            -spline_order * h,
            1.0 + spline_order * h,
            grid_size + 2 * spline_order + 1,
        ).expand(in_features, -1).contiguous()   # (in, G+2k+1)
        self.register_buffer('grid', grid)

        # Learnable B-spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.n_basis)
            / (in_features * self.n_basis) ** 0.5
        )
        # Learnable residual (SiLU base) weights
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) / in_features ** 0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features), expected ∈ [0, 1]

        # SiLU residual path
        base = F.linear(F.silu(x), self.base_weight)            # (B, out)

        # B-spline path
        splines     = b_splines(x, self.grid, self.k)           # (B, in, G+k)
        splines_f   = splines.view(x.shape[0], -1)              # (B, in*(G+k))
        weight_f    = self.spline_weight.view(self.out_features, -1)
        spline_out  = F.linear(splines_f, weight_f)             # (B, out)

        return base + spline_out


# ── KAN-PINN Model ─────────────────────────────────────────────────────────────

class KANPINNModel(nn.Module):
    """
    KAN-based Physics-Informed Neural Network for NILM.

    Feature extraction:
        AdaptiveAvgPool1d → N_POOL local means of the mains window
        Global stats: mean, std, min, max

    KAN encoder (3 layers with sigmoid activations):
        Sigmoid keeps hidden activations in (0,1) so they stay within
        the B-spline grid's main range [0,1] for the next layer.

    Output: K_OUT power predictions.
        - Index K_OUT//2 is the supervised target (window midpoint).
        - All K_OUT values are used in the physics loss.
    """

    def __init__(self, win: int = WIN, k_out: int = K_OUT,
                 hidden: int = KAN_HIDDEN, n_pool: int = N_POOL,
                 grid_size: int = GRID_SIZE, spline_order: int = SPLINE_ORDER):
        super().__init__()
        self.k_out = k_out
        self.mid   = k_out // 2   # supervised midpoint index

        self.pool = nn.AdaptiveAvgPool1d(n_pool)
        in_dim    = n_pool + 4    # pooled segments + (mean, std, min, max)

        self.kan1 = KANLayer(in_dim, hidden, grid_size, spline_order)
        self.kan2 = KANLayer(hidden, hidden, grid_size, spline_order)
        self.kan3 = KANLayer(hidden, k_out,  grid_size, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, win, 1), already MinMax-scaled to [0,1]
        x_flat = x.squeeze(-1)                                    # (B, win)

        mu  = x_flat.mean(dim=1, keepdim=True)
        sig = x_flat.std(dim=1, keepdim=True).clamp(min=1e-6)
        mn  = x_flat.min(dim=1).values.unsqueeze(1)
        mx  = x_flat.max(dim=1).values.unsqueeze(1)

        pooled = self.pool(x_flat.unsqueeze(1)).squeeze(1)        # (B, n_pool)
        feat   = torch.cat([pooled, mu, sig, mn, mx], dim=1)      # (B, in_dim)
        feat   = feat.clamp(0.0, 1.0)                             # keep ∈ [0,1]

        h = torch.sigmoid(self.kan1(feat))                        # (B, hidden)
        h = torch.sigmoid(self.kan2(h))                           # (B, hidden)
        p = self.kan3(h)                                          # (B, k_out)
        return p

    def predict_midpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the supervised midpoint prediction: (B, 1)."""
        return self.forward(x)[:, self.mid:self.mid + 1]


# ── Physics Loss ───────────────────────────────────────────────────────────────

def compute_physics_loss(p: torch.Tensor) -> torch.Tensor:
    """
    Penalise velocity, acceleration, and jerk over the K_OUT predictions.

    p: (B, K_OUT) — power predictions in scaled space [0,1]

    L_phys = λ_v · mean(Δp²) + λ_a · mean(Δ²p²) + λ_j · mean(Δ³p²)

    Why C² splines matter: computing Δ²p and Δ³p during backprop requires
    differentiating through the model twice/three times.  ReLU activations
    produce zero second derivatives almost everywhere, making these terms
    uninformative.  Cubic B-splines (C² smooth) preserve gradient signal
    through all three derivative levels.
    """
    vel  = torch.diff(p, n=1, dim=-1)   # (B, K-1)  — first differences
    acc  = torch.diff(p, n=2, dim=-1)   # (B, K-2)  — second differences
    jerk = torch.diff(p, n=3, dim=-1)   # (B, K-3)  — third differences

    return (LAMBDA_V * vel.pow(2).mean()  +
            LAMBDA_A * acc.pow(2).mean()  +
            LAMBDA_J * jerk.pow(2).mean())


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading REDD data (specific splits)...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(f'data/redd/{split}_small.pkl', 'rb') as f:
            splits[split] = pickle.load(f)[0]
    print(f"  Train: {splits['train'].index.min()} → {splits['train'].index.max()}")
    print(f"  Val  : {splits['val'].index.min()} → {splits['val'].index.max()}")
    print(f"  Test : {splits['test'].index.min()} → {splits['test'].index.max()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(data, appliance_name, window_size=WIN):
    mains = data['main'].values
    app   = data[appliance_name].values
    X, y  = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        y.append(app[i + window_size // 2])   # midpoint target
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


class NILMDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Fast per-epoch metrics ─────────────────────────────────────────────────────

def _fast_metrics(y_true, y_pred, threshold):
    y_true = y_true.flatten();  y_pred = y_pred.flatten()
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    t_bin  = y_true > threshold;  p_bin = y_pred > threshold
    tp = int(np.sum(t_bin & p_bin));  fp = int(np.sum(~t_bin & p_bin))
    fn = int(np.sum(t_bin & ~p_bin))
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return {'mae': mae, 'f1': f1, 'precision': pr, 'recall': rc}


# ── Training ───────────────────────────────────────────────────────────────────

def train_appliance(appliance_name, splits, device, epochs, hidden):
    thr       = THRESHOLDS[appliance_name]
    bce_lam   = BCE_LAMBDA[appliance_name]
    bce_alpha = BCE_ALPHA[appliance_name]

    print(f"\n{'='*60}")
    print(f"  {appliance_name}  |  hidden={hidden}  K_OUT={K_OUT}")
    print(f"  λ_phys={LAMBDA_PHYS} (v={LAMBDA_V} a={LAMBDA_A} j={LAMBDA_J})")
    print(f"  λ_bce={bce_lam}  α_bce={bce_alpha}  warmup={WARMUP_EPOCHS}")
    print(f"{'='*60}")

    X_tr, y_tr = create_sequences(splits['train'], appliance_name)
    X_va, y_va = create_sequences(splits['val'],   appliance_name)
    X_te, y_te = create_sequences(splits['test'],  appliance_name)

    x_scaler = MinMaxScaler();  y_scaler = MinMaxScaler()

    X_tr_n = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va_n = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te_n = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_tr_n = y_scaler.fit_transform(y_tr)
    y_va_n = y_scaler.transform(y_va)
    y_te_n = y_scaler.transform(y_te)

    thr_scaled = float((thr - y_scaler.data_min_[0]) / y_scaler.data_range_[0])
    print(f"  Train: {X_tr_n.shape}  Val: {X_va_n.shape}  Test: {X_te_n.shape}")

    tr_loader = torch.utils.data.DataLoader(
        NILMDataset(X_tr_n, y_tr_n), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        NILMDataset(X_va_n, y_va_n), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        NILMDataset(X_te_n, y_te_n), batch_size=BATCH, shuffle=False, drop_last=False)

    model = KANPINNModel(
        win=WIN, k_out=K_OUT, hidden=hidden, n_pool=N_POOL,
        grid_size=GRID_SIZE, spline_order=SPLINE_ORDER,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  B-spline basis per edge: {GRID_SIZE + SPLINE_ORDER}  "
          f"(grid={GRID_SIZE}, order={SPLINE_ORDER})")

    optimizer        = torch.optim.Adam(model.parameters(), lr=LR)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-5)

    def _loss(p_all, yb, epoch):
        p_mid = p_all[:, model.mid:model.mid + 1]          # (B, 1)

        mse   = F.mse_loss(p_mid, yb)
        phys  = LAMBDA_PHYS * compute_physics_loss(p_all)

        if epoch < WARMUP_EPOCHS:
            return mse + phys, mse.item(), phys.item()

        prob  = torch.sigmoid(p_mid / (thr_scaled + 1e-8))
        y_bin = (yb > thr_scaled).float()
        w     = torch.where(y_bin == 1,
                            torch.full_like(y_bin, bce_alpha),
                            torch.ones_like(y_bin))
        bce   = F.binary_cross_entropy(prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
        return mse + phys + bce_lam * bce, mse.item(), phys.item()

    train_losses, val_losses = [], []
    val_mae_h, val_f1_h, val_p_h, val_r_h = [], [], [], []
    val_time_h, phys_loss_h = [], []

    best_val    = float('inf')
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for epoch in range(epochs):
        model.train()
        ep_loss = ep_mse = ep_phys = 0.0
        bar = tqdm(tr_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            p_all         = model(xb)
            loss, mse, ph = _loss(p_all, yb, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss  += loss.item()
            ep_mse   += mse
            ep_phys  += ph
            bar.set_postfix({'loss': f'{loss.item():.5f}', 'phys': f'{ph:.5f}'})

        avg_tr   = ep_loss / len(tr_loader)
        avg_phys = ep_phys / len(tr_loader)
        train_losses.append(avg_tr)
        phys_loss_h.append(avg_phys)

        # ── Validation ──────────────────────────────────────────────────────
        vt0 = time.time()
        model.eval()
        vl_loss = 0.0
        preds_v, trues_v = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb  = xb.to(device), yb.to(device)
                p_all   = model(xb)
                vl, _, _ = _loss(p_all, yb, epoch)
                vl_loss += vl.item()
                preds_v.append(p_all[:, model.mid:model.mid + 1].cpu().numpy())
                trues_v.append(yb.cpu().numpy())

        avg_va = vl_loss / len(va_loader)
        val_losses.append(avg_va)

        if epoch >= WARMUP_EPOCHS:
            cosine_scheduler.step()

        raw_pred = y_scaler.inverse_transform(
            np.concatenate(preds_v).reshape(-1, 1)).flatten()
        raw_true = y_scaler.inverse_transform(
            np.concatenate(trues_v).reshape(-1, 1)).flatten()
        vm = _fast_metrics(raw_true, raw_pred, threshold=thr)
        val_mae_h.append(vm['mae']);  val_f1_h.append(vm['f1'])
        val_p_h.append(vm['precision']);  val_r_h.append(vm['recall'])
        val_time_h.append(time.time() - vt0)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr:.5f}  phys={avg_phys:.5f}  val={avg_va:.5f}  "
              f"F1={vm['f1']:.4f}  P={vm['precision']:.4f}  R={vm['recall']:.4f}  "
              f"MAE={vm['mae']:.1f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # Switch early-stopping criterion at warmup boundary
        if epoch == WARMUP_EPOCHS:
            best_val = float('inf');  best_val_f1 = -1.0;  no_improve = 0

        improved = (avg_va < best_val) if epoch < WARMUP_EPOCHS \
                   else (vm['f1'] > best_val_f1)

        if improved:
            best_val    = avg_va
            best_val_f1 = vm['f1']
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    infer_t0 = time.time()
    preds_t, trues_t = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds_t.append(
                model(xb.to(device))[:, model.mid:model.mid + 1].cpu().numpy())
            trues_t.append(yb.numpy())
    infer_s = time.time() - infer_t0

    raw_pred_te = y_scaler.inverse_transform(
        np.concatenate(preds_t).reshape(-1, 1)).flatten()
    raw_true_te = y_scaler.inverse_transform(
        np.concatenate(trues_t).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_true_te, raw_pred_te, threshold=thr)

    print(f"\n  Test  F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}")

    return {
        'metrics':      test_metrics,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'phys_loss_h':  phys_loss_h,
        'val_mae_h':    val_mae_h,
        'val_f1_h':     val_f1_h,
        'val_p_h':      val_p_h,
        'val_r_h':      val_r_h,
        'val_time_h':   val_time_h,
        'infer_s':      infer_s,
        'epochs_run':   len(train_losses),
        'num_params':   n_params,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, app in zip(axes.flatten(), APPLIANCES):
        if app not in results:
            continue
        r  = results[app];  ep = range(1, r['epochs_run'] + 1)
        ax.plot(ep, r['train_losses'], label='Train total', color='steelblue', lw=1.5)
        ax.plot(ep, r['val_losses'],   label='Val total',   color='tomato',    lw=1.5, ls='--')
        if 'phys_loss_h' in r:
            ax2 = ax.twinx()
            ax2.plot(ep, r['phys_loss_h'], color='grey', lw=1.0, ls=':', alpha=0.7,
                     label='Phys loss')
            ax2.set_ylabel('L_phys', fontsize=7)
        ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1, label='BCE start')
        ax.set_title(app.title());  ax.set_xlabel('Epoch');  ax.set_ylabel('Loss')
        ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    fig.suptitle('Training Curves — KAN-PINN (REDD)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics(results, save_dir):
    fig, axes = plt.subplots(len(APPLIANCES), 3,
                              figsize=(15, 4 * len(APPLIANCES)))
    for row, app in enumerate(APPLIANCES):
        if app not in results:
            continue
        r  = results[app];  ep = range(1, r['epochs_run'] + 1)
        for col, (key, label) in enumerate(
                [('val_mae_h', 'MAE (W)'), ('val_f1_h', 'F1'),
                 ('val_p_h', 'Precision / Recall')]):
            ax = axes[row, col]
            ax.plot(ep, r[key], color=COLOR, lw=1.5, label=label)
            if label == 'Precision / Recall':
                ax.plot(ep, r['val_r_h'], color='orange', lw=1.5, ls='--',
                        label='Recall')
                ax.legend(fontsize=7)
            ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1)
            ax.set_title(f'{app.title()} — {label}')
            ax.set_xlabel('Epoch');  ax.set_ylabel(label);  ax.grid(True, alpha=0.3)
    fig.suptitle('Val Metrics — KAN-PINN (REDD)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'epoch_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved: {path}")


def plot_bar_chart(results, save_dir):
    metrics_cfg = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1'),
                   ('precision', 'Precision'), ('recall', 'Recall')]
    x = np.arange(len(APPLIANCES))
    fig, axes = plt.subplots(1, len(metrics_cfg), figsize=(18, 5))
    for ax, (mk, ml) in zip(axes, metrics_cfg):
        vals = [results[app]['metrics'].get(mk, np.nan)
                if app in results else np.nan for app in APPLIANCES]
        bars = ax.bar(x, vals, color=COLOR, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x);  ax.set_xticklabels(APP_LABELS, rotation=12, ha='right')
        ax.set_ylabel(ml);  ax.set_title(ml)
        ax.grid(axis='y', alpha=0.3);  ax.set_axisbelow(True)
    fig.suptitle('Final Test Metrics — KAN-PINN (REDD)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved: {path}")


def print_table(results):
    for metric, label in [('f1', 'F1'), ('precision', 'Precision'),
                           ('recall', 'Recall'), ('mae', 'MAE'), ('sae', 'SAE')]:
        print(f"\n{'='*70}")
        print(f"  {label} — KAN-PINN (REDD)")
        print(f"{'='*70}")
        print(f"  {'Appliance':<20}{'Value':>12}  (epochs)")
        print('─' * 70)
        vals = []
        for app in APPLIANCES:
            if app in results:
                v  = results[app]['metrics'].get(metric, float('nan'))
                ep = results[app]['epochs_run']
                print(f"  {app.title():<20}{v:>12.4f}  ({ep})")
                vals.append(v)
        print('─' * 70)
        if vals:
            print(f"  {'Average':<20}{np.nanmean(vals):>12.4f}")


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _save_json(app, r, hidden, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{app.replace(" ", "_")}.json')
    with open(path, 'w') as f:
        json.dump({
            'appliance':     app,
            'dataset':       'REDD',
            'architecture':  'KANPINNModel',
            'hidden_size':   hidden,
            'k_out':         K_OUT,
            'grid_size':     GRID_SIZE,
            'spline_order':  SPLINE_ORDER,
            'lambda_phys':   LAMBDA_PHYS,
            'lambda_v':      LAMBDA_V,
            'lambda_a':      LAMBDA_A,
            'lambda_j':      LAMBDA_J,
            'epochs':        epochs,
            'epochs_run':    r['epochs_run'],
            'num_params':    r['num_params'],
            'bce_lambda':    BCE_LAMBDA[app],
            'bce_alpha':     BCE_ALPHA[app],
            'warmup_epochs': WARMUP_EPOCHS,
            'metrics':       {k: float(v) for k, v in r['metrics'].items()},
            'train_losses':  r['train_losses'],
            'val_losses':    r['val_losses'],
            'phys_loss_h':   r['phys_loss_h'],
            'val_mae_h':     r['val_mae_h'],
            'val_f1_h':      r['val_f1_h'],
            'val_p_h':       r['val_p_h'],
            'val_r_h':       r['val_r_h'],
            'val_time_h':    r['val_time_h'],
            'infer_s':       r['infer_s'],
        }, f, indent=2)
    print(f'  JSON saved → {path}')


def _load_all_jsons(save_dir):
    results = {}
    for app in APPLIANCES:
        path = os.path.join(save_dir, f'{app.replace(" ", "_")}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[app] = json.load(f)
            print(f'  Loaded {app} ← {path}')
        else:
            print(f'  Missing: {path}  (run --appliance "{app}" first)')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='KAN-PINN — REDD')
    parser.add_argument('--epochs',    type=int,  default=EPOCHS)
    parser.add_argument('--hidden',    type=int,  default=KAN_HIDDEN,
                        help='KAN hidden size (default 64)')
    parser.add_argument('--appliance', type=str,  default=None,
                        choices=APPLIANCES)
    parser.add_argument('--plot',      action='store_true',
                        help='Skip training — load saved JSONs and plot.')
    parser.add_argument('--save_dir',  type=str,  default=SAVE_DIR)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.plot:
        print('Plot mode — loading saved results...')
        results = _load_all_jsons(save_dir)
        if not results:
            print('No results found.  Run at least one appliance first.')
            sys.exit(1)
        plot_training_curves(results, save_dir)
        plot_epoch_metrics(results, save_dir)
        plot_bar_chart(results, save_dir)
        print_table(results)
        return

    for fp in ['data/redd/train_small.pkl',
               'data/redd/val_small.pkl',
               'data/redd/test_small.pkl']:
        if not os.path.exists(fp):
            print(f'ERROR: {fp} not found');  sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Hidden: {args.hidden}  |  K_OUT: {K_OUT}')
    print(f'Architecture: KAN (B-spline order {SPLINE_ORDER}) + Physics loss '
          f'(vel/acc/jerk, λ={LAMBDA_PHYS})')

    splits      = load_data()
    apps_to_run = [args.appliance] if args.appliance else APPLIANCES

    total_t0   = time.time()
    all_results: dict = {}

    for app in apps_to_run:
        t0 = time.time()
        r  = train_appliance(app, splits, device, args.epochs, args.hidden)
        r['time_s'] = time.time() - t0
        m = r['metrics']
        print(f"\n  DONE {app:<15} | F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
              f"({r['num_params']:,} params  {r['time_s']:.0f}s)")
        _save_json(app, r, args.hidden, args.epochs, save_dir)
        all_results[app] = r

    total_s = time.time() - total_t0
    print(f'\nTotal time: {total_s:.0f}s ({total_s/60:.1f} min)')

    all_done = all(
        os.path.exists(os.path.join(save_dir, f'{a.replace(" ", "_")}.json'))
        for a in APPLIANCES
    )
    if all_done:
        full = _load_all_jsons(save_dir)
        print('\nAll appliances complete — generating plots...')
        plot_training_curves(full, save_dir)
        plot_epoch_metrics(full, save_dir)
        plot_bar_chart(full, save_dir)
        print_table(full)
    else:
        missing = [a for a in APPLIANCES
                   if not os.path.exists(
                       os.path.join(save_dir, f'{a.replace(" ", "_")}.json'))]
        print(f'\nStill missing: {missing}')
        print('Run those, then: python test_kan_pinn_redd_specific_splits.py --plot')

    print(f'\nAll outputs → {save_dir}/')


if __name__ == '__main__':
    main()
