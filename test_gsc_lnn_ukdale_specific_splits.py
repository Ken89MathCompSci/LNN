"""
GSC-LNN: Gated + Spectral-Stable + Continuous (Stateful) LNN — UKDALE specific splits
========================================================================================
Extends GS-LNN with stateful hidden continuity — the third orthogonal improvement.

─────────────────────────────────────────────────────────────────────────────
Improvement 1 — Residual gated transport  (already in GS-LNN)
    g_t     = σ(W_g h_t + b_g)
    h_{t+1} = h_t + g_t ⊙ dt · dh/dt

Improvement 2 — Spectral stability  (already in GS-LNN)
    After every optimizer step: ‖U‖₂ clipped to ρ_max = 0.99

─────────────────────────────────────────────────────────────────────────────
Improvement 3 — Stateful hidden continuity  (NEW)

    Base / GS-LNN behaviour:
        Window k is processed with h_0 = 0 (zero reset).
        Every window starts blind — no memory of what came before.

    Physical problem:
        Appliances have persistent states that span many windows:
            fridge compressor cycling  (~10–30 min cycles)
            washer dryer spin phases   (multi-hour programs)
            dishwasher fill/wash/rinse (60+ min programs)
        Resetting hidden state every 100 samples destroys this context.

    Fix: carry final hidden state to next window's initial state
        h_{k+1}(0) = h_k(T)

        out, h_last = model(x_batch, h_prev)
        h_prev      = h_last.detach()          ← detach = stop BPTT there

    Why .detach() matters:
        Backprop through the entire time series would require O(N) memory.
        Detach creates a "truncated BPTT" — gradients flow through the
        current window only, but the hidden STATE is still correctly
        seeded from the previous window's output.

    Batching scheme:
        DataLoader uses shuffle=False during training so that
        batch position j at batch b is the temporal predecessor of
        position j at batch b+1 (windows are indexed sequentially).
        Each of the B=128 parallel slots runs its own temporal trajectory.
        The hidden state tensor h_prev (B, H) carries B trajectories
        simultaneously — no sequential bottleneck.

    Last-batch size change:
        If the final batch has fewer than B samples, h_prev is trimmed
        to match: h_init = h_prev[:B_actual].
        State is reset (h_prev = None) at the start of each epoch.

    Val / Test:
        Separate stateful passes for validation and test.
        Each starts from h=0 and carries state through in temporal order.
        This mirrors inference on a live stream.

─────────────────────────────────────────────────────────────────────────────
Why the three improvements are complementary
    Gate    → limits per-step transport magnitude  (∥Δh∥ ≤ g·dt·∥dh∥)
    Spectral → limits recurrent Jacobian norm       (∥∂h'/∂h∥ ≲ ρ)
    Stateful → provides temporal context across windows
    Together: stable, memory-preserving, physically continuous dynamics.

Loss schedule (identical to base):
    Epochs 0 … WARMUP_EPOCHS-1 : MSE only
    Epochs WARMUP_EPOCHS …     : MSE + λ_bce × weighted-BCE

LR schedule:
    Epochs 0 … WARMUP_EPOCHS-1 : constant LR
    Epochs WARMUP_EPOCHS …     : CosineAnnealingLR (η_min=1e-5)

Gradient clipping: max_norm=1.0 every step.

Model selection:
    Warmup phase — best val_loss
    BCE phase    — best val F1

Diagnostics saved to JSON:
    spec_h — σ_max(U) per epoch (post-clip)

Usage:
    python test_gsc_lnn_ukdale_specific_splits.py --appliance fridge
    python test_gsc_lnn_ukdale_specific_splits.py --appliance microwave
    python test_gsc_lnn_ukdale_specific_splits.py --appliance "dish washer"
    python test_gsc_lnn_ukdale_specific_splits.py --appliance "washer dryer"
    python test_gsc_lnn_ukdale_specific_splits.py --plot
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
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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

SPEC_RADIUS   = 0.99

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
APP_LABELS = ['Dish Washer', 'Fridge', 'Microwave', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {
    'dish washer':  0.3,
    'fridge':       0.5,
    'microwave':    0.3,
    'washer dryer': 0.5,
}

BCE_ALPHA = {
    'dish washer':  10.0,
    'fridge':        0.5,
    'microwave':     4.0,
    'washer dryer':  3.0,
}

SAVE_DIR = os.path.join('results', 'gsc_lnn_ukdale')
COLOR    = '#1F77B4'   # blue — distinct from G(orange) S(purple) GS(red)


# ── GSC-LNN Layer ─────────────────────────────────────────────────────────────

class GatedSpectralLiquidTimeLayer(nn.Module):
    """
    Liquid Time-Constant layer with residual gating + spectral-norm constraint.

    ODE:  dh/dt = -h/τ + tanh(W_in x + U h)
    Gate: g     = σ(W_g h + b_g)           ∈ (0,1)^H
    Step: h'    = clamp(h + g ⊙ dt · dh,  -10, 10)

    U is nn.Linear(bias=False) so its .weight is accessible for spectral clipping.
    Gate bias is zero-initialised so σ(0) = 0.5 matches the base LNN at epoch 0.
    """

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt          = dt

        self.W_in = nn.Linear(input_size, hidden_size)
        self.tau  = nn.Parameter(torch.ones(hidden_size))

        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.U.weight)

        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        f_t  = torch.tanh(self.W_in(x) + self.U(hidden))
        dh   = -hidden / self.tau.unsqueeze(0) + f_t
        gate = torch.sigmoid(self.gate_proj(hidden))

        return (hidden + gate * self.dt * dh).clamp(-10.0, 10.0)


# ── GSC-LNN Model ─────────────────────────────────────────────────────────────

class GSCLiquidNetworkModel(nn.Module):
    """
    Stateful GS-LNN: accepts an initial hidden state and returns the final
    hidden state so it can be carried to the next batch/window.

    Signature change vs GS-LNN:
        forward(x, h_init=None) → (output, h_final)
            h_init : (B, hidden_size) or None  → zero-init if None
            h_final: (B, hidden_size)          → pass as h_init for next call
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dt: float = 0.1):
        super().__init__()
        self.liquid_layer = GatedSpectralLiquidTimeLayer(input_size, hidden_size, dt)
        self.fc           = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor,
                h_init: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x      : (B, WIN, input_size)
        h_init : (B, hidden_size) or None
        returns: (output (B,1),  h_final (B, hidden_size))
        """
        hidden = h_init
        for t in range(x.size(1)):
            hidden = self.liquid_layer(x[:, t, :], hidden)
        return self.fc(hidden), hidden


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading UKDALE data (specific splits)...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
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
        y.append(app[i + window_size // 2])
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
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    t_bin  = y_true > threshold
    p_bin  = y_pred > threshold
    tp = int(np.sum(t_bin & p_bin))
    fp = int(np.sum(~t_bin & p_bin))
    fn = int(np.sum(t_bin & ~p_bin))
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return {'mae': mae, 'f1': f1, 'precision': pr, 'recall': rc}


def _spectral_norm(model: GSCLiquidNetworkModel) -> float:
    with torch.no_grad():
        return float(torch.linalg.matrix_norm(
            model.liquid_layer.U.weight, ord=2))


# ── Training ───────────────────────────────────────────────────────────────────

def train_appliance(appliance_name, splits, device, epochs, hidden_size):
    thr       = THRESHOLDS[appliance_name]
    bce_lam   = BCE_LAMBDA[appliance_name]
    bce_alpha = BCE_ALPHA[appliance_name]

    print(f"\n{'='*60}")
    print(f"  {appliance_name}  |  hidden={hidden_size}  ρ_max={SPEC_RADIUS}")
    print(f"  λ_bce={bce_lam}  α_bce={bce_alpha}  warmup={WARMUP_EPOCHS}")
    print(f"  Stateful: h carried between batches (shuffle=False)")
    print(f"{'='*60}")

    X_tr, y_tr = create_sequences(splits['train'], appliance_name)
    X_va, y_va = create_sequences(splits['val'],   appliance_name)
    X_te, y_te = create_sequences(splits['test'],  appliance_name)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_tr_n = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va_n = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te_n = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_tr_n = y_scaler.fit_transform(y_tr)
    y_va_n = y_scaler.transform(y_va)
    y_te_n = y_scaler.transform(y_te)

    thr_scaled = float((thr - y_scaler.data_min_[0]) / y_scaler.data_range_[0])

    print(f"  Train: {X_tr_n.shape}  Val: {X_va_n.shape}  Test: {X_te_n.shape}")

    # shuffle=False is REQUIRED for stateful continuity — windows must stay
    # in temporal order so batch-position j at batch b precedes position j
    # at batch b+1 along the time axis.
    tr_loader = torch.utils.data.DataLoader(
        NILMDataset(X_tr_n, y_tr_n), batch_size=BATCH, shuffle=False, drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        NILMDataset(X_va_n, y_va_n), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        NILMDataset(X_te_n, y_te_n), batch_size=BATCH, shuffle=False, drop_last=False)

    model = GSCLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size, output_size=1, dt=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Initial σ_max(U) = {_spectral_norm(model):.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-5)

    def _loss(out, yb, epoch):
        mse = F.mse_loss(out, yb)
        if epoch < WARMUP_EPOCHS:
            return mse
        prob  = torch.sigmoid(out / (thr_scaled + 1e-8))
        y_bin = (yb > thr_scaled).float()
        w     = torch.where(y_bin == 1,
                            torch.full_like(y_bin, bce_alpha),
                            torch.ones_like(y_bin))
        bce   = F.binary_cross_entropy(prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
        return mse + bce_lam * bce

    train_losses, val_losses = [], []
    val_mae_h, val_f1_h, val_p_h, val_r_h = [], [], [], []
    val_time_h = []
    spec_h     = []

    best_val    = float('inf')
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0

        # Reset hidden state at the start of each epoch
        h_prev = None

        bar = tqdm(tr_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            B_actual = xb.shape[0]

            # Trim h_prev when the last batch is smaller than BATCH
            h_init = h_prev[:B_actual] if h_prev is not None else None

            optimizer.zero_grad()
            out, h_last = model(xb, h_init)

            # Carry hidden state to next batch; detach to stop BPTT at
            # the window boundary (truncated BPTT — gradient flows within
            # the window, state value flows across windows)
            h_prev = h_last.detach()

            loss = _loss(out, yb, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── Spectral clipping ──────────────────────────────────────────
            with torch.no_grad():
                W    = model.liquid_layer.U.weight
                spec = torch.linalg.matrix_norm(W, ord=2)
                if spec > SPEC_RADIUS:
                    W.mul_(SPEC_RADIUS / spec)

            ep_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_tr = ep_loss / len(tr_loader)
        train_losses.append(avg_tr)
        spec_h.append(_spectral_norm(model))

        # ── Validation (stateful pass, reset per epoch) ──
        vt0 = time.time()
        model.eval()
        vl_loss  = 0.0
        preds_v, trues_v = [], []
        h_prev_val = None

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb   = xb.to(device), yb.to(device)
                B_actual = xb.shape[0]
                h_init_val = h_prev_val[:B_actual] if h_prev_val is not None else None

                out, h_last_val = model(xb, h_init_val)
                h_prev_val = h_last_val   # no detach needed inside no_grad

                vl_loss += _loss(out, yb, epoch).item()
                preds_v.append(out.cpu().numpy())
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
        val_mae_h.append(vm['mae'])
        val_f1_h.append(vm['f1'])
        val_p_h.append(vm['precision'])
        val_r_h.append(vm['recall'])
        val_time_h.append(time.time() - vt0)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr:.5f}  val={avg_va:.5f}  "
              f"F1={vm['f1']:.4f}  P={vm['precision']:.4f}  R={vm['recall']:.4f}  "
              f"MAE={vm['mae']:.1f}  σ(U)={spec_h[-1]:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Reset at warmup→BCE boundary; switch criterion to val F1
        if epoch == WARMUP_EPOCHS:
            best_val    = float('inf')
            best_val_f1 = -1.0
            no_improve  = 0

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

    # ── Test (stateful pass, fresh start) ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    infer_t0 = time.time()
    preds_t, trues_t = [], []
    h_prev_te = None

    with torch.no_grad():
        for xb, yb in te_loader:
            xb       = xb.to(device)
            B_actual = xb.shape[0]
            h_init_te = h_prev_te[:B_actual] if h_prev_te is not None else None

            out, h_last_te = model(xb, h_init_te)
            h_prev_te = h_last_te

            preds_t.append(out.cpu().numpy())
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
        'val_mae_h':    val_mae_h,
        'val_f1_h':     val_f1_h,
        'val_p_h':      val_p_h,
        'val_r_h':      val_r_h,
        'val_time_h':   val_time_h,
        'spec_h':       spec_h,
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
        r  = results[app]
        ep = range(1, r['epochs_run'] + 1)
        ax.plot(ep, r['train_losses'], label='Train', color='steelblue', lw=1.5)
        ax.plot(ep, r['val_losses'],   label='Val',   color='tomato',    lw=1.5, ls='--')
        ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1, label='BCE start')
        ax.set_title(app.title())
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Training Curves — GSC-LNN (UKDALE)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics(results, save_dir):
    fig, axes = plt.subplots(len(APPLIANCES), 4,
                              figsize=(18, 4 * len(APPLIANCES)))
    for row, app in enumerate(APPLIANCES):
        if app not in results:
            continue
        r  = results[app]
        ep = range(1, r['epochs_run'] + 1)
        for col, (key, label) in enumerate(
                [('val_mae_h', 'MAE (W)'), ('val_f1_h', 'F1'),
                 ('val_p_h', 'Precision / Recall'), ('spec_h', 'σ_max(U)')]):
            ax = axes[row, col]
            ax.plot(ep, r[key], color=COLOR, lw=1.5, label=label)
            if label == 'Precision / Recall':
                ax.plot(ep, r['val_r_h'], color='steelblue', lw=1.5,
                        ls='--', label='Recall')
                ax.legend(fontsize=7)
            if label == 'σ_max(U)':
                ax.axhline(SPEC_RADIUS, color='grey', ls='--', lw=1,
                           label=f'ceiling {SPEC_RADIUS}')
                ax.legend(fontsize=7)
            ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1)
            ax.set_title(f'{app.title()} — {label}')
            ax.set_xlabel('Epoch'); ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Val Metrics per Epoch — GSC-LNN (UKDALE)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'epoch_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_bar_chart(results, save_dir):
    metrics_cfg = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1'),
                   ('precision', 'Precision'), ('recall', 'Recall')]
    x   = np.arange(len(APPLIANCES))
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
        ax.set_xticks(x)
        ax.set_xticklabels(APP_LABELS, rotation=12, ha='right')
        ax.set_ylabel(ml); ax.set_title(ml)
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle('Final Test Metrics — GSC-LNN (UKDALE)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_table(results):
    divider = '─' * 70
    for metric, label in [('f1', 'F1'), ('precision', 'Precision'),
                           ('recall', 'Recall'), ('mae', 'MAE'), ('sae', 'SAE')]:
        print(f"\n{'='*70}")
        print(f"  {label} — GSC-LNN (UKDALE)")
        print(f"{'='*70}")
        print(f"  {'Appliance':<20}{'Value':>12}  (epochs)")
        print(divider)
        vals = []
        for app in APPLIANCES:
            if app in results:
                v  = results[app]['metrics'].get(metric, float('nan'))
                ep = results[app]['epochs_run']
                print(f"  {app.title():<20}{v:>12.4f}  ({ep})")
                vals.append(v)
        print(divider)
        if vals:
            print(f"  {'Average':<20}{np.nanmean(vals):>12.4f}")


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _save_json(app, r, hidden, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{app.replace(" ", "_")}.json')
    with open(path, 'w') as f:
        json.dump({
            'appliance':     app,
            'dataset':       'UKDALE',
            'architecture':  'GSCLiquidNetworkModel',
            'hidden_size':   hidden,
            'spec_radius':   SPEC_RADIUS,
            'stateful':      True,
            'epochs':        epochs,
            'epochs_run':    r['epochs_run'],
            'num_params':    r['num_params'],
            'bce_lambda':    BCE_LAMBDA[app],
            'bce_alpha':     BCE_ALPHA[app],
            'warmup_epochs': WARMUP_EPOCHS,
            'metrics':       {k: float(v) for k, v in r['metrics'].items()},
            'train_losses':  r['train_losses'],
            'val_losses':    r['val_losses'],
            'val_mae_h':     r['val_mae_h'],
            'val_f1_h':      r['val_f1_h'],
            'val_p_h':       r['val_p_h'],
            'val_r_h':       r['val_r_h'],
            'val_time_h':    r['val_time_h'],
            'spec_h':        r['spec_h'],
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
    parser = argparse.ArgumentParser(description='GSC-LNN — UKDALE')
    parser.add_argument('--epochs',    type=int, default=EPOCHS)
    parser.add_argument('--hidden',    type=int, default=256,
                        help='Hidden size (default 256)')
    parser.add_argument('--appliance', type=str, default=None,
                        choices=APPLIANCES,
                        help='Train a single appliance and save its JSON.')
    parser.add_argument('--plot',      action='store_true',
                        help='Skip training — load saved JSONs and plot.')
    parser.add_argument('--save_dir',  type=str, default=SAVE_DIR)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.plot:
        print('Plot mode — loading saved results...')
        results = _load_all_jsons(save_dir)
        if not results:
            print('No results found. Run at least one appliance first.')
            sys.exit(1)
        plot_training_curves(results, save_dir)
        plot_epoch_metrics(results, save_dir)
        plot_bar_chart(results, save_dir)
        print_table(results)
        return

    for fp in ['data/ukdale/train_small.pkl',
               'data/ukdale/val_small.pkl',
               'data/ukdale/test_small.pkl']:
        if not os.path.exists(fp):
            print(f'ERROR: {fp} not found')
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Hidden: {args.hidden}  |  Epochs: {args.epochs}')
    print('Architecture: GSCLiquidNetworkModel (Gated + Spectral + Stateful) → FC')
    print(f'  Gate:     g = σ(W_g h + b_g),  h\' = h + g ⊙ dt·dh/dt')
    print(f'  Spectral: σ_max(U) clipped to {SPEC_RADIUS} after every optimizer step')
    print(f'  Stateful: h_prev carried between batches (shuffle=False)')

    splits      = load_data()
    apps_to_run = [args.appliance] if args.appliance else APPLIANCES

    total_t0 = time.time()
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
        print('Run those, then: python test_gsc_lnn_ukdale_specific_splits.py --plot')

    print(f'\nAll outputs → {save_dir}/')


if __name__ == '__main__':
    main()
