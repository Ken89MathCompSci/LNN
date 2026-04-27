"""
Physics-Informed LNN with Residual Background Head — UKDALE specific splits.

Identical to test_pinn_lnn_ukdale_specific_splits.py except the physics
constraint is replaced by direct supervision of the background/residual load.

Original physics loss (soft, one-sided):
    L_phys = mean( ReLU( Σ p̂_k_raw − P_mains_raw − ε ) )
    — only punishes over-assignment; does not pull predictions toward correct sum.

This file (residual head, supervised):
    y_background = P_mains_midpoint − Σ_k y_k_true   (computed in raw Watts)
    L_bg = MSE( p̂_background, y_background )          (scales like any other head)

    Total loss:
        L = L_MSE_4apps + λ_bg · L_bg + L_BCE

Why this is stronger:
    • The model must now account for ALL power at every training step, not just
      avoid the rare over-assignment case.
    • y_background is real signal — it contains lighting, boiler, TV, computers.
      The 5th head learns the shape of unlabelled loads, which helps the other
      four heads not absorb that signal by mistake.
    • The physics constraint becomes an equality (in expectation) rather than a
      one-sided inequality.

Architecture:
    Input (batch, WIN, 1)
         ↓
    Shared AdvancedLiquidTimeLayer encoder
         ↓
    LayerNorm(hidden)
         ↓
    ┌────┬────┬────┬────┬────────────┐
    │ DW │ FR │ MW │ WD │ background │
    └────┴────┴────┴────┴────────────┘
    output: (batch, 5)   — col 4 is background, scaled by its own MinMaxScaler
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS      = 80
PATIENCE    = 20
LR          = 1e-3
BATCH       = 32
WIN         = 100
STRIDE      = 5

LAMBDA_BG     = 0.1    # background head loss weight (higher than old λ_phys=0.01
                       # because this is direct regression, not a soft penalty)
WARMUP_EPOCHS = 20     # Stage 1: MSE on 4 apps only; background + BCE added after

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.5, 'fridge': 0.3, 'microwave': 0.0, 'washer dryer': 0.0}
BCE_ALPHA  = {'dish washer': 2.0, 'fridge': 2.0, 'microwave': 1.0, 'washer dryer': 1.0}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualHeadPINNLiquidNetworkModel(nn.Module):
    """
    Five-output model: four appliance heads + one background head.

    The background head is supervised by:
        y_bg = P_mains_midpoint − Σ_k y_k_true   (raw Watts, then scaled)

    Forcing the model to predict the background constrains the decomposition:
        Σ_k p̂_k + p̂_bg  ≈  P_mains   (by regression, not by penalty)

    ODE (unchanged from original PINN-LNN):
        τ   = softplus(τ_base) ⊙ sigmoid(W_τ · x_t)
        g   = sigmoid(W_gate · [x_t ; h])
        f_t = tanh(W_in · x_t + W_rec · h)
        dh  = (−h/τ + g ⊙ f_t) · dt
        h   ← clamp(h + dh, −10, 10)
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances   # 4 — does not include background
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

        # 4 appliance heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])
        # 5th head — background / residual
        self.background_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, n_appliances + 1) — cols 0..3 = appliances, col 4 = background
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]

            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)

            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod  = torch.sigmoid(self.tau_mod(x_t))
            tau      = (tau_base * tau_mod).clamp(min=self.dt)

            gate = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t  = torch.tanh(input_proj + rec_proj)

            dh = ((-h / tau) + gate * f_t) * self.dt
            h  = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)

        app_preds = torch.cat([head(h) for head in self.heads], dim=1)  # (B, 4)
        bg_pred   = self.background_head(h)                              # (B, 1)
        return torch.cat([app_preds, bg_pred], dim=1)                   # (B, 5)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)   # (N, 5): cols 0-3 apps, col 4 background

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data():
    print("Loading UKDALE data...")
    with open('data/ukdale/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/ukdale/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/ukdale/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=WIN):
    """
    Returns:
        X:      (N, window, 1)  — scaled mains windows
        Y_apps: (N, 4)          — raw appliance power at midpoint
        Y_bg:   (N, 1)          — raw background = mains_mid − Σ appliances_mid
    """
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y_apps, Y_bg = [], [], []

    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid       = i + window_size // 2
        app_pwr   = [app_vals[app][mid] for app in APPLIANCES]
        bg        = mains[mid] - sum(app_pwr)
        Y_apps.append(app_pwr)
        Y_bg.append([bg])

    return (
        np.array(X,      dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y_apps, dtype=np.float32),    # (N, 4)
        np.array(Y_bg,   dtype=np.float32),    # (N, 1)
    )


# ---------------------------------------------------------------------------
# Per-appliance metrics helper  (4 appliances only)
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(
            raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, dt=0.1, lambda_bg=LAMBDA_BG):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"ODE: original LTC  dt={dt}  λ_bg={lambda_bg}  hidden={hidden_size}")

    # ── Sequences ──
    X_tr, Y_tr_apps, Y_tr_bg = create_sequences(data_dict['train'], WIN)
    X_va, Y_va_apps, Y_va_bg = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te_apps, Y_te_bg = create_sequences(data_dict['test'],  WIN)

    # ── Scale mains ──
    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    # ── Scale 4 appliances ──
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr_apps[:, i:i+1] = ys.fit_transform(Y_tr_apps[:, i:i+1])
        Y_va_apps[:, i:i+1] = ys.transform(Y_va_apps[:, i:i+1])
        Y_te_apps[:, i:i+1] = ys.transform(Y_te_apps[:, i:i+1])
        y_scalers.append(ys)

    # ── Scale background (may be negative — MinMaxScaler handles it) ──
    bg_scaler = MinMaxScaler()
    Y_tr_bg = bg_scaler.fit_transform(Y_tr_bg)
    Y_va_bg = bg_scaler.transform(Y_va_bg)
    Y_te_bg = bg_scaler.transform(Y_te_bg)

    # Concatenate: Y[:, 0:4] = appliances, Y[:, 4] = background
    Y_tr = np.concatenate([Y_tr_apps, Y_tr_bg], axis=1)   # (N, 5)
    Y_va = np.concatenate([Y_va_apps, Y_va_bg], axis=1)
    Y_te = np.concatenate([Y_te_apps, Y_te_bg], axis=1)

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")
    bg_mean = float(bg_scaler.inverse_transform([[0.5]])[0, 0])
    print(f"Background: min={bg_scaler.data_min_[0]:.1f}W  "
          f"max={bg_scaler.data_max_[0]:.1f}W  "
          f"train_mean≈{float(data_dict['train']['main'].mean() - sum(data_dict['train'][a].mean() for a in APPLIANCES)):.1f}W")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = ResidualHeadPINNLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_bg': [],
        'val_loss':   [], 'val_mse':   [], 'val_bg':   [],
        'val_metrics': [],
    }
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting PINN-LNN (residual head) training...")

    for epoch in range(EPOCHS):
        model.train()
        ep_mse = ep_bg = ep_total = 0.0
        progress_bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)                       # (batch, 5)

            # MSE over 4 appliance heads
            mse_loss = mse_criterion(pred[:, :4], yb[:, :4])

            # Background head loss — direct supervision of residual power
            bg_loss = mse_criterion(pred[:, 4], yb[:, 4])

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss
            else:
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_i = pred[:, i].clamp(1e-7, 1 - 1e-7)
                        thr_s  = thresholds_scaled[i]
                        y_bin  = (yb[:, i] > thr_s).float()
                        w      = torch.where(y_bin == 1,
                                             torch.full_like(y_bin, BCE_ALPHA[app]),
                                             torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_i, y_bin, weight=w)
                loss = mse_loss + lambda_bg * bg_loss + bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_bg    += bg_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'mse': f'{mse_loss.item():.5f}',
                'bg':  f'{bg_loss.item():.5f}',
            })

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_bg    = ep_bg    / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_bg'].append(avg_tr_bg)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_mse = vl_bg = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)

                mse_loss = mse_criterion(pred[:, :4], yb[:, :4])
                bg_loss  = mse_criterion(pred[:, 4],  yb[:, 4])
                loss     = mse_loss + lambda_bg * bg_loss

                vl_mse   += mse_loss.item()
                vl_bg    += bg_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_bg    = vl_bg    / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_bg'].append(avg_va_bg)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)   # (N, 5)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(
            y_true_all[:, :4], y_pred_all[:, :4], y_scalers)
        history['val_metrics'].append(per_app_metrics)

        # Background MAE in raw Watts
        bg_pred_raw = bg_scaler.inverse_transform(y_pred_all[:, 4:5]).flatten()
        bg_true_raw = bg_scaler.inverse_transform(y_true_all[:, 4:5]).flatten()
        bg_mae = float(np.mean(np.abs(bg_pred_raw - bg_true_raw)))

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} bg={avg_tr_bg:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} bg={avg_va_bg:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  bgMAE={bg_mae:.1f}W  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_va_mse < best_val_loss:
            best_val_loss = avg_va_mse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            test_preds.append(model(xb.to(device)).cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    y_pred_te = np.concatenate(test_preds)
    y_true_te = np.concatenate(test_trues)

    test_metrics = compute_per_appliance_metrics(
        y_true_te[:, :4], y_pred_te[:, :4], y_scalers)

    bg_pred_raw_te = bg_scaler.inverse_transform(y_pred_te[:, 4:5]).flatten()
    bg_true_raw_te = bg_scaler.inverse_transform(y_true_te[:, 4:5]).flatten()
    bg_test_mae    = float(np.mean(np.abs(bg_pred_raw_te - bg_true_raw_te)))

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")
    print(f"{'background':<15} {'—':>8} {'—':>10} {'—':>8} "
          f"{bg_test_mae:>8.2f} {'—':>8}")

    _plot_training(history, test_metrics, bg_test_mae, save_dir)

    config = {
        'dataset': 'UKDALE',
        'model': 'ResidualHeadPINNLiquidNetworkModel',
        'description': 'LTC encoder + 4 appliance heads + 1 background head',
        'physics': 'L_bg = MSE(p_hat_background, P_mains - sum(y_true_apps))',
        'loss': f'MSE_4apps + {lambda_bg} * L_bg + L_BCE',
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_bg': lambda_bg,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
        'background_test_mae_watts': bg_test_mae,
    }
    with open(os.path.join(save_dir, 'pinn_lnn_residual_ukdale_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, bg_test_mae, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE (4 apps)', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE (4 apps)',   color='red')
    plt.title('Appliance MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_bg'], label='Train BG MSE', color='blue')
    plt.plot(epochs_x, history['val_bg'],   label='Val BG MSE',   color='red')
    plt.title('Background Head MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_residual_ukdale_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN Residual Head UKDALE — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.set_title(f'{app} — F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.set_title(f'{app} — MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_residual_ukdale_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_lnn_residual_ukdale_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_bg   = LAMBDA_BG,
    )

    print(f"\nResults saved to {save_dir}")
