"""
Hybrid Fourier Neural Operator + LNN (FNO-LNN) for NILM — REDD specific splits.

The aggregate mains window is processed in parallel by two branches:

  Spectral Branch (FNO)
      FFT → learnable complex weights on the first N_MODES frequency modes → IFFT
      → global average pool → frequency-domain feature vector.
      Captures the global "fingerprint" of each appliance in the frequency domain
      (e.g. the compressor cycle of a fridge, the magnetron signature of a MW).

  Temporal Branch (LNN)
      AdvancedLiquidTimeLayer with input-dependent time constants and gating.
      Captures fine-grained adaptive transitions that the FNO misses.

  Fusion MLP → per-appliance regression heads (all 4 appliances simultaneously).

Architecture summary:
    Input (batch, WIN, 1)
         ↓
    ┌──────────────────┬───────────────────────────┐
    │  FNO Branch       │  LNN Branch               │
    │  lift → N×FNO1d   │  AdvancedLiquidTimeLayer  │
    │  → project        │                           │
    │  → global avgpool │                           │
    │  → (B, FNO_CH)    │  → (B, HIDDEN_LNN)        │
    └────────┬──────────┴──────────┬────────────────┘
             └──── cat ────────────┘
                      ↓
             Fusion MLP (B, FNO_CH + HIDDEN_LNN → FUSION_SIZE)
                      ↓
           ┌────┬────┬────┬────┐
           │ DW │ FR │ MW │ WD │  one Linear head per appliance
           └────┴────┴────┴────┘

Note on FNO resolution:
    With WIN=100, rfft produces 51 unique frequency bins. N_MODES=16 retains the
    16 lowest-frequency modes (DC + slow oscillations spanning ~6-100 samples).
    Increasing WIN to 500+ would give richer spectral discrimination.

Differences from test_lnn_redd_specific_splits.py:
    - All 4 appliances trained simultaneously (multi-head)
    - Midpoint-targeted sequences (y[i] = appliance power at window centre)
    - FNO spectral branch fused with LNN temporal branch
    - Inline LNN — no import from models.py
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
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS      = 80
PATIENCE    = 20
LR          = 1e-3
BATCH       = 32
WIN         = 100   # window length; 51 unique FFT bins available
STRIDE      = 5

# FNO hyperparameters
N_MODES     = 16    # frequency modes retained (out of WIN//2+1 = 51)
FNO_CH      = 16    # channel width inside FNO blocks
N_FNO       = 3     # number of stacked FNO blocks

# LNN hyperparameters
HIDDEN_LNN  = 64
DT          = 0.1

FUSION_SIZE = 64    # hidden size of fusion MLP

APPLIANCES  = ['dish washer', 'fridge', 'microwave', 'washer dryer']
THRESHOLDS  = {'dish washer': 10.0, 'fridge': 10.0,
               'microwave': 10.0, 'washer dryer': 10.0}


# ---------------------------------------------------------------------------
# FNO components
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    1-D Spectral Convolution — the core FNO layer.

    Applies a learnable linear map in the frequency domain:
        u_out = IFFT( W · FFT(u)[0:n_modes] )

    W is stored as two real tensors (weight_r + i·weight_i) so autograd
    works without any special complex-parameter handling.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.n_modes      = n_modes
        self.out_channels = out_channels
        scale = 1.0 / (in_channels * out_channels)
        self.weight_r = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes))
        self.weight_i = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        B, C, L = x.shape
        n = min(self.n_modes, L // 2 + 1)

        x_ft = torch.fft.rfft(x, dim=-1)                        # (B, C, L//2+1) cfloat
        W    = torch.complex(self.weight_r, self.weight_i)       # (in, out, n_modes)

        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        # contract over in_channels, keep (batch, out_channels, n_modes)
        out_ft[:, :, :n] = torch.einsum('bin,ion->bon',
                                         x_ft[:, :, :n], W[:, :, :n])

        return torch.fft.irfft(out_ft, n=L, dim=-1)             # (B, out_channels, L)


class FNO1dBlock(nn.Module):
    """
    One FNO layer: spectral conv + pointwise local conv + residual + GELU.

    The local (W·u) branch ensures the layer can represent the identity
    mapping when the spectral branch is not yet useful.
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, n_modes)
        self.local    = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm     = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


# ---------------------------------------------------------------------------
# LNN branch (inline — avoids dependency on models.py)
# ---------------------------------------------------------------------------

class LiquidBranch(nn.Module):
    """
    AdvancedLiquidTimeLayer: processes a (batch, seq_len, 1) window
    step-by-step and returns the final hidden state (batch, hidden_size).

    Time constants τ are input-dependent (liquid) and modulated by a
    sigmoid gate, giving the model adaptive memory over the window.
    """

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt          = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            x_t        = x[:, t, :]
            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)

            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod  = torch.sigmoid(self.tau_mod(x_t))
            tau      = (tau_base * tau_mod).clamp(min=self.dt)

            gate = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t  = torch.tanh(input_proj + rec_proj)
            dh   = ((-h / tau) + gate * f_t) * self.dt
            h    = (h + dh).clamp(-10.0, 10.0)

        return self.norm(h)   # (batch, hidden_size)


# ---------------------------------------------------------------------------
# Hybrid FNO-LNN model
# ---------------------------------------------------------------------------

class FNO_LNN_Model(nn.Module):
    """
    Hybrid Fourier Neural Operator + Liquid Neural Network for NILM.

    FNO branch:  global frequency features  (batch, FNO_CH)
    LNN branch:  adaptive temporal features (batch, HIDDEN_LNN)
    Fusion MLP:  combined representation    (batch, FUSION_SIZE)
    Heads:       per-appliance predictions  (batch, n_appliances)
    """

    def __init__(self, window_size: int = WIN,
                 fno_ch: int = FNO_CH, n_modes: int = N_MODES, n_fno: int = N_FNO,
                 hidden_lnn: int = HIDDEN_LNN, fusion_size: int = FUSION_SIZE,
                 n_appliances: int = 4, dt: float = DT):
        super().__init__()

        # ── FNO branch ──
        self.lift       = nn.Conv1d(1, fno_ch, kernel_size=1)          # 1 → fno_ch channels
        self.fno_blocks = nn.ModuleList(
            [FNO1dBlock(fno_ch, n_modes) for _ in range(n_fno)])
        self.project    = nn.Conv1d(fno_ch, fno_ch, kernel_size=1)     # fno_ch → fno_ch
        # global average pool along time → (batch, fno_ch)

        # ── LNN branch ──
        self.lnn = LiquidBranch(1, hidden_lnn, dt)

        # ── Fusion ──
        self.fusion = nn.Sequential(
            nn.Linear(fno_ch + hidden_lnn, fusion_size),
            nn.GELU(),
            nn.LayerNorm(fusion_size),
        )

        # ── Per-appliance regression heads ──
        self.heads = nn.ModuleList(
            [nn.Linear(fusion_size, 1) for _ in range(n_appliances)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)

        # FNO branch expects (batch, channels, seq_len)
        x_fno = x.permute(0, 2, 1)              # (B, 1, L)
        x_fno = self.lift(x_fno)                # (B, fno_ch, L)
        for block in self.fno_blocks:
            x_fno = block(x_fno)                # (B, fno_ch, L)
        x_fno = self.project(x_fno)             # (B, fno_ch, L)
        spectral_feat = x_fno.mean(dim=-1)      # (B, fno_ch)  — global avg pool

        # LNN branch
        lnn_feat = self.lnn(x)                  # (B, hidden_lnn)

        # Fusion and heads
        fused = self.fusion(
            torch.cat([spectral_feat, lnn_feat], dim=-1))  # (B, fusion_size)
        return torch.cat([h(fused) for h in self.heads], dim=-1)  # (B, n_apps)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data():
    print("Loading REDD data...")
    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/redd/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size: int = WIN):
    """Midpoint targeting: y[i] is the appliance power at the window centre."""
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
    )


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

def train_fno_lnn_model(data_dict, save_dir: str,
                        fno_ch: int = FNO_CH, n_modes: int = N_MODES,
                        n_fno: int = N_FNO, hidden_lnn: int = HIDDEN_LNN,
                        fusion_size: int = FUSION_SIZE, dt: float = DT):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"FNO: {n_fno} blocks × {fno_ch}ch × {n_modes} modes  |  "
          f"LNN: hidden={hidden_lnn}  |  fusion={fusion_size}")

    # ── Sequences ──
    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # ── Scaling ──
    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH,
        shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH,
        shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH,
        shuffle=False, drop_last=False)

    # ── Model ──
    model = FNO_LNN_Model(
        window_size=WIN, fno_ch=fno_ch, n_modes=n_modes, n_fno=n_fno,
        hidden_lnn=hidden_lnn, fusion_size=fusion_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print("Starting FNO-LNN training (all appliances simultaneously)...")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_tr = ep_loss / len(tr_loader)
        history['train_loss'].append(avg_tr)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred    = model(xb)
                vl_loss += criterion(pred, yb).item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va = vl_loss / len(va_loader)
        history['val_loss'].append(avg_va)
        scheduler.step(avg_va)

        y_pred_va = np.concatenate(val_preds)
        y_true_va = np.concatenate(val_trues)
        per_app   = compute_per_appliance_metrics(y_true_va, y_pred_va, y_scalers)
        history['val_metrics'].append(per_app)

        avg_f1  = np.mean([per_app[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr:.5f}  val={avg_va:.5f}  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_va < best_val_loss:
            best_val_loss = avg_va
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    # ── Test evaluation ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            pred = model(xb.to(device))
            test_preds.append(pred.cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    y_pred_te  = np.concatenate(test_preds)
    y_true_te  = np.concatenate(test_trues)
    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    # ── Save model weights ──
    torch.save(best_state or model.state_dict(),
               os.path.join(save_dir, 'fno_lnn_redd_best.pth'))

    # ── Plots ──
    _plot_training(history, test_metrics, save_dir)

    # ── JSON results ──
    config = {
        'dataset':     'REDD',
        'model':       'FNO_LNN_Model',
        'description': 'Hybrid FNO spectral branch + LNN temporal branch, all appliances',
        'window_size': WIN,
        'model_params': {
            'fno_ch': fno_ch, 'n_modes': n_modes, 'n_fno': n_fno,
            'hidden_lnn': hidden_lnn, 'fusion_size': fusion_size, 'dt': dt,
        },
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'batch': BATCH, 'stride': STRIDE},
        'thresholds':  THRESHOLDS,
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'fno_lnn_redd_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print(f"\nResults saved to {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)
    colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val',   color='red')
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    for i, app in enumerate(APPLIANCES):
        f1_series = [m[app]['f1'] for m in history['val_metrics']]
        plt.plot(epochs_x, f1_series, label=app, color=colors[i])
    plt.title('Val F1 per Appliance')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    maes = [test_metrics[a]['mae'] for a in APPLIANCES]
    plt.bar(range(len(APPLIANCES)), maes, color=colors)
    plt.xticks(range(len(APPLIANCES)),
               [a.replace(' ', '\n') for a in APPLIANCES], fontsize=8)
    plt.title('Test MAE (W)')
    plt.ylabel('MAE (W)'); plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fno_lnn_redd_training.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FNO-LNN for NILM — REDD dataset, specific splits")
    print("=" * 60)

    for f in ['data/redd/train_small.pkl', 'data/redd/val_small.pkl',
              'data/redd/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    data_dict = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/fno_lnn_redd_{timestamp}"

    test_metrics, _ = train_fno_lnn_model(
        data_dict, save_dir=save_dir,
        fno_ch=FNO_CH, n_modes=N_MODES, n_fno=N_FNO,
        hidden_lnn=HIDDEN_LNN, fusion_size=FUSION_SIZE, dt=DT,
    )

    print(f"\nSummary — FNO-LNN on REDD:")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")
