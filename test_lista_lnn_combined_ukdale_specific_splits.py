"""
LISTA-LNN (LISTA encoder + Advanced Liquid Neural Network) for NILM — UKDALE specific splits.

Architecture:
  Sequence of T mains windows (each WIN=100 samples)
    → LISTAEncoder  × T   →  x_t ∈ R^n_atoms   (sparse codes, one per window)
    → Linear projection   →  z_t ∈ R^hidden_size
    → AdvancedLiquidTimeLayer × num_lnn_layers   (temporal ODE dynamics over sparse codes)
    → MLP head            →  p̂  (appliance power at centre of last window)

LISTA role:   within-window sparse decomposition (what patterns exist right now)
LNN role:     across-window temporal dynamics     (how patterns evolve over time)

Loss (four terms):
    L = MSE(p̂, p*)
      + λ_recon  · (1/T) Σ_t ||y_t − x_t @ D||²  — dict reconstruction
      + λ_sparse · (1/T) Σ_t ||x_t||₁              — sparsity
      + λ_bce    · BCE(σ(p̂/thr), s*)               — on/off supervision (post-warmup)

Key differences from test_lista_lnn_ukdale_specific_splits.py:
  1. Dataset returns sequences of T=10 consecutive windows for temporal context
  2. AdvancedLiquidTimeLayer (adaptive τ, input gate) runs over T sparse codes
  3. MLP decoder head (hidden→32→1) instead of a single linear layer
  4. Recon / sparse losses averaged over T windows

Trained per-appliance (same UKDALE splits as other scripts).
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
from models import AdvancedLiquidTimeLayer
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 100      # samples per window
WIN_STEP      = 50       # stride between consecutive windows in a sequence (50% overlap)
SEQ_LEN       = 10       # T: number of windows per sample (covers ~550 timesteps)
STRIDE        = 5        # stride between samples in the dataset
N_ATOMS       = 64       # LISTA sparse code dimension
K_LAYERS      = 8        # unrolled ISTA iterations
HIDDEN_SIZE   = 64       # LNN hidden state size
NUM_LNN_LAYERS = 2       # stacked AdvancedLiquidTimeLayer count
DT            = 0.1      # LNN ODE time step

LAMBDA_RECON  = 0.1
LAMBDA_SPARSE = 0.01
WARMUP_EPOCHS = 15       # epochs before BCE loss is added

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.3, 'fridge': 0.5, 'microwave': 2.0, 'washer dryer': 2.0}
BCE_ALPHA  = {'dish washer': 1.0, 'fridge': 1.5, 'microwave': 10.0, 'washer dryer': 8.0}


# ---------------------------------------------------------------------------
# Soft threshold activation
# ---------------------------------------------------------------------------

def soft_threshold(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Element-wise soft threshold. lam must broadcast onto z."""
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


# ---------------------------------------------------------------------------
# LISTA components
# ---------------------------------------------------------------------------

class LISTALayer(nn.Module):
    """One unrolled ISTA iteration with fully learnable weights."""

    def __init__(self, signal_len: int, n_atoms: int):
        super().__init__()
        self.We        = nn.Linear(signal_len, n_atoms, bias=True)
        self.Wr        = nn.Linear(n_atoms,    n_atoms, bias=False)
        self.threshold = nn.Parameter(torch.full((n_atoms,), 0.1))

    def forward(self, y: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        z = self.We(y) + self.Wr(x_prev)
        return soft_threshold(z, F.softplus(self.threshold))


class LISTAEncoder(nn.Module):
    """
    K-layer LISTA stack + jointly learned dictionary D.

    encode(y) : (batch, WIN) -> x^(K) (batch, n_atoms)
    reconstruct(x) : (batch, n_atoms) -> y_hat (batch, WIN)
    """

    def __init__(self, signal_len: int, n_atoms: int, k_layers: int):
        super().__init__()
        self.n_atoms = n_atoms
        self.layers  = nn.ModuleList([
            LISTALayer(signal_len, n_atoms) for _ in range(k_layers)
        ])
        self.D = nn.Parameter(torch.randn(n_atoms, signal_len) * 0.01)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """y: (batch, WIN) -> x: (batch, n_atoms)"""
        x = torch.zeros(y.size(0), self.n_atoms, device=y.device)
        for layer in self.layers:
            x = layer(y, x)
        return x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_atoms) -> y_hat: (batch, WIN)"""
        return x @ self.D


# ---------------------------------------------------------------------------
# LISTA-LNN model
# ---------------------------------------------------------------------------

class LISTALNNModel(nn.Module):
    """
    LISTA encoder (per window) + AdvancedLiquidTimeLayer stack (over time).

    Forward:
        y_seq : (batch, T, WIN) — T consecutive mains windows
        returns:
            power : (batch, 1)        — predicted appliance power (scaled)
            codes : (batch, T, n_atoms) — sparse codes for auxiliary losses
    """

    def __init__(self, signal_len: int = WIN, n_atoms: int = N_ATOMS,
                 k_layers: int = K_LAYERS, hidden_size: int = HIDDEN_SIZE,
                 num_lnn_layers: int = NUM_LNN_LAYERS, dt: float = DT):
        super().__init__()
        self.num_lnn_layers = num_lnn_layers

        # ── LISTA encoder ──
        self.lista = LISTAEncoder(signal_len, n_atoms, k_layers)

        # ── Project sparse codes into LNN input space ──
        self.proj = nn.Linear(n_atoms, hidden_size)

        # ── Stacked AdvancedLiquidTimeLayer (inter-layer LayerNorm) ──
        self.lnn_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size  = hidden_size,
                hidden_size = hidden_size,
                dt          = dt,
            ) for _ in range(num_lnn_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_lnn_layers)
        ])

        # ── MLP decoder head ──
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, y_seq: torch.Tensor):
        """y_seq: (batch, T, WIN)"""
        B, T, _ = y_seq.shape

        # Encode each window with LISTA
        codes = []
        for t in range(T):
            codes.append(self.lista.encode(y_seq[:, t]))   # (B, n_atoms)
        codes = torch.stack(codes, dim=1)                  # (B, T, n_atoms)

        # Project to LNN input size
        z_seq = self.proj(codes)                           # (B, T, hidden_size)

        # Run LNN over T timesteps
        hidden_states = [None] * self.num_lnn_layers
        for t in range(T):
            z_t = z_seq[:, t]                              # (B, hidden_size)
            for i in range(self.num_lnn_layers):
                inp = z_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                hidden_states[i] = self.lnn_layers[i](inp, hidden_states[i])

        # Decode final hidden state
        h_last = self.layer_norms[-1](hidden_states[-1])  # (B, hidden_size)
        power  = self.head(h_last)                         # (B, 1)

        return power, codes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKDALESequenceDataset(torch.utils.data.Dataset):
    """Returns sequences of T windows (X: float32) and scalar labels (y: float32)."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)   # (N, T, WIN)
        self.y = torch.FloatTensor(y)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ukdale_specific_splits():
    print("Loading UKDALE data with specific splits...")
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


def create_sequences(data, appliance_name,
                     window_size: int = WIN,
                     win_step: int    = WIN_STEP,
                     seq_len: int     = SEQ_LEN,
                     stride: int      = STRIDE):
    """
    Build (X, y) where:
      X shape : (N, seq_len, window_size)   — T consecutive overlapping windows
      y shape : (N, 1)                      — appliance power at centre of last window

    total_span = (seq_len-1)*win_step + window_size
    """
    mains    = data['main'].values
    app_vals = data[appliance_name].values

    total_span = (seq_len - 1) * win_step + window_size
    X, y = [], []

    for i in range(0, len(mains) - total_span, stride):
        windows = [mains[i + t * win_step : i + t * win_step + window_size]
                   for t in range(seq_len)]
        X.append(windows)
        last_start = i + (seq_len - 1) * win_step
        y.append(app_vals[last_start + window_size // 2])

    return (
        np.array(X, dtype=np.float32),           # (N, T, WIN)
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name,
                       n_atoms      = N_ATOMS,
                       k_layers     = K_LAYERS,
                       hidden_size  = HIDDEN_SIZE,
                       num_lnn_layers = NUM_LNN_LAYERS,
                       epochs       = EPOCHS,
                       lr           = LR,
                       patience     = PATIENCE,
                       lambda_recon = LAMBDA_RECON,
                       lambda_sparse= LAMBDA_SPARSE,
                       save_dir     = 'models/lista_lnn_combined_ukdale'):
    os.makedirs(save_dir, exist_ok=True)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold  = THRESHOLDS[appliance_name]
    bce_lambda = BCE_LAMBDA[appliance_name]
    bce_alpha  = BCE_ALPHA[appliance_name]

    print(f"\nDevice: {device}  |  appliance: {appliance_name}")
    print(f"n_atoms={n_atoms}  K={k_layers}  hidden={hidden_size}  lnn_layers={num_lnn_layers}")
    print(f"T={SEQ_LEN}  win_step={WIN_STEP}  λ_recon={lambda_recon}  "
          f"λ_sparse={lambda_sparse}  λ_bce={bce_lambda}  α_bce={bce_alpha}  "
          f"warmup={WARMUP_EPOCHS}")

    # ── Sequences ──
    X_tr, y_tr = create_sequences(data_dict['train'], appliance_name)
    X_va, y_va = create_sequences(data_dict['val'],   appliance_name)
    X_te, y_te = create_sequences(data_dict['test'],  appliance_name)

    # ── Scaling ──
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    N_tr, T, W = X_tr.shape
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(N_tr, T, W)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_tr = y_scaler.fit_transform(y_tr)
    y_va = y_scaler.transform(y_va)
    y_te = y_scaler.transform(y_te)

    thr_scaled = (threshold - float(y_scaler.data_min_[0])) / float(y_scaler.data_range_[0])

    print(f"Train: {X_tr.shape} → {y_tr.shape}")
    print(f"Val:   {X_va.shape} → {y_va.shape}")
    print(f"Test:  {X_te.shape} → {y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        UKDALESequenceDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        UKDALESequenceDataset(X_va, y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        UKDALESequenceDataset(X_te, y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = LISTALNNModel(
        signal_len     = WIN,
        n_atoms        = n_atoms,
        k_layers       = k_layers,
        hidden_size    = hidden_size,
        num_lnn_layers = num_lnn_layers,
        dt             = DT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history       = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print(f"Starting LISTA-LNN training for {appliance_name}...")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)    # (B,T,WIN), (B,1)
            optimizer.zero_grad()

            power, codes = model(xb)                  # (B,1), (B,T,n_atoms)

            mse_loss = F.mse_loss(power, yb)

            # Reconstruction + sparsity averaged over T windows
            recon_loss  = 0.0
            sparse_loss = 0.0
            for t in range(SEQ_LEN):
                y_hat        = model.lista.reconstruct(codes[:, t])
                recon_loss  += F.mse_loss(y_hat, xb[:, t])
                sparse_loss += codes[:, t].abs().mean()
            recon_loss  = recon_loss  / SEQ_LEN
            sparse_loss = sparse_loss / SEQ_LEN

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss + lambda_recon * recon_loss + lambda_sparse * sparse_loss
            else:
                pred_prob = torch.sigmoid(power / (thr_scaled + 1e-8))
                y_bin     = (yb > thr_scaled).float()
                w         = torch.where(y_bin == 1,
                                        torch.full_like(y_bin, bce_alpha),
                                        torch.ones_like(y_bin))
                bce_loss  = F.binary_cross_entropy(
                    pred_prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
                loss = (mse_loss
                        + lambda_recon  * recon_loss
                        + lambda_sparse * sparse_loss
                        + bce_lambda    * bce_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.5f}',
                             'mse':  f'{mse_loss.item():.5f}'})

        avg_tr_loss = ep_loss / len(tr_loader)
        history['train_loss'].append(avg_tr_loss)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                power, codes = model(xb)

                mse_loss    = F.mse_loss(power, yb)
                recon_loss  = 0.0
                sparse_loss = 0.0
                for t in range(SEQ_LEN):
                    y_hat        = model.lista.reconstruct(codes[:, t])
                    recon_loss  += F.mse_loss(y_hat, xb[:, t])
                    sparse_loss += codes[:, t].abs().mean()
                recon_loss  = recon_loss  / SEQ_LEN
                sparse_loss = sparse_loss / SEQ_LEN

                if epoch < WARMUP_EPOCHS:
                    loss = mse_loss + lambda_recon * recon_loss + lambda_sparse * sparse_loss
                else:
                    pred_prob = torch.sigmoid(power / (thr_scaled + 1e-8))
                    y_bin     = (yb > thr_scaled).float()
                    w         = torch.where(y_bin == 1,
                                            torch.full_like(y_bin, bce_alpha),
                                            torch.ones_like(y_bin))
                    bce_loss  = F.binary_cross_entropy(
                        pred_prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
                    loss = (mse_loss
                            + lambda_recon  * recon_loss
                            + lambda_sparse * sparse_loss
                            + bce_lambda    * bce_loss)

                vl_loss += loss.item()
                all_preds.append(power.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        avg_va_loss = vl_loss / len(va_loader)
        history['val_loss'].append(avg_va_loss)
        scheduler.step(avg_va_loss)

        raw_true = y_scaler.inverse_transform(
            np.concatenate(all_trues).reshape(-1, 1)).flatten()
        raw_pred = y_scaler.inverse_transform(
            np.concatenate(all_preds).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(raw_true, raw_pred, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr_loss:.5f}  val={avg_va_loss:.5f}  "
              f"F1={metrics['f1']:.4f}  MAE={metrics['mae']:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_va_loss < best_val_loss:
            best_val_loss = avg_va_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
            save_model(
                model,
                {'signal_len': WIN, 'n_atoms': n_atoms, 'k_layers': k_layers,
                 'hidden_size': hidden_size, 'num_lnn_layers': num_lnn_layers},
                {'lr': lr, 'epochs': epochs, 'patience': patience,
                 'appliance': appliance_name},
                metrics,
                os.path.join(save_dir,
                    f"lista_lnn_combined_ukdale_{appliance_name.replace(' ', '_')}_best.pth")
            )
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            power, _ = model(xb.to(device))
            test_preds.append(power.cpu().numpy())
            test_trues.append(yb.numpy())

    raw_true_te = y_scaler.inverse_transform(
        np.concatenate(test_trues).reshape(-1, 1)).flatten()
    raw_pred_te = y_scaler.inverse_transform(
        np.concatenate(test_preds).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_true_te, raw_pred_te, threshold=threshold)

    print(f"\nTest  F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}")

    _plot_results(history, test_metrics, appliance_name, save_dir)

    config = {
        'appliance': appliance_name,
        'dataset':   'UKDALE',
        'model':     'LISTALNNModel',
        'description': (
            'LISTA encoder (K-layer unrolled ISTA) per window '
            '+ AdvancedLiquidTimeLayer stack over T-window sequence'
        ),
        'loss': 'MSE + λ_recon·Recon + λ_sparse·L1 + λ_bce·BCE',
        'window_size': WIN,
        'model_params': {
            'signal_len': WIN, 'n_atoms': n_atoms, 'k_layers': k_layers,
            'hidden_size': hidden_size, 'num_lnn_layers': num_lnn_layers, 'dt': DT,
        },
        'train_params': {
            'lr': lr, 'epochs': epochs, 'patience': patience,
            'seq_len': SEQ_LEN, 'win_step': WIN_STEP,
            'lambda_recon': lambda_recon, 'lambda_sparse': lambda_sparse,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': bce_lambda, 'bce_alpha': bce_alpha,
        },
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': {
                'train_loss_mean':    float(np.mean(history['train_loss'])),
                'val_loss_mean':      float(np.mean(history['val_loss'])),
                'val_mae_mean':       float(np.mean([m['mae'] for m in history['val_metrics']])),
                'val_f1_mean':        float(np.mean([m['f1']  for m in history['val_metrics']])),
            },
        },
    }
    with open(os.path.join(save_dir,
            f'lista_lnn_combined_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, appliance_name, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs_x, val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs_x, val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(epochs_x, val_f1_series,        label='Val F1',        color='red')
    plt.plot(epochs_x, val_precision_series, label='Val Precision',  color='blue')
    plt.plot(epochs_x, val_recall_series,    label='Val Recall',     color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"lista_lnn_combined_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def test_on_all_appliances(n_atoms       = N_ATOMS,
                           k_layers      = K_LAYERS,
                           hidden_size   = HIDDEN_SIZE,
                           num_lnn_layers= NUM_LNN_LAYERS,
                           epochs        = EPOCHS,
                           lr            = LR,
                           patience      = PATIENCE,
                           lambda_recon  = LAMBDA_RECON,
                           lambda_sparse = LAMBDA_SPARSE):
    data_dict = load_ukdale_specific_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/lista_lnn_combined_ukdale_specific_test_{timestamp}"

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"LISTA-LNN (combined) on {appliance_name}")
        print(f"{'='*60}")

        app_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(app_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_on_appliance(
                data_dict,
                appliance_name  = appliance_name,
                n_atoms         = n_atoms,
                k_layers        = k_layers,
                hidden_size     = hidden_size,
                num_lnn_layers  = num_lnn_layers,
                epochs          = epochs,
                lr              = lr,
                patience        = patience,
                lambda_recon    = lambda_recon,
                lambda_sparse   = lambda_sparse,
                save_dir        = app_dir,
            )
            all_results[appliance_name] = {
                'model_path': os.path.join(
                    app_dir,
                    f"lista_lnn_combined_ukdale_{appliance_name.replace(' ', '_')}_best.pth"),
                'final_metrics': {k: float(v) for k, v in test_metrics.items()},
            }
        except Exception as e:
            print(f"Error on {appliance_name}: {e}")
            import traceback; traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset':   'UKDALE',
        'model':     'LISTALNNModel',
        'description': (
            'LISTA sparse encoder (K unrolled ISTA) per window '
            '+ AdvancedLiquidTimeLayer (adaptive tau, gate, inter-LayerNorm) '
            'over T-window sequence'
        ),
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'},
        },
        'window_size': WIN,
        'model_params': {
            'n_atoms': n_atoms, 'k_layers': k_layers,
            'hidden_size': hidden_size, 'num_lnn_layers': num_lnn_layers, 'dt': DT,
        },
        'train_params': {
            'epochs': epochs, 'lr': lr, 'patience': patience,
            'seq_len': SEQ_LEN, 'win_step': WIN_STEP,
            'lambda_recon': lambda_recon, 'lambda_sparse': lambda_sparse,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': BCE_LAMBDA, 'bce_alpha': BCE_ALPHA,
        },
        'results': all_results,
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nLISTA-LNN (combined) UKDALE testing complete. Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]['final_metrics']
            print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    results = test_on_all_appliances(
        n_atoms        = N_ATOMS,
        k_layers       = K_LAYERS,
        hidden_size    = HIDDEN_SIZE,
        num_lnn_layers = NUM_LNN_LAYERS,
        epochs         = EPOCHS,
        lr             = LR,
        patience       = PATIENCE,
        lambda_recon   = LAMBDA_RECON,
        lambda_sparse  = LAMBDA_SPARSE,
    )

    print(f"\nSummary — LISTA-LNN (combined) on UKDALE:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        m = result['final_metrics']
        print(f"  {appliance}:")
        print(f"    F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")
