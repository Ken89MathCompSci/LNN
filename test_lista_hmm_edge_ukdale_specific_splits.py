"""
LISTA + HMM + Edge Detection for NILM — UKDALE specific splits.

Pipeline per appliance:
  1. CUSUM change-point detection on mains windows → edge-density scalar
  2. GaussianHMM trained on training appliance power → state posteriors
  3. Context vector c = [p_hmm (2), edge_density (1)]
  4. LISTAHMMNILMModel:
       x^(0)  = tanh(hmm_proj(c))           ← HMM warm-start
       x^(K)  = K × LISTA(y, x)             ← sparse refinement
       gate   = σ(W_g · [x^(K), c])         ← learned blend
       p̂     = gate · head_lista(x^(K))
              + (1-gate) · head_hmm(c)

Loss:
  L = MSE(p̂, p*) + λ_recon·||y - x@D||² + λ_sparse·||x||₁
    + λ_bce·BCE(σ(p̂/thr), s*)   [after warmup]
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

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    print("WARNING: hmmlearn not installed — run: pip install hmmlearn")
    HMM_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 5
N_ATOMS       = 64
K_LAYERS      = 8
N_HMM_STATES  = 2

LAMBDA_RECON  = 0.1
LAMBDA_SPARSE = 0.01
WARMUP_EPOCHS = 15

CUSUM_THRESHOLD = 50.0
CUSUM_DRIFT     =  5.0

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.3, 'fridge': 0.5, 'microwave': 2.0, 'washer dryer': 2.0}
BCE_ALPHA  = {'dish washer': 2.0, 'fridge': 0.3, 'microwave': 20.0, 'washer dryer': 8.0}

CONTEXT_DIM = N_HMM_STATES + 1   # [p_off, p_on, edge_density]


# ---------------------------------------------------------------------------
# CUSUM edge detection
# ---------------------------------------------------------------------------

def cusum_edge_detection(signal: np.ndarray,
                          threshold: float = CUSUM_THRESHOLD,
                          drift: float     = CUSUM_DRIFT) -> np.ndarray:
    """CUSUM on first differences. Returns binary float32 edge array."""
    edges = np.zeros(len(signal), dtype=np.float32)
    s_pos, s_neg = 0.0, 0.0
    for i in range(1, len(signal)):
        diff  = float(signal[i]) - float(signal[i - 1])
        s_pos = max(0.0, s_pos + diff  - drift)
        s_neg = max(0.0, s_neg - diff  - drift)
        if s_pos > threshold or s_neg > threshold:
            edges[i] = 1.0
            s_pos, s_neg = 0.0, 0.0
    return edges


# ---------------------------------------------------------------------------
# HMM helpers
# ---------------------------------------------------------------------------

def train_appliance_hmm(app_power: np.ndarray,
                         n_states: int = N_HMM_STATES) -> "GaussianHMM":
    """Fit diagonal GaussianHMM; state 0 = OFF (lower mean), state 1 = ON."""
    hmm = GaussianHMM(n_components=n_states, covariance_type='diag',
                      n_iter=200, random_state=42)
    hmm.fit(app_power.reshape(-1, 1).astype(np.float64))

    means = hmm.means_.flatten()
    if means[0] > means[1]:
        order = [1, 0]
        hmm.means_     = hmm.means_[order]
        hmm.covars_    = hmm.covars_[order]
        hmm.startprob_ = hmm.startprob_[order]
        hmm.transmat_  = hmm.transmat_[order][:, order]
    return hmm


def compute_hmm_posteriors(hmm_model, app_windows: np.ndarray) -> np.ndarray:
    """
    app_windows : (N, WIN)  raw appliance power
    Returns     : (N, n_states) posterior at window midpoint
    """
    mid      = WIN // 2
    n_states = hmm_model.n_components
    uniform  = np.ones(n_states, dtype=np.float32) / n_states
    posts    = []
    for seq in app_windows:
        try:
            probs = hmm_model.predict_proba(
                seq.reshape(-1, 1).astype(np.float64))
            posts.append(probs[mid].astype(np.float32))
        except Exception:
            posts.append(uniform.copy())
    return np.array(posts, dtype=np.float32)


def build_context(raw_mains_windows: np.ndarray,
                  raw_app_windows:   np.ndarray,
                  hmm_model) -> np.ndarray:
    """
    Returns (N, CONTEXT_DIM) = [hmm_posterior(2), edge_density(1)].
    Both inputs must be in raw Watts (unscaled).
    """
    hmm_posts    = compute_hmm_posteriors(hmm_model, raw_app_windows)

    N            = len(raw_mains_windows)
    edge_density = np.zeros((N, 1), dtype=np.float32)
    for i, win in enumerate(raw_mains_windows):
        edge_density[i, 0] = float(cusum_edge_detection(win).mean())

    return np.concatenate([hmm_posts, edge_density], axis=1)


# ---------------------------------------------------------------------------
# Soft threshold  /  LISTA layer
# ---------------------------------------------------------------------------

def soft_threshold(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


class LISTALayer(nn.Module):
    def __init__(self, signal_len: int, n_atoms: int):
        super().__init__()
        self.We        = nn.Linear(signal_len, n_atoms, bias=True)
        self.Wr        = nn.Linear(n_atoms,    n_atoms, bias=False)
        self.threshold = nn.Parameter(torch.full((n_atoms,), 0.1))

    def forward(self, y: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        z = self.We(y) + self.Wr(x_prev)
        return soft_threshold(z, F.softplus(self.threshold))


# ---------------------------------------------------------------------------
# LISTA-HMM-Edge model
# ---------------------------------------------------------------------------

class LISTAHMMNILMModel(nn.Module):
    """
    context  = [p_hmm (n_hmm_states), edge_density (1)]
    x^(0)    = tanh(hmm_proj(context))
    x^(K)    ← K LISTA iterations
    gate     = σ(gate_net([x^(K), context]))
    p̂       = gate · head_lista(x^(K))  +  (1-gate) · head_hmm(context)
    """
    def __init__(self, signal_len: int = WIN, n_atoms: int = N_ATOMS,
                 k_layers: int = K_LAYERS, context_dim: int = CONTEXT_DIM):
        super().__init__()
        self.signal_len  = signal_len
        self.n_atoms     = n_atoms
        self.context_dim = context_dim

        self.hmm_proj   = nn.Linear(context_dim, n_atoms)
        self.layers     = nn.ModuleList([
            LISTALayer(signal_len, n_atoms) for _ in range(k_layers)
        ])
        self.D          = nn.Parameter(torch.randn(n_atoms, signal_len) * 0.01)
        self.head_lista = nn.Linear(n_atoms, 1)
        self.head_hmm   = nn.Linear(context_dim, 1)
        self.gate_net   = nn.Linear(n_atoms + context_dim, 1)

    def forward(self, y: torch.Tensor, context: torch.Tensor):
        if y.dim() == 3:
            y = y.squeeze(-1)

        x = torch.tanh(self.hmm_proj(context))
        for layer in self.layers:
            x = layer(y, x)

        lista_power = self.head_lista(x)
        hmm_power   = self.head_hmm(context)
        gate        = torch.sigmoid(
            self.gate_net(torch.cat([x, context], dim=1))
        )
        return gate * lista_power + (1.0 - gate) * hmm_power, x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.D


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKDALEHMMDataset(torch.utils.data.Dataset):
    def __init__(self, X, context, y):
        self.X       = torch.FloatTensor(X)
        self.context = torch.FloatTensor(context)
        self.y       = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.context[idx], self.y[idx]


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


def create_sequences(data, appliance_name, window_size=WIN):
    mains    = data['main'].values
    app_vals = data[appliance_name].values
    X, y, app_wins = [], [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        y.append(app_vals[i + window_size // 2])
        app_wins.append(app_vals[i:i + window_size])
    return (
        np.array(X,        dtype=np.float32).reshape(-1, window_size, 1),
        np.array(y,        dtype=np.float32).reshape(-1, 1),
        np.array(app_wins, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name,
                       n_atoms=N_ATOMS, k_layers=K_LAYERS,
                       epochs=EPOCHS, lr=LR, patience=PATIENCE,
                       lambda_recon=LAMBDA_RECON,
                       lambda_sparse=LAMBDA_SPARSE,
                       save_dir='models/lista_hmm_edge_ukdale'):
    assert HMM_AVAILABLE, "hmmlearn is required — pip install hmmlearn"
    os.makedirs(save_dir, exist_ok=True)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold  = THRESHOLDS[appliance_name]
    bce_lambda = BCE_LAMBDA[appliance_name]
    bce_alpha  = BCE_ALPHA[appliance_name]

    print(f"\nDevice: {device}  |  appliance: {appliance_name}")
    print(f"n_atoms={n_atoms}  K={k_layers}  context_dim={CONTEXT_DIM}  "
          f"λ_recon={lambda_recon}  λ_sparse={lambda_sparse}  "
          f"λ_bce={bce_lambda}  α_bce={bce_alpha}  warmup={WARMUP_EPOCHS}")

    X_tr_raw, y_tr_raw, app_tr = create_sequences(data_dict['train'], appliance_name)
    X_va_raw, y_va_raw, app_va = create_sequences(data_dict['val'],   appliance_name)
    X_te_raw, y_te_raw, app_te = create_sequences(data_dict['test'],  appliance_name)

    print(f"  Training HMM on {len(app_tr)} windows...")
    hmm_model = train_appliance_hmm(app_tr.flatten())
    print(f"  HMM means (OFF/ON): {hmm_model.means_.flatten().tolist()}")

    mains_tr_raw = X_tr_raw.squeeze(-1)
    mains_va_raw = X_va_raw.squeeze(-1)
    mains_te_raw = X_te_raw.squeeze(-1)

    print("  Building context vectors (HMM posteriors + CUSUM edge density)...")
    ctx_tr = build_context(mains_tr_raw, app_tr, hmm_model)
    ctx_va = build_context(mains_va_raw, app_va, hmm_model)
    ctx_te = build_context(mains_te_raw, app_te, hmm_model)
    print(f"  Context shape: {ctx_tr.shape}")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_tr = x_scaler.fit_transform(X_tr_raw.reshape(-1, 1)).reshape(X_tr_raw.shape)
    X_va = x_scaler.transform(X_va_raw.reshape(-1, 1)).reshape(X_va_raw.shape)
    X_te = x_scaler.transform(X_te_raw.reshape(-1, 1)).reshape(X_te_raw.shape)

    y_tr = y_scaler.fit_transform(y_tr_raw)
    y_va = y_scaler.transform(y_va_raw)
    y_te = y_scaler.transform(y_te_raw)

    thr_scaled = (threshold - float(y_scaler.data_min_[0])) / float(y_scaler.data_range_[0])

    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        UKDALEHMMDataset(X_tr, ctx_tr, y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        UKDALEHMMDataset(X_va, ctx_va, y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        UKDALEHMMDataset(X_te, ctx_te, y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = LISTAHMMNILMModel(
        signal_len=WIN, n_atoms=n_atoms, k_layers=k_layers, context_dim=CONTEXT_DIM
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history    = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val   = float('inf')
    best_state = None
    counter    = 0

    def _compute_loss(power, x, xb, yb, epoch):
        mse_loss    = F.mse_loss(power, yb)
        recon_loss  = F.mse_loss(model.reconstruct(x), xb.squeeze(-1))
        sparse_loss = x.abs().mean()
        if epoch < WARMUP_EPOCHS:
            return mse_loss + lambda_recon * recon_loss + lambda_sparse * sparse_loss
        pred_prob = torch.sigmoid(power / (thr_scaled + 1e-8))
        y_bin     = (yb > thr_scaled).float()
        w         = torch.where(y_bin == 1,
                                torch.full_like(y_bin, bce_alpha),
                                torch.ones_like(y_bin))
        bce_loss  = F.binary_cross_entropy(
            pred_prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
        return (mse_loss
                + lambda_recon  * recon_loss
                + lambda_sparse * sparse_loss
                + bce_lambda    * bce_loss)

    print(f"  Starting training for {appliance_name}...")
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, cb, yb in bar:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            power, x = model(xb, cb)
            loss     = _compute_loss(power, x, xb, yb, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_tr = ep_loss / len(tr_loader)
        history['train_loss'].append(avg_tr)

        model.eval()
        vl_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, cb, yb in va_loader:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                power, x  = model(xb, cb)
                loss       = _compute_loss(power, x, xb, yb, epoch)
                vl_loss   += loss.item()
                all_preds.append(power.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        avg_va = vl_loss / len(va_loader)
        history['val_loss'].append(avg_va)
        scheduler.step(avg_va)

        raw_true = y_scaler.inverse_transform(
            np.concatenate(all_trues).reshape(-1, 1)).flatten()
        raw_pred = y_scaler.inverse_transform(
            np.concatenate(all_preds).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(raw_true, raw_pred, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr:.5f}  val={avg_va:.5f}  "
              f"F1={metrics['f1']:.4f}  MAE={metrics['mae']:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_va < best_val:
            best_val   = avg_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
            save_model(
                model,
                {'signal_len': WIN, 'n_atoms': n_atoms, 'k_layers': k_layers,
                 'context_dim': CONTEXT_DIM},
                {'lr': lr, 'epochs': epochs, 'patience': patience,
                 'appliance': appliance_name},
                metrics,
                os.path.join(save_dir,
                    f"lista_hmm_edge_ukdale_{appliance_name.replace(' ', '_')}_best.pth")
            )
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("  Training complete.")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, cb, yb in te_loader:
            power, _ = model(xb.to(device), cb.to(device))
            test_preds.append(power.cpu().numpy())
            test_trues.append(yb.numpy())

    raw_true_te = y_scaler.inverse_transform(
        np.concatenate(test_trues).reshape(-1, 1)).flatten()
    raw_pred_te = y_scaler.inverse_transform(
        np.concatenate(test_preds).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_true_te, raw_pred_te, threshold=threshold)

    print(f"\n  Test  F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}")

    _plot_results(history, test_metrics, appliance_name, save_dir)

    config = {
        'appliance':  appliance_name,
        'dataset':    'UKDALE',
        'model':      'LISTAHMMNILMModel',
        'description': ('K-layer LISTA warm-started from GaussianHMM posterior + '
                        'CUSUM edge density; gated blend with HMM direct head'),
        'hmm_params': {
            'n_states': N_HMM_STATES,
            'off_mean': float(hmm_model.means_[0, 0]),
            'on_mean':  float(hmm_model.means_[1, 0]),
        },
        'cusum_params': {'threshold': CUSUM_THRESHOLD, 'drift': CUSUM_DRIFT},
        'model_params': {'signal_len': WIN, 'n_atoms': n_atoms, 'k_layers': k_layers,
                         'context_dim': CONTEXT_DIM},
        'train_params': {
            'lr': lr, 'epochs': epochs, 'patience': patience,
            'lambda_recon': lambda_recon, 'lambda_sparse': lambda_sparse,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': bce_lambda, 'bce_alpha': bce_alpha,
        },
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': {
                'val_mae_mean': float(np.mean([m['mae'] for m in history['val_metrics']])),
                'val_f1_mean':  float(np.mean([m['f1']  for m in history['val_metrics']])),
            },
        },
    }
    with open(os.path.join(save_dir,
              f'lista_hmm_edge_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, appliance_name, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    val_mae = [m['mae']       for m in history['val_metrics']]
    val_sae = [m['sae']       for m in history['val_metrics']]
    val_f1  = [m['f1']        for m in history['val_metrics']]
    val_p   = [m['precision'] for m in history['val_metrics']]
    val_r   = [m['recall']    for m in history['val_metrics']]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs_x, val_mae, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs_x, val_sae, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(epochs_x, val_f1, label='Val F1',        color='red')
    plt.plot(epochs_x, val_p,  label='Val Precision', color='blue')
    plt.plot(epochs_x, val_r,  label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / P / R — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir,
            f"lista_hmm_edge_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def test_on_all_appliances(n_atoms=N_ATOMS, k_layers=K_LAYERS,
                           epochs=EPOCHS, lr=LR, patience=PATIENCE,
                           lambda_recon=LAMBDA_RECON,
                           lambda_sparse=LAMBDA_SPARSE):
    data_dict   = load_ukdale_specific_splits()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir    = f"models/lista_hmm_edge_ukdale_specific_test_{timestamp}"
    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"LISTA-HMM-Edge on {appliance_name}")
        print(f"{'='*60}")

        app_dir = os.path.join(base_dir, appliance_name.replace(' ', '_'))
        os.makedirs(app_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_on_appliance(
                data_dict,
                appliance_name=appliance_name,
                n_atoms=n_atoms,
                k_layers=k_layers,
                epochs=epochs,
                lr=lr,
                patience=patience,
                lambda_recon=lambda_recon,
                lambda_sparse=lambda_sparse,
                save_dir=app_dir,
            )
            all_results[appliance_name] = {
                'model_path': os.path.join(
                    app_dir,
                    f"lista_hmm_edge_ukdale_{appliance_name.replace(' ', '_')}_best.pth"),
                'final_metrics': {k: float(v) for k, v in test_metrics.items()},
            }
        except Exception as e:
            print(f"Error on {appliance_name}: {e}")
            import traceback; traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset':   'UKDALE',
        'model':     'LISTAHMMNILMModel',
        'description': 'LISTA warm-started from HMM posterior + CUSUM edge density',
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2013-04-12'},
            'validation': {'house': 1, 'date': '2013-04-13'},
            'testing':    {'house': 1, 'date': '2013-04-14'},
        },
        'window_size':  WIN,
        'model_params': {'n_atoms': n_atoms, 'k_layers': k_layers,
                         'context_dim': CONTEXT_DIM},
        'train_params': {
            'epochs': epochs, 'lr': lr, 'patience': patience,
            'lambda_recon': lambda_recon, 'lambda_sparse': lambda_sparse,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': BCE_LAMBDA,
            'bce_alpha': BCE_ALPHA,
        },
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nLISTA-HMM-Edge UKDALE testing complete. Results in {base_dir}\n")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
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
    if not HMM_AVAILABLE:
        print("Install hmmlearn first: pip install hmmlearn")
        sys.exit(1)

    for f in ['data/ukdale/train_small.pkl',
              'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    results = test_on_all_appliances(
        n_atoms=N_ATOMS, k_layers=K_LAYERS,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        lambda_recon=LAMBDA_RECON, lambda_sparse=LAMBDA_SPARSE,
    )

    print(f"\nSummary — LISTA-HMM-Edge on UKDALE:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        m = result['final_metrics']
        print(f"  {appliance}:")
        print(f"    F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")
