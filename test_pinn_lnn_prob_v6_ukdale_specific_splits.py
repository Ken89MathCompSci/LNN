"""
Probabilistic Physics-Informed LNN (Prob-PINN-LNN) v6 — UKDALE specific splits
================================================================================
Fixes the three failure modes of v5 (avg test F1=0.345 vs base 0.673):

v5 failure analysis:
    1. Fix4 (log1p) raised scaled thresholds — MW threshold jumped from ~0.009
       to 0.3244, so NLL warmup never crossed it → MW/DW F1=0.000 every warmup epoch.
    2. Fix5 reset bug — best_val_nll=inf at BCE start made epoch 20 (warmup end)
       the BCE best immediately; counter hit PATIENCE=20 by epoch 40 with 0 BCE progress.
    3. Fix1 alpha=0.3 too aggressive — combined with the saved epoch-20 always-OFF
       model, caused DW R=0.012 on test.

v6 fixes (based on base prob model — no log1p):

Fix A — DW/FR BCE_ALPHA moderated (replaces v5's too-aggressive flip)
    base:  BCE_ALPHA = {DW: 2.0, FR: 1.5, MW: 15.0, WD: 2.0}
    v6:    BCE_ALPHA = {DW: 0.8, FR: 0.8, MW: 15.0, WD: 2.0}
    DW/FR alpha < 1 means OFF samples get 1/0.8 = 1.25× more gradient than ON.
    Sufficient to nudge mu below the very low raw-MinMax threshold without
    suppressing recall to near zero (as alpha=0.3 did).

Fix B — Two-sided physics loss (same as v5 Fix2, but no expm1 — no log1p here)
    |Σ mu_i_raw − P_agg_raw| / sqrt(Σ sigma_i_raw^2)
    Symmetric, always active, no dead ReLU zone.

Fix C — Sigma clamp [0.01, 5.0] (same as v5 Fix3)
    sigma = softplus(log_s).clamp(0.01, 5.0) + 1e-6
    Prevents sigma from absorbing errors (upper) or NLL spikes (lower).

Fix D — Early stopping: val NLL throughout, counter reset only (no inf reset)
    Warmup tracks best val NLL and saves checkpoint.
    At warmup→BCE transition: counter resets to 0, best_val_nll kept at warmup best.
    BCE gets PATIENCE epochs to improve over the warmup NLL.
    If BCE never beats warmup NLL, the warmup checkpoint is used for test.

Base model results (for reference):
    dish washer   F1=0.202  P=0.113  R=1.000   ← always-ON, target improvement
    fridge        F1=0.582  P=0.410  R=1.000   ← over-predicts ON
    microwave     F1=0.945  P=0.962  R=0.928   ← excellent, must not regress
    washer dryer  F1=0.962  P=0.965  R=0.959   ← excellent, must not regress
    avg           F1=0.673
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

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 5

LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0   # not used in two-sided loss; kept for API compat
WARMUP_EPOCHS = 20

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.5, 'fridge': 0.3, 'microwave': 0.1, 'washer dryer': 0.2}

# Fix A: alpha < 1 upweights the OFF class for always-ON appliances.
# DW/FR: 1/0.8 ≈ 1.25× more gradient on OFF samples vs ON.
# MW/WD unchanged — already excellent in base model.
BCE_ALPHA = {'dish washer': 0.8, 'fridge': 0.8, 'microwave': 15.0, 'washer dryer': 2.0}


# ---------------------------------------------------------------------------
# Probabilistic Physics Consistency Loss  (Fix B — two-sided, raw MinMax)
# ---------------------------------------------------------------------------

class ProbabilisticPhysicsLoss(nn.Module):
    """
    Two-sided uncertainty-normalised physics reconstruction penalty:
        |Σ mu_i_raw − P_agg_raw| / sqrt(Σ sigma_i_raw^2)

    Symmetric and always active (no dead ReLU zone).
    Raw MinMaxScaler inverse (no log1p): scaled → linear MinMax⁻¹ → raw Watts.
    """

    def __init__(self, x_scaler, y_scalers, appliances, epsilon_w=EPSILON_W):
        super().__init__()

        x_min   = float(x_scaler.data_min_[0])
        x_range = float(x_scaler.data_range_[0])
        self.register_buffer('x_min',   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(x_range, dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, mu_scaled, sigma_scaled):
        """
        x_mid_scaled : (batch,)        — MinMax scaled mains midpoint
        mu_scaled    : (batch, n_apps) — predicted means in scaled space
        sigma_scaled : (batch, n_apps) — predicted stds  in scaled space
        """
        x_raw     = x_mid_scaled * self.x_range + self.x_min    # (batch,)
        mu_raw    = mu_scaled    * self.y_ranges + self.y_mins   # (batch, n_apps)
        sigma_raw = sigma_scaled * self.y_ranges                 # (batch, n_apps)

        mu_sum    = mu_raw.sum(dim=1)                            # (batch,)
        sigma_sum = torch.sqrt((sigma_raw ** 2).sum(dim=1) + 1e-8)

        # Fix B: two-sided symmetric penalty
        return ((mu_sum - x_raw).abs() / (sigma_sum + 1e-8)).mean()


# ---------------------------------------------------------------------------
# Model  (Fix C — sigma clamp)
# ---------------------------------------------------------------------------

class ProbPINNLiquidNetworkModel(nn.Module):
    """
    Shared AdvancedLiquidTimeLayer encoder with separate mu and sigma heads.

    Fix C: sigma = softplus(log_s).clamp(0.01, 5.0) + 1e-6
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

        self.mu_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

        self.sigma_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])
        for head in self.sigma_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, -2.0)

    def forward(self, x):
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

            f_t = torch.tanh(input_proj + rec_proj)
            dh  = ((-h / tau) + gate * f_t) * self.dt
            h   = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)

        mu    = torch.cat([head(h) for head in self.mu_heads],    dim=1)
        log_s = torch.cat([head(h) for head in self.sigma_heads], dim=1)
        # Fix C: clamp prevents sigma collapse (lower) and error absorption (upper)
        sigma = F.softplus(log_s).clamp(0.01, 5.0) + 1e-6

        return mu, sigma


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
    print("Loading UKDALE data...")
    with open('data/ukdale/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/ukdale/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/ukdale/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train: {train_data.index.min()} → {train_data.index.max()}")
    print(f"Val  : {val_data.index.min()} → {val_data.index.max()}")
    print(f"Test : {test_data.index.min()} → {test_data.index.max()}")
    print(f"Columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=WIN):
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


# ---------------------------------------------------------------------------
# Metrics  (raw MinMaxScaler — no expm1)
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_mu, y_scalers):
    """y_true, y_mu: (N, n_appliances) MinMax scaled. Metrics on inverse-transformed Watts."""
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_mu[:,   i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn_prob_model(data_dict, save_dir,
                          hidden_size=64, dt=0.1,
                          lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: Prob-PINN-LNN v6  λ_phys={lambda_phys}  ε={epsilon_w}W  "
          f"hidden={hidden_size}  dt={dt}")
    print(f"Fixes: DW/FR alpha=0.8, two-sided physics, sigma clamp, NLL ES (counter-reset only)")

    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # Raw MinMaxScaler — no log1p (log1p raised thresholds too high in v5)
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

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")
    for i, app in enumerate(APPLIANCES):
        print(f"  threshold {app}: {THRESHOLDS[app]:.1f} W → scaled={thresholds_scaled[i]:.4f}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = ProbPINNLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    nll_criterion  = nn.GaussianNLLLoss(full=False, reduction='mean')
    phys_criterion = ProbabilisticPhysicsLoss(
        x_scaler, y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_nll': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_nll':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
        'val_sigma': [],
    }
    best_val_nll = float('inf')
    best_state   = None
    counter      = 0
    bce_phase    = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting Prob-PINN-LNN v6 training...")

    for epoch in range(EPOCHS):
        model.train()
        ep_nll = ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            mu, sigma = model(xb)

            nll_loss = nll_criterion(mu, yb, sigma ** 2)
            mse_mon  = F.mse_loss(mu.detach(), yb).item()

            x_mid     = xb[:, WIN // 2, 0]
            phys_loss = phys_criterion(x_mid, mu, sigma)

            if epoch < WARMUP_EPOCHS:
                loss = nll_loss
            else:
                # Fix A: alpha < 1 upweights OFF samples for DW/FR
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_i = mu[:, i].clamp(1e-7, 1 - 1e-7)
                        thr_s  = thresholds_scaled[i]
                        y_bin  = (yb[:, i] > thr_s).float()
                        w      = torch.where(y_bin == 1,
                                             torch.full_like(y_bin, BCE_ALPHA[app]),
                                             torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_i, y_bin, weight=w)
                loss = nll_loss + lambda_phys * phys_loss + bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_nll   += nll_loss.item()
            ep_mse   += mse_mon
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'nll':   f'{nll_loss.item():.4f}',
                'sigma': f'{sigma.mean().item():.4f}',
            })

        avg_tr_nll   = ep_nll   / len(tr_loader)
        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_nll'].append(avg_tr_nll)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_nll = vl_mse = vl_phys = vl_total = 0.0
        val_mus, val_sigmas, val_trues = [], [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model(xb)

                nll_loss  = nll_criterion(mu, yb, sigma ** 2)
                mse_mon   = F.mse_loss(mu, yb).item()
                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, mu, sigma)

                # Match training schedule: no physics in val during warmup
                if bce_phase:
                    val_loss = nll_loss + lambda_phys * phys_loss
                else:
                    val_loss = nll_loss

                vl_nll   += nll_loss.item()
                vl_mse   += mse_mon
                vl_phys  += phys_loss.item()
                vl_total += val_loss.item()
                val_mus.append(mu.cpu().numpy())
                val_sigmas.append(sigma.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_nll   = vl_nll   / len(va_loader)
        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_nll'].append(avg_va_nll)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_nll)

        mu_all     = np.concatenate(val_mus)
        sigma_all  = np.concatenate(val_sigmas)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(y_true_all, mu_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        mean_sigma = {app: float(sigma_all[:, i].mean()) for i, app in enumerate(APPLIANCES)}
        history['val_sigma'].append(mean_sigma)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_sig = float(sigma_all.mean())

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (nll={avg_tr_nll:.5f} phys={avg_tr_phys:.5f})  "
            f"val_nll={avg_va_nll:.5f}  avgF1={avg_f1:.4f}  avgSigma={avg_sig:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m   = per_app_metrics[app]
            sig = mean_sigma[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  sigma={sig:.4f}")

        # Fix D: val NLL drives both phases; only counter resets at BCE transition.
        # best_val_nll is NOT reset to inf — BCE must genuinely beat warmup's best.
        # This prevents the v5 bug where epoch 20 became BCE best immediately.
        if epoch + 1 == WARMUP_EPOCHS:
            bce_phase = True
            counter   = 0   # fresh PATIENCE window for BCE phase
            print(f"  [switching to BCE phase — counter reset, best_val_nll={best_val_nll:.5f} kept]")

        if avg_va_nll < best_val_nll:
            best_val_nll = avg_va_nll
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
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

    test_mus, test_sigmas, test_trues = [], [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            mu, sigma = model(xb)
            test_mus.append(mu.cpu().numpy())
            test_sigmas.append(sigma.cpu().numpy())
            test_trues.append(yb.numpy())

    y_mu_te    = np.concatenate(test_mus)
    y_sigma_te = np.concatenate(test_sigmas)
    y_true_te  = np.concatenate(test_trues)

    test_metrics = compute_per_appliance_metrics(y_true_te, y_mu_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8} {'mean_s':>8} {'std_s':>7}")
    print("-" * 82)
    for i, app in enumerate(APPLIANCES):
        m   = test_metrics[app]
        sig = y_sigma_te[:, i]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f} "
              f"{sig.mean():>8.4f} {sig.std():>7.4f}")

    avg_f1_test = np.mean([test_metrics[a]['f1'] for a in APPLIANCES])
    print(f"\n  avg test F1 = {avg_f1_test:.4f}  (base prob: 0.6727)")

    _plot_training(history, test_metrics, save_dir)
    _plot_uncertainty(y_true_te, y_mu_te, y_sigma_te, y_scalers, save_dir)

    sigma_stats = {
        app: {
            'mean': float(y_sigma_te[:, i].mean()),
            'std':  float(y_sigma_te[:, i].std()),
            'min':  float(y_sigma_te[:, i].min()),
            'max':  float(y_sigma_te[:, i].max()),
        }
        for i, app in enumerate(APPLIANCES)
    }

    config = {
        'dataset':  'UKDALE',
        'model':    'ProbPINNLiquidNetworkModel',
        'description': (
            'heteroscedastic Gaussian PINN-LNN v6; '
            'GaussianNLL + two-sided probabilistic physics loss + BCE; '
            'FixA: DW/FR alpha=0.8 (slight OFF upweight); '
            'FixB: |Σmu-agg|/σ two-sided physics; '
            'FixC: sigma clamp [0.01,5.0]; '
            'FixD: val NLL ES both phases, counter-reset only at BCE transition'
        ),
        'loss': f'GaussianNLL + {lambda_phys} * |Σmu-agg|/σ + BCE',
        'window_size': WIN,
        'bce_settings': {'lambda': BCE_LAMBDA, 'alpha': BCE_ALPHA},
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
        'test_uncertainty': sigma_stats,
    }
    with open(os.path.join(save_dir, 'pinn_lnn_prob_v6_ukdale_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history, y_sigma_te


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1, label='BCE start')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(epochs_x, history['train_nll'], label='Train NLL', color='blue')
    plt.plot(epochs_x, history['val_nll'],   label='Val NLL',   color='red')
    plt.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1)
    plt.title('Gaussian NLL Loss')
    plt.xlabel('Epoch'); plt.ylabel('NLL')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1)
    plt.title('Physics Loss (two-sided |Σmu-agg|/σ)')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, app in enumerate(APPLIANCES):
        sig_series = [m[app] for m in history['val_sigma']]
        plt.plot(epochs_x, sig_series, label=app, color=colors[i], linewidth=1.5)
    plt.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1, label='BCE start')
    plt.title('Mean Val sigma per Appliance')
    plt.xlabel('Epoch'); plt.ylabel('sigma (scaled)')
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v6_ukdale_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Prob-PINN-LNN v6 UKDALE — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.set_title(f'{app} — F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.set_title(f'{app} — MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v6_ukdale_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_uncertainty(y_true, y_mu, y_sigma, y_scalers, save_dir):
    """Per-appliance: sigma histogram (left) + |error| vs sigma scatter (right)."""
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Prob-PINN-LNN v6 UKDALE — Test Uncertainty Analysis', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        i = APPLIANCES.index(app)

        raw_true  = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_mu    = y_scalers[i].inverse_transform(y_mu[:,   i:i+1]).flatten()
        sigma_raw = y_sigma[:, i] * float(y_scalers[i].data_range_[0])  # approx Watts
        abs_err   = np.abs(raw_true - raw_mu)

        ax_hist = axes[row][0]
        ax_scat = axes[row][1]

        ax_hist.hist(sigma_raw, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
        ax_hist.axvline(sigma_raw.mean(), color='red', linestyle='--',
                        label=f'mean={sigma_raw.mean():.1f} W')
        ax_hist.set_title(f'{app} — sigma distribution (W)')
        ax_hist.set_xlabel('sigma (W)'); ax_hist.set_ylabel('Count')
        ax_hist.legend(); ax_hist.grid(True, alpha=0.3)

        ax_scat.scatter(sigma_raw, abs_err, alpha=0.3, s=5, color='steelblue')
        ax_scat.set_title(f'{app} — |error| vs sigma')
        ax_scat.set_xlabel('sigma (W)'); ax_scat.set_ylabel('|mu - y_true| (W)')
        ax_scat.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v6_ukdale_uncertainty.png'),
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
    save_dir  = f"models/pinn_lnn_prob_v6_ukdale_{timestamp}"

    data_dict = load_data()

    test_metrics, history, test_sigmas = train_pinn_prob_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        epsilon_w   = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
