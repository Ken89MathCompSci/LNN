"""
Physics-Informed LNN (PINN-LNN) for NILM — REDD specific splits.

Same architecture as test_pinn_lnn_ukdale_specific_splits.py but adapted for
the REDD dataset, with one key improvement: ε is computed adaptively from the
training data rather than being a fixed constant.

Adaptive ε:
    residual_n = max(0, P_agg_n - Σ p_i_true_n)
    ε = μ_residual + k · σ_residual

    A residual of zero means the monitored appliances account for all power;
    a positive residual means some background load is present. Using μ + kσ
    sets ε just above the typical background noise level so the physics
    constraint only fires on genuine over-predictions.

Architecture:
    Input (batch, WIN, 1)  — scaled mains window
         ↓
    Shared AdvancedLiquidTimeLayer encoder (adaptive tau, input-dependent gate)
         ↓
    LayerNorm(hidden)
         ↓
    ┌────┬────┬────┬────┐
    │ DW │ FR │ MW │ WD │  — one Linear head per appliance
    └────┴────┴────┴────┘
    output: (batch, 4)

Loss:
    Stage 1 (epochs 1–WARMUP_EPOCHS): MSE only
    Stage 2 (remaining epochs):       MSE + λ·L_phys + weighted BCE
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

EPOCHS        = 120
PATIENCE      = 30     # extended: need headroom for BCE to fire and improve MW
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 5

LAMBDA_PHYS   = 0.01   # physics loss weight
BETA_PHYS     = 0.1    # underestimation penalty weight (two-sided physics loss)
ALPHA_L1      = 0.3    # L1 fraction in regression loss (1-ALPHA_L1 is MSE fraction)
EPSILON_K     = 0.5    # k in ε = μ + k·σ
EPSILON_CAP   = 150.0  # tighter ceiling — forces active physics constraint
WARMUP_EPOCHS = 15     # shorter warmup so BCE fires while there are still epochs left
BCE_ANNEAL    = 10     # ramp BCE from 0→full weight over this many epochs after warmup

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  50.0,   # DW idles near 0, active draw is 1200W+
    'fridge':       50.0,   # fridge compressor draw is 100-200W
    'microwave':    50.0,   # microwave is either OFF or 600W+
    'washer dryer': 200.0,  # WD is off or drawing 300-500W+ — 0.5W caused always-ON
}

# BCE applied to state_head (sigmoid) — separated from regression so it can't
# distort power magnitude.
# WD BCE disabled: positive class weight was biasing state_head toward always-ON;
#                  regression + physics loss alone handles WD detection.
BCE_LAMBDA = {'dish washer': 0.05, 'fridge': 0.05, 'microwave': 0.20, 'washer dryer': 0.0}
BCE_ALPHA  = {'dish washer': 1.5,  'fridge': 1.5,  'microwave': 5.0,  'washer dryer': 1.5}


# ---------------------------------------------------------------------------
# Adaptive ε
# ---------------------------------------------------------------------------

def compute_epsilon_schedule(train_data, k=EPSILON_K, cap=EPSILON_CAP, min_samples=20):
    """
    Returns a (24,) float32 array — one ε per hour of day.

    For each hour h:
        ε(h) = min(μ_residual(h) + k·σ_residual(h), cap)

    Hours with fewer than min_samples fall back to the global ε, which avoids
    unstable estimates from sparse night-time data.  The schedule is printed
    at startup so it is easy to audit.
    """
    mains   = train_data['main'].values.astype(np.float64)
    app_sum = np.zeros(len(mains), dtype=np.float64)
    for app in APPLIANCES:
        app_sum += train_data[app].values.astype(np.float64)

    residuals = np.maximum(0.0, mains - app_sum)
    hours     = train_data.index.hour   # local hour (0-23), works with tz-aware index

    global_eps = min(float(residuals.mean() + k * residuals.std()), cap)
    schedule   = np.full(24, global_eps, dtype=np.float32)

    for h in range(24):
        mask = (hours == h)
        if mask.sum() >= min_samples:
            r = residuals[mask]
            schedule[h] = float(min(r.mean() + k * r.std(), cap))

    print("  Per-hour ε schedule (W):")
    for h in range(0, 24, 6):
        row = "  " + "  ".join(
            f"{h+j:02d}h={schedule[h+j]:.0f}" for j in range(6) if h + j < 24
        )
        print(row)
    return schedule


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty:  ReLU(Σ p_hat_i_raw - P_agg_raw - ε)

    All arithmetic is in raw Watts via differentiable linear inverse-scaling.
    MinMaxScaler inverse: x_raw = x_scaled * data_range_ + data_min_
    """

    def __init__(self, x_scaler, y_scalers, appliances):
        super().__init__()

        x_min   = float(x_scaler.data_min_[0])
        x_range = float(x_scaler.data_range_[0])
        self.register_buffer('x_min',   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(x_range, dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, pred_scaled, eps_per_sample, beta=BETA_PHYS):
        """
        Two-sided soft penalty:
            over:  ReLU(Σp_i - P_agg - ε)          — penalise over-prediction
            under: ReLU(P_agg - Σp_i - ε/2) × β   — penalise severe under-prediction

        Args:
            x_mid_scaled:   (batch,)
            pred_scaled:    (batch, n_apps)
            eps_per_sample: (batch,)  — per-hour upper tolerance (raw W)
            beta:           float     — relative weight of under-penalty
        """
        x_raw = x_mid_scaled * self.x_range + self.x_min           # (batch,)
        p_raw = pred_scaled  * self.y_ranges + self.y_mins          # (batch, n_apps)
        # clamp: no negative watts, no single appliance exceeding aggregate
        p_raw = torch.minimum(p_raw.clamp(min=0.0), x_raw.unsqueeze(1))
        p_sum = p_raw.sum(dim=1)                                    # (batch,)

        over  = F.relu(p_sum - x_raw - eps_per_sample)
        under = F.relu(x_raw - p_sum - eps_per_sample * 0.5)
        return over.mean() + beta * under.mean()


# ---------------------------------------------------------------------------
# Physics-Informed LNN Model
# ---------------------------------------------------------------------------

class PhysicsInformedLiquidNetworkModel(nn.Module):
    """
    Shared AdvancedLiquidTimeLayer encoder → per-appliance linear heads.
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

        # Dual heads per appliance:
        #   power_head — regression (raw scaled watts)
        #   state_head — classification (sigmoid → ON probability)
        # Combined prediction: power * state, so OFF states yield ~0 watts.
        self.power_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])
        self.state_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            combined: (batch, n_appliances) — power × state, used for MSE/physics
            state:    (batch, n_appliances) — ON probability in [0,1], used for BCE
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

            f_t = torch.tanh(input_proj + rec_proj)
            dh  = ((-h / tau) + gate * f_t) * self.dt
            h   = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)
        power = torch.cat([head(h) for head in self.power_heads], dim=1)   # (batch, n_apps)
        state = torch.sigmoid(
            torch.cat([head(h) for head in self.state_heads], dim=1))      # (batch, n_apps)
        return power * state, state


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, H):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.H = torch.LongTensor(H)   # hour-of-day index (0-23) for each window

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.H[idx]


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


def create_sequences(data, window_size=WIN):
    """Midpoint targeting — y[i] is the appliance values at the window centre.
    Also returns H: the local hour-of-day at the midpoint, used for time-aware ε.
    """
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    hours    = data.index.hour          # tz-aware index → local hour (0-23)
    X, Y, H = [], [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
        H.append(int(hours[mid]))
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
        np.array(H, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Per-appliance metrics helper
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
                     hidden_size=64, dt=0.1,
                     lambda_phys=LAMBDA_PHYS, epsilon_k=EPSILON_K,
                     epsilon_cap=EPSILON_CAP):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Per-hour ε schedule from raw training data (before scaling) ──
    eps_schedule = compute_epsilon_schedule(data_dict['train'], k=epsilon_k, cap=epsilon_cap)
    print(f"λ_phys={lambda_phys}  hidden={hidden_size}  dt={dt}")

    # ── Sequences (now also returns hour-of-day H for each window) ──
    X_tr, Y_tr, H_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va, H_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te, H_te = create_sequences(data_dict['test'],  WIN)

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

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr, H_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va, H_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te, H_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model + losses ──
    model = PhysicsInformedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_criterion  = nn.MSELoss()
    l1_criterion   = nn.L1Loss()
    phys_criterion = PhysicsConsistencyLoss(x_scaler, y_scalers, APPLIANCES).to(device)

    # Per-hour ε schedule as a device tensor — indexed by hour each batch
    eps_tensor = torch.tensor(eps_schedule, dtype=torch.float32, device=device)  # (24,)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_loss = -float('inf')   # maximising avg val F1
    best_state    = None
    counter       = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting PINN-LNN training (all appliances simultaneously)...")

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb, hb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            hb = hb.to(device)
            optimizer.zero_grad()

            pred_combined, state_pred = model(xb)    # (batch, n_apps) each

            # 0.7·MSE + 0.3·L1 — aligns training closer to MAE evaluation metric
            reg_loss  = (1 - ALPHA_L1) * mse_criterion(pred_combined, yb) \
                      + ALPHA_L1       * l1_criterion(pred_combined, yb)
            x_mid     = xb[:, WIN // 2, 0]           # (batch,)
            eps_batch = eps_tensor[hb]                # (batch,) — per-sample ε
            phys_loss = phys_criterion(x_mid, pred_combined, eps_batch)

            if epoch < WARMUP_EPOCHS:
                loss = reg_loss + lambda_phys * phys_loss
            else:
                # Anneal BCE from 0→1 over BCE_ANNEAL epochs to prevent gradient spike
                bce_scale = min(1.0, (epoch - WARMUP_EPOCHS + 1) / BCE_ANNEAL)
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        # state_pred already sigmoid — no clamping needed
                        state_i = state_pred[:, i].clamp(1e-7, 1 - 1e-7)
                        thr_s   = thresholds_scaled[i]
                        y_bin   = (yb[:, i] > thr_s).float()
                        w       = torch.where(y_bin == 1,
                                              torch.full_like(y_bin, BCE_ALPHA[app]),
                                              torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            state_i, y_bin, weight=w)
                loss = reg_loss + lambda_phys * phys_loss + bce_scale * bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += reg_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'reg': f'{reg_loss.item():.5f}',
                'phys': f'{phys_loss.item():.5f}',
            })

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb, hb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                hb = hb.to(device)
                pred_combined, _ = model(xb)

                reg_loss  = (1 - ALPHA_L1) * mse_criterion(pred_combined, yb) \
                          + ALPHA_L1       * l1_criterion(pred_combined, yb)
                x_mid     = xb[:, WIN // 2, 0]
                eps_batch = eps_tensor[hb]
                phys_loss = phys_criterion(x_mid, pred_combined, eps_batch)
                loss      = reg_loss + lambda_phys * phys_loss

                vl_mse   += reg_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred_combined.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_mse)   # LR decay on MSE — less noisy than F1

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(
            y_true_all, y_pred_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_f1 > best_val_loss:
            best_val_loss = avg_f1
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
        for xb, yb, _ in te_loader:
            pred_combined, _ = model(xb.to(device))
            test_preds.append(pred_combined.cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    y_pred_te = np.concatenate(test_preds)
    y_true_te = np.concatenate(test_trues)

    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    # ── Plots ──
    _plot_training(history, test_metrics, save_dir)

    # ── Save JSON ──
    config = {
        'dataset': 'REDD',
        'model': 'PhysicsInformedLiquidNetworkModel',
        'description': 'shared AdvancedLiquidTimeLayer encoder + per-appliance heads + adaptive L_phys',
        'loss': f'MSE + {lambda_phys} * PhysicsConsistency(per-hour ε, k={epsilon_k})',
        'window_size': WIN,
        'epsilon_schedule': {f'{h:02d}h': float(eps_schedule[h]) for h in range(24)},
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'epsilon_k': epsilon_k, 'epsilon_cap': epsilon_cap,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_lnn_redd_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss (MSE + λ·Phys)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.title('Physics Consistency Loss')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_redd_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN REDD — Per-Appliance Val Metrics', fontsize=13)

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
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_redd_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for f in ['data/redd/train_small.pkl', 'data/redd/val_small.pkl',
              'data/redd/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_lnn_redd_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        epsilon_k   = EPSILON_K,
        epsilon_cap = EPSILON_CAP,
    )

    print(f"\nResults saved to {save_dir}")
