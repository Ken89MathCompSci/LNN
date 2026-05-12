"""
Physics-Informed LNN (PINN-LNN) for NILM — UKDALE, seq-to-seq with gated heads.

Extends test_pinn_lnn_s2s_ukdale_specific_splits.py with high-impact improvements:

  1. Separate power_heads (softplus regression) + state_heads (classification logit)
       power      = softplus(power_head(h))          — always non-negative
       state_logit = state_head(h)                   — raw logit for BCEWithLogits
       gated_power = sigmoid(state_logit) × power    — zero when OFF, smooth otherwise

  2. BCEWithLogitsLoss on state logits for ALL four appliances
       - MW and WD now enabled (BCE_LAMBDA > 0)
       - Uses pos_weight instead of per-sample weight tensor (numerically cleaner)

  3. Smoothness loss: ((pred[:, 1:] − pred[:, :-1])²).mean()
       Penalises step-to-step jitter in power predictions.

  4. Two-sided physics loss: |Σpred_raw − agg_raw| / (agg_raw + 1)
       Scale-invariant and always active (no dead zone unlike one-sided ReLU).

  5. Two-phase early stopping:
       Warmup  → val MSE  (best model = lowest MSE)
       BCE phase → avg val F1  (counter resets at phase switch; best_val_mse kept)
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

LAMBDA_PHYS   = 0.01    # physics loss weight
LAMBDA_SMOOTH = 0.01    # smoothness loss weight
EPSILON_W     = 50.0    # not used in two-sided loss, kept for reference
WARMUP_EPOCHS = 20      # Stage 1: MSE+phys only; BCE added after this epoch

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

# All four appliances enabled; DW alpha raised (2→10) and lambda raised (0.5→1.0)
# to overcome DW's near-zero scaled threshold (0.0038) that caused always-ON in v1
BCE_LAMBDA = {'dish washer': 1.0, 'fridge': 0.3, 'microwave': 0.2, 'washer dryer': 0.2}
BCE_ALPHA  = {'dish washer': 10.0, 'fridge': 1.5, 'microwave': 15.0, 'washer dryer': 2.0}


# ---------------------------------------------------------------------------
# Sequence reconstruction
# ---------------------------------------------------------------------------

def reconstruct_sequence(window_preds, stride=STRIDE, window_size=WIN):
    """
    Reconstruct a full power trace from overlapping window predictions by
    averaging all predictions that cover each timestep.

    Args:
        window_preds: (N_windows, WIN, n_apps) ndarray
    Returns:
        (T, n_apps) ndarray  where T = (N_windows - 1) * stride + window_size
    """
    n_windows, _, n_apps = window_preds.shape
    data_len = (n_windows - 1) * stride + window_size
    full = np.zeros((data_len, n_apps), dtype=np.float32)
    cnt  = np.zeros((data_len, n_apps), dtype=np.float32)
    for i in range(n_windows):
        s = i * stride
        full[s:s + window_size] += window_preds[i]
        cnt[s:s + window_size]  += 1
    return full / np.maximum(cnt, 1)


# ---------------------------------------------------------------------------
# Two-sided Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Scale-invariant two-sided penalty over every timestep:
        mean_{batch,t}( |Σ_i p_hat_i_raw(t) − P_agg_raw(t)| / (P_agg_raw(t) + 1) )

    Always active (no dead zone), and small loads don't dominate large loads.
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

    def forward(self, x_scaled, pred_scaled):
        """
        Args:
            x_scaled:    (batch, WIN)         — scaled mains at every timestep
            pred_scaled: (batch, WIN, n_apps) — scaled gated power predictions
        Returns:
            scalar physics loss
        """
        x_raw = x_scaled * self.x_range + self.x_min           # (batch, WIN)
        p_raw = pred_scaled * self.y_ranges + self.y_mins       # (batch, WIN, n_apps)
        p_sum = p_raw.sum(dim=-1)                               # (batch, WIN)
        return ((p_sum - x_raw).abs() / (x_raw + 1.0)).mean()


# ---------------------------------------------------------------------------
# Seq-to-Seq PINN-LNN with Separate State + Power Heads
# ---------------------------------------------------------------------------

class PhysicsInformedLiquidNetworkModel(nn.Module):
    """
    Shared AdvancedLiquidTimeLayer — hidden state collected at every step.

    Two separate head sets per appliance:
        power_heads[i]:  hidden → softplus → power magnitude (always ≥ 0)
        state_heads[i]:  hidden → raw logit (used for BCEWithLogitsLoss)

    Gated output:
        gated_power = sigmoid(state_logit) × softplus(power_head_output)

    Returns: (gated_power, state_logits)
        gated_power:   (batch, WIN, n_apps) — for MSE, physics, smoothness
        state_logits:  (batch, WIN, n_apps) — for BCEWithLogitsLoss
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        # LNN core
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

        # Separate heads per appliance
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
            gated_power:  (batch, seq_len, n_apps)
            state_logits: (batch, seq_len, n_apps)
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_seq = []
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

            h_seq.append(h)

        H = torch.stack(h_seq, dim=1)   # (batch, seq_len, hidden)
        H = self.norm(H)

        # Power heads: non-negative via softplus
        power = torch.cat(
            [F.softplus(head(H)) for head in self.power_heads], dim=-1
        )  # (batch, seq_len, n_apps)

        # State heads: raw logits
        state_logits = torch.cat(
            [head(H) for head in self.state_heads], dim=-1
        )  # (batch, seq_len, n_apps)

        # Gated power: zero when OFF, smooth regression when ON
        gated_power = torch.sigmoid(state_logits) * power   # (batch, seq_len, n_apps)

        return gated_power, state_logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)   # (N, WIN, 1)
        self.Y = torch.FloatTensor(Y)   # (N, WIN, n_apps)

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
    Seq-to-seq: Y contains appliance labels at EVERY timestep in each window.

    Returns:
        X: (N, WIN, 1)       — mains window
        Y: (N, WIN, n_apps)  — appliance power at every timestep
    """
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        window_labels = np.stack(
            [app_vals[app][i:i + window_size] for app in APPLIANCES],
            axis=-1,
        )   # (WIN, n_apps)
        Y.append(window_labels)
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),   # (N, WIN, n_apps)
    )


# ---------------------------------------------------------------------------
# Metrics on reconstructed full sequences
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true_recon, y_pred_recon, y_scalers):
    """
    Evaluate on the full reconstructed time-series (after overlap averaging).

    Args:
        y_true_recon: (T, n_apps) — reconstructed true sequence (scaled)
        y_pred_recon: (T, n_apps) — reconstructed predicted sequence (scaled)
        y_scalers: list of MinMaxScaler, one per appliance
    Returns:
        dict {appliance_name: metrics_dict}
    """
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(
            y_true_recon[:, i:i+1].clip(0, 1)).flatten()
        raw_pred = y_scalers[i].inverse_transform(
            y_pred_recon[:, i:i+1].clip(0, 1)).flatten()
        metrics[app] = calculate_nilm_metrics(
            raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, dt=0.1,
                     lambda_phys=LAMBDA_PHYS,
                     lambda_smooth=LAMBDA_SMOOTH):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"λ_phys={lambda_phys}  λ_smooth={lambda_smooth}  "
          f"hidden={hidden_size}  dt={dt}")

    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # ── Scaling ──
    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    # Y is (N, WIN, n_apps) — fit scaler on all N*WIN values per appliance
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        n_tr, n_va, n_te = Y_tr.shape[0], Y_va.shape[0], Y_te.shape[0]
        Y_tr[:, :, i] = ys.fit_transform(
            Y_tr[:, :, i].reshape(-1, 1)).reshape(n_tr, WIN)
        Y_va[:, :, i] = ys.transform(
            Y_va[:, :, i].reshape(-1, 1)).reshape(n_va, WIN)
        Y_te[:, :, i] = ys.transform(
            Y_te[:, :, i].reshape(-1, 1)).reshape(n_te, WIN)
        y_scalers.append(ys)

    # Scaled ON/OFF thresholds for BCE
    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")
    print("Scaled thresholds:")
    for i, app in enumerate(APPLIANCES):
        print(f"  {app:<14}: {thresholds_scaled[i]:.4f}  "
              f"(BCE_LAMBDA={BCE_LAMBDA[app]}, BCE_ALPHA={BCE_ALPHA[app]})")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = PhysicsInformedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_criterion  = nn.MSELoss()
    phys_criterion = PhysicsConsistencyLoss(
        x_scaler, y_scalers, APPLIANCES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [], 'train_smooth': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_mse = float('inf')
    best_val_f1  = -float('inf')
    best_state   = None
    counter      = 0
    bce_phase    = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting PINN-LNN S2S-Gated training...")

    for epoch in range(EPOCHS):

        # Detect transition into BCE phase
        if not bce_phase and epoch >= WARMUP_EPOCHS:
            bce_phase = True
            counter   = 0
            print(f"  [Phase switch] Entering BCE phase at epoch {epoch+1}. "
                  f"Resetting patience counter.")

        # ── Training ──
        model.train()
        ep_mse = ep_phys = ep_smooth = ep_total = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            # xb: (batch, WIN, 1),  yb: (batch, WIN, n_apps)
            optimizer.zero_grad()

            gated_power, state_logits = model(xb)   # both (batch, WIN, n_apps)

            mse_loss    = mse_criterion(gated_power, yb)
            x_all       = xb[:, :, 0]               # (batch, WIN)
            phys_loss   = phys_criterion(x_all, gated_power)
            smooth_loss = ((gated_power[:, 1:, :] - gated_power[:, :-1, :]) ** 2).mean()

            if not bce_phase:
                loss = mse_loss + lambda_phys * phys_loss
            else:
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        thr_s  = thresholds_scaled[i]
                        y_bin  = (yb[:, :, i] > thr_s).float()   # (batch, WIN)
                        pos_w  = torch.tensor(
                            [BCE_ALPHA[app]], dtype=torch.float32, device=device)
                        bce_i  = F.binary_cross_entropy_with_logits(
                            state_logits[:, :, i], y_bin, pos_weight=pos_w)
                        bce_loss = bce_loss + BCE_LAMBDA[app] * bce_i
                loss = (mse_loss
                        + lambda_phys  * phys_loss
                        + lambda_smooth * smooth_loss
                        + bce_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse    += mse_loss.item()
            ep_phys   += phys_loss.item()
            ep_smooth += smooth_loss.item()
            ep_total  += loss.item()
            progress_bar.set_postfix({
                'mse':    f'{mse_loss.item():.5f}',
                'phys':   f'{phys_loss.item():.5f}',
                'smooth': f'{smooth_loss.item():.5f}',
            })

        avg_tr_mse    = ep_mse    / len(tr_loader)
        avg_tr_phys   = ep_phys   / len(tr_loader)
        avg_tr_smooth = ep_smooth / len(tr_loader)
        avg_tr_total  = ep_total  / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_smooth'].append(avg_tr_smooth)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                gated_power, _ = model(xb)

                mse_loss  = mse_criterion(gated_power, yb)
                x_all     = xb[:, :, 0]
                phys_loss = phys_criterion(x_all, gated_power)
                loss      = mse_loss + lambda_phys * phys_loss

                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(gated_power.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_mse)

        # Reconstruct full val sequence by averaging overlapping predictions
        y_pred_all = np.concatenate(val_preds)   # (N_val, WIN, n_apps)
        y_true_all = np.concatenate(val_trues)

        y_pred_recon = reconstruct_sequence(y_pred_all)   # (T_val, n_apps)
        y_true_recon = reconstruct_sequence(y_true_all)

        per_app_metrics = compute_per_appliance_metrics(
            y_true_recon, y_pred_recon, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        phase_tag = 'BCE' if bce_phase else 'MSE'
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS} [{phase_tag}]  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} "
            f"phys={avg_tr_phys:.5f} sm={avg_tr_smooth:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} "
            f"phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        # ── Two-phase early stopping ──
        if not bce_phase:
            # Warmup: minimise val MSE
            if avg_va_mse < best_val_mse:
                best_val_mse = avg_va_mse
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                counter      = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"  Early stopping (MSE warmup) at epoch {epoch+1}")
                    break
        else:
            # BCE phase: maximise avg val F1
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                counter     = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"  Early stopping (BCE F1) at epoch {epoch+1}")
                    break

    print("Training completed!")

    # ── Test evaluation ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            gated_power, _ = model(xb.to(device))
            test_preds.append(gated_power.cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    # Reconstruct full test sequence
    y_pred_te = np.concatenate(test_preds)   # (N_test, WIN, n_apps)
    y_true_te = np.concatenate(test_trues)

    y_pred_recon = reconstruct_sequence(y_pred_te)
    y_true_recon = reconstruct_sequence(y_true_te)

    test_metrics = compute_per_appliance_metrics(y_true_recon, y_pred_recon, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    avg_f1_test = 0.0
    for app in APPLIANCES:
        m = test_metrics[app]
        avg_f1_test += m['f1']
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")
    print(f"{'Average':<15} {avg_f1_test/len(APPLIANCES):>8.4f}")

    _plot_training(history, test_metrics, save_dir)

    config = {
        'dataset': 'UKDALE',
        'model': 'PhysicsInformedLiquidNetworkModel_S2S_Gated',
        'description': (
            'seq-to-seq with gated heads: sigmoid(state_logit) × softplus(power); '
            'two-sided physics; smoothness loss; two-phase ES (MSE→F1); '
            'BCEWithLogitsLoss on all four appliances'
        ),
        'window_size': WIN,
        'stride': STRIDE,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'lambda_smooth': lambda_smooth,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': BCE_LAMBDA, 'bce_alpha': BCE_ALPHA,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_lnn_s2s_gated_v2_ukdale_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6,
                label='BCE start')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6)
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6)
    plt.title('Physics Loss (two-sided)')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(epochs_x, history['train_smooth'], label='Train Smooth', color='purple')
    plt.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6)
    plt.title('Smoothness Loss')
    plt.xlabel('Epoch'); plt.ylabel('L_smooth')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_s2s_gated_v2_ukdale_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN S2S-Gated UKDALE — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6,
                      label='BCE start')
        ax_f1.set_title(f'{app} — F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.6)
        ax_mae.set_title(f'{app} — MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_s2s_gated_v2_ukdale_per_appliance.png'),
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
    save_dir  = f"models/pinn_lnn_s2s_gated_v2_ukdale_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir      = save_dir,
        hidden_size   = 64,
        dt            = 0.1,
        lambda_phys   = LAMBDA_PHYS,
        lambda_smooth = LAMBDA_SMOOTH,
    )

    print(f"\nResults saved to {save_dir}")
