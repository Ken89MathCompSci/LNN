"""
Probabilistic PINN-LNN v3 — UKDALE specific splits
====================================================
Extends v2 with two targeted fixes for the microwave always-OFF failure.

Root cause of v2 MW failure
-----------------------------
Microwave is ~95% OFF.  GaussianNLL on the full sequence rewards mu_final ≈ 0
(minimises error on the dominant OFF class).  BCE_LAMBDA=0.1 with pos_weight=15
was not strong enough to overcome this: MW p_on stayed at 0.10–0.19 throughout
all 49 epochs, so mu_final = p_on * mu_power never crossed the 10 W threshold.

Additionally, training NLL showed large spikes (26→79→141→173) caused by sigma
occasionally collapsing to near-zero, making the NLL loss blow up.

Fix 1 — Increase BCE_LAMBDA[microwave] 0.1 → 0.4
    Stronger classification pressure pushes p_on higher for ON samples.
    Total BCE contribution for MW is now 0.4 × 15 × BCE = 6× more than v2.

Fix 2 — Clamp sigma: softplus(·).clamp(0.01, 5.0) + 1e-6
    Prevents sigma collapse (σ < 0.01) that caused NLL spikes.
    Upper clamp (5.0) prevents sigma from becoming a trivial error absorber.

Everything else identical to v2:
    — Gated output: mu_final = p_on * mu_power
    — Separate state_head (BCEWithLogitsLoss with pos_weight)
    — log1p + MinMax scaling
    — Probabilistic physics (log1p Jacobian)
    — Two-phase early stopping (warmup NLL → avg F1)
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
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

# v3 change: BCE_LAMBDA['microwave'] raised 0.1 → 0.4
BCE_LAMBDA = {'dish washer': 0.5, 'fridge': 0.3, 'microwave': 0.4, 'washer dryer': 0.2}
BCE_ALPHA  = {'dish washer': 2.0, 'fridge': 1.5, 'microwave': 15.0, 'washer dryer': 2.0}


# ---------------------------------------------------------------------------
# Probabilistic Physics Consistency Loss (log1p-aware)
# ---------------------------------------------------------------------------

class ProbabilisticPhysicsLoss(nn.Module):
    """
    Uncertainty-normalised one-sided physics penalty with log1p inversion.

    sigma_raw = sigma_scaled * y_range * (1 + mu_raw)
               ← Jacobian of expm1 evaluated at mu_raw (chain rule)
    """

    def __init__(self, x_scaler, y_scalers, appliances, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w

        x_min   = float(x_scaler.data_min_[0])
        x_range = float(x_scaler.data_range_[0])
        self.register_buffer('x_min',   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(x_range, dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, mu_scaled, sigma_scaled):
        x_log = x_mid_scaled * self.x_range + self.x_min
        x_raw = torch.expm1(x_log.clamp(min=0))                         # (batch,)

        mu_log = mu_scaled * self.y_ranges + self.y_mins                 # (batch, n_apps)
        mu_raw = torch.expm1(mu_log.clamp(min=0))                        # (batch, n_apps)

        sigma_raw = sigma_scaled * self.y_ranges * (1.0 + mu_raw)        # (batch, n_apps)

        mu_sum    = mu_raw.sum(dim=1)
        violation = F.relu(mu_sum - x_raw - self.epsilon)
        sigma_sum = torch.sqrt((sigma_raw ** 2).sum(dim=1) + 1e-8)
        return (violation / (sigma_sum + 1e-8)).mean()


# ---------------------------------------------------------------------------
# Model: gated probabilistic PINN-LNN
# ---------------------------------------------------------------------------

class GatedProbPINNLiquidNetworkModel(nn.Module):
    """
    Shared LNN encoder with three per-appliance heads:
        state_head  → logit  (classification, BCEWithLogitsLoss)
        mu_head     → mu_power  (ON-state power regression)
        sigma_head  → σ = softplus(·).clamp(0.01, 5.0) + 1e-6  (clamped, v3 fix)

    Forward output:
        mu_final    = sigmoid(state_logit) * mu_power   (gated power mean)
        sigma                                           (clamped uncertainty)
        state_logits                                    (raw, for BCE)
        p_on        = sigmoid(state_logits)             (monitoring)
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
        self.norm        = nn.LayerNorm(hidden_size)

        self.state_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])
        self.mu_heads    = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])
        self.sigma_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])

        for head in self.sigma_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, -2.0)  # softplus(-2) ≈ 0.127

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)
            tau_base   = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod    = torch.sigmoid(self.tau_mod(x_t))
            tau        = (tau_base * tau_mod).clamp(min=self.dt)
            gate       = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t        = torch.tanh(input_proj + rec_proj)
            dh         = ((-h / tau) + gate * f_t) * self.dt
            h          = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)

        state_logits = torch.cat([head(h) for head in self.state_heads], dim=1)
        mu_power     = torch.cat([head(h) for head in self.mu_heads],    dim=1)
        log_s        = torch.cat([head(h) for head in self.sigma_heads],  dim=1)

        p_on     = torch.sigmoid(state_logits)
        sigma    = F.softplus(log_s).clamp(0.01, 5.0) + 1e-6  # v3: clamped
        mu_final = p_on * mu_power

        return mu_final, sigma, state_logits, p_on


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
# Per-appliance metrics (log1p inverse)
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true_scaled, y_mu_scaled, y_scalers):
    """Inverse-scale (expm1 ∘ MinMax⁻¹) before computing NILM metrics."""
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        log_true = y_scalers[i].inverse_transform(y_true_scaled[:, i:i+1])
        log_pred = y_scalers[i].inverse_transform(y_mu_scaled[:,   i:i+1])
        raw_true = np.expm1(np.clip(log_true, 0, None)).flatten()
        raw_pred = np.expm1(np.clip(log_pred, 0, None)).flatten()
        metrics[app] = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn_prob_v3_model(data_dict, save_dir,
                              hidden_size=64, dt=0.1,
                              lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: Gated Prob-PINN-LNN v3  λ_phys={lambda_phys}  ε={epsilon_w}W  "
          f"hidden={hidden_size}  dt={dt}")
    print(f"Scaling: log1p → MinMax")
    print(f"v3 fixes: BCE_LAMBDA[MW]={BCE_LAMBDA['microwave']}  sigma clamped [0.01, 5.0]")

    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # ── log1p + MinMax scaling ──
    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(
        np.log1p(X_tr.reshape(-1, 1))).reshape(X_tr.shape)
    X_va = x_scaler.transform(
        np.log1p(X_va.reshape(-1, 1))).reshape(X_va.shape)
    X_te = x_scaler.transform(
        np.log1p(X_te.reshape(-1, 1))).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(np.log1p(Y_tr[:, i:i+1]))
        Y_va[:, i:i+1] = ys.transform(np.log1p(Y_va[:, i:i+1]))
        Y_te[:, i:i+1] = ys.transform(np.log1p(Y_te[:, i:i+1]))
        y_scalers.append(ys)

    thresholds_scaled = [
        (np.log1p(THRESHOLDS[app]) - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = GatedProbPINNLiquidNetworkModel(
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
        'train_loss': [], 'train_nll': [], 'train_bce': [], 'train_phys': [],
        'val_loss':   [], 'val_nll':   [], 'val_bce':   [], 'val_phys':   [],
        'val_metrics': [],
        'val_sigma':   [],
        'val_p_on':    [],
    }

    best_val_nll = float('inf')
    best_val_f1  = -1.0
    best_state   = None
    counter      = 0
    f1_phase     = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting Gated Prob-PINN-LNN v3 training...")

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        ep_nll = ep_bce = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            mu_final, sigma, state_logits, p_on = model(xb)

            nll_loss = nll_criterion(mu_final, yb, sigma ** 2)

            bce_loss = torch.tensor(0.0, device=device)
            for i, app in enumerate(APPLIANCES):
                if BCE_LAMBDA[app] > 0:
                    y_bin = (yb[:, i] > thresholds_scaled[i]).float()
                    pos_w = torch.tensor(BCE_ALPHA[app], device=device)
                    bce_i = F.binary_cross_entropy_with_logits(
                        state_logits[:, i], y_bin, pos_weight=pos_w)
                    bce_loss = bce_loss + BCE_LAMBDA[app] * bce_i

            x_mid     = xb[:, WIN // 2, 0]
            phys_loss = phys_criterion(x_mid, mu_final, sigma)

            if epoch < WARMUP_EPOCHS:
                loss = nll_loss + bce_loss
            else:
                loss = nll_loss + bce_loss + lambda_phys * phys_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_nll   += nll_loss.item()
            ep_bce   += bce_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'nll':  f'{nll_loss.item():.4f}',
                'bce':  f'{bce_loss.item():.4f}',
                'p_on': f'{p_on.mean().item():.3f}',
            })

        avg_tr_nll   = ep_nll   / len(tr_loader)
        avg_tr_bce   = ep_bce   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_nll'].append(avg_tr_nll)
        history['train_bce'].append(avg_tr_bce)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_nll = vl_bce = vl_phys = vl_total = 0.0
        val_mus, val_sigmas, val_p_ons, val_trues = [], [], [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu_final, sigma, state_logits, p_on = model(xb)

                nll_loss = nll_criterion(mu_final, yb, sigma ** 2)

                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        y_bin = (yb[:, i] > thresholds_scaled[i]).float()
                        pos_w = torch.tensor(BCE_ALPHA[app], device=device)
                        bce_i = F.binary_cross_entropy_with_logits(
                            state_logits[:, i], y_bin, pos_weight=pos_w)
                        bce_loss = bce_loss + BCE_LAMBDA[app] * bce_i

                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, mu_final, sigma)
                loss      = nll_loss + bce_loss + lambda_phys * phys_loss

                vl_nll   += nll_loss.item()
                vl_bce   += bce_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_mus.append(mu_final.cpu().numpy())
                val_sigmas.append(sigma.cpu().numpy())
                val_p_ons.append(p_on.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_nll   = vl_nll   / len(va_loader)
        avg_va_bce   = vl_bce   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_nll'].append(avg_va_nll)
        history['val_bce'].append(avg_va_bce)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_nll)

        mu_all     = np.concatenate(val_mus)
        sigma_all  = np.concatenate(val_sigmas)
        p_on_all   = np.concatenate(val_p_ons)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(y_true_all, mu_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        mean_sigma = {app: float(sigma_all[:, i].mean()) for i, app in enumerate(APPLIANCES)}
        mean_p_on  = {app: float(p_on_all[:,  i].mean()) for i, app in enumerate(APPLIANCES)}
        history['val_sigma'].append(mean_sigma)
        history['val_p_on'].append(mean_p_on)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])
        avg_sig = float(sigma_all.mean())
        avg_pon = float(p_on_all.mean())

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"tr_nll={avg_tr_nll:.4f} bce={avg_tr_bce:.4f} phys={avg_tr_phys:.4f}  "
            f"va_nll={avg_va_nll:.4f}  avgF1={avg_f1:.4f}  "
            f"sigma={avg_sig:.4f}  p_on={avg_pon:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m   = per_app_metrics[app]
            sig = mean_sigma[app]
            pon = mean_p_on[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"sigma={sig:.4f}  p_on={pon:.4f}")

        # ── Two-phase early stopping ──
        if (not f1_phase) and (epoch + 1 == WARMUP_EPOCHS):
            f1_phase    = True
            counter     = 0
            best_val_f1 = -1.0
            print("  [switching to F1 phase — early stopping now tracks avg val F1]")

        if not f1_phase:
            if avg_va_nll < best_val_nll:
                best_val_nll = avg_va_nll
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                counter      = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} (warmup phase)")
                    break
        else:
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                counter     = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} (F1 phase)")
                    break

    print("Training completed!")

    # ── Test evaluation ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_mus, test_sigmas, test_p_ons, test_trues = [], [], [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            mu_final, sigma, state_logits, p_on = model(xb)
            test_mus.append(mu_final.cpu().numpy())
            test_sigmas.append(sigma.cpu().numpy())
            test_p_ons.append(p_on.cpu().numpy())
            test_trues.append(yb.numpy())

    y_mu_te    = np.concatenate(test_mus)
    y_sigma_te = np.concatenate(test_sigmas)
    y_pon_te   = np.concatenate(test_p_ons)
    y_true_te  = np.concatenate(test_trues)

    test_metrics = compute_per_appliance_metrics(y_true_te, y_mu_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8} {'mean_s':>8} {'mean_pon':>9}")
    print("-" * 88)
    for i, app in enumerate(APPLIANCES):
        m   = test_metrics[app]
        sig = y_sigma_te[:, i].mean()
        pon = y_pon_te[:,  i].mean()
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f} "
              f"{sig:>8.4f} {pon:>9.4f}")

    _plot_training(history, test_metrics, save_dir)
    _plot_uncertainty(y_true_te, y_mu_te, y_sigma_te, y_pon_te, y_scalers, save_dir)

    config = {
        'dataset':  'UKDALE',
        'model':    'GatedProbPINNLiquidNetworkModel',
        'version':  'prob_v3',
        'description': (
            'Prob-PINN-LNN v3: gated output (p_on * mu_power); '
            'BCE_LAMBDA[MW]=0.4 (up from 0.1); sigma clamped [0.01,5.0]; '
            'separate state BCE head (BCEWithLogitsLoss); '
            'log1p + MinMax scaling; probabilistic physics (log1p Jacobian); '
            'two-phase early stopping (warmup NLL → avg F1)'
        ),
        'loss': (
            f'GaussianNLL(mu_final, y, σ²) + Σ λ_i·BCE(logit_i, y_bin_i) '
            f'+ {lambda_phys}·ProbPhysics(log1p, ε={epsilon_w}W)'
        ),
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
            'bce_lambda': BCE_LAMBDA, 'bce_alpha': BCE_ALPHA,
            'warmup_epochs': WARMUP_EPOCHS,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
        'test_uncertainty': {
            app: {
                'mean_sigma': float(y_sigma_te[:, i].mean()),
                'std_sigma':  float(y_sigma_te[:, i].std()),
                'mean_p_on':  float(y_pon_te[:,  i].mean()),
                'std_p_on':   float(y_pon_te[:,  i].std()),
            }
            for i, app in enumerate(APPLIANCES)
        },
    }
    with open(os.path.join(save_dir, 'pinn_lnn_prob_v3_ukdale_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history, y_sigma_te, y_pon_te


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle('Gated Prob-PINN-LNN v3 — Training Losses', fontsize=12)

    axes[0].plot(epochs_x, history['train_loss'], label='Train', color='blue')
    axes[0].plot(epochs_x, history['val_loss'],   label='Val',   color='red')
    axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, history['train_nll'], label='Train NLL', color='blue')
    axes[1].plot(epochs_x, history['val_nll'],   label='Val NLL',   color='red')
    axes[1].set_title('Gaussian NLL'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_x, history['train_bce'], label='Train BCE', color='blue')
    axes[2].plot(epochs_x, history['val_bce'],   label='Val BCE',   color='red')
    axes[2].set_title('State BCE Loss'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    axes[3].plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    axes[3].set_title('Physics Loss'); axes[3].set_xlabel('Epoch')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v3_ukdale_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Gated Prob-PINN-LNN v3 — Uncertainty & State Monitoring', fontsize=12)

    for i, app in enumerate(APPLIANCES):
        sig_series = [m[app] for m in history['val_sigma']]
        pon_series = [m[app] for m in history['val_p_on']]
        ax1.plot(epochs_x, sig_series, label=app, color=colors[i], linewidth=1.5)
        ax2.plot(epochs_x, pon_series, label=app, color=colors[i], linewidth=1.5)

    ax1.set_title('Val mean σ (should diversify per appliance)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('σ (scaled)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_title('Val mean p_on (should match duty cycle)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('mean p_on')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v3_ukdale_monitoring.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Gated Prob-PINN-LNN v3 UKDALE — Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        axes[row][0].plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].set_xlabel('Epoch'); axes[row][0].set_ylabel('F1')
        axes[row][0].legend(); axes[row][0].grid(True, alpha=0.3)

        axes[row][1].plot(epochs_x, mae_series, color='red', linewidth=1.5)
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].set_xlabel('Epoch'); axes[row][1].set_ylabel('MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v3_ukdale_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_uncertainty(y_true, y_mu, y_sigma, y_p_on, y_scalers, save_dir):
    """Per appliance: σ histogram, |error| vs σ scatter, p_on histogram."""
    fig, axes = plt.subplots(len(APPLIANCES), 3,
                             figsize=(18, 4 * len(APPLIANCES)))
    fig.suptitle('Gated Prob-PINN-LNN v3 — Test Uncertainty Analysis', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        i = APPLIANCES.index(app)
        y_range = float(y_scalers[i].data_range_[0])
        y_min_v = float(y_scalers[i].data_min_[0])

        log_true = y_scalers[i].inverse_transform(y_true[:, i:i+1])
        log_pred = y_scalers[i].inverse_transform(y_mu[:,   i:i+1])
        raw_true = np.expm1(np.clip(log_true, 0, None)).flatten()
        raw_pred = np.expm1(np.clip(log_pred, 0, None)).flatten()
        abs_err  = np.abs(raw_true - raw_pred)

        mu_log    = y_mu[:, i] * y_range + y_min_v
        mu_raw    = np.expm1(np.clip(mu_log, 0, None))
        sigma_raw = y_sigma[:, i] * y_range * (1.0 + mu_raw)
        p_on_arr  = y_p_on[:, i]

        axes[row][0].hist(sigma_raw, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
        axes[row][0].axvline(sigma_raw.mean(), color='red', linestyle='--',
                             label=f'mean={sigma_raw.mean():.1f} W')
        axes[row][0].set_title(f'{app} — σ distribution (W)')
        axes[row][0].set_xlabel('σ (W)'); axes[row][0].set_ylabel('Count')
        axes[row][0].legend(); axes[row][0].grid(True, alpha=0.3)

        axes[row][1].scatter(sigma_raw, abs_err, alpha=0.3, s=5, color='steelblue')
        axes[row][1].set_title(f'{app} — |error| vs σ')
        axes[row][1].set_xlabel('σ (W)'); axes[row][1].set_ylabel('|μ_final − y_true| (W)')
        axes[row][1].grid(True, alpha=0.3)

        axes[row][2].hist(p_on_arr, bins=50, color='coral', alpha=0.75, edgecolor='white')
        axes[row][2].axvline(p_on_arr.mean(), color='red', linestyle='--',
                             label=f'mean={p_on_arr.mean():.3f}')
        axes[row][2].set_title(f'{app} — p_on distribution')
        axes[row][2].set_xlabel('p_on'); axes[row][2].set_ylabel('Count')
        axes[row][2].legend(); axes[row][2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_prob_v3_ukdale_uncertainty.png'),
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
    save_dir  = f"models/pinn_lnn_prob_v3_ukdale_{timestamp}"

    data_dict = load_data()

    test_metrics, history, test_sigmas, test_p_ons = train_pinn_prob_v3_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        epsilon_w   = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
