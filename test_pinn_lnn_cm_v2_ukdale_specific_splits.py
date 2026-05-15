"""
PINN-LNN-CM v2 — four fixes over test_pinn_lnn_cm_ukdale_specific_splits.py

  Fix 1: WARMUP_EPOCHS 20 -> 40
          Let MSE converge fully before BCE is introduced. The v1 training log
          shows a clear MSE shock at epoch 21 — MSE jumped from 0.01 to 0.06.

  Fix 2: Two-phase early stopping
          Warmup  (epoch < WARMUP_EPOCHS): monitor val MSE  — lower is better
          BCE phase (epoch >= WARMUP_EPOCHS): monitor avg val F1 — higher is better
          Counter resets on phase switch so BCE phase gets a full PATIENCE window.

  Fix 3: BCEWithLogitsLoss with pos_weight, BCE_LAMBDA reduced to 0.05
          pos_weight[i] = n_off / n_on computed from training labels.
          DW: ~16x, Fridge: ~1.5x. BCE_LAMBDA kept small (0.05) so BCE
          guides rather than overrides MSE — avoids MAE blowup seen when
          BCE_LAMBDA=0.5 caused raw outputs to be pushed to extreme logit values.

  Fix 4: Freeze shared encoder for first BCE_FREEZE_EPOCHS after warmup
          DW/fridge BCE was corrupting the shared LNN hidden state and collapsing
          MW recall. Freezing the encoder lets only the heads adapt first;
          after BCE_FREEZE_EPOCHS the encoder unfreezes for joint fine-tuning.
"""

import sys
import os
import random
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS      = 120
PATIENCE    = 20
LR          = 1e-3
BATCH       = 32
WIN         = 100
STRIDE      = 5

LAMBDA_PHYS      = 0.01
EPSILON_W        = 50.0
WARMUP_EPOCHS    = 40   # Fix 1: was 20
BCE_FREEZE_EPOCHS = 10  # Fix 4: freeze encoder for this many epochs after warmup

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.05, 'fridge': 0.05, 'microwave': 0.0, 'washer dryer': 0.0}


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
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

    def forward(self, x_mid_scaled, pred_scaled):
        x_raw     = x_mid_scaled * self.x_range + self.x_min
        p_raw     = pred_scaled  * self.y_ranges + self.y_mins
        p_sum     = p_raw.sum(dim=1)
        violation = F.relu(p_sum - x_raw - self.epsilon)
        return violation.mean()


# ---------------------------------------------------------------------------
# Physics-Informed LNN Model
# ---------------------------------------------------------------------------

class PhysicsInformedLiquidNetworkModel(nn.Module):
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

        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

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
        return torch.cat([head(h) for head in self.heads], dim=1)


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

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
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
# Per-appliance metrics — includes TP, TN, FP, FN
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(
            y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(
            y_pred[:, i:i+1]).flatten()

        m = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLDS[app])

        thr = THRESHOLDS[app]
        y_true_bin = raw_true > thr
        y_pred_bin = raw_pred > thr
        m['tp'] = int(np.sum( y_true_bin &  y_pred_bin))
        m['tn'] = int(np.sum(~y_true_bin & ~y_pred_bin))
        m['fp'] = int(np.sum(~y_true_bin &  y_pred_bin))
        m['fn'] = int(np.sum( y_true_bin & ~y_pred_bin))

        metrics[app] = m
    return metrics


def _print_cm_table(metrics, header="Test Results"):
    print(f"\n{'='*80}")
    print(f"  {header}")
    print(f"{'='*80}")
    print(f"  {'Appliance':<16} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8} "
          f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'MAE':>7}")
    print(f"  {'-'*76}")
    for app in APPLIANCES:
        m = metrics[app]
        print(f"  {app:<16} {m['tp']:>8,} {m['tn']:>8,} {m['fp']:>8,} {m['fn']:>8,} "
              f"{m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['mae']:>7.2f}")
    avg_f1 = np.mean([metrics[a]['f1'] for a in APPLIANCES])
    print(f"  {'-'*76}")
    print(f"  {'Average F1':<16} {'':>8} {'':>8} {'':>8} {'':>8} {avg_f1:>7.4f}")


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, dt=0.1,
                     lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"lambda_phys={lambda_phys}  epsilon={epsilon_w}W  hidden={hidden_size}  dt={dt}")
    print(f"WARMUP_EPOCHS={WARMUP_EPOCHS}  BCE_FREEZE_EPOCHS={BCE_FREEZE_EPOCHS}")

    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

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

    # Fix 3: pos_weight from training label balance
    pos_weights = []
    print("\nClass balance (training set):")
    for i, app in enumerate(APPLIANCES):
        thr_s = thresholds_scaled[i]
        n_on  = int((Y_tr[:, i] > thr_s).sum())
        n_off = len(Y_tr) - n_on
        pw    = max(float(n_off) / max(float(n_on), 1.0), 1.0)
        pos_weights.append(pw)
        print(f"  {app:<16}  ON={n_on:,}  OFF={n_off:,}  pos_weight={pw:.1f}")
    print()

    print(f"Train: {X_tr.shape} -> {Y_tr.shape}")
    print(f"Val:   {X_va.shape} -> {Y_va.shape}")
    print(f"Test:  {X_te.shape} -> {Y_te.shape}")

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
        x_scaler, y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

    # Fix 3: per-appliance BCEWithLogitsLoss with pos_weight
    bce_criteria = [
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weights[i]], device=device)
        )
        for i in range(len(APPLIANCES))
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }

    # Fix 2: two-phase early stopping state
    best_val_mse = float('inf')   # warmup phase metric
    best_val_f1  = -float('inf')  # BCE phase metric
    best_state   = None
    counter      = 0
    encoder_frozen = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting PINN-LNN-CM-v2 training...")

    for epoch in range(EPOCHS):

        # Fix 4: freeze / unfreeze encoder at phase boundaries
        if epoch == WARMUP_EPOCHS and not encoder_frozen:
            for name, p in model.named_parameters():
                if 'heads' not in name:
                    p.requires_grad = False
            encoder_frozen = True
            counter = 0   # Fix 2: reset counter on phase switch
            print(f"  [Epoch {epoch+1}] Encoder frozen — only heads train for {BCE_FREEZE_EPOCHS} epochs")

        if epoch == WARMUP_EPOCHS + BCE_FREEZE_EPOCHS and encoder_frozen:
            for p in model.parameters():
                p.requires_grad = True
            encoder_frozen = False
            print(f"  [Epoch {epoch+1}] Encoder unfrozen — joint fine-tuning resumes")

        # ── Training ──
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred      = model(xb)
            mse_loss  = mse_criterion(pred, yb)
            x_mid     = xb[:, WIN // 2, 0]
            phys_loss = phys_criterion(x_mid, pred)

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss
            else:
                # Fix 3: BCEWithLogitsLoss on raw logits with pos_weight
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        thr_s = thresholds_scaled[i]
                        y_bin = (yb[:, i] > thr_s).float()
                        bce_loss = bce_loss + BCE_LAMBDA[app] * bce_criteria[i](
                            pred[:, i], y_bin)
                loss = mse_loss + lambda_phys * phys_loss + bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'mse': f'{mse_loss.item():.5f}',
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
            for xb, yb in va_loader:
                xb, yb    = xb.to(device), yb.to(device)
                pred       = model(xb)
                mse_loss   = mse_criterion(pred, yb)
                x_mid      = xb[:, WIN // 2, 0]
                phys_loss  = phys_criterion(x_mid, pred)
                loss       = mse_loss + lambda_phys * phys_loss

                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(
            y_true_all, y_pred_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        phase = "warmup" if epoch < WARMUP_EPOCHS else (
            "freeze" if encoder_frozen else "joint")
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS} [{phase}]  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  "
                  f"TP={m['tp']:,}  TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}")

        # Fix 2: two-phase early stopping
        if epoch < WARMUP_EPOCHS:
            if avg_va_mse < best_val_mse:
                best_val_mse = avg_va_mse
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                counter      = 0
            else:
                counter += 1
        else:
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                counter     = 0
            else:
                counter += 1

        if counter >= PATIENCE and epoch >= WARMUP_EPOCHS:
            print(f"  Early stopping at epoch {epoch+1} (no F1 improvement for {PATIENCE} epochs)")
            break

    print("Training completed!")

    # ── Test evaluation ──
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

    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    _print_cm_table(test_metrics, header="Test Set — Confusion Matrix + Metrics")

    # ── Plots ──
    _plot_training(history, test_metrics, save_dir)

    # ── Save JSON ──
    config = {
        'dataset': 'UKDALE',
        'model': 'PhysicsInformedLiquidNetworkModel_CM_v2',
        'fixes': [
            'WARMUP_EPOCHS=40',
            'two-phase early stopping (val MSE -> avg val F1)',
            'BCEWithLogitsLoss with pos_weight from training balance',
            f'encoder frozen for first {BCE_FREEZE_EPOCHS} BCE epochs',
        ],
        'pos_weights': {app: pos_weights[i] for i, app in enumerate(APPLIANCES)},
        'loss': f'MSE + {lambda_phys} * PhysicsConsistency(epsilon={epsilon_w}W)',
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_freeze_epochs': BCE_FREEZE_EPOCHS,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
        },
        'test_metrics': {
            app: {k: float(v) if not isinstance(v, int) else v
                  for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_lnn_cm_v2_ukdale_results.json'),
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
    plt.axvline(WARMUP_EPOCHS, color='orange', linestyle='--', alpha=0.7, label='BCE start')
    plt.axvline(WARMUP_EPOCHS + BCE_FREEZE_EPOCHS, color='green',
                linestyle='--', alpha=0.7, label='Encoder unfreeze')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.axvline(WARMUP_EPOCHS, color='orange', linestyle='--', alpha=0.7)
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.axvline(WARMUP_EPOCHS, color='orange', linestyle='--', alpha=0.7)
    plt.title('Physics Consistency Loss')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_cm_v2_ukdale_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN-CM-v2 UKDALE — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.axvline(WARMUP_EPOCHS, color='orange', linestyle='--',
                      alpha=0.7, label='BCE start')
        ax_f1.set_title(f'{app} — F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(fontsize=8); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.axvline(WARMUP_EPOCHS, color='orange', linestyle='--', alpha=0.7)
        ax_mae.set_title(f'{app} — MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(fontsize=8); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_cm_v2_ukdale_per_appliance.png'),
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
    save_dir  = f"models/pinn_lnn_cm_v2_ukdale_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        epsilon_w   = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
