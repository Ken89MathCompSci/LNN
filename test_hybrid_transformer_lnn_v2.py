"""
Hybrid Transformer-LNN v2 — Paper-Faithful Implementation — 80 Epochs
=======================================================================
Trains and evaluates HybridTransformerLNNv2Model on all REDD appliances.

Key differences from v1 (test_hybrid_transformer_lnn.py):
  - LNN reservoir embedded INSIDE attention (modifies Q and K)
  - Adaptive positional encoding: PE = alpha * PE_sinusoidal + beta
  - CNN encoder uses ReLU + MaxPool1d (per paper)
  - GLU feedforward in each Transformer layer
  - Final prediction uses last-token output (no LiquidODECell at end)

Reference:
  Gabriel et al. (2025) "Hybrid transformer model with liquid neural networks
  and learnable encodings for buildings' energy forecasting", Energy and AI.
  DOI: 10.1016/j.egyai.2025.100489

Usage:
    !python test_hybrid_transformer_lnn_v2.py
    !python test_hybrid_transformer_lnn_v2.py --epochs 80 --hidden 256
"""

import sys
import os
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

sys.path.append('Source Code')

from models import HybridTransformerLNNv2Model
from utils import calculate_nilm_metrics

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCHS    = 80
PATIENCE  = 20
LR        = 1e-3
BATCH     = 128
WIN       = 100

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
APP_LABELS = ['Dish Washer', 'Fridge', 'Microwave', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

SAVE_DIR = os.path.join('results', 'hybrid_transformer_lnn_v2')
COLOR    = '#E377C2'

# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading REDD data...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(f'data/redd/{split}_small.pkl', 'rb') as f:
            splits[split] = pickle.load(f)[0]
    return splits


def create_sequences(mains, appliance, window_size=WIN):
    X, y = [], []
    for i in range(len(mains) - window_size):
        X.append(mains[i:i + window_size])
        y.append(appliance[i + window_size])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(y, dtype=np.float32),
    )


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training ───────────────────────────────────────────────────────────────────

def train_appliance(appliance_name, splits, device, epochs, hidden_size):
    thr = THRESHOLDS[appliance_name]

    X_tr, y_tr = create_sequences(splits['train']['main'].values,
                                   splits['train'][appliance_name].values)
    X_va, y_va = create_sequences(splits['val']['main'].values,
                                   splits['val'][appliance_name].values)
    X_te, y_te = create_sequences(splits['test']['main'].values,
                                   splits['test'][appliance_name].values)

    # Normalise using training stats
    x_mean, x_std = float(X_tr.mean()), float(X_tr.std()) + 1e-8
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8

    X_tr = (X_tr - x_mean) / x_std
    X_va = (X_va - x_mean) / x_std
    X_te = (X_te - x_mean) / x_std
    y_tr_n = (y_tr - y_mean) / y_std
    y_va_n = (y_va - y_mean) / y_std
    y_te_n = (y_te - y_mean) / y_std

    tr_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_tr, y_tr_n.reshape(-1, 1)), batch_size=BATCH, shuffle=True)
    va_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_va, y_va_n.reshape(-1, 1)), batch_size=BATCH, shuffle=False)
    te_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_te, y_te_n.reshape(-1, 1)), batch_size=BATCH, shuffle=False)

    model = HybridTransformerLNNv2Model(
        input_size=1, hidden_size=hidden_size, output_size=1,
        dt=0.1, num_conv_layers=2, num_encoder_layers=2, num_heads=4,
        spectral_radius=0.9,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []
    val_mae_hist, val_sae_hist, val_f1_hist = [], [], []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for xb, yb in tqdm(tr_loader,
                            desc=f"  [{appliance_name}] Epoch {epoch+1}/{epochs}",
                            leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        avg_tr = ep_loss / len(tr_loader)
        train_losses.append(avg_tr)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        val_preds_ep, val_trues_ep = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl_loss += criterion(out, yb).item()
                val_preds_ep.append(out.cpu().numpy())
                val_trues_ep.append(yb.cpu().numpy())
        avg_va = vl_loss / len(va_loader)
        val_losses.append(avg_va)
        scheduler.step(avg_va)

        # Epoch-wise metrics (denormalized)
        ep_pred = np.concatenate(val_preds_ep) * y_std + y_mean
        ep_true = np.concatenate(val_trues_ep) * y_std + y_mean
        ep_m = calculate_nilm_metrics(ep_true, ep_pred, threshold=thr)
        val_mae_hist.append(ep_m['mae'])
        val_sae_hist.append(ep_m['sae'])
        val_f1_hist.append(ep_m['f1'])

        print(f"  [{appliance_name}] Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr:.5f}  val={avg_va:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_va < best_val:
            best_val   = avg_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds) * y_std + y_mean
    y_true = np.concatenate(trues) * y_std + y_mean
    metrics = calculate_nilm_metrics(y_true, y_pred, threshold=thr)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'metrics':      metrics,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'val_mae_hist': val_mae_hist,
        'val_sae_hist': val_sae_hist,
        'val_f1_hist':  val_f1_hist,
        'epochs_run':   len(train_losses),
        'num_params':   n_params,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, app in zip(axes, APPLIANCES):
        r = results[app]
        ep = range(1, r['epochs_run'] + 1)
        ax.plot(ep, r['train_losses'], label='Train', color='steelblue', lw=1.5)
        ax.plot(ep, r['val_losses'],   label='Val',   color='tomato',    lw=1.5, linestyle='--')
        ax.set_title(app.title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Training Curves — Hybrid Transformer-LNN v2 (80 epochs)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics(results, save_dir):
    metric_info = [
        ('val_mae_hist', 'MAE (W)'),
        ('val_sae_hist', 'SAE'),
        ('val_f1_hist',  'F1'),
    ]
    fig, axes = plt.subplots(len(APPLIANCES), 3, figsize=(15, 4 * len(APPLIANCES)))
    for row, app in enumerate(APPLIANCES):
        r = results[app]
        for col, (hist_key, mk_label) in enumerate(metric_info):
            ax = axes[row, col]
            vals = r.get(hist_key, [])
            if vals:
                ax.plot(range(1, len(vals) + 1), vals, color=COLOR, lw=1.5)
            ax.set_title(f'{app.title()} — {mk_label}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(mk_label)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Val Metrics per Epoch — Hybrid Transformer-LNN v2 (80 epochs)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'epoch_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_bar_chart(results, save_dir):
    metrics_cfg = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1')]
    x = np.arange(len(APPLIANCES))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (mk, ml) in zip(axes, metrics_cfg):
        vals = [results[app]['metrics'].get(mk, np.nan) for app in APPLIANCES]
        bars = ax.bar(x, vals, color=COLOR, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(v for v in vals if not np.isnan(v)),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(APP_LABELS, rotation=10, ha='right')
        ax.set_ylabel(ml)
        ax.set_title(f'{ml} per Appliance')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle('Final Metrics — Hybrid Transformer-LNN v2 (80 epochs)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Console table ──────────────────────────────────────────────────────────────

def print_table(results):
    divider = '─' * 75
    for metric, label in [('f1', 'F1'), ('mae', 'MAE'), ('sae', 'SAE')]:
        print(f"\n{'='*75}")
        print(f"  {label} — Hybrid Transformer-LNN v2")
        print(f"{'='*75}")
        print(f"  {'Appliance':<20}{'Value':>12}  (epochs run)")
        print(divider)
        vals = []
        for app in APPLIANCES:
            v  = results[app]['metrics'].get(metric, float('nan'))
            ep = results[app]['epochs_run']
            print(f"  {app.title():<20}{v:>12.4f}  ({ep} epochs)")
            vals.append(v)
        print(divider)
        print(f"  {'Average':<20}{np.nanmean(vals):>12.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Transformer-LNN v2 (paper-faithful) — 80 Epochs')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Max training epochs (default: {EPOCHS})')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Hidden size (default: 256)')
    args = parser.parse_args()

    for fp in ['data/redd/train_small.pkl',
               'data/redd/val_small.pkl',
               'data/redd/test_small.pkl']:
        if not os.path.exists(fp):
            print(f'ERROR: Missing {fp}')
            sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Hidden size: {args.hidden}  |  Max epochs: {args.epochs}')
    print('Architecture: Input Embedding → CNN (ReLU+MaxPool) → Adaptive PE '
          '→ LiquidTransformer (LNN in Q,K + GLU FFN) → FC')

    splits = load_data()
    results = {}

    print('\n' + '=' * 70)
    print('  Hybrid Transformer-LNN v2 — Training')
    print('=' * 70)

    for app in APPLIANCES:
        print(f'\n  ▶  {app}')
        r = train_appliance(app, splits, device, args.epochs, args.hidden)
        results[app] = r
        m = r['metrics']
        print(f"  ✅ {app:15s} | F1={m['f1']:.4f}  MAE={m['mae']:.2f}  "
              f"SAE={m['sae']:.4f}  (params={r['num_params']:,})")

    # ── Plots ──
    print('\nGenerating plots...')
    plot_training_curves(results, SAVE_DIR)
    plot_epoch_metrics(results, SAVE_DIR)
    plot_bar_chart(results, SAVE_DIR)

    # ── Console table ──
    print_table(results)

    # ── Save JSON ──
    json_path = os.path.join(SAVE_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model': 'HybridTransformerLNNv2',
            'hidden_size': args.hidden,
            'epochs': args.epochs,
            'results': {
                app: {
                    'epochs_run': r['epochs_run'],
                    'num_params': r['num_params'],
                    'metrics': {k: float(v) for k, v in r['metrics'].items()},
                }
                for app, r in results.items()
            }
        }, f, indent=2)
    print(f'\nJSON saved → {json_path}')
    print(f'All plots  → {SAVE_DIR}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
