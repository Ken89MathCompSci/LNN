"""
Hybrid Transformer-LNN v4 — Modest Improvement over v1
=========================================================
Same as test_hybrid_transformer_lnn.py with one addition:
  1. --appliance / --plot flags  (run one appliance per Colab session)

Architecture (Gabriel et al., 2025 — Energy and AI, DOI: 10.1016/j.egyai.2025.100489):
  Learnable Encoding → CNN Encoder → Transformer Encoder → LNN Reservoir → FC Output

Usage:
    # Run one appliance per session:
    !python test_hybrid_transformer_lnn_v4.py --appliance fridge
    !python test_hybrid_transformer_lnn_v4.py --appliance microwave
    !python test_hybrid_transformer_lnn_v4.py --appliance "dish washer"
    !python test_hybrid_transformer_lnn_v4.py --appliance "washer dryer"

    # After all four done, generate plots:
    !python test_hybrid_transformer_lnn_v4.py --plot

    # Or run all at once (if session allows):
    !python test_hybrid_transformer_lnn_v4.py
"""

import sys
import os
import time
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

from models import HybridTransformerLNNModel
from utils import calculate_nilm_metrics

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCHS       = 80
PATIENCE     = 20
LR           = 1e-3
WEIGHT_DECAY = 0.0    # Adam (no weight decay)
BATCH        = 128
WIN          = 100

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
APP_LABELS = ['Dish Washer', 'Fridge', 'Microwave', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

SAVE_DIR = os.path.join('results', 'hybrid_transformer_lnn_v4')
COLOR    = '#17BECF'

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


# ── Fast per-epoch metrics (no sklearn, vectorised SAE) ───────────────────────

def _fast_epoch_metrics(y_true, y_pred, threshold):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    N = 100
    num_periods = len(y_true) // N
    if num_periods > 0:
        t_b = y_true[:num_periods * N].reshape(num_periods, N)
        p_b = y_pred[:num_periods * N].reshape(num_periods, N)
        sae = float(np.abs(t_b.sum(1) - p_b.sum(1)).sum() / (N * num_periods))
    else:
        sae = 0.0
    t_bin = y_true > threshold
    p_bin = y_pred > threshold
    tp = int(np.sum(t_bin & p_bin))
    fp = int(np.sum(~t_bin & p_bin))
    fn = int(np.sum(t_bin & ~p_bin))
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return {'mae': mae, 'sae': sae, 'f1': f1}


# ── Training ───────────────────────────────────────────────────────────────────

def train_appliance(appliance_name, splits, device, epochs, hidden_size):
    thr = THRESHOLDS[appliance_name]

    X_tr, y_tr = create_sequences(splits['train']['main'].values,
                                   splits['train'][appliance_name].values)
    X_va, y_va = create_sequences(splits['val']['main'].values,
                                   splits['val'][appliance_name].values)
    X_te, y_te = create_sequences(splits['test']['main'].values,
                                   splits['test'][appliance_name].values)

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

    model = HybridTransformerLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        dt=0.1, num_conv_layers=2, num_encoder_layers=2, num_heads=4,
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

        ep_pred = np.concatenate(val_preds_ep) * y_std + y_mean
        ep_true = np.concatenate(val_trues_ep) * y_std + y_mean
        ep_m = _fast_epoch_metrics(ep_true, ep_pred, threshold=thr)
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
    fig.suptitle('Training Curves — Hybrid Transformer-LNN v4 (80 epochs)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics(results, save_dir):
    metric_info = [('val_mae_hist', 'MAE (W)'), ('val_sae_hist', 'SAE'), ('val_f1_hist', 'F1')]
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
    fig.suptitle('Val Metrics per Epoch — Hybrid Transformer-LNN v4 (80 epochs)', fontsize=12)
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
    fig.suptitle('Final Metrics — Hybrid Transformer-LNN v4 (80 epochs)', fontsize=12)
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
        print(f"  {label} — Hybrid Transformer-LNN v4")
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


# ── Per-appliance JSON helpers ─────────────────────────────────────────────────

def _save_appliance_json(app, r, hidden, epochs):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f'{app.replace(" ", "_")}.json')
    with open(path, 'w') as f:
        json.dump({
            'appliance':    app,
            'hidden_size':  hidden,
            'epochs':       epochs,
            'epochs_run':   r['epochs_run'],
            'num_params':   r['num_params'],
            'metrics':      {k: float(v) for k, v in r['metrics'].items()},
            'train_losses': r['train_losses'],
            'val_losses':   r['val_losses'],
            'val_mae_hist': r['val_mae_hist'],
            'val_sae_hist': r['val_sae_hist'],
            'val_f1_hist':  r['val_f1_hist'],
            'time_s':       r.get('time_s', None),
        }, f, indent=2)
    print(f'  JSON saved → {path}')


def _load_all_appliance_jsons():
    results = {}
    for app in APPLIANCES:
        path = os.path.join(SAVE_DIR, f'{app.replace(" ", "_")}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[app] = json.load(f)
            print(f'  Loaded {app} ← {path}')
        else:
            print(f'  Missing: {path}  (run --appliance "{app}" first)')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Transformer-LNN v4 — Adam + per-appliance flags')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Max training epochs (default: {EPOCHS})')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Hidden size (default: 256)')
    parser.add_argument('--appliance', type=str, default=None,
                        choices=APPLIANCES,
                        help='Train a single appliance and save its JSON.')
    parser.add_argument('--plot', action='store_true',
                        help='Skip training — load saved JSONs and regenerate plots.')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Plot-only mode ──
    if args.plot:
        print('Plot mode — loading saved per-appliance results...')
        results = _load_all_appliance_jsons()
        if not results:
            print('No results found. Run at least one appliance first.')
            sys.exit(1)
        print('\nGenerating plots...')
        plot_training_curves(results, SAVE_DIR)
        plot_epoch_metrics(results, SAVE_DIR)
        plot_bar_chart(results, SAVE_DIR)
        print_table(results)
        return

    for fp in ['data/redd/train_small.pkl',
               'data/redd/val_small.pkl',
               'data/redd/test_small.pkl']:
        if not os.path.exists(fp):
            print(f'ERROR: Missing {fp}')
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Hidden size: {args.hidden}  |  Max epochs: {args.epochs}')
    print(f'Optimiser: Adam')

    splits = load_data()
    apps_to_run = [args.appliance] if args.appliance else APPLIANCES

    print('\n' + '=' * 70)
    print('  Hybrid Transformer-LNN v4 — Training')
    print('=' * 70)

    total_start = time.time()
    for app in apps_to_run:
        print(f'\n  ▶  {app}')
        t0 = time.time()
        r = train_appliance(app, splits, device, args.epochs, args.hidden)
        elapsed = time.time() - t0
        r['time_s'] = elapsed
        m = r['metrics']
        print(f"  ✅ {app:15s} | F1={m['f1']:.4f}  MAE={m['mae']:.2f}  "
              f"SAE={m['sae']:.4f}  (params={r['num_params']:,})  "
              f"time={elapsed:.1f}s")
        _save_appliance_json(app, r, args.hidden, args.epochs)
    total_elapsed = time.time() - total_start
    print(f'\n  Total training time: {total_elapsed:.1f}s '
          f'({total_elapsed/60:.1f} min)')

    all_done = all(
        os.path.exists(os.path.join(SAVE_DIR, f'{a.replace(" ", "_")}.json'))
        for a in APPLIANCES
    )
    if all_done:
        all_results = _load_all_appliance_jsons()
        print('\nGenerating plots...')
        plot_training_curves(all_results, SAVE_DIR)
        plot_epoch_metrics(all_results, SAVE_DIR)
        plot_bar_chart(all_results, SAVE_DIR)
        print_table(all_results)
    else:
        missing = [a for a in APPLIANCES
                   if not os.path.exists(
                       os.path.join(SAVE_DIR, f'{a.replace(" ", "_")}.json'))]
        print(f'\nStill missing: {missing}')
        print('Run those, then: !python test_hybrid_transformer_lnn_v4.py --plot')

    print(f'All outputs → {SAVE_DIR}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
