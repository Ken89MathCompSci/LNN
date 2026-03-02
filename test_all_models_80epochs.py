"""
Comprehensive Comparison of ALL LNN Models - 80 Epochs with LR Scheduler
=========================================================================
Tests 6 LNN architectures with:
  - 80 epochs (up from 20)
  - ReduceLROnPlateau learning rate scheduler
  - Early stopping (patience=20)
  - MAE, SAE, F1 bar charts per appliance
  - Summary bar charts averaged across all appliances
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

sys.path.append('Source Code')

from models import (
    LiquidNetworkModel,
    AdvancedLiquidNetworkModel,
    AttentionLiquidNetworkModel,
    CNNEncoderLiquidNetworkModel,
    TransformerEncoderLiquidNetworkModel,
    BidirectionalEncoderLiquidNetworkModel
)
from utils import calculate_nilm_metrics

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_TYPES = [
    'standard_lnn',
    'advanced_lnn',
    'attention_lnn',
    'cnn_encoder',
    'transformer_encoder',
    'bidirectional_encoder',
]

MODEL_LABELS = {
    'standard_lnn':        'Standard LNN',
    'advanced_lnn':        'Advanced LNN',
    'attention_lnn':       'Attention LNN',
    'cnn_encoder':         'CNN + LNN',
    'transformer_encoder': 'Transformer + LNN',
    'bidirectional_encoder': 'Bidir + LNN',
}

MODEL_COLORS = {
    'standard_lnn':        '#4C72B0',
    'advanced_lnn':        '#DD8452',
    'attention_lnn':       '#55A868',
    'cnn_encoder':         '#C44E52',
    'transformer_encoder': '#8172B2',
    'bidirectional_encoder': '#937860',
}

APPLIANCES    = ['dish washer', 'fridge', 'microwave', 'washer dryer']
APP_LABELS    = ['Dish Washer', 'Fridge', 'Microwave', 'Washer Dryer']

THRESHOLDS = {
    'dish washer': 10.0,
    'fridge':      10.0,
    'microwave':   10.0,
    'washer dryer': 0.5,
}

AUG_PROBS = {
    'dish washer': 0.3,
    'fridge':      0.6,
    'microwave':   0.3,
    'washer dryer': 0.3,
}

# ── Data helpers ─────────────────────────────────────────────────────────────

def load_data():
    print("Loading REDD data...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(f'data/redd/{split}_small.pkl', 'rb') as f:
            splits[split] = pickle.load(f)[0]
    return splits


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(mains, appliance, window_size=100):
    X, y = [], []
    for i in range(len(mains) - window_size):
        X.append(mains[i:i + window_size])
        y.append(appliance[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)


# ── Augmentation ─────────────────────────────────────────────────────────────

def vertical_scale(x):
    mu, sigma = 1, 0.2
    lo, hi = mu - 2 * sigma, mu + 2 * sigma
    tn = truncnorm((lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
    return x * tn.rvs(1)[0]


def horizontal_scale(signal, target_length):
    mu, sigma = 1, 0.2
    lo, hi = mu - 2 * sigma, mu + 2 * sigma
    tn = truncnorm((lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)[0]
    y = signal.reshape(-1)
    f = interp1d(np.arange(len(y)), y, kind='linear', fill_value='extrapolate')
    ynew = f(np.arange(0, len(y) - 1, scale))
    if len(ynew) >= target_length:
        return ynew[:target_length]
    return np.pad(ynew, (0, target_length - len(ynew)), 'constant')


def augment_batch(X_batch, mode):
    if mode == 'none':
        return X_batch
    X_aug = X_batch.copy()
    bsz, seq_len, _ = X_batch.shape
    for i in range(bsz):
        sig = X_batch[i, :, 0]
        choice = np.random.choice(['none', 'vertical', 'horizontal', 'both'],
                                  p=[0.25, 0.25, 0.25, 0.25]) if mode == 'mixed' else mode
        if choice == 'vertical':
            sig = vertical_scale(sig)
        elif choice == 'horizontal':
            sig = horizontal_scale(sig, seq_len)
        elif choice == 'both':
            sig = horizontal_scale(sig, seq_len)
            sig = vertical_scale(sig)
        X_aug[i, :, 0] = sig
    return X_aug


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type, hidden_size=256, dt=0.1):
    input_size, output_size = 1, 1
    if model_type == 'standard_lnn':
        return LiquidNetworkModel(input_size, hidden_size, output_size, dt=dt)
    elif model_type == 'advanced_lnn':
        return AdvancedLiquidNetworkModel(input_size, hidden_size, output_size, num_layers=2, dt=dt)
    elif model_type == 'attention_lnn':
        return AttentionLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt, num_heads=4)
    elif model_type == 'cnn_encoder':
        return CNNEncoderLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt, num_conv_layers=3)
    elif model_type == 'transformer_encoder':
        # Reduced complexity to aid convergence
        return TransformerEncoderLiquidNetworkModel(
            input_size, hidden_size, output_size,
            dt=dt, num_encoder_layers=1, num_heads=4
        )
    elif model_type == 'bidirectional_encoder':
        return BidirectionalEncoderLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    data_dict, appliance_name, model_type,
    epochs=80, lr=0.001, patience=20,
    augmentation='none', hidden_size=256, dt=0.1,
    window_size=100
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    # Sequences
    X_tr, y_tr = create_sequences(train_data['main'].values, train_data[appliance_name].values, window_size)
    X_va, y_va = create_sequences(val_data['main'].values,   val_data[appliance_name].values,   window_size)
    X_te, y_te = create_sequences(test_data['main'].values,  test_data[appliance_name].values,  window_size)

    # Normalise
    x_mean, x_std = X_tr.mean(), X_tr.std() + 1e-8
    y_mean, y_std = y_tr.mean(), y_tr.std()  + 1e-8

    X_tr = (X_tr - x_mean) / x_std
    X_va = (X_va - x_mean) / x_std
    X_te = (X_te - x_mean) / x_std
    y_tr = (y_tr - y_mean) / y_std
    y_va = (y_va - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_tr, y_tr.reshape(-1, 1)), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_va, y_va.reshape(-1, 1)), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_te, y_te.reshape(-1, 1)), batch_size=batch_size, shuffle=False)

    model     = build_model(model_type, hidden_size, dt).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── LR scheduler ──
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )

    aug_prob        = AUG_PROBS.get(appliance_name, 0.3) if augmentation != 'none' else 0.0
    best_val_loss   = float('inf')
    best_state      = None
    counter         = 0
    train_losses    = []
    val_losses      = []
    val_mae_hist, val_sae_hist, val_f1_hist = [], [], []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            if augmentation != 'none' and np.random.rand() < aug_prob:
                inputs = torch.FloatTensor(
                    augment_batch(inputs.cpu().numpy(), augmentation))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_preds_ep, val_trues_ep = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                val_loss += criterion(out, targets).item()
                val_preds_ep.append(out.cpu().numpy())
                val_trues_ep.append(targets.cpu().numpy())
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # Epoch-wise metrics (denormalized)
        ep_pred = np.concatenate(val_preds_ep) * y_std + y_mean
        ep_true = np.concatenate(val_trues_ep) * y_std + y_mean
        ep_m = _fast_epoch_metrics(ep_true, ep_pred, threshold=THRESHOLDS[appliance_name])
        val_mae_hist.append(ep_m['mae'])
        val_sae_hist.append(ep_m['sae'])
        val_f1_hist.append(ep_m['f1'])

        # Step scheduler
        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"      Early stop at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test ──
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(model(inputs).cpu().numpy())

    y_true = np.concatenate(all_targets) * y_std + y_mean
    y_pred = np.concatenate(all_preds)   * y_std + y_mean

    metrics = calculate_nilm_metrics(
        y_true, y_pred, threshold=THRESHOLDS[appliance_name]
    )

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'model_name':   MODEL_LABELS[model_type],
        'model_type':   model_type,
        'num_params':   n_params,
        'metrics':      metrics,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'epochs_run':   len(train_losses),
        'val_mae_hist': val_mae_hist,
        'val_sae_hist': val_sae_hist,
        'val_f1_hist':  val_f1_hist,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metric_per_appliance(all_results, metric_key, metric_label, save_dir):
    """One grouped bar chart per appliance for a given metric."""
    n_apps   = len(APPLIANCES)
    n_models = len(MODEL_TYPES)
    x        = np.arange(n_apps)
    width    = 0.13
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for j, mt in enumerate(MODEL_TYPES):
        vals = []
        for app in APPLIANCES:
            if mt in all_results[app]:
                vals.append(all_results[app][mt]['metrics'].get(metric_key, 0))
            else:
                vals.append(0)
        bars = ax.bar(x + offsets[j], vals, width,
                      label=MODEL_LABELS[mt],
                      color=MODEL_COLORS[mt],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(APP_LABELS, fontsize=11)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} per Appliance — All LNN Models (80 epochs)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = os.path.join(save_dir, f'{metric_key}_per_appliance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bar(all_results, save_dir):
    """Single chart with MAE, SAE, F1 averages across all appliances."""
    metrics_cfg = [
        ('mae', 'MAE (↓ better)',  '#E05A5A'),
        ('sae', 'SAE (↓ better)',  '#5A9AE0'),
        ('f1',  'F1  (↑ better)',  '#5AC87B'),
    ]

    n_models = len(MODEL_TYPES)
    n_metrics = len(metrics_cfg)
    x      = np.arange(n_models)
    width  = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for k, (mkey, mlabel, mcolor) in enumerate(metrics_cfg):
        avgs = []
        for mt in MODEL_TYPES:
            vals = [
                all_results[app][mt]['metrics'].get(mkey, 0)
                for app in APPLIANCES if mt in all_results[app]
            ]
            avgs.append(np.mean(vals) if vals else 0)

        bars = ax.bar(x + offsets[k], avgs, width,
                      label=mlabel, color=mcolor,
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_TYPES],
                       rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Score (averaged across 4 appliances)', fontsize=11)
    ax.set_title('Summary — MAE / SAE / F1 Averaged Across All Appliances (80 epochs)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = os.path.join(save_dir, 'summary_all_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves(all_results, appliance_name, save_dir):
    """Loss curves for all models on one appliance."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, mt in enumerate(MODEL_TYPES):
        ax = axes[idx]
        if mt not in all_results[appliance_name]:
            ax.set_visible(False)
            continue
        res = all_results[appliance_name][mt]
        ep  = range(1, res['epochs_run'] + 1)
        ax.plot(ep, res['train_losses'], label='Train', color='#4C72B0', lw=1.5)
        ax.plot(ep, res['val_losses'],   label='Val',   color='#DD8452', lw=1.5, linestyle='--')
        ax.set_title(MODEL_LABELS[mt], fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Training Curves — {appliance_name.title()} (80 epochs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = appliance_name.replace(' ', '_')
    path = os.path.join(save_dir, f'training_curves_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_f1_heatmap(all_results, save_dir):
    """Heatmap of F1 scores: models × appliances."""
    import matplotlib.colors as mcolors

    data = np.zeros((len(MODEL_TYPES), len(APPLIANCES)))
    for i, mt in enumerate(MODEL_TYPES):
        for j, app in enumerate(APPLIANCES):
            if mt in all_results[app]:
                data[i, j] = all_results[app][mt]['metrics'].get('f1', 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='F1 Score')

    ax.set_xticks(range(len(APPLIANCES)))
    ax.set_xticklabels(APP_LABELS, fontsize=11)
    ax.set_yticks(range(len(MODEL_TYPES)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_TYPES], fontsize=10)

    for i in range(len(MODEL_TYPES)):
        for j in range(len(APPLIANCES)):
            ax.text(j, i, f'{data[i, j]:.3f}',
                    ha='center', va='center',
                    color='black' if 0.3 < data[i, j] < 0.7 else 'white',
                    fontsize=9, fontweight='bold')

    ax.set_title('F1 Score Heatmap — All Models × All Appliances (80 epochs)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'f1_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics_combined(all_results, save_dir):
    """Val MAE, SAE, F1 across epochs — all models overlaid, one row per appliance."""
    metric_info = [
        ('val_mae_hist', 'MAE (W)'),
        ('val_sae_hist', 'SAE'),
        ('val_f1_hist',  'F1'),
    ]
    n = len(APPLIANCES)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    for row, app in enumerate(APPLIANCES):
        for col, (hist_key, mk_label) in enumerate(metric_info):
            ax = axes[row, col]
            for mt in MODEL_TYPES:
                if mt not in all_results[app]:
                    continue
                vals = all_results[app][mt].get(hist_key, [])
                if vals:
                    ax.plot(range(1, len(vals) + 1), vals,
                            label=MODEL_LABELS[mt], color=MODEL_COLORS[mt])
            ax.set_title(f'{app.title()} — {mk_label}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(mk_label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Val Metrics per Epoch — All LNN Models (80 epochs)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'epoch_metrics_all_models.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Print tables ──────────────────────────────────────────────────────────────

def print_table(all_results, metric_key, title):
    header = f"{'Model':<30s} | {'DishWasher':>12s} | {'Fridge':>12s} | {'Microwave':>12s} | {'Washer':>12s} | {'Avg':>10s}"
    print(f"\n{'='*90}")
    print(f"FINAL COMPARISON — {title}")
    print('=' * 90)
    print(header)
    print('-' * 90)
    for mt in MODEL_TYPES:
        row   = f"{MODEL_LABELS[mt]:<30s}"
        vals  = []
        for app in APPLIANCES:
            if mt in all_results[app]:
                v = all_results[app][mt]['metrics'].get(metric_key, float('nan'))
                row += f" | {v:>12.4f}"
                vals.append(v)
            else:
                row += f" | {'N/A':>12s}"
        avg = np.mean(vals) if vals else float('nan')
        row += f" | {avg:>10.4f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(augmentation='none', epochs=80):
    data_dict = load_data()

    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir     = f'models/comparison_80epochs_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    all_results = {app: {} for app in APPLIANCES}

    print('\n' + '=' * 80)
    print('LNN COMPREHENSIVE COMPARISON — 80 EPOCHS + LR SCHEDULER')
    print('=' * 80)
    print(f'Augmentation : {augmentation}')
    print(f'Max Epochs   : {epochs}')
    print(f'Patience     : 20')
    print(f'LR Scheduler : ReduceLROnPlateau (factor=0.5, patience=8)')
    print('=' * 80)

    for app in APPLIANCES:
        print(f"\n{'='*80}")
        print(f"  APPLIANCE: {app.upper()}")
        print('=' * 80)

        for mt in MODEL_TYPES:
            print(f"\n  >> {MODEL_LABELS[mt]}")
            try:
                result = train_and_evaluate(
                    data_dict, app, mt,
                    epochs=epochs,
                    lr=0.001,
                    patience=20,
                    augmentation=augmentation,
                    hidden_size=256,
                )
                all_results[app][mt] = result
                m = result['metrics']
                print(f"     F1={m['f1']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}"
                      f"  (epochs={result['epochs_run']}, params={result['num_params']:,})")
            except Exception as e:
                print(f"     ERROR: {e}")

    # ── Print summary tables ──
    print_table(all_results, 'f1',  'F1 SCORES')
    print_table(all_results, 'mae', 'MAE SCORES')
    print_table(all_results, 'sae', 'SAE SCORES')

    # ── Generate plots ──
    print('\nGenerating plots...')
    plot_metric_per_appliance(all_results, 'mae', 'MAE (Watts)',  save_dir)
    plot_metric_per_appliance(all_results, 'sae', 'SAE',          save_dir)
    plot_metric_per_appliance(all_results, 'f1',  'F1 Score',     save_dir)
    plot_summary_bar(all_results, save_dir)
    plot_f1_heatmap(all_results, save_dir)
    plot_epoch_metrics_combined(all_results, save_dir)

    for app in APPLIANCES:
        plot_training_curves(all_results, app, save_dir)

    # ── Save JSON ──
    results_file = os.path.join(save_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp':     timestamp,
            'augmentation':  augmentation,
            'epochs':        epochs,
            'results': {
                app: {
                    mt: {
                        'model_name': v['model_name'],
                        'num_params': v['num_params'],
                        'epochs_run': v['epochs_run'],
                        'metrics':    {k: float(val) for k, val in v['metrics'].items()},
                    }
                    for mt, v in mods.items()
                }
                for app, mods in all_results.items()
            }
        }, f, indent=4)

    print(f'\n✅ All results and plots saved to: {save_dir}')
    print('=' * 80)
    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='LNN Comprehensive Comparison — 80 epochs with LR scheduler and plots')
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'vertical', 'horizontal', 'both', 'mixed'],
                        help='Data augmentation mode (default: none)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Max training epochs (default: 80)')
    args = parser.parse_args()

    for fp in ['data/redd/train_small.pkl',
               'data/redd/val_small.pkl',
               'data/redd/test_small.pkl']:
        if not os.path.exists(fp):
            print(f'❌ Missing: {fp}')
            sys.exit(1)

    run(augmentation=args.augmentation, epochs=args.epochs)
