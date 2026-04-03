import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))

from models import TCNAdvancedLiquidNetworkModel
from utils import save_model


# ---------------------------------------------------------------------------
# Constants  (mirrors run_baseline_models_80epochs.py)
# ---------------------------------------------------------------------------

EPOCHS   = 80
PATIENCE = 20
LR       = 1e-3
BATCH    = 32
WIN      = 100
STRIDE   = 5

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

# Per-appliance asymmetric BCE parameters
# beta > alpha  -> penalise false positives  (dish washer: rarely ON)
# alpha > beta  -> penalise false negatives  (microwave, washer dryer)
# alpha == beta -> symmetric                 (fridge: balanced)
APPLIANCE_LOSS_PARAMS = {
    'dish washer':  {'alpha': 0.25, 'beta': 4.0},
    'fridge':       {'alpha': 0.75, 'beta': 0.75},
    'microwave':    {'alpha': 2.0,  'beta': 0.5},
    'washer dryer': {'alpha': 1.5,  'beta': 0.5},
}


# ---------------------------------------------------------------------------
# Asymmetric Weighted BCE Loss
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """
    L_total = L_MSE + bce_lambda * L_BCE_asymmetric

    L_BCE_asymmetric = -1/N * sum[
        alpha * y_i     * log(sigmoid(output_i))       +
        beta  * (1-y_i) * log(1 - sigmoid(output_i))
    ]

    beta > alpha  -> penalise false positives more (dish washer)
    alpha > beta  -> penalise false negatives more (microwave, washer dryer)
    """
    def __init__(self, alpha=0.5, beta=2.0, bce_lambda=0.1):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_lambda = bce_lambda
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, threshold):
        loss_mse = self.mse(outputs, targets)

        y = (targets >= threshold).float()
        p = torch.clamp(torch.sigmoid(outputs), min=1e-7, max=1.0 - 1e-7)

        loss_bce = -(
            self.alpha * y       * torch.log(p) +
            self.beta  * (1 - y) * torch.log(1 - p)
        ).mean()

        return loss_mse + self.bce_lambda * loss_bce


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data helpers  (z-score X, no y-scaling — consistent with baseline)
# ---------------------------------------------------------------------------

def load_data():
    print("Loading REDD data...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(f'data/redd/{split}_small.pkl', 'rb') as f:
            splits[split] = pickle.load(f)[0]
    print(f"Train date range: {splits['train'].index.min()} to {splits['train'].index.max()}")
    print(f"Val   date range: {splits['val'].index.min()}   to {splits['val'].index.max()}")
    print(f"Test  date range: {splits['test'].index.min()}  to {splits['test'].index.max()}")
    print(f"Available columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(df, appliance_name, window_size=WIN, stride=STRIDE):
    mains   = df['main'].values
    targets = df[appliance_name].values
    X, y = [], []
    for i in range(0, len(mains) - window_size, stride):
        X.append(mains[i:i + window_size])
        midpoint = i + window_size // 2
        y.append(targets[midpoint])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Metrics  (matches run_baseline_models_80epochs.py exactly)
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, threshold):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))

    N = 100
    num_periods = len(y_true) // N
    diff = sum(
        abs(np.sum(y_true[i*N:(i+1)*N]) - np.sum(y_pred[i*N:(i+1)*N]))
        for i in range(num_periods)
    )
    sae = float(diff / (N * num_periods)) if num_periods > 0 else 0.0

    t_bin = (y_true > threshold).astype(int)
    p_bin = (y_pred > threshold).astype(int)
    tp = int(np.sum((t_bin == 1) & (p_bin == 1)))
    fp = int(np.sum((t_bin == 0) & (p_bin == 1)))
    fn = int(np.sum((t_bin == 1) & (p_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'mae': mae, 'sae': sae, 'f1': float(f1),
            'precision': float(precision), 'recall': float(recall)}


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_on_appliance(splits, appliance_name, save_dir,
                       hidden_size=64, num_layers=2, dt=0.1,
                       num_channels=None, kernel_size=3, dropout=0.2,
                       bce_lambda=0.1):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    thr      = THRESHOLDS[appliance_name]
    loss_p   = APPLIANCE_LOSS_PARAMS[appliance_name]
    mode_str = 'FP-penalised' if loss_p['beta'] > loss_p['alpha'] else 'FN-penalised'

    # Sequences
    X_tr, y_tr = create_sequences(splits['train'], appliance_name)
    X_va, y_va = create_sequences(splits['val'],   appliance_name)
    X_te, y_te = create_sequences(splits['test'],  appliance_name)

    # Z-score normalise X using training stats (no y-scaling)
    mu    = float(X_tr.mean())
    sigma = float(X_tr.std()) + 1e-8
    X_tr = (X_tr - mu) / sigma
    X_va = (X_va - mu) / sigma
    X_te = (X_te - mu) / sigma

    print(f"Training sequences:   {X_tr.shape} -> {y_tr.shape}")
    print(f"Validation sequences: {X_va.shape} -> {y_va.shape}")
    print(f"Test sequences:       {X_te.shape} -> {y_te.shape}")

    num_on  = int((y_tr >= thr).sum())
    num_off = int((y_tr <  thr).sum())
    print(f"Threshold: {thr}W  |  ON: {num_on}  |  OFF: {num_off}")
    print(f"Asymmetric BCE -> alpha={loss_p['alpha']}  beta={loss_p['beta']}  "
          f"bce_lambda={bce_lambda}  [{mode_str}]")

    tr_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_va, y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_te, y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = TCNAdvancedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        dt=dt, num_channels=num_channels, kernel_size=kernel_size,
        dropout=dropout, num_layers=num_layers
    ).to(device)

    criterion = AsymmetricLoss(alpha=loss_p['alpha'], beta=loss_p['beta'],
                               bce_lambda=bce_lambda)
    mse_only  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print(f"Starting TCN-Advanced-LNN (Asymmetric Loss) training for {appliance_name}...")

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"[{appliance_name}] Epoch {epoch+1}/{EPOCHS}",
                            leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb, thr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_tr = ep_loss / len(tr_loader)
        history['train_loss'].append(avg_tr)

        model.eval()
        vl_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl_loss += mse_only(out, yb).item()
                val_preds.append(out.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va = vl_loss / len(va_loader)
        history['val_loss'].append(avg_va)
        scheduler.step(avg_va)

        ep_pred = np.concatenate(val_preds)
        ep_true = np.concatenate(val_trues)
        m = calculate_metrics(ep_true, ep_pred, thr)
        history['val_metrics'].append(m)

        print(f"  [{appliance_name}] Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={avg_tr:.5f}  val_mse={avg_va:.5f}  "
              f"F1={m['f1']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_va < best_val_loss:
            best_val_loss = avg_va
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    # Evaluate best val loss model on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    test_metrics = calculate_metrics(y_true, y_pred, thr)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean':      float(np.mean(history['train_loss'])),
        'train_loss_var':       float(np.var(history['train_loss'])),
        'val_loss_mean':        float(np.mean(history['val_loss'])),
        'val_loss_var':         float(np.var(history['val_loss'])),
        'val_mae_mean':         float(np.mean(val_mae_series)),
        'val_mae_var':          float(np.var(val_mae_series)),
        'val_sae_mean':         float(np.mean(val_sae_series)),
        'val_sae_var':          float(np.var(val_sae_series)),
        'val_f1_mean':          float(np.mean(val_f1_series)),
        'val_f1_var':           float(np.var(val_f1_series)),
        'val_precision_mean':   float(np.mean(val_precision_series)),
        'val_precision_var':    float(np.var(val_precision_series)),
        'val_recall_mean':      float(np.mean(val_recall_series)),
        'val_recall_var':       float(np.var(val_recall_series)),
        'test_mae':             float(test_metrics['mae']),
        'test_sae':             float(test_metrics['sae']),
        'test_f1':              float(test_metrics['f1']),
        'test_precision':       float(test_metrics['precision']),
        'test_recall':          float(test_metrics['recall']),
    }

    print(f"  Test MAE={test_metrics['mae']:.4f}  SAE={test_metrics['sae']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}")

    # Plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val MSE',    color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series,        label='Val F1',        color='red')
    plt.plot(val_precision_series, label='Val Precision', color='blue')
    plt.plot(val_recall_series,    label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"tcn_advanced_lnn_asymmetric_loss_redd_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'REDD',
        'loss': 'MSE + Asymmetric BCE',
        'loss_params': {
            'alpha': loss_p['alpha'], 'beta': loss_p['beta'],
            'bce_lambda': bce_lambda, 'mode': mode_str
        },
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'final_metrics': {
            'test_metrics': test_metrics,
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'tcn_advanced_lnn_asymmetric_loss_redd_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(hidden_size=64, num_layers=2, dt=0.1,
            num_channels=None, kernel_size=3, dropout=0.2,
            bce_lambda=0.1):
    if num_channels is None:
        num_channels = [32, 64, 128]

    splits = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_advanced_lnn_asymmetric_loss_redd_test_{timestamp}"

    all_results = {}

    for appliance_name in APPLIANCES:
        lp   = APPLIANCE_LOSS_PARAMS[appliance_name]
        mode = 'FP-penalised' if lp['beta'] > lp['alpha'] else 'FN-penalised'
        print(f"\n{'='*60}")
        print(f"Testing TCN-Advanced-LNN (Asymmetric Loss) on {appliance_name}")
        print(f"  alpha={lp['alpha']}  beta={lp['beta']}  [{mode}]")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))

        try:
            test_metrics, _ = train_on_appliance(
                splits, appliance_name, appliance_dir,
                hidden_size=hidden_size, num_layers=num_layers, dt=dt,
                num_channels=num_channels, kernel_size=kernel_size,
                dropout=dropout, bce_lambda=bce_lambda
            )
            all_results[appliance_name] = test_metrics
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    summary = {
        'timestamp': timestamp,
        'dataset': 'REDD',
        'loss': 'MSE + Asymmetric BCE',
        'loss_params': {app: APPLIANCE_LOSS_PARAMS[app] for app in APPLIANCES},
        'bce_lambda': bce_lambda,
        'model_params': {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'epochs': EPOCHS, 'lr': LR, 'patience': PATIENCE},
        'results': all_results
    }
    os.makedirs(base_save_dir, exist_ok=True)
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    # Console table
    print(f"\nTCN-Advanced-LNN (Asymmetric Loss) REDD testing completed.")
    print(f"Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    f1s, maes, saes, precs, recs = [], [], [], [], []
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]
            print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")
            f1s.append(m['f1']); maes.append(m['mae']); saes.append(m['sae'])
            precs.append(m['precision']); recs.append(m['recall'])
    if f1s:
        print("-" * 65)
        print(f"{'Average':<15} {np.mean(f1s):>8.4f} {np.mean(precs):>10.4f} "
              f"{np.mean(recs):>8.4f} {np.mean(maes):>8.2f} {np.mean(saes):>8.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for f in ['data/redd/train_small.pkl', 'data/redd/val_small.pkl',
              'data/redd/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    run_all(
        hidden_size=64,
        num_layers=2,
        dt=0.1,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        bce_lambda=0.1,
    )
