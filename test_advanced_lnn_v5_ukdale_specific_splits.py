"""
Advanced LNN v5 — UKDALE specific splits (single-phase training)

Same MultiTimescaleLNNModel architecture as test_advanced_lnn_v5_finetune.py:
  - Multi-timescale hidden state (fast + slow coupled LNN streams)
  - Appliance-specific tau ranges (from v5 observations)
  - Event-aware tau modulation (e_t = |x_t - x_{t-1}|)
  - Attentive pooling over all hidden states

Uses UKDALE pickle data format and single-phase training structure from
test_lnn_ukdale_specific_splits.py: EPOCHS=80, PATIENCE=20, LR=0.001.

tau parameterisation per stream:
    tau = tau_min + (tau_max - tau_min) * sigmoid(W_tau ctx)
    where ctx = cat([xe, h_fast, h_slow]) and xe = cat([x_t, e_t])
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
# Config
# ---------------------------------------------------------------------------

EPOCHS   = 80
PATIENCE = 20
LR       = 0.001
BATCH    = 32
WIN      = 100
STRIDE   = 5

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

# Appliance-specific tau ranges (adapted from v5 observations)
APPLIANCE_TAU = {
    'dish washer':  {'fast': (0.01, 1.0),  'slow': (0.5,  8.0)},
    'fridge':       {'fast': (0.05, 2.0),  'slow': (0.5,  6.0)},
    'microwave':    {'fast': (0.01, 0.5),  'slow': (0.1,  2.0)},
    'washer dryer': {'fast': (0.05, 2.0),  'slow': (1.0, 12.0)},
}


def get_threshold(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTimescaleLiquidLayer(nn.Module):
    """
    Two coupled LNN streams (fast and slow) sharing a combined context.

    Input:  xe (B, input_size)   where xe = cat([x_t, e_t])
    State:  h_fast (B, half),  h_slow (B, half)
    Context: ctx = cat([xe, h_fast, h_slow])

    Fast/slow each use sigmoid-bounded tau within appliance-specific range.
    """

    def __init__(self, input_size, hidden_size, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size must be even"
        self.half = hidden_size // 2
        self.dt   = dt
        self.tau_fast_min, self.tau_fast_max = tau_fast
        self.tau_slow_min, self.tau_slow_max = tau_slow

        ctx_size = input_size + hidden_size   # xe + h_fast + h_slow

        # Fast stream
        self.fast_proj = nn.Linear(input_size, self.half)
        self.fast_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.fast_tau  = nn.Linear(ctx_size, self.half)
        self.fast_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.fast_rec)

        # Slow stream
        self.slow_proj = nn.Linear(input_size, self.half)
        self.slow_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.slow_tau  = nn.Linear(ctx_size, self.half)
        self.slow_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.slow_rec)

    def forward(self, xe, h_fast=None, h_slow=None):
        B = xe.size(0)
        if h_fast is None: h_fast = torch.zeros(B, self.half, device=xe.device)
        if h_slow is None: h_slow = torch.zeros(B, self.half, device=xe.device)

        ctx = torch.cat([xe, h_fast, h_slow], dim=1)

        # Fast stream
        f_inp  = self.fast_proj(xe)
        f_rec  = torch.matmul(h_fast, self.fast_rec)
        f_gate = torch.sigmoid(self.fast_gate(ctx))
        f_tau  = (self.tau_fast_min
                  + (self.tau_fast_max - self.tau_fast_min)
                  * torch.sigmoid(self.fast_tau(ctx)))
        dh_f   = ((-h_fast / f_tau) + f_gate * torch.tanh(f_inp + f_rec)) * self.dt
        h_fast_new = (h_fast + dh_f).clamp(-10.0, 10.0)

        # Slow stream
        s_inp  = self.slow_proj(xe)
        s_rec  = torch.matmul(h_slow, self.slow_rec)
        s_gate = torch.sigmoid(self.slow_gate(ctx))
        s_tau  = (self.tau_slow_min
                  + (self.tau_slow_max - self.tau_slow_min)
                  * torch.sigmoid(self.slow_tau(ctx)))
        dh_s   = ((-h_slow / s_tau) + s_gate * torch.tanh(s_inp + s_rec)) * self.dt
        h_slow_new = (h_slow + dh_s).clamp(-10.0, 10.0)

        return h_fast_new, h_slow_new


class MultiTimescaleLNNModel(nn.Module):
    """
    Multi-timescale LNN with event signal and attentive pooling.

    x (B,T,1) → xe=[x,e] → fast+slow LNN streams → attentive pool → fc → output
    """

    def __init__(self, input_size=1, hidden_size=64, output_size=1, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        self.hidden_size = hidden_size
        # +1 for the event signal e_t appended to x_t
        self.lnn  = MultiTimescaleLiquidLayer(
            input_size + 1, hidden_size, dt, tau_fast, tau_slow)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.size()

        # Event signal: rate-of-change of aggregate; e_0 = 0
        e = torch.zeros_like(x)
        e[:, 1:, :] = (x[:, 1:, :] - x[:, :-1, :]).abs()

        h_fast = h_slow = None
        states = []
        for t in range(T):
            xe = torch.cat([x[:, t, :], e[:, t, :]], dim=1)
            h_fast, h_slow = self.lnn(xe, h_fast, h_slow)
            states.append(torch.cat([h_fast, h_slow], dim=1))

        states  = torch.stack(states, dim=1)           # (B, T, hidden)
        scores  = self.attn(states)                    # (B, T, 1)
        weights = F.softmax(scores, dim=1)             # (B, T, 1)
        context = (weights * states).sum(dim=1)        # (B, hidden)
        return self.fc(context)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKDALEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, appliance_name, window_size=WIN):
    mains    = data['main'].values
    app_vals = data[appliance_name].values
    X, y = [], []
    for i in range(0, len(mains) - window_size + 1, STRIDE):
        X.append(mains[i:i + window_size])
        y.append(app_vals[i + window_size // 2])
    return (np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ---------------------------------------------------------------------------
# Per-appliance training
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name, hidden_size=64, dt=0.1,
                       save_dir='models/advanced_lnn_v5_ukdale_specific'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_cfg   = APPLIANCE_TAU[appliance_name]
    tau_fast  = tau_cfg['fast']
    tau_slow  = tau_cfg['slow']
    threshold = get_threshold(appliance_name)

    print(f"\nAppliance: {appliance_name}  |  device: {device}  "
          f"tau_fast={tau_fast}  tau_slow={tau_slow}")

    X_train, y_train = create_sequences(data_dict['train'], appliance_name)
    X_val,   y_val   = create_sequences(data_dict['val'],   appliance_name)
    X_test,  y_test  = create_sequences(data_dict['test'],  appliance_name)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    print(f"  Training sequences:   {X_train.shape}")
    print(f"  Validation sequences: {X_val.shape}")
    print(f"  Test sequences:       {X_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        UKDALEDataset(X_val,   y_val),   batch_size=BATCH, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(
        UKDALEDataset(X_test,  y_test),  batch_size=BATCH, shuffle=False)

    model = MultiTimescaleLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1, dt=dt,
        tau_fast=tau_fast, tau_slow=tau_slow).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print(f"  Training MultiTimescaleLNNModel...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        model.eval()
        val_loss = 0.0
        all_tgts, all_outs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                all_tgts.append(yb.cpu().numpy())
                all_outs.append(out.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)
        scheduler.step(avg_val)

        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_tgts).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outs).reshape(-1, 1)).flatten()
        m = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=threshold)
        history['val_metrics'].append(m)

        print(f"Epoch {epoch+1}/{EPOCHS}  train={avg_train:.6f}  val={avg_val:.6f}  "
              f"MAE={m['mae']:.2f}  SAE={m['sae']:.2f}  F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            save_model(
                model,
                {'input_size': 1, 'output_size': 1, 'hidden_size': hidden_size,
                 'dt': dt, 'tau_fast': list(tau_fast), 'tau_slow': list(tau_slow)},
                {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                 'appliance': appliance_name},
                m,
                os.path.join(save_dir,
                    f"advanced_lnn_v5_{appliance_name.replace(' ', '_')}_best.pth"))
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {PATIENCE}")
            if counter >= PATIENCE:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_test_tgts, all_test_outs = [], []
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            test_loss += criterion(out, yb).item()
            all_test_tgts.append(yb.cpu().numpy())
            all_test_outs.append(out.cpu().numpy())

    raw_test_tgts = y_scaler.inverse_transform(
        np.concatenate(all_test_tgts).reshape(-1, 1)).flatten()
    raw_test_outs = y_scaler.inverse_transform(
        np.concatenate(all_test_outs).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_test_tgts, raw_test_outs, threshold=threshold)

    # Plots
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(ep, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(ep, history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(ep, [m['mae'] for m in history['val_metrics']], label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(ep, [m['sae'] for m in history['val_metrics']], label='Val SAE', color='purple')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(ep, [m['f1']        for m in history['val_metrics']], label='Val F1',        color='red')
    plt.plot(ep, [m['precision'] for m in history['val_metrics']], label='Val Precision', color='blue')
    plt.plot(ep, [m['recall']    for m in history['val_metrics']], label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"advanced_lnn_v5_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()

    # JSON
    aggregates = {
        'train_loss_mean': float(np.mean(history['train_loss'])),
        'train_loss_var':  float(np.var(history['train_loss'])),
        'val_loss_mean':   float(np.mean(history['val_loss'])),
        'val_loss_var':    float(np.var(history['val_loss'])),
        'val_f1_mean':     float(np.mean([m['f1']        for m in history['val_metrics']])),
        'val_f1_var':      float(np.var( [m['f1']        for m in history['val_metrics']])),
        'val_mae_mean':    float(np.mean([m['mae']       for m in history['val_metrics']])),
        'val_mae_var':     float(np.var( [m['mae']       for m in history['val_metrics']])),
        'val_sae_mean':    float(np.mean([m['sae']       for m in history['val_metrics']])),
        'val_sae_var':     float(np.var( [m['sae']       for m in history['val_metrics']])),
        'test_f1':         float(test_metrics['f1']),
        'test_mae':        float(test_metrics['mae']),
        'test_sae':        float(test_metrics['sae']),
        'test_precision':  float(test_metrics['precision']),
        'test_recall':     float(test_metrics['recall']),
    }
    config = {
        'appliance': appliance_name,
        'dataset': 'UKDALE',
        'model': 'MultiTimescaleLNNModel (Advanced LNN v5)',
        'architecture': {
            'multi_timescale': 'fast + slow hidden streams, each (hidden/2)',
            'event_aware_tau': 'e_t = |x_t - x_{t-1}| fed into tau and gate',
            'attentive_pooling': 'scalar softmax attention over all T states',
            'appliance_specific_tau': True,
        },
        'tau': {'fast': list(tau_fast), 'slow': list(tau_slow)},
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'aggregates': aggregates,
    }
    with open(os.path.join(save_dir,
            f'advanced_lnn_v5_{appliance_name.replace(" ", "_")}_results.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(hidden_size=64, dt=0.1):
    data_dict = load_ukdale_specific_splits()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = f'models/advanced_lnn_v5_ukdale_specific_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)

    all_results = {}

    for app in APPLIANCES:
        print(f"\n{'='*60}\nAdvanced LNN v5 — {app}\n{'='*60}")
        app_dir = os.path.join(base_dir, app.replace(' ', '_'))
        try:
            _, _, test_metrics = train_on_appliance(
                data_dict, app, hidden_size=hidden_size, dt=dt, save_dir=app_dir)
            all_results[app] = {k: float(v) for k, v in test_metrics.items()}
        except Exception as e:
            print(f"Error on {app}: {e}")
            import traceback; traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'model': 'MultiTimescaleLNNModel (Advanced LNN v5)',
        'dataset': 'UKDALE',
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'},
        },
        'improvements': [
            'multi_timescale_hidden_state (fast + slow streams)',
            'appliance_specific_tau_ranges',
            'event_aware_tau (e_t = |x_t - x_{t-1}|)',
            'attentive_pooling_over_all_states',
        ],
        'appliance_tau': {k: {'fast': list(v['fast']), 'slow': list(v['slow'])}
                          for k, v in APPLIANCE_TAU.items()},
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'train_params': {'epochs': EPOCHS, 'lr': LR, 'patience': PATIENCE},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN v5 UKDALE complete.  Results → {base_dir}")
    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]
            print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    valid = [all_results[a]['f1'] for a in APPLIANCES if a in all_results]
    if valid:
        print(f"\n  Avg F1: {np.mean(valid):.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    run_all(hidden_size=64, dt=0.1)
