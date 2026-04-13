"""
Selective LNN — learnable input-dependent dt (inspired by Mamba's selective discretisation).

Enhancement over AdvancedLiquidNetworkModelTwo:
  - dt is no longer a fixed hyperparameter
  - dt = softplus(W_dt · x_t)  per timestep, per layer
  - allows each layer to adapt its integration step to the input timescale
    (e.g. long dt for slow appliances like washing machine, short dt for microwave bursts)
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


# ---------------------------------------------------------------------------
# Selective Liquid Time Layer  (learnable input-dependent dt)
# ---------------------------------------------------------------------------

class SelectiveLiquidTimeLayer(nn.Module):
    """
    Enhanced LNN cell with input-dependent integration step (selective dt).

    Standard AdvancedLiquidTimeLayer uses a fixed dt hyperparameter:
        dh = (-h/tau + gate * f) * dt_fixed

    This layer replaces it with a learned, input-dependent step:
        dt = softplus(W_dt * x_t)          # scalar per timestep
        dh = (-h/tau + gate * f) * dt

    Inspired by Mamba's selective discretisation (Delta parameter).
    Allows the model to slow down (large dt) for slow dynamics like a
    washing machine cycle and speed up (small dt) for fast bursts like
    a microwave.
    """
    def __init__(self, input_size, hidden_size):
        super(SelectiveLiquidTimeLayer, self).__init__()
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Recurrent weights (not a full Linear to match AdvancedLiquidTimeLayer)
        self.rec_weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        # Adaptive time constant (tau) — same as AdvancedLiquidTimeLayer
        self.tau_base = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod  = nn.Linear(input_size, hidden_size)

        # Input-dependent gate
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)

        # *** Selective dt: maps input to a scalar step size ***
        self.W_dt = nn.Linear(input_size, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Input and recurrent projections
        input_proj = self.input_proj(x)
        rec_proj   = torch.matmul(hidden, self.rec_weights)

        # Adaptive tau (same as base model)
        tau_base = F.softplus(self.tau_base).unsqueeze(0)
        tau_mod  = self.sigmoid(self.tau_mod(x))
        tau      = (tau_base * tau_mod).clamp(min=1e-3)

        # Input-dependent gate
        combined = torch.cat([x, hidden], dim=1)
        gate     = self.sigmoid(self.gate(combined))

        # ODE target
        f_t = self.tanh(input_proj + rec_proj)

        # *** Selective dt: input determines integration step size ***
        dt = F.softplus(self.W_dt(x))          # (batch, 1) — broadcasts over hidden_size

        dh = (-hidden / tau + gate * f_t) * dt

        new_hidden = (hidden + dh).clamp(-10.0, 10.0)

        return new_hidden


# ---------------------------------------------------------------------------
# Selective LNN  (inter + intra LayerNorm, learnable dt)
# ---------------------------------------------------------------------------

class SelectiveLiquidNetworkModel(nn.Module):
    """
    Stacked SelectiveLiquidTimeLayer with inter-layer and intra-layer LayerNorm.

    Same structure as AdvancedLiquidNetworkModelTwo but with selective dt
    instead of a fixed hyperparameter.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SelectiveLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.liquid_layers = nn.ModuleList([
            SelectiveLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
            ) for i in range(num_layers)
        ])

        # Inter-layer LayerNorm: normalises hidden state crossing layer boundaries
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Intra-layer LayerNorm: normalises h_{t-1} before each timestep recurrence
        self.intra_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        hidden_states = [None] * self.num_layers

        for t in range(seq_length):
            x_t = x[:, t, :]

            for i in range(self.num_layers):
                # Intra-layer norm: normalise h_{t-1} before feeding back
                if hidden_states[i] is not None:
                    h_in = self.intra_norms[i](hidden_states[i])
                else:
                    h_in = hidden_states[i]

                # Inter-layer norm: use normed output of previous layer as input
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])

                hidden_states[i] = self.liquid_layers[i](inp, h_in)

        output = self.fc(self.layer_norms[-1](hidden_states[-1]))
        return output


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimpleDataset(torch.utils.data.Dataset):
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
    mains = data['main'].values
    X = []
    for i in range(0, len(mains) - window_size + 1, STRIDE):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def get_threshold_for_appliance(appliance_name):
    return THRESHOLDS[appliance_name]


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name, save_dir,
                       hidden_size=64, num_layers=2):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    threshold = get_threshold_for_appliance(appliance_name)

    X_tr = create_sequences(data_dict['train'], WIN)
    X_va = create_sequences(data_dict['val'],   WIN)
    X_te = create_sequences(data_dict['test'],  WIN)

    y_tr = data_dict['train'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_tr)]
    y_va = data_dict['val'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_va)]
    y_te = data_dict['test'][appliance_name].iloc[::STRIDE].values.reshape(-1, 1)[:len(X_te)]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_tr = y_scaler.fit_transform(y_tr)
    y_va = y_scaler.transform(y_va)
    y_te = y_scaler.transform(y_te)

    print(f"Training sequences:   {X_tr.shape} -> {y_tr.shape}")
    print(f"Validation sequences: {X_va.shape} -> {y_va.shape}")
    print(f"Test sequences:       {X_te.shape} -> {y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_va, y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_te, y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = SelectiveLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        output_size=1, num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print(f"Starting Selective-LNN training for {appliance_name}...")

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"[{appliance_name}] Epoch {epoch+1}/{EPOCHS}",
                            leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
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
                vl_loss += criterion(out, yb).item()
                val_preds.append(out.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va = vl_loss / len(va_loader)
        history['val_loss'].append(avg_va)
        scheduler.step(avg_va)

        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(val_trues).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(val_preds).reshape(-1, 1)).flatten()
        m = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=threshold)
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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred = y_scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(np.concatenate(trues).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(y_true, y_pred, threshold=threshold)

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
        f"selective_lnn_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'UKDALE',
        'model': 'SelectiveLiquidNetworkModel',
        'enhancement': 'learnable input-dependent dt (Mamba-inspired selective discretisation)',
        'norm': 'inter + intra LayerNorm',
        'loss': 'MSE',
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'num_layers': num_layers
        },
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'selective_lnn_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(hidden_size=64, num_layers=2):
    data_dict = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/selective_lnn_ukdale_test_{timestamp}"

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Testing Selective-LNN on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))

        try:
            test_metrics, _ = train_on_appliance(
                data_dict, appliance_name, appliance_dir,
                hidden_size=hidden_size, num_layers=num_layers,
            )
            all_results[appliance_name] = {k: float(v) for k, v in test_metrics.items()}
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset': 'UKDALE',
        'model': 'SelectiveLiquidNetworkModel',
        'enhancement': 'learnable input-dependent dt',
        'model_params': {'hidden_size': hidden_size, 'num_layers': num_layers},
        'train_params': {'epochs': EPOCHS, 'lr': LR, 'patience': PATIENCE},
        'results': all_results
    }
    os.makedirs(base_save_dir, exist_ok=True)
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nSelective-LNN UKDALE testing completed.")
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
    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    run_all(
        hidden_size=64,
        num_layers=2,
    )
