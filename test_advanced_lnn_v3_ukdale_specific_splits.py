"""
Advanced LNN v3 — UKDALE specific splits (single-phase training)

Same FullyAdaptiveLiquidTimeLayer architecture as test_advanced_lnn_v3_finetune.py
(hidden-and-input-adaptive tau: tau_min + softplus(tau_base + W_tau[x,h])),
but uses the UKDALE pickle data format and single-phase training structure
from test_lnn_ukdale_specific_splits.py.

tau parameterisation:
    Old: softplus(tau_base) * sigmoid(W_tau x_t)
    New: tau_min + softplus(tau_base + W_tau [x_t, h_t])
         — tau can grow OR shrink, conditioned on both input and hidden state.
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

EPOCHS     = 80
PATIENCE   = 20
LR         = 0.001
BATCH      = 32
WIN        = 100
STRIDE     = 5
TAU_MIN    = 0.01

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']


def get_threshold(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


# ---------------------------------------------------------------------------
# Model: fully adaptive tau LNN
# ---------------------------------------------------------------------------

class FullyAdaptiveLiquidTimeLayer(nn.Module):
    """
    LNN cell with hidden-and-input-adaptive tau.

        tau_t = tau_min + softplus(tau_base + W_tau [x_t, h_t])

    Unlike the original where tau_mod only saw x_t and was multiplicative
    (capped at tau_base), this formulation is additive and sees h_t, so
    tau can grow above tau_base when the hidden state signals an active cycle.
    """

    def __init__(self, input_size, hidden_size, dt=0.1, tau_min=TAU_MIN):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt      = dt
        self.tau_min = tau_min

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size + hidden_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden=None):
        B = x.size(0)
        if hidden is None:
            hidden = torch.zeros(B, self.hidden_size, device=x.device)

        combined = torch.cat([x, hidden], dim=1)
        inp  = self.input_proj(x)
        rec  = torch.matmul(hidden, self.rec_weights)
        gate = torch.sigmoid(self.gate(combined))
        tau  = self.tau_min + F.softplus(
            self.tau_base.unsqueeze(0) + self.tau_mod(combined))

        f_t = torch.tanh(inp + rec)
        dh  = ((-hidden / tau) + gate * f_t) * self.dt
        return (hidden + dh).clamp(-10.0, 10.0)


class FullyAdaptiveLNNModel(nn.Module):
    """Stacks N FullyAdaptiveLiquidTimeLayer cells; final hidden state → linear output."""

    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, dt=0.1, tau_min=TAU_MIN):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        self.liquid_layers = nn.ModuleList([
            FullyAdaptiveLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size, dt, tau_min)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.size()
        hs = [None] * self.num_layers
        for t in range(T):
            x_t = x[:, t, :]
            for i, lyr in enumerate(self.liquid_layers):
                inp  = x_t if i == 0 else hs[i - 1]
                hs[i] = lyr(inp, hs[i])
        return self.fc(hs[-1])


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

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, appliance_name):
    """Seq-to-point: window of WIN mains samples → appliance power at midpoint."""
    mains    = data['main'].values
    app_vals = data[appliance_name].values
    X, y = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        y.append(app_vals[i + WIN // 2])
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ---------------------------------------------------------------------------
# Per-appliance training
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name,
                       hidden_size=64, num_layers=2, dt=0.1, tau_min=TAU_MIN,
                       save_dir='models/advanced_lnn_v3_ukdale_specific'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_val, y_val), batch_size=BATCH, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_test, y_test), batch_size=BATCH, shuffle=False)

    model = FullyAdaptiveLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        num_layers=num_layers, dt=dt, tau_min=tau_min
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    threshold = get_threshold(appliance_name)
    print(f"Starting Advanced LNN v3 training for {appliance_name}...")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += criterion(model(inputs), targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(model(inputs).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train={avg_train_loss:.6f}  "
              f"val={avg_val_loss:.6f}  "
              f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"MAE={metrics['mae']:.2f}  SAE={metrics['sae']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
            save_model(
                model,
                {'input_size': 1, 'output_size': 1, 'hidden_size': hidden_size,
                 'num_layers': num_layers, 'dt': dt, 'tau_min': tau_min},
                {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                 'window_size': WIN, 'appliance': appliance_name},
                metrics,
                os.path.join(save_dir,
                             f"advanced_lnn_v3_ukdale_{appliance_name.replace(' ', '_')}_best.pth"))
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {PATIENCE}")
            if counter >= PATIENCE:
                print("Early stopping triggered")
                break

    print("Training completed!")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test ──
    print("Evaluating on test set...")
    model.eval()
    all_test_targets, all_test_outputs = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            test_loss += criterion(out, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(out.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    raw_test_tgts = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    raw_test_outs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_test_tgts, raw_test_outs, threshold=threshold)

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics — F1={test_metrics['f1']:.4f}  "
          f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  "
          f"MAE={test_metrics['mae']:.2f}  SAE={test_metrics['sae']:.4f}")

    # ── Plot ──
    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series,        label='Val F1',        color='red')
    plt.plot(val_precision_series, label='Val Precision', color='blue')
    plt.plot(val_recall_series,    label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall — {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"advanced_lnn_v3_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON ──
    aggregates = {
        'train_loss_mean':    float(np.mean(history['train_loss'])),
        'train_loss_var':     float(np.var(history['train_loss'])),
        'val_loss_mean':      float(np.mean(history['val_loss'])),
        'val_loss_var':       float(np.var(history['val_loss'])),
        'val_mae_mean':       float(np.mean(val_mae_series)),
        'val_mae_var':        float(np.var(val_mae_series)),
        'val_sae_mean':       float(np.mean(val_sae_series)),
        'val_sae_var':        float(np.var(val_sae_series)),
        'val_f1_mean':        float(np.mean(val_f1_series)),
        'val_f1_var':         float(np.var(val_f1_series)),
        'val_precision_mean': float(np.mean(val_precision_series)),
        'val_precision_var':  float(np.var(val_precision_series)),
        'val_recall_mean':    float(np.mean(val_recall_series)),
        'val_recall_var':     float(np.var(val_recall_series)),
        'test_mae':           float(test_metrics['mae']),
        'test_sae':           float(test_metrics['sae']),
        'test_f1':            float(test_metrics['f1']),
        'test_precision':     float(test_metrics['precision']),
        'test_recall':        float(test_metrics['recall']),
        'test_loss':          float(avg_test_loss),
    }

    config = {
        'appliance': appliance_name,
        'dataset': 'UKDALE',
        'model': 'FullyAdaptiveLNNModel (Advanced LNN v3)',
        'tau_parameterisation': {
            'formula': 'tau_min + softplus(tau_base + W_tau[x,h] + b_tau)',
            'tau_min': tau_min,
            'vs_original': 'softplus(tau_base) * sigmoid(W_tau x + b_tau)',
        },
        'loss': 'MSE',
        'window_size': WIN,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'dt': dt, 'tau_min': tau_min,
        },
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates,
        },
    }
    with open(os.path.join(save_dir,
              f'advanced_lnn_v3_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def test_on_all_appliances(hidden_size=64, num_layers=2, dt=0.1, tau_min=TAU_MIN):
    data_dict = load_ukdale_specific_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/advanced_lnn_v3_ukdale_specific_{timestamp}"

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Advanced LNN v3 — {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        try:
            _, _, test_metrics = train_on_appliance(
                data_dict,
                appliance_name=appliance_name,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dt=dt,
                tau_min=tau_min,
                save_dir=appliance_dir,
            )
            all_results[appliance_name] = {
                'model_path': os.path.join(
                    appliance_dir,
                    f"advanced_lnn_v3_ukdale_{appliance_name.replace(' ', '_')}_best.pth"),
                'final_metrics': {k: float(v) for k, v in test_metrics.items()},
            }
        except Exception as e:
            print(f"Error on {appliance_name}: {e}")
            import traceback; traceback.print_exc()

    os.makedirs(base_save_dir, exist_ok=True)
    summary = {
        'timestamp': timestamp,
        'dataset': 'UKDALE',
        'model': 'FullyAdaptiveLNNModel (Advanced LNN v3)',
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'},
        },
        'tau_parameterisation': {
            'formula': 'tau_min + softplus(tau_base + W_tau[x,h] + b_tau)',
            'tau_min': tau_min,
        },
        'model_params': {'hidden_size': hidden_size, 'num_layers': num_layers,
                         'dt': dt, 'tau_min': tau_min},
        'train_params': {'epochs': EPOCHS, 'lr': LR, 'patience': PATIENCE},
        'results': all_results,
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN v3 UKDALE testing completed. Results → {base_save_dir}\n")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]['final_metrics']
            print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Advanced LNN v3 on UKDALE dataset with specific splits...")

    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            import sys; sys.exit(1)

    results = test_on_all_appliances(
        hidden_size=64,
        num_layers=2,
        dt=0.1,
        tau_min=TAU_MIN,
    )

    print(f"\nSummary — Advanced LNN v3 UKDALE:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        m = result['final_metrics']
        print(f"  {appliance}:")
        print(f"    Test F1:        {m['f1']:.4f}")
        print(f"    Test Precision: {m['precision']:.4f}")
        print(f"    Test Recall:    {m['recall']:.4f}")
        print(f"    Test MAE:       {m['mae']:.4f}")
        print(f"    Test SAE:       {m['sae']:.4f}")
