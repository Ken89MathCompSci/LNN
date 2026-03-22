"""
Gated GRU-GNN-LNN on REDD with specific splits.

Architecture:
    GRU branch  : BiGRU → projection → LiquidODECell → h_t  (temporal embedding)
    GNN branch  : TCN → node init → GCN layers → LiquidODECell per node → r_t
    Gate        : g_t[n] = σ( W_g · [h_t ; r_t[n]] + b_g )
    Mix         : z_t[n] = g_t[n] ⊙ h_t + (1 − g_t[n]) ⊙ r_t[n]
    Decoder     : z_t[n] → FC → appliance power

The gate g_t is saved and visualised post-training.  A mean gate value near 1
for a node means the model relied primarily on the GRU-LNN temporal branch;
near 0 means it relied on the GNN-LNN relational branch.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add Source Code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))

from models import GRUGNNLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
THRESHOLDS = {'dish washer': 10.0, 'fridge': 10.0,
              'microwave': 10.0, 'washer dryer': 0.5}


class PerApplianceWeightedMSELoss(torch.nn.Module):
    """
    Per-appliance weighted MSE loss that upweights ON-state samples.

    For each appliance i:
        weight[i, sample] = on_weight[i]  if target > threshold_norm[i]
                          = 1.0           otherwise
        loss += mean(weight * (pred - target)^2)

    on_weight is computed from training data class imbalance:
        on_weight = (1 - on_ratio) / on_ratio,  capped at 30x
    """
    def __init__(self, on_weights, on_thresholds_norm):
        super().__init__()
        self.on_weights         = [min(float(w), 30.0) for w in on_weights]
        self.on_thresholds_norm = [float(t) for t in on_thresholds_norm]

    def forward(self, pred, target):
        # pred, target: (B, N)
        total = 0.0
        for i in range(pred.shape[1]):
            p_i = pred[:, i]
            t_i = target[:, i]
            w_i = torch.where(
                t_i > self.on_thresholds_norm[i],
                torch.full_like(t_i, self.on_weights[i]),
                torch.ones_like(t_i)
            )
            total = total + torch.mean(w_i * (p_i - t_i) ** 2)
        return total / pred.shape[1]


class REDDMultiDataset(torch.utils.data.Dataset):
    """Dataset returning (aggregate_window, all_appliance_targets)."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)   # (N, window, 1)
        self.y = torch.FloatTensor(y)   # (N, 4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_redd_specific_splits():
    print("Loading REDD data with specific splits...")

    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/redd/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")

    for app in APPLIANCES:
        if app not in train_data.columns:
            print(f"Warning: {app} not in columns")
    print(f"Available columns: {list(train_data.columns)}")

    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=100):
    """Return X (N, window, 1) using aggregate mains, stride=5."""
    mains = data['main'].values
    X = []
    stride = 5
    for i in range(0, len(mains) - window_size, stride):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def build_targets(data, n_samples):
    """Return y (n_samples, 4) — one column per appliance, iloc[::5] aligned."""
    cols = []
    for app in APPLIANCES:
        vals = data[app].iloc[::5].values[:n_samples]
        cols.append(vals)
    return np.column_stack(cols)


def compute_adjacency(train_data):
    """
    Build a normalised GCN adjacency matrix from appliance co-activation.
    Steps: binarise → |Pearson correlation| → self-loops → D^{-1/2} A D^{-1/2}.
    """
    binary = pd.DataFrame({
        app: (train_data[app] > THRESHOLDS[app]).astype(float)
        for app in APPLIANCES
    })
    corr = binary.corr().values.astype(np.float32)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)

    d = corr.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
    adj = (d_inv_sqrt @ corr @ d_inv_sqrt).astype(np.float32)

    print("Adjacency matrix (appliance co-activation):")
    print(pd.DataFrame(adj, index=APPLIANCES, columns=APPLIANCES).round(3).to_string())
    return adj


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gru_gnn_lnn(data_dict, window_size=100,
                      gru_hidden=64, num_gru_layers=2,
                      num_channels=None, kernel_size=3, dropout=0.2,
                      hidden_size=64, dt=0.1, num_gcn_layers=2,
                      epochs=80, lr=0.001, patience=20,
                      save_dir='models/gru_gnn_lnn_redd_specific'):

    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    adj = compute_adjacency(train_data)

    print("Creating sequences...")
    X_train = create_sequences(train_data, window_size)
    X_val   = create_sequences(val_data,   window_size)
    X_test  = create_sequences(test_data,  window_size)

    y_train = build_targets(train_data, len(X_train))
    y_val   = build_targets(val_data,   len(X_val))
    y_test  = build_targets(test_data,  len(X_test))

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_scalers = {}
    on_weights         = []
    on_thresholds_norm = []
    for i, app in enumerate(APPLIANCES):
        sc = MinMaxScaler()
        # Compute class imbalance weight BEFORE normalising
        raw_vals  = y_train[:, i]
        on_ratio  = float((raw_vals > THRESHOLDS[app]).mean())
        on_ratio  = max(on_ratio, 1e-6)
        on_weight = (1.0 - on_ratio) / on_ratio
        on_weights.append(on_weight)

        y_train[:, i:i+1] = sc.fit_transform(y_train[:, i:i+1])
        y_val[:,   i:i+1] = sc.transform(y_val[:,   i:i+1])
        y_test[:,  i:i+1] = sc.transform(y_test[:,  i:i+1])
        y_scalers[app] = sc

        thresh_norm = float(sc.transform([[THRESHOLDS[app]]])[0][0])
        on_thresholds_norm.append(thresh_norm)
        print(f"  {app}: on_ratio={on_ratio:.3f}  on_weight={min(on_weight,30):.1f}x  "
              f"threshold_norm={thresh_norm:.4f}")

    train_loader = torch.utils.data.DataLoader(
        REDDMultiDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        REDDMultiDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        REDDMultiDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = GRUGNNLiquidNetworkModel(
        input_size=1,
        hidden_size=hidden_size,
        num_nodes=len(APPLIANCES),
        gru_hidden=gru_hidden,
        num_gru_layers=num_gru_layers,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        dt=dt,
        num_gcn_layers=num_gcn_layers,
        adj_matrix=adj
    ).to(device)

    criterion = PerApplianceWeightedMSELoss(on_weights, on_thresholds_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    mse_loss = torch.nn.MSELoss()   # used for val/test tracking only
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    counter = 0

    print(f"\nStarting GRU-GNN-LNN training ({len(APPLIANCES)} appliances simultaneously)...")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pb:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)          # ignore gates during training
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pb.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_targets_list, all_outputs_list = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                val_loss += mse_loss(outputs, targets).item()
                all_targets_list.append(targets.cpu().numpy())
                all_outputs_list.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        all_targets_np = np.concatenate(all_targets_list)
        all_outputs_np = np.concatenate(all_outputs_list)

        epoch_metrics = {}
        for i, app in enumerate(APPLIANCES):
            t = y_scalers[app].inverse_transform(
                all_targets_np[:, i:i+1]).flatten()
            o = y_scalers[app].inverse_transform(
                all_outputs_np[:, i:i+1]).flatten()
            epoch_metrics[app] = calculate_nilm_metrics(t, o, threshold=THRESHOLDS[app])

        history['val_metrics'].append(epoch_metrics)

        avg_mae = np.mean([epoch_metrics[a]['mae']       for a in APPLIANCES])
        avg_sae = np.mean([epoch_metrics[a]['sae']       for a in APPLIANCES])
        avg_f1  = np.mean([epoch_metrics[a]['f1']        for a in APPLIANCES])
        avg_pre = np.mean([epoch_metrics[a]['precision'] for a in APPLIANCES])
        avg_rec = np.mean([epoch_metrics[a]['recall']    for a in APPLIANCES])
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Avg MAE: {avg_mae:.2f}, Avg SAE: {avg_sae:.2f}, Avg F1: {avg_f1:.4f}, "
              f"Avg Precision: {avg_pre:.4f}, Avg Recall: {avg_rec:.4f}")

        for app in APPLIANCES:
            m = epoch_metrics[app]
            print(f"  {app:15s}: MAE={m['mae']:.2f}  SAE={m['sae']:.2f}  F1={m['f1']:.4f}  "
                  f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_path = os.path.join(save_dir, 'gru_gnn_lnn_redd_best.pth')
            torch.save({'model_state': model.state_dict(),
                        'adj': adj,
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss}, best_path)
            print(f"  Model saved to {best_path}")
        else:
            counter += 1
            print(f"  EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    # --- Test ---
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    all_test_t, all_test_o, all_test_gates = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, gates = model(inputs)
            test_loss += mse_loss(outputs, targets).item()
            all_test_t.append(targets.cpu().numpy())
            all_test_o.append(outputs.cpu().numpy())
            all_test_gates.append(gates.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_test_t     = np.concatenate(all_test_t)      # (N_test, 4)
    all_test_o     = np.concatenate(all_test_o)      # (N_test, 4)
    all_test_gates = np.concatenate(all_test_gates)  # (N_test, 4, hidden)

    # Mean gate value per appliance (averaged over hidden dim and test samples)
    mean_gate_per_app = all_test_gates.mean(axis=(0, 2))   # (4,)

    test_metrics = {}
    for i, app in enumerate(APPLIANCES):
        t = y_scalers[app].inverse_transform(all_test_t[:, i:i+1]).flatten()
        o = y_scalers[app].inverse_transform(all_test_o[:, i:i+1]).flatten()
        test_metrics[app] = calculate_nilm_metrics(t, o, threshold=THRESHOLDS[app])

    def agg(key, app):
        vals = [m[app][key] for m in history['val_metrics']]
        return float(np.mean(vals)), float(np.var(vals))

    aggregates = {}
    for app in APPLIANCES:
        mae_mean, mae_var = agg('mae', app)
        sae_mean, sae_var = agg('sae', app)
        f1_mean,  f1_var  = agg('f1',  app)
        pre_mean, pre_var = agg('precision', app)
        rec_mean, rec_var = agg('recall',    app)
        aggregates[app] = {
            'val_mae_mean': mae_mean, 'val_mae_var': mae_var,
            'val_sae_mean': sae_mean, 'val_sae_var': sae_var,
            'val_f1_mean':  f1_mean,  'val_f1_var':  f1_var,
            'val_precision_mean': pre_mean, 'val_precision_var': pre_var,
            'val_recall_mean':    rec_mean, 'val_recall_var':    rec_var,
            'test_mae':       float(test_metrics[app]['mae']),
            'test_sae':       float(test_metrics[app]['sae']),
            'test_f1':        float(test_metrics[app]['f1']),
            'test_precision': float(test_metrics[app]['precision']),
            'test_recall':    float(test_metrics[app]['recall']),
            'mean_gate':      float(mean_gate_per_app[APPLIANCES.index(app)])
        }

    print(f"\nTest Loss: {avg_test_loss:.6f}")
    print("\nTest Metrics per appliance:")
    for app in APPLIANCES:
        m = test_metrics[app]
        g = mean_gate_per_app[APPLIANCES.index(app)]
        print(f"  {app:15s}: MAE={m['mae']:.2f}  SAE={m['sae']:.2f}  F1={m['f1']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
              f"Gate={g:.3f} ({'GRU' if g > 0.5 else 'GNN'})")
    print("\nAggregates (mean/variance over epochs):")
    print(json.dumps(aggregates, indent=2))

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'],   label='Val Loss',   color='red')
    axes[0, 0].set_title('Loss'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Val MAE per appliance
    for app, c in zip(APPLIANCES, colors):
        mae_series = [m[app]['mae'] for m in history['val_metrics']]
        axes[0, 1].plot(mae_series, label=app, color=c)
    axes[0, 1].set_title('Val MAE per appliance'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (W)'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Val F1 per appliance
    for app, c in zip(APPLIANCES, colors):
        f1_series = [m[app]['f1'] for m in history['val_metrics']]
        axes[0, 2].plot(f1_series, label=app, color=c)
    axes[0, 2].set_title('Val F1 per appliance'); axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1'); axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # Adjacency heatmap
    im = axes[1, 0].imshow(adj, cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_xticks(range(len(APPLIANCES)))
    axes[1, 0].set_yticks(range(len(APPLIANCES)))
    short = ['dish\nwasher', 'fridge', 'micro\nwave', 'washer\ndryer']
    axes[1, 0].set_xticklabels(short, fontsize=8)
    axes[1, 0].set_yticklabels(short, fontsize=8)
    axes[1, 0].set_title('Adjacency Matrix')
    plt.colorbar(im, ax=axes[1, 0])

    # Mean gate per appliance (bar chart)
    bar_colors = ['#2196F3' if g > 0.5 else '#4CAF50' for g in mean_gate_per_app]
    bars = axes[1, 1].bar(range(len(APPLIANCES)), mean_gate_per_app,
                          color=bar_colors, edgecolor='black', linewidth=0.8)
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', linewidth=1.2, label='neutral (0.5)')
    axes[1, 1].set_xticks(range(len(APPLIANCES)))
    axes[1, 1].set_xticklabels(['dish\nwasher', 'fridge', 'micro\nwave', 'washer\ndryer'],
                                fontsize=9)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel('Mean gate value')
    axes[1, 1].set_title('Gate Interpretation\n(blue=GRU dominant, green=GNN dominant)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, mean_gate_per_app):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{g:.3f}', ha='center', va='bottom', fontsize=9)

    # Gate distribution across hidden dims per appliance (box plot)
    gate_by_app = [all_test_gates[:, i, :].flatten() for i in range(len(APPLIANCES))]
    axes[1, 2].boxplot(gate_by_app, labels=['dish\nwasher', 'fridge', 'micro\nwave', 'washer\ndryer'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#E3F2FD', color='#1565C0'),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 2].axhline(0.5, color='gray', linestyle='--', linewidth=1.2)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_ylabel('Gate value')
    axes[1, 2].set_title('Gate Distribution per Appliance')
    axes[1, 2].grid(axis='y', alpha=0.3)

    plt.suptitle('GRU-GNN-LNN (Gated Fusion) — REDD', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gru_gnn_lnn_redd_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save config ---
    config = {
        'dataset': 'REDD',
        'dataset_splits': {
            'training':   {'house': 3, 'time_start': '2011-04-21 19:41:24', 'time_end': '2011-04-22 19:41:21'},
            'validation': {'house': 3, 'time_start': '2011-05-23 10:31:24', 'time_end': '2011-05-24 10:31:21'},
            'testing':    {'house': 1, 'time_start': '2011-04-18 09:22:12', 'time_end': '2011-05-23 09:21:51'}
        },
        'appliances': {
            'dish washer':  {'on': 1143,  'off': 27657, 'total': 28800},
            'fridge':       {'on': 10471, 'off': 18329, 'total': 28800},
            'microwave':    {'on': 531,   'off': 28269, 'total': 28800},
            'washer dryer': {'on': 1879,  'off': 26921, 'total': 28800}
        },
        'window_size': window_size,
        'model_params': {
            'gru_hidden': gru_hidden, 'num_gru_layers': num_gru_layers,
            'num_channels': num_channels, 'kernel_size': kernel_size,
            'dropout': dropout, 'hidden_size': hidden_size,
            'dt': dt, 'num_gcn_layers': num_gcn_layers,
            'num_nodes': len(APPLIANCES)
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'adjacency': adj.tolist(),
        'mean_gate_per_appliance': {
            app: float(mean_gate_per_app[i]) for i, app in enumerate(APPLIANCES)
        },
        'test_loss': float(avg_test_loss),
        'aggregates': aggregates
    }
    with open(os.path.join(save_dir, 'gru_gnn_lnn_redd_history.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, mean_gate_per_app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing GRU-GNN-LNN (gated fusion) on REDD dataset with specific splits...")

    required_files = [
        'data/redd/train_small.pkl',
        'data/redd/val_small.pkl',
        'data/redd/test_small.pkl'
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    data_dict = load_redd_specific_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/gru_gnn_lnn_redd_specific_test_{timestamp}"

    model, history, test_metrics, mean_gate = train_gru_gnn_lnn(
        data_dict,
        window_size=100,
        gru_hidden=64,
        num_gru_layers=2,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        hidden_size=64,
        dt=0.1,
        num_gcn_layers=2,
        epochs=80,
        lr=0.001,
        patience=20,
        save_dir=save_dir
    )

    print(f"\nSummary — GRU-GNN-LNN (Gated Fusion) on REDD:")
    print(f"{'Appliance':<15}  {'MAE':>7}  {'SAE':>7}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}  {'Gate':>6}  {'Branch'}")
    print("-" * 75)
    for i, app in enumerate(APPLIANCES):
        m = test_metrics[app]
        g = mean_gate[i]
        branch = 'GRU' if g > 0.5 else 'GNN'
        print(f"  {app:<13}  {m['mae']:>7.4f}  {m['sae']:>7.4f}  {m['f1']:>7.4f}  "
              f"{m['precision']:>9.4f}  {m['recall']:>7.4f}  {g:>6.3f}  {branch}")
