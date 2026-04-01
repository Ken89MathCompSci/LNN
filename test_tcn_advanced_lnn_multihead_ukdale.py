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
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))

from models import TCNAdvancedLiquidNetworkMultiHeadModel
from utils import calculate_nilm_metrics, save_model


APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']


class UKDALEMultiApplianceDataset(torch.utils.data.Dataset):
    """Returns (X_window, y_all_appliances) where y shape is (num_appliances,)."""
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)   # (N, window, 1)
        self.Y = torch.FloatTensor(Y)   # (N, num_appliances)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


def create_sequences(data, window_size=100):
    mains = data['main'].values
    X = []
    stride = 5
    for i in range(0, len(mains) - window_size + 1, stride):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def get_threshold_for_appliance(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


def train_multihead(data_dict, window_size=100,
                    hidden_size=64, num_layers=2, dt=0.1,
                    num_channels=None, kernel_size=3, dropout=0.2,
                    epochs=80, lr=0.001, patience=20,
                    save_dir='models/tcn_advanced_lnn_multihead_ukdale'):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    # Shared X sequences
    X_train = create_sequences(train_data, window_size)
    X_val   = create_sequences(val_data,   window_size)
    X_test  = create_sequences(test_data,  window_size)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    # Per-appliance y — each gets its own MinMaxScaler
    y_scalers = {}
    Y_train = np.zeros((n_train, len(APPLIANCES)))
    Y_val   = np.zeros((n_val,   len(APPLIANCES)))
    Y_test  = np.zeros((n_test,  len(APPLIANCES)))

    for a_idx, app in enumerate(APPLIANCES):
        ys = MinMaxScaler()
        y_scalers[app] = ys
        y_tr = train_data[app].iloc[::5].values.reshape(-1, 1)[:n_train]
        y_va = val_data[app].iloc[::5].values.reshape(-1, 1)[:n_val]
        y_te = test_data[app].iloc[::5].values.reshape(-1, 1)[:n_test]
        Y_train[:, a_idx] = ys.fit_transform(y_tr).flatten()
        Y_val[:,   a_idx] = ys.transform(y_va).flatten()
        Y_test[:,  a_idx] = ys.transform(y_te).flatten()

    # Shared X scaler
    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    print(f"Training sequences:   {X_train.shape} -> {Y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {Y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {Y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEMultiApplianceDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEMultiApplianceDataset(X_val, Y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEMultiApplianceDataset(X_test, Y_test), batch_size=32, shuffle=False)

    model = TCNAdvancedLiquidNetworkMultiHeadModel(
        input_size=1,
        hidden_size=hidden_size,
        num_appliances=len(APPLIANCES),
        dt=dt,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_path = os.path.join(save_dir, 'tcn_advanced_lnn_multihead_ukdale_best.pth')

    print("Starting TCN-Advanced-LNN Multi-Head training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)           # (batch, num_appliances)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets_list = [[] for _ in APPLIANCES]
        all_outputs_list = [[] for _ in APPLIANCES]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                for a_idx in range(len(APPLIANCES)):
                    all_targets_list[a_idx].append(targets[:, a_idx].cpu().numpy())
                    all_outputs_list[a_idx].append(outputs[:, a_idx].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Per-appliance val metrics
        epoch_metrics = {}
        for a_idx, app in enumerate(APPLIANCES):
            tgts = y_scalers[app].inverse_transform(
                np.concatenate(all_targets_list[a_idx]).reshape(-1, 1)).flatten()
            outs = y_scalers[app].inverse_transform(
                np.concatenate(all_outputs_list[a_idx]).reshape(-1, 1)).flatten()
            threshold = get_threshold_for_appliance(app)
            epoch_metrics[app] = calculate_nilm_metrics(tgts, outs, threshold=threshold)
        history['val_metrics'].append(epoch_metrics)

        avg_f1 = np.mean([epoch_metrics[app]['f1'] for app in APPLIANCES])
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, Avg Val F1: {avg_f1:.4f}")
        for app in APPLIANCES:
            m = epoch_metrics[app]
            print(f"  {app:15s}: MAE={m['mae']:.1f}  SAE={m['sae']:.1f}  "
                  f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_params': {
                    'input_size': 1, 'hidden_size': hidden_size,
                    'num_appliances': len(APPLIANCES), 'dt': dt,
                    'num_channels': num_channels, 'kernel_size': kernel_size,
                    'dropout': dropout, 'num_layers': num_layers
                },
                'appliances': APPLIANCES,
                'val_metrics': epoch_metrics
            }, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    # Evaluate best val loss model on test set
    print(f"Loading best val loss model from {best_model_path} for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0.0
    all_test_targets_list = [[] for _ in APPLIANCES]
    all_test_outputs_list = [[] for _ in APPLIANCES]
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            for a_idx in range(len(APPLIANCES)):
                all_test_targets_list[a_idx].append(targets[:, a_idx].cpu().numpy())
                all_test_outputs_list[a_idx].append(outputs[:, a_idx].cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    test_metrics = {}
    for a_idx, app in enumerate(APPLIANCES):
        tgts = y_scalers[app].inverse_transform(
            np.concatenate(all_test_targets_list[a_idx]).reshape(-1, 1)).flatten()
        outs = y_scalers[app].inverse_transform(
            np.concatenate(all_test_outputs_list[a_idx]).reshape(-1, 1)).flatten()
        threshold = get_threshold_for_appliance(app)
        test_metrics[app] = calculate_nilm_metrics(tgts, outs, threshold=threshold)

    print(f"\nTest Loss: {avg_test_loss:.6f}")
    print("Test Metrics per appliance:")
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"  {app:15s}: MAE={m['mae']:.4f}  SAE={m['sae']:.4f}  "
              f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    avg_mae = np.mean([test_metrics[app]['mae'] for app in APPLIANCES])
    avg_sae = np.mean([test_metrics[app]['sae'] for app in APPLIANCES])
    avg_f1  = np.mean([test_metrics[app]['f1']  for app in APPLIANCES])
    avg_pre = np.mean([test_metrics[app]['precision'] for app in APPLIANCES])
    avg_rec = np.mean([test_metrics[app]['recall']    for app in APPLIANCES])
    print(f"\n  {'Average':15s}: MAE={avg_mae:.4f}  SAE={avg_sae:.4f}  "
          f"F1={avg_f1:.4f}  P={avg_pre:.4f}  R={avg_rec:.4f}")

    # Plots — one figure per appliance
    for app in APPLIANCES:
        val_f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        val_mae_series = [m[app]['mae'] for m in history['val_metrics']]
        val_sae_series = [m[app]['sae'] for m in history['val_metrics']]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'],   label='Val Loss',   color='red')
        plt.title(f'Loss - {app}')
        plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(val_mae_series, label='Val MAE', color='red')
        plt.axhline(test_metrics[app]['mae'], label='Test MAE', color='green', linestyle='--')
        plt.title(f'MAE - {app}')
        plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(val_f1_series, label='Val F1', color='red')
        plt.axhline(test_metrics[app]['f1'], label='Test F1', color='green', linestyle='--')
        plt.title(f'F1 - {app}')
        plt.xlabel('Epoch'); plt.ylabel('F1')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
            f"tcn_advanced_lnn_multihead_ukdale_{app.replace(' ', '_')}_metrics.png"),
            dpi=300, bbox_inches='tight')
        plt.close()

    # Save config + results
    config = {
        'dataset': 'UKDALE',
        'architecture': 'Shared TCN + Separate LNN heads per appliance',
        'appliances': APPLIANCES,
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'}
        },
        'window_size': window_size,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'num_appliances': len(APPLIANCES), 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size,
            'dropout': dropout, 'num_layers': num_layers
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'test_loss': avg_test_loss,
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
        'averages': {
            'mae': float(avg_mae), 'sae': float(avg_sae),
            'f1': float(avg_f1), 'precision': float(avg_pre),
            'recall': float(avg_rec)
        }
    }
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


if __name__ == "__main__":
    print("Testing TCN-Advanced-LNN Multi-Head on UKDALE dataset...")

    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/tcn_advanced_lnn_multihead_ukdale_{timestamp}"

    data_dict = load_ukdale_specific_splits()

    model, history, test_metrics = train_multihead(
        data_dict,
        window_size=100,
        hidden_size=64,
        num_layers=2,
        dt=0.1,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        epochs=80,
        lr=0.001,
        patience=20,
        save_dir=save_dir
    )

    print(f"\nSummary of TCN-Advanced-LNN Multi-Head testing on UKDALE dataset:")
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"  {app}:")
        print(f"    Test MAE:       {m['mae']:.4f}")
        print(f"    Test SAE:       {m['sae']:.4f}")
        print(f"    Test F1:        {m['f1']:.4f}")
        print(f"    Test Precision: {m['precision']:.4f}")
        print(f"    Test Recall:    {m['recall']:.4f}")
