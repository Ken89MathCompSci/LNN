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

from models import TCNAdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


class UKDALEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    L_total = L_MSE + focal_lambda * L_focal

    L_focal = -alpha * (1-p)^gamma * y * log(p)
              - (1-alpha) * p^gamma * (1-y) * log(1-p)

    p = sigmoid(output)  — maps scaled regression output to (0,1)
    """
    def __init__(self, alpha=0.75, gamma=2.0, focal_lambda=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_lambda = focal_lambda
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, threshold_scaled):
        loss_mse = self.mse(outputs, targets)

        y = (targets >= threshold_scaled).float()
        p = torch.clamp(torch.sigmoid(outputs), min=1e-7, max=1.0 - 1e-7)

        focal_pos = -self.alpha       * (1 - p) ** self.gamma * y       * torch.log(p)
        focal_neg = -(1 - self.alpha) * p       ** self.gamma * (1 - y) * torch.log(1 - p)
        loss_focal = (focal_pos + focal_neg).mean()

        return loss_mse + self.focal_lambda * loss_focal


# ---------------------------------------------------------------------------
# HMM Viterbi post-processing
# ---------------------------------------------------------------------------

def viterbi_decode(observations, threshold, p_off_to_on=0.01, p_on_to_on=0.9):
    """
    Two-state HMM (OFF=0, ON=1) Viterbi decoder.

    Transition matrix:
        A[0,0] = 1 - p_off_to_on    A[0,1] = p_off_to_on
        A[1,0] = 1 - p_on_to_on     A[1,1] = p_on_to_on

    Emission: Gaussian per state, parameters estimated from observations.
    """
    obs = np.array(observations, dtype=np.float64)
    N   = len(obs)

    on_mask  = obs >= threshold
    off_mask = ~on_mask

    mu_on  = obs[on_mask].mean()  if on_mask.any()  else threshold + 1.0
    mu_off = obs[off_mask].mean() if off_mask.any() else 0.0
    sig_on  = max(obs[on_mask].std()  if on_mask.any()  else threshold * 0.5, 1e-3)
    sig_off = max(obs[off_mask].std() if off_mask.any() else threshold * 0.5, 1e-3)

    log_A = np.log(np.array([
        [1 - p_off_to_on, p_off_to_on   ],
        [1 - p_on_to_on,  p_on_to_on    ]
    ]) + 1e-300)

    def log_emit(o, mu, sig):
        return -0.5 * ((o - mu) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))

    log_emit_arr = np.stack([log_emit(obs, mu_off, sig_off),
                              log_emit(obs, mu_on,  sig_on)], axis=1)

    log_delta = np.zeros((N, 2))
    psi       = np.zeros((N, 2), dtype=int)

    log_delta[0, 0] = log_emit_arr[0, 0] + np.log(0.5)
    log_delta[0, 1] = log_emit_arr[0, 1] + np.log(0.5)

    for t in range(1, N):
        for s in range(2):
            cands           = log_delta[t - 1] + log_A[:, s]
            psi[t, s]       = int(np.argmax(cands))
            log_delta[t, s] = cands[psi[t, s]] + log_emit_arr[t, s]

    states = np.zeros(N, dtype=int)
    states[-1] = int(np.argmax(log_delta[-1]))
    for t in range(N - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def hmm_postprocess(outputs, threshold, p_off_to_on=0.01, p_on_to_on=0.9):
    states = viterbi_decode(outputs, threshold, p_off_to_on, p_on_to_on)
    on_mask = outputs >= threshold
    mu_on   = outputs[on_mask].mean() if on_mask.any() else threshold
    smoothed = np.where(states == 1, mu_on, 0.0)
    return smoothed, states


# ---------------------------------------------------------------------------
# Per-appliance parameters
# ---------------------------------------------------------------------------

APPLIANCE_FOCAL_PARAMS = {
    'dish washer':  {'alpha': 0.90, 'gamma': 4.0},
    'fridge':       {'alpha': 0.75, 'gamma': 2.0},
    'microwave':    {'alpha': 0.75, 'gamma': 2.0},
    'washer dryer': {'alpha': 0.75, 'gamma': 2.0},
}

APPLIANCE_HMM_PARAMS = {
    'dish washer':  {'p_off_to_on': 0.005, 'p_on_to_on': 0.95},
    'fridge':       {'p_off_to_on': 0.05,  'p_on_to_on': 0.85},
    'microwave':    {'p_off_to_on': 0.02,  'p_on_to_on': 0.60},
    'washer dryer': {'p_off_to_on': 0.005, 'p_on_to_on': 0.97},
}


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


def create_sequences(data, window_size=100):
    mains = data['main'].values
    X, stride = [], 5
    for i in range(0, len(mains) - window_size + 1, stride):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def get_threshold_for_appliance(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name, window_size=100,
                       hidden_size=64, num_layers=2, dt=0.1,
                       num_channels=None, kernel_size=3, dropout=0.2,
                       focal_lambda=0.1,
                       epochs=80, lr=0.001, patience=20,
                       save_dir='models/tcn_advanced_lnn_focal_hmm_ukdale'):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")
    X_train = create_sequences(train_data, window_size)
    X_val   = create_sequences(val_data,   window_size)
    X_test  = create_sequences(test_data,  window_size)

    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val   = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test  = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

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

    raw_threshold    = get_threshold_for_appliance(appliance_name)
    threshold_scaled = float(y_scaler.transform([[raw_threshold]])[0][0])

    focal_p  = APPLIANCE_FOCAL_PARAMS[appliance_name]
    hmm_p    = APPLIANCE_HMM_PARAMS[appliance_name]

    print(f"Focal -> alpha={focal_p['alpha']}  gamma={focal_p['gamma']}  "
          f"focal_lambda={focal_lambda}")
    print(f"HMM   -> p_off_to_on={hmm_p['p_off_to_on']}  "
          f"p_on_to_on={hmm_p['p_on_to_on']}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = TCNAdvancedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        dt=dt, num_channels=num_channels, kernel_size=kernel_size,
        dropout=dropout, num_layers=num_layers
    ).to(device)

    criterion = FocalLoss(alpha=focal_p['alpha'], gamma=focal_p['gamma'],
                          focal_lambda=focal_lambda)
    mse_only  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [],
               'val_metrics_raw': [], 'val_metrics_hmm': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_path = os.path.join(
        save_dir,
        f"tcn_advanced_lnn_focal_hmm_ukdale_{appliance_name.replace(' ', '_')}_best.pth")

    print(f"Starting TCN-Advanced-LNN (Focal + HMM) training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, threshold_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += mse_only(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()

        metrics_raw = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=raw_threshold)
        history['val_metrics_raw'].append(metrics_raw)

        hmm_outs, _ = hmm_postprocess(raw_outs, raw_threshold,
                                       hmm_p['p_off_to_on'], hmm_p['p_on_to_on'])
        metrics_hmm = calculate_nilm_metrics(raw_tgts, hmm_outs, threshold=raw_threshold)
        history['val_metrics_hmm'].append(metrics_hmm)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val MSE: {avg_val_loss:.6f}")
        print(f"  Raw -> MAE:{metrics_raw['mae']:.1f}  SAE:{metrics_raw['sae']:.1f}  "
              f"F1:{metrics_raw['f1']:.4f}  P:{metrics_raw['precision']:.4f}  "
              f"R:{metrics_raw['recall']:.4f}")
        print(f"  HMM -> MAE:{metrics_hmm['mae']:.1f}  SAE:{metrics_hmm['sae']:.1f}  "
              f"F1:{metrics_hmm['f1']:.4f}  P:{metrics_hmm['precision']:.4f}  "
              f"R:{metrics_hmm['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            save_model(model,
                       {'input_size': 1, 'output_size': 1,
                        'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
                        'num_channels': num_channels, 'kernel_size': kernel_size,
                        'dropout': dropout},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name,
                        'focal_alpha': focal_p['alpha'], 'focal_gamma': focal_p['gamma'],
                        'focal_lambda': focal_lambda},
                       metrics_raw, best_model_path)
            print(f"  Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"  EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    print(f"Loading best val loss model from {best_model_path} for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0.0
    all_test_targets, all_test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += mse_only(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    raw_test_tgts = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    raw_test_outs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()

    test_metrics_raw = calculate_nilm_metrics(raw_test_tgts, raw_test_outs,
                                              threshold=raw_threshold)
    hmm_test_outs, _ = hmm_postprocess(raw_test_outs, raw_threshold,
                                        hmm_p['p_off_to_on'], hmm_p['p_on_to_on'])
    test_metrics_hmm = calculate_nilm_metrics(raw_test_tgts, hmm_test_outs,
                                              threshold=raw_threshold)

    print(f"\nTest Loss: {avg_test_loss:.6f}")
    print(f"Test Raw -> {test_metrics_raw}")
    print(f"Test HMM -> {test_metrics_hmm}")

    val_f1_raw = [m['f1'] for m in history['val_metrics_raw']]
    val_f1_hmm = [m['f1'] for m in history['val_metrics_hmm']]
    val_mae_raw = [m['mae'] for m in history['val_metrics_raw']]
    val_sae_raw = [m['sae'] for m in history['val_metrics_raw']]

    aggregates = {
        'train_loss_mean':    float(np.mean(history['train_loss'])),
        'train_loss_var':     float(np.var(history['train_loss'])),
        'val_loss_mean':      float(np.mean(history['val_loss'])),
        'val_loss_var':       float(np.var(history['val_loss'])),
        'val_f1_raw_mean':    float(np.mean(val_f1_raw)),
        'val_f1_hmm_mean':    float(np.mean(val_f1_hmm)),
        'val_mae_mean':       float(np.mean(val_mae_raw)),
        'val_sae_mean':       float(np.mean(val_sae_raw)),
        'test_raw_mae':       float(test_metrics_raw['mae']),
        'test_raw_sae':       float(test_metrics_raw['sae']),
        'test_raw_f1':        float(test_metrics_raw['f1']),
        'test_raw_precision': float(test_metrics_raw['precision']),
        'test_raw_recall':    float(test_metrics_raw['recall']),
        'test_hmm_mae':       float(test_metrics_hmm['mae']),
        'test_hmm_sae':       float(test_metrics_hmm['sae']),
        'test_hmm_f1':        float(test_metrics_hmm['f1']),
        'test_hmm_precision': float(test_metrics_hmm['precision']),
        'test_hmm_recall':    float(test_metrics_hmm['recall']),
        'test_loss':          float(avg_test_loss)
    }
    print("Aggregates:")
    print(json.dumps(aggregates, indent=2))

    # Plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val MSE',    color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_raw, label='Val MAE (raw)', color='red')
    plt.axhline(test_metrics_raw['mae'], label='Test MAE (raw)',
                color='orange', linestyle='--')
    plt.axhline(test_metrics_hmm['mae'], label='Test MAE (HMM)',
                color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_f1_raw, label='Val F1 (raw)', color='red')
    plt.plot(val_f1_hmm, label='Val F1 (HMM)', color='blue')
    plt.axhline(test_metrics_raw['f1'], label='Test F1 (raw)',
                color='orange', linestyle='--')
    plt.axhline(test_metrics_hmm['f1'], label='Test F1 (HMM)',
                color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    n_plot = min(500, len(raw_test_outs))
    plt.plot(raw_test_tgts[:n_plot], label='Ground Truth', color='black', alpha=0.7)
    plt.plot(raw_test_outs[:n_plot], label='Raw Output',   color='red',   alpha=0.5)
    plt.plot(hmm_test_outs[:n_plot], label='HMM Output',   color='green', alpha=0.8)
    plt.title(f'Test Predictions - {appliance_name} (first {n_plot} samples)')
    plt.xlabel('Sample'); plt.ylabel('Power (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"tcn_advanced_lnn_focal_hmm_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=300, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'UKDALE',
        'loss': 'MSE + Focal Loss',
        'postprocessing': 'HMM Viterbi (2-state)',
        'focal_params': {**focal_p, 'focal_lambda': focal_lambda},
        'hmm_params': hmm_p,
        'window_size': window_size,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'test_loss': avg_test_loss,
            'test_raw':  {k: float(v) for k, v in test_metrics_raw.items()},
            'test_hmm':  {k: float(v) for k, v in test_metrics_hmm.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'tcn_advanced_lnn_focal_hmm_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics_raw, test_metrics_hmm


def test_on_all_appliances(window_size=100, hidden_size=64, num_layers=2, dt=0.1,
                           num_channels=None, kernel_size=3, dropout=0.2,
                           focal_lambda=0.1, epochs=80, lr=0.001, patience=20):
    if num_channels is None:
        num_channels = [32, 64, 128]

    data_dict = load_ukdale_specific_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_advanced_lnn_focal_hmm_ukdale_test_{timestamp}"

    all_results = {}
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing TCN-Advanced-LNN (Focal + HMM) on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_raw, test_hmm = train_on_appliance(
                data_dict,
                appliance_name=appliance_name,
                window_size=window_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dt=dt,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                focal_lambda=focal_lambda,
                epochs=epochs,
                lr=lr,
                patience=patience,
                save_dir=appliance_dir
            )
            if model is not None:
                all_results[appliance_name] = {
                    'raw': {k: float(v) for k, v in test_raw.items()},
                    'hmm': {k: float(v) for k, v in test_hmm.items()}
                }
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset': 'UKDALE',
        'loss': 'MSE + Focal Loss',
        'postprocessing': 'HMM Viterbi (2-state)',
        'focal_params': APPLIANCE_FOCAL_PARAMS,
        'hmm_params': APPLIANCE_HMM_PARAMS,
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'}
        },
        'window_size': window_size,
        'model_params': {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results': all_results
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nTCN-Advanced-LNN (Focal + HMM) UKDALE testing completed. "
          f"Results saved to {base_save_dir}")
    return all_results


if __name__ == "__main__":
    print("Testing TCN-Advanced-LNN with Focal Loss + HMM post-processing on UKDALE...")

    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    results = test_on_all_appliances(
        window_size=100,
        hidden_size=64,
        num_layers=2,
        dt=0.1,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        focal_lambda=0.1,
        epochs=80,
        lr=0.001,
        patience=20
    )

    print(f"\nSummary of TCN-Advanced-LNN (Focal + HMM) on UKDALE dataset:")
    print(f"{'Appliance':<15} {'Raw F1':>8} {'HMM F1':>8} {'Raw MAE':>9} {'HMM MAE':>9}")
    print("-" * 55)
    for appliance, result in results.items():
        print(f"{appliance:<15} "
              f"{result['raw']['f1']:>8.4f} "
              f"{result['hmm']['f1']:>8.4f} "
              f"{result['raw']['mae']:>9.2f} "
              f"{result['hmm']['mae']:>9.2f}")
