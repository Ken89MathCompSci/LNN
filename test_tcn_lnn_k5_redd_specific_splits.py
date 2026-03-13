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
sys.path.append('Source Code')

# Import from our modules
from models import TCNLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model

# kernel_size=5 variant: num_channels=[32,64,128], dilation=[1,2,4]
# Receptive field = 1 + (5-1)*(1+2+4) = 29 steps (vs 15 with k=3)

class REDDSpecificDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_redd_specific_splits():
    """
    Load REDD data with the specific splits:

    Dataset House # Time Start Time End
    Training 3 2011-04-21 19:41:24 2011-04-22 19:41:21
    Validation 3 2011-05-23 10:31:24 2011-05-24 10:31:21
    Testing 1 2011-04-18 09:22:12 2011-05-23 09:21:51
    """
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
    print(f"Val date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test date range: {test_data.index.min()} to {test_data.index.max()}")

    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']
    for appliance in appliances:
        if appliance not in train_data.columns:
            print(f"Warning: {appliance} not found in training data columns")

    print(f"Available columns: {list(train_data.columns)}")

    return {
        'train': train_data,
        'val':   val_data,
        'test':  test_data,
        'appliances': appliances
    }

def create_sequences(data, window_size=100, target_size=1):
    mains = data['main'].values
    X, y  = [], []
    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])
        if target_size == 1:
            midpoint = i + window_size // 2
            y.append(mains[midpoint:midpoint+1])
        else:
            y.append(mains[i+window_size:i+window_size+target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def get_threshold_for_appliance(appliance_name):
    if appliance_name == 'washer dryer':
        return 0.5
    else:
        return 10.0

def train_tcn_lnn_k5_on_specific_redd_appliance(data_dict, appliance_name, window_size=100,
                                                num_channels=[32, 64, 128], kernel_size=5, dropout=0.2,
                                                hidden_size=64, dt=0.1,
                                                epochs=80, lr=0.001, patience=20,
                                                save_dir='models/tcn_lnn_k5_redd_specific'):
    """
    Train TCN-LNN with kernel_size=5 on a specific REDD appliance.
    RF = 1 + (5-1)*(1+2+4) = 29 steps.
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")

    X_train, y_train = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val,   y_val   = create_sequences(val_data,   window_size=window_size, target_size=1)
    X_test,  y_test  = create_sequences(test_data,  window_size=window_size, target_size=1)

    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val   = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test  = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    # Normalise to [0,1] — critical for LNN clamp stability
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    print(f"Training sequences: {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences: {X_test.shape} -> {y_test.shape}")

    train_dataset = REDDSpecificDataset(X_train, y_train)
    val_dataset   = REDDSpecificDataset(X_val,   y_val)
    test_dataset  = REDDSpecificDataset(X_test,  y_test)

    batch_size   = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    input_size  = 1
    output_size = 1

    model = TCNLiquidNetworkModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dt=dt,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    best_val_loss   = float('inf')
    counter         = 0
    best_model_path = None

    print(f"Starting TCN-LNN (k=5, RF=29) training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss    = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                val_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Inverse transform to watts before metrics
        all_targets = y_scaler.inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()
        all_outputs = y_scaler.inverse_transform(np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        threshold   = get_threshold_for_appliance(appliance_name)
        metrics     = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Val MAE: {metrics['mae']:.2f}, Val SAE: {metrics['sae']:.4f}, Val F1: {metrics['f1']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_model_path = os.path.join(save_dir, f"tcn_lnn_k5_redd_{appliance_name.replace(' ', '_')}_best.pth")

            model_params = {
                'input_size':   input_size,
                'output_size':  output_size,
                'num_channels': num_channels,
                'kernel_size':  kernel_size,
                'dropout':      dropout
            }
            train_params = {
                'lr':          lr,
                'epochs':      epochs,
                'patience':    patience,
                'window_size': window_size,
                'appliance':   appliance_name
            }
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    # Test evaluation
    print("Evaluating on test set...")
    model.eval()
    test_loss        = 0.0
    all_test_targets = []
    all_test_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            test_loss += loss.item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # Inverse transform to watts then calculate test metrics
    all_test_targets = y_scaler.inverse_transform(np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    threshold        = get_threshold_for_appliance(appliance_name)
    test_metrics     = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    val_mae_series = [m['mae'] for m in history['val_metrics']]
    val_sae_series = [m['sae'] for m in history['val_metrics']]
    val_f1_series  = [m['f1']  for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean': float(np.mean(history['train_loss'])) if history['train_loss'] else None,
        'train_loss_var':  float(np.var(history['train_loss']))  if history['train_loss'] else None,
        'val_loss_mean':   float(np.mean(history['val_loss']))   if history['val_loss'] else None,
        'val_loss_var':    float(np.var(history['val_loss']))    if history['val_loss'] else None,
        'val_mae_mean':    float(np.mean(val_mae_series))        if val_mae_series else None,
        'val_mae_var':     float(np.var(val_mae_series))         if val_mae_series else None,
        'val_sae_mean':    float(np.mean(val_sae_series))        if val_sae_series else None,
        'val_sae_var':     float(np.var(val_sae_series))         if val_sae_series else None,
        'val_f1_mean':     float(np.mean(val_f1_series))         if val_f1_series else None,
        'val_f1_var':      float(np.var(val_f1_series))          if val_f1_series else None,
        'test_mae':        float(test_metrics['mae']),
        'test_sae':        float(test_metrics['sae']),
        'test_f1':         float(test_metrics['f1']),
        'test_loss':       float(avg_test_loss)
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")
    print("Aggregates (mean/variance):")
    print(json.dumps(aggregates, indent=2))

    # Plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tcn_lnn_k5_redd_{appliance_name.replace(' ', '_')}_metrics.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    config = {
        'appliance':    appliance_name,
        'window_size':  window_size,
        'model_params': {
            'input_size':        input_size,
            'output_size':       output_size,
            'num_channels':      num_channels,
            'kernel_size':       kernel_size,
            'dropout':           dropout,
            'receptive_field':   1 + (kernel_size - 1) * sum(2**i for i in range(len(num_channels)))
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'train_loss':   history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss':     history['val_loss'][-1]   if history['val_loss'] else None,
            'test_loss':    avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates':   aggregates
        }
    }

    with open(os.path.join(save_dir, f'tcn_lnn_k5_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_tcn_lnn_k5_on_all_redd_appliances(window_size=100, num_channels=[32, 64, 128], kernel_size=5,
                                           dropout=0.2, hidden_size=64, dt=0.1,
                                           epochs=80, lr=0.001, patience=20):
    """
    Test TCN-LNN (kernel_size=5, RF=29) on all REDD appliances with specific splits.
    """
    print("Loading REDD data with specific splits...")
    data_dict = load_redd_specific_splits()

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_lnn_k5_redd_specific_test_{timestamp}"

    all_results = {}
    appliances  = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing TCN-LNN (k=5) on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_tcn_lnn_k5_on_specific_redd_appliance(
                data_dict,
                appliance_name=appliance_name,
                window_size=window_size,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                hidden_size=hidden_size,
                dt=dt,
                epochs=epochs,
                lr=lr,
                patience=patience,
                save_dir=appliance_dir
            )

            if model is not None:
                all_results[appliance_name] = {
                    'model_path':    os.path.join(appliance_dir, f"tcn_lnn_k5_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"Successfully tested TCN-LNN (k=5) on {appliance_name}")
            else:
                print(f"Failed to test TCN-LNN (k=5) on {appliance_name}")

        except Exception as e:
            print(f"Error testing TCN-LNN (k=5) on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    summary = {
        'timestamp':  timestamp,
        'model':      'TCN-LNN kernel_size=5, dilation=[1,2,4], RF=29',
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
        'window_size':  window_size,
        'model_params': {'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout},
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results':      all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nTCN-LNN (k=5) testing completed on all REDD appliances!")
    print(f"Results saved to {base_save_dir}")

    return all_results


if __name__ == "__main__":
    print("Testing TCN-LNN (kernel_size=5, RF=29) on REDD dataset with specific splits...")

    required_files = [
        'data/redd/train_small.pkl',
        'data/redd/val_small.pkl',
        'data/redd/test_small.pkl'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            print("Please ensure the REDD dataset pickle files are in the data/redd/ directory.")
            sys.exit(1)

    print("Running comprehensive TCN-LNN (k=5) test on all REDD appliances...")
    results = test_tcn_lnn_k5_on_all_redd_appliances(
        window_size=100,
        num_channels=[32, 64, 128],
        kernel_size=5,
        dropout=0.2,
        hidden_size=64,
        dt=0.1,
        epochs=80,
        lr=0.001,
        patience=20
    )

    print(f"\nSummary of TCN-LNN (k=5) testing on REDD dataset:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")
        print(f"    Test F1:  {result['final_metrics']['f1']:.4f}")
