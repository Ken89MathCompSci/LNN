import sys
import os
import torch
import torch.nn.functional as F
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


class WeightedMAEEnergyLoss(torch.nn.Module):
    """
    Loss = lambda1 * weighted_MAE
         + lambda2 * |sum(pred) - sum(target)| / N   (energy penalty -> SAE)
         + alpha_bce * BCE(pred, binary_target)       (classification -> F1)

    weighted_MAE: on_weight applied to ON samples, 1.0 to OFF samples.
    BCE term: treats the [0,1] regression output as a soft probability for the
              ON/OFF state, directly optimising the classification boundary for F1.
    on_weight is capped at 30x to avoid gradient instability.
    All terms operate in normalised [0,1] space during training.
    """
    def __init__(self, on_weight, on_threshold_norm, lambda1=1.0, lambda2=0.1, alpha_bce=0.5):
        super().__init__()
        self.on_weight = min(float(on_weight), 30.0)
        self.on_threshold_norm = float(on_threshold_norm)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha_bce = alpha_bce

    def forward(self, pred, target):
        weights = torch.where(
            target > self.on_threshold_norm,
            torch.full_like(target, self.on_weight),
            torch.ones_like(target)
        )
        weighted_mae = torch.mean(weights * torch.abs(pred - target))
        energy_penalty = torch.abs(pred.sum() - target.sum()) / pred.numel()

        binary_target = (target > self.on_threshold_norm).float()
        bce = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), binary_target)

        return self.lambda1 * weighted_mae + self.lambda2 * energy_penalty + self.alpha_bce * bce


class REDDSpecificDataset(torch.utils.data.Dataset):
    """
    Dataset class for REDD data with specific appliance targets
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_redd_specific_splits():
    """
    Load REDD data with the specific splits and appliances you mentioned:

    Dataset House # Time Start Time End
    Training 3 2011-04-21 19:41:24 2011-04-22 19:41:21
    Validation 3 2011-05-23 10:31:24 2011-05-24 10:31:21
    Testing 1 2011-04-18 09:22:12 2011-05-23 09:21:51

    APPLIANCE ON OFF TOTAL
    Dishwasher 1143 27657 28800
    Fridge 10471 18329 28800
    Microwave 531 28269 28800
    Washing Machine 1879 26921 28800
    """

    print("Loading REDD data with specific splits...")

    # Load the pre-split data
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

    # Verify the specific appliances are present
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']
    for appliance in appliances:
        if appliance not in train_data.columns:
            print(f"Warning: {appliance} not found in training data columns")

    print(f"Available columns: {list(train_data.columns)}")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'appliances': appliances
    }

def create_sequences(data, window_size=100, target_size=1):
    """
    Create sequences for sequence-to-point prediction

    Args:
        data: DataFrame with 'main' and appliance columns
        window_size: Size of input window
        target_size: Size of target (1 for sequence-to-point)

    Returns:
        X: Input sequences (n_samples, window_size, 1)
        y: Target values (n_samples, target_size)
    """
    mains = data['main'].values
    X, y = [], []

    # Use stride to reduce correlation between samples
    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])

        if target_size == 1:
            # Sequence-to-point: predict the middle point
            midpoint = i + window_size // 2
            y.append(mains[midpoint:midpoint+1])
        else:
            # Sequence-to-sequence
            y.append(mains[i+window_size:i+window_size+target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def get_threshold_for_appliance(appliance_name):
    """
    Get appropriate power threshold for on/off detection based on appliance characteristics

    Args:
        appliance_name: Name of the appliance

    Returns:
        Power threshold in watts
    """
    # Washer dryer has very low power consumption (max 4W)
    if appliance_name == 'washer dryer':
        return 0.5  # Very low threshold for washer dryer
    else:
        return 10.0  # Standard threshold for other appliances


def get_post_processing_threshold(appliance_name):
    """
    Minimum power (W) below which predictions are zeroed out.

    Rationale: tiny persistent predictions (e.g. 2 W when appliance is off)
    accumulate over thousands of samples and inflate SAE without affecting MAE
    much. Zeroing sub-threshold values removes this systematic bias.
    """
    thresholds = {
        'dish washer': 10.0,
        'fridge':      10.0,
        'microwave':   10.0,
        'washer dryer': 0.5,   # max power is ~4 W so keep threshold very low
    }
    return thresholds.get(appliance_name, 10.0)


def train_tcn_lnn_on_specific_redd_appliance(data_dict, appliance_name, window_size=100,
                                           num_channels=[32, 64, 128], kernel_size=3, dropout=0.2,
                                           hidden_size=64, dt=0.1,
                                           epochs=20, lr=0.001, patience=5,
                                           lambda_energy=0.1, alpha_bce=0.5,
                                           save_dir='models/tcn_lnn_redd_specific'):
    """
    Train TCN model on a specific REDD appliance with the exact splits you specified.

    Improvements over baseline:
      1. CombinedMAEEnergyLoss: MAE + lambda_energy * |sum(pred)-sum(target)|/N
         Forces the model to match total energy as well as point-wise values.
      2. Post-processing threshold: predictions below get_post_processing_threshold()
         are zeroed out after inverse-transform to eliminate systematic bias in SAE.

    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        window_size: Size of input sequence window
        num_channels: List of channel numbers for each TCN layer
        kernel_size: Kernel size for TCN convolutions
        dropout: Dropout rate for regularization
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        lambda_energy: Weight for the energy penalty term in the combined loss
        save_dir: Directory to save model and results

    Returns:
        Trained model, training history, and evaluation metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract data for this appliance
    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    # Create sequences for each split
    print(f"Creating sequences for {appliance_name}...")

    # For NILM, we use aggregate as input and appliance as target
    X_train, y_train = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val, y_val = create_sequences(val_data, window_size=window_size, target_size=1)
    X_test, y_test = create_sequences(test_data, window_size=window_size, target_size=1)

    # Use appliance power as target instead of aggregate
    # This is the key difference - we're predicting the appliance from the aggregate
    y_train_raw = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    # Compute class imbalance weight BEFORE normalisation (uses raw watts)
    on_off_threshold = get_threshold_for_appliance(appliance_name)
    on_ratio = float((y_train_raw.flatten() > on_off_threshold).mean())
    on_ratio = max(on_ratio, 1e-6)  # guard against all-zero appliance
    on_weight = (1.0 - on_ratio) / on_ratio
    print(f"Class balance for {appliance_name}: {on_ratio*100:.1f}% ON  ->  on_weight = {min(on_weight, 30.0):.1f}x")

    # Normalise inputs and targets to [0, 1] — critical for LNN clamp stability
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    # Normalised ON threshold (for use inside the loss function)
    on_threshold_norm = float(y_scaler.transform([[on_off_threshold]])[0][0])

    print(f"Training sequences: {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences: {X_test.shape} -> {y_test.shape}")

    # Create datasets and dataloaders
    train_dataset = REDDSpecificDataset(X_train, y_train)
    val_dataset = REDDSpecificDataset(X_val, y_val)
    test_dataset = REDDSpecificDataset(X_test, y_test)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create TCN model
    input_size = 1  # Single feature (aggregate power)
    output_size = 1  # Single target (appliance power)

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

    # Weighted MAE + energy penalty + BCE loss; Adam optimizer
    criterion = WeightedMAEEnergyLoss(
        on_weight=on_weight,
        on_threshold_norm=on_threshold_norm,
        lambda1=1.0,
        lambda2=lambda_energy,
        alpha_bce=alpha_bce
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    post_threshold = get_post_processing_threshold(appliance_name)
    print(f"Post-processing threshold for {appliance_name}: {post_threshold} W")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_path = None

    print(f"Starting TCN-LNN training for {appliance_name}...")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss (MAE + energy penalty)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Update statistics
                val_loss += loss.item()

                # Store targets and outputs for metrics calculation
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Inverse transform to watts, apply post-processing threshold, then compute metrics
        all_targets = y_scaler.inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()
        all_outputs = y_scaler.inverse_transform(np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        all_outputs = np.where(all_outputs < post_threshold, 0.0, all_outputs)

        threshold = get_threshold_for_appliance(appliance_name)
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history['val_metrics'].append(metrics)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Val MAE: {metrics['mae']:.2f}, Val F1: {metrics['f1']:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(save_dir, f"tcn_lnn_redd_{appliance_name.replace(' ', '_')}_best.pth")

            # Save model with metadata
            model_params = {
                'input_size': input_size,
                'output_size': output_size,
                'num_channels': num_channels,
                'kernel_size': kernel_size,
                'dropout': dropout
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience,
                'window_size': window_size,
                'appliance': appliance_name
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

    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    all_test_targets = []
    all_test_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Update statistics
            test_loss += loss.item()

            # Store targets and outputs
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Inverse transform to watts, apply post-processing threshold, then compute metrics
    all_test_targets = y_scaler.inverse_transform(np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    all_test_outputs = np.where(all_test_outputs < post_threshold, 0.0, all_test_outputs)

    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    # Aggregates (val/test) — train per-epoch metrics not collected
    val_mae_series = [m['mae'] for m in history['val_metrics']]
    val_sae_series = [m['sae'] for m in history['val_metrics']]
    val_f1_series = [m['f1'] for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean': float(np.mean(history['train_loss'])) if history['train_loss'] else None,
        'train_loss_var': float(np.var(history['train_loss'])) if history['train_loss'] else None,
        'val_loss_mean': float(np.mean(history['val_loss'])) if history['val_loss'] else None,
        'val_loss_var': float(np.var(history['val_loss'])) if history['val_loss'] else None,
        'val_mae_mean': float(np.mean(val_mae_series)) if val_mae_series else None,
        'val_mae_var': float(np.var(val_mae_series)) if val_mae_series else None,
        'val_sae_mean': float(np.mean(val_sae_series)) if val_sae_series else None,
        'val_sae_var': float(np.var(val_sae_series)) if val_sae_series else None,
        'val_f1_mean': float(np.mean(val_f1_series)) if val_f1_series else None,
        'val_f1_var': float(np.var(val_f1_series)) if val_f1_series else None,
        'test_mae': float(test_metrics['mae']),
        'test_sae': float(test_metrics['sae']),
        'test_f1': float(test_metrics['f1']),
        'test_loss': float(avg_test_loss)
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")
    print("Aggregates (mean/variance):")
    print(json.dumps(aggregates, indent=2))

    # Plot training history and metrics
    plt.figure(figsize=(15, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # MAE
    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SAE
    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('SAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1
    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tcn_lnn_redd_{appliance_name.replace(' ', '_')}_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.savefig(os.path.join(save_dir, f"tcn_lnn_redd_{appliance_name.replace(' ', '_')}_training_history.png"))
    plt.close()

    # Plot some prediction examples
    plt.figure(figsize=(12, 6))
    num_examples = min(5, len(all_test_targets))
    for i in range(num_examples):
        plt.subplot(2, 3, i+1)
        plt.plot(all_test_targets[i], 'b-', label='Actual')
        plt.plot(all_test_outputs[i], 'r--', label='Predicted')
        plt.title(f'Example {i+1}')
        plt.legend()
        if i == 0:
            plt.ylabel('Power')
    plt.suptitle(f'TCN Predictions vs Actual - {appliance_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tcn_lnn_redd_{appliance_name.replace(' ', '_')}_predictions.png"))
    plt.close()

    # Save training history to JSON
    config = {
        'appliance': appliance_name,
        'window_size': window_size,
        'improvements': {
            'loss': 'WeightedMAEEnergyLoss',
            'on_weight': min(on_weight, 30.0),
            'on_threshold_norm': on_threshold_norm,
            'lambda1': 1.0,
            'lambda2': lambda_energy,
            'alpha_bce': alpha_bce,
            'post_processing_threshold_W': post_threshold
        },
        'model_params': {
            'input_size': input_size,
            'output_size': output_size,
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout
        },
        'train_params': {
            'lr': lr,
            'epochs': epochs,
            'patience': patience
        },
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_loss': avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }

    with open(os.path.join(save_dir, f'tcn_lnn_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics

def test_tcn_lnn_on_all_redd_appliances(window_size=100, num_channels=[32, 64, 128], kernel_size=3,
                                       dropout=0.2, hidden_size=64, dt=0.1,
                                       epochs=80, lr=0.001, patience=20,
                                       lambda_energy=0.1, alpha_bce=0.5):
    """
    Test TCN model on all specified REDD appliances with the exact splits you mentioned.
    Uses CombinedMAEEnergyLoss and post-processing thresholding.

    Args:
        window_size: Size of input sequence window
        num_channels: List of channel numbers for each TCN layer
        kernel_size: Kernel size for TCN convolutions
        dropout: Dropout rate for regularization
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        lambda_energy: Weight for energy penalty term (lambda2 in combined loss)

    Returns:
        Dictionary of results for all appliances
    """
    # Load data with specific splits
    print("Loading REDD data with specific splits...")
    data_dict = load_redd_specific_splits()

    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_lnn_redd_specific_energy_loss_{timestamp}"

    # Dictionary to store results
    all_results = {}

    # Test TCN model on each specified appliance
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing TCN-LNN on {appliance_name}")
        print(f"{'='*60}\n")

        # Create appliance-specific save directory
        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            # Train and evaluate TCN model
            model, history, test_metrics = train_tcn_lnn_on_specific_redd_appliance(
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
                lambda_energy=lambda_energy,
                alpha_bce=alpha_bce,
                save_dir=appliance_dir
            )

            if model is not None:
                # Store results
                all_results[appliance_name] = {
                    'model_path': os.path.join(appliance_dir, f"tcn_lnn_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"Successfully tested TCN-LNN on {appliance_name}")
            else:
                print(f"Failed to test TCN-LNN on {appliance_name}")

        except Exception as e:
            print(f"Error testing TCN-LNN on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary of results
    summary = {
        'timestamp': timestamp,
        'improvements': {
            'loss': 'WeightedMAEEnergyLoss (on_weight*MAE_on + MAE_off + lambda_energy*energy_penalty + alpha_bce*BCE)',
            'lambda_energy': lambda_energy,
            'alpha_bce': alpha_bce,
            'on_weight': 'computed per-appliance from training class imbalance, capped at 30x',
            'post_processing': 'Zero predictions below appliance-specific threshold'
        },
        'dataset_splits': {
            'training': {
                'house': 3,
                'time_start': '2011-04-21 19:41:24',
                'time_end': '2011-04-22 19:41:21'
            },
            'validation': {
                'house': 3,
                'time_start': '2011-05-23 10:31:24',
                'time_end': '2011-05-24 10:31:21'
            },
            'testing': {
                'house': 1,
                'time_start': '2011-04-18 09:22:12',
                'time_end': '2011-05-23 09:21:51'
            }
        },
        'appliances': {
            'dish washer': {'on': 1143, 'off': 27657, 'total': 28800},
            'fridge': {'on': 10471, 'off': 18329, 'total': 28800},
            'microwave': {'on': 531, 'off': 28269, 'total': 28800},
            'washer dryer': {'on': 1879, 'off': 26921, 'total': 28800}
        },
        'window_size': window_size,
        'model_params': {
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout
        },
        'train_params': {
            'epochs': epochs,
            'lr': lr,
            'patience': patience
        },
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nTCN-LNN testing completed on all REDD appliances with specific splits!")
    print(f"Results saved to {base_save_dir}")

    return all_results

if __name__ == "__main__":
    print("Testing TCN-LNN algorithm on REDD dataset with specific splits...")
    print("Improvements: WeightedMAEEnergyLoss (class-balanced) + energy penalty + BCE (F1) + post-processing threshold")

    # Check if the required pickle files exist
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

    # Run comprehensive test on all specified appliances
    print("Running comprehensive TCN-LNN test on all REDD appliances with specific splits...")
    results = test_tcn_lnn_on_all_redd_appliances(
        window_size=100,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        hidden_size=64,
        dt=0.1,
        epochs=80,
        lr=0.001,
        patience=20,
        lambda_energy=0.1,
        alpha_bce=0.5
    )

    # Print summary
    print(f"\nSummary of TCN-LNN testing on REDD dataset with specific splits:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")
