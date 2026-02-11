"""
Test Attention-Enhanced Liquid Neural Networks on REDD Dataset
With MatNILM-Exact Data Augmentation

This implementation EXACTLY matches MatNILM's augmentation strategy:
- 4 modes with equal 25% probability (none, vertical, horizontal, both)
- Appliance-specific probabilities (dishwasher: 0.3, fridge: 0.6, microwave: 0.3, washer: 0.3)
- Truncated normal scaling (μ=1, σ=0.2, range=[0.6, 1.4])
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
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

# Add Source Code to path
sys.path.append('Source Code')

# Import models
from models import (
    LiquidNetworkModel,
    AdvancedLiquidNetworkModel,
    AttentionLiquidNetworkModel
)
from utils import calculate_nilm_metrics, save_model


def get_threshold_for_appliance(appliance_name):
    """Get appropriate power threshold for on/off detection"""
    if appliance_name == 'washer dryer':
        return 0.5
    else:
        return 10.0


def get_matnilm_augmentation_probability(appliance_name):
    """
    Get MatNILM-specific augmentation probability for each appliance

    MatNILM uses different probabilities per appliance:
    - Dishwasher: 0.3
    - Fridge: 0.6
    - Microwave: 0.3
    - Washer Dryer: 0.3

    Args:
        appliance_name: Name of the appliance

    Returns:
        Augmentation probability (float)
    """
    matnilm_probs = {
        'dish washer': 0.3,
        'fridge': 0.6,
        'microwave': 0.3,
        'washer dryer': 0.3
    }
    return matnilm_probs.get(appliance_name, 0.3)  # Default to 0.3 if not found


def load_redd_specific_splits():
    """Load REDD data with specific splits"""
    print("Loading REDD data with specific splits...")

    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]

    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]

    with open('data/redd/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'appliances': ['dish washer', 'fridge', 'microwave', 'washer dryer']
    }


def create_sequences(data, window_size=100, target_size=1):
    """Create sequences for sequence-to-point prediction"""
    mains = data['main'].values
    X, y = [], []

    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])

        if target_size == 1:
            midpoint = i + window_size // 2
            y.append(mains[midpoint:midpoint+1])
        else:
            y.append(mains[i+window_size:i+window_size+target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)


class REDDSpecificDataset(torch.utils.data.Dataset):
    """Dataset class for REDD data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# DATA AUGMENTATION (MatNILM-Exact Implementation)
# ============================================================================

def vertical_scale(x):
    """
    Vertically scale signal amplitude using truncated normal distribution
    Mean=1, Std=0.2, range=[0.6, 1.4]
    """
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)[0]
    return x * scale


def horizontal_scale(signal, target_length=None):
    """
    Horizontally scale (time-stretch/compress) signal using interpolation
    Scale factor from truncated normal: Mean=1, Std=0.2, range=[0.6, 1.4]
    """
    if target_length is None:
        target_length = len(signal)

    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)[0]

    y = signal.reshape(-1)
    x = np.arange(0, len(y))
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')

    xnew = np.arange(0, len(y) - 1, scale)
    ynew = f(xnew)

    if len(ynew) > target_length:
        return ynew[:target_length]
    else:
        # Pad with zeros if too short
        diffSize = target_length - len(ynew)
        ynew = np.pad(ynew, (0, diffSize), 'constant')
        return ynew


def add_noise(signal, noise_factor=0.02):
    """Add Gaussian noise to signal"""
    noise = np.random.normal(0, noise_factor, signal.shape)
    return signal + noise


def apply_augmentation_to_batch(X_batch, mode='none'):
    """
    Apply data augmentation to a batch of input sequences (MatNILM-aligned)

    Args:
        X_batch: Input batch of shape (batch_size, seq_len, 1)
        mode: Augmentation mode:
            - 'none': No augmentation
            - 'vertical': Vertical scaling only
            - 'horizontal': Horizontal scaling only
            - 'both': Both vertical and horizontal
            - 'mixed': Randomly choose augmentation (MatNILM strategy)

    Returns:
        Augmented batch
    """
    if mode == 'none':
        return X_batch

    X_aug = X_batch.copy()
    batch_size, seq_len, _ = X_batch.shape

    for i in range(batch_size):
        signal = X_batch[i, :, 0]

        # Choose augmentation strategy (MatNILM: 4 modes with equal probability)
        if mode == 'mixed':
            aug_choice = np.random.choice(['none', 'vertical', 'horizontal', 'both'],
                                         p=[0.25, 0.25, 0.25, 0.25])
        else:
            aug_choice = mode

        # Apply augmentation
        if aug_choice == 'vertical':
            signal = vertical_scale(signal)
        elif aug_choice == 'horizontal':
            signal = horizontal_scale(signal, target_length=seq_len)
        elif aug_choice == 'both':
            signal = horizontal_scale(signal, target_length=seq_len)
            signal = vertical_scale(signal)
        # 'none' mode: no transformation applied

        X_aug[i, :, 0] = signal

    return X_aug


def train_attention_liquidnn_on_specific_redd_appliance(
    data_dict, appliance_name, model_type='attention',
    window_size=100, hidden_size=256, num_layers=3, dt=0.1,
    num_heads=4, dropout=0.1, epochs=20, lr=0.001, patience=10,
    seed=42, save_dir='models/attention_liquidnn_redd_specific',
    augmentation='none', aug_probability=0.5
):
    """
    Train Attention-Enhanced Liquid Neural Network on specific REDD appliance

    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        model_type: 'attention' (single-layer + attention)
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        num_layers: Number of layers (not used for single-layer)
        dt: Time step for liquid dynamics
        num_heads: Number of attention heads
        dropout: Dropout rate
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        seed: Random seed
        save_dir: Directory to save model and results
        augmentation: Data augmentation mode (MatNILM-aligned):
            'none': No augmentation (default)
            'vertical': Vertical scaling only
            'horizontal': Horizontal scaling only
            'both': Both vertical and horizontal
            'mixed': Randomly choose augmentation (MatNILM: equal 25% for each mode)
        aug_probability: Probability of applying augmentation to each batch
                        (MatNILM defaults: 0.3 for dishwasher/microwave/washer, 0.6 for fridge)

    Returns:
        Trained model, training history, and evaluation metrics
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract data
    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    # Create sequences
    print(f"Creating sequences for {appliance_name}...")
    X_train, y_train_temp = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val, y_val_temp = create_sequences(val_data, window_size=window_size, target_size=1)
    X_test, y_test_temp = create_sequences(test_data, window_size=window_size, target_size=1)

    # Use appliance power as target
    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    # Normalize inputs
    X_mean = X_train.mean()
    X_std = X_train.std() + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    print(f"Input normalization: mean={X_mean:.2f}, std={X_std:.2f}")

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

    # Create model
    input_size = 1
    output_size = 1

    model = AttentionLiquidNetworkModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dt=dt,
        num_heads=num_heads,
        dropout=dropout
    )
    model_name = "attention_liquid"
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'best': None
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = None
    best_metrics = None
    counter = 0
    best_model_path = None
    consecutive_nan_epochs = 0

    print(f"Starting {model_name} training for {appliance_name}...")
    if augmentation != 'none':
        print(f"Data augmentation: {augmentation} (probability={aug_probability})")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Apply data augmentation (training only)
            if augmentation != 'none' and np.random.rand() < aug_probability:
                inputs_np = inputs.cpu().numpy()
                inputs_aug = apply_augmentation_to_batch(inputs_np, mode=augmentation)
                inputs = torch.FloatTensor(inputs_aug)

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at batch {batch_idx}! Skipping batch...")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # Collect for metrics
            train_targets.append(targets.detach().cpu().numpy())
            train_outputs.append(outputs.detach().cpu().numpy())

        # Calculate average training loss
        num_train_batches = len(train_targets)
        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0
        history['train_loss'].append(avg_train_loss)

        # Check for NaN in training loss
        if np.isnan(avg_train_loss):
            consecutive_nan_epochs += 1
            if consecutive_nan_epochs >= 3:
                print(f"\n❌ Training aborted: Model produced NaN loss for {consecutive_nan_epochs} consecutive epochs")
                raise ValueError(f"Training failed with persistent NaN losses for {appliance_name}")
        else:
            consecutive_nan_epochs = 0

        # Training metrics
        train_targets = np.concatenate(train_targets)
        train_outputs = np.concatenate(train_outputs)
        train_threshold = get_threshold_for_appliance(appliance_name)
        train_metrics = calculate_nilm_metrics(train_targets, train_outputs, threshold=train_threshold)
        history['train_metrics'].append(train_metrics)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Calculate NILM metrics
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        val_threshold = get_threshold_for_appliance(appliance_name)
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=val_threshold)
        history['val_metrics'].append(metrics)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Val MAE: {metrics['mae']:.2f}, Val F1: {metrics['f1']:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_metrics = metrics
            counter = 0

            # Save best model
            best_model_path = os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_best.pth")

            model_params = {
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers if model_type == 'advanced_attention' else 1,
                'dt': dt,
                'num_heads': num_heads,
                'dropout': dropout,
                'model_type': model_type
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience
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

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Calculate test metrics
    all_test_targets = np.concatenate(all_test_targets)
    all_test_outputs = np.concatenate(all_test_outputs)
    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")

    # Store best info
    history['best'] = {
        'epoch': int(best_epoch) if best_epoch is not None else None,
        'val_loss': float(best_val_loss) if best_epoch is not None else None,
        'val_metrics': {k: float(v) for k, v in best_metrics.items()} if best_metrics is not None else None,
        'model_path': best_model_path
    }

    # Plot metrics
    plot_training_metrics(history, test_metrics, appliance_name, model_name, save_dir)

    # Save training history
    save_training_history(history, test_metrics, appliance_name, model_name, model_params, train_params, save_dir)

    return model, history, test_metrics


def plot_training_metrics(history, test_metrics, appliance_name, model_name, save_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # MAE
    plt.subplot(2, 2, 2)
    train_mae = [m['mae'] for m in history['train_metrics']]
    val_mae = [m['mae'] for m in history['val_metrics']]
    plt.plot(train_mae, label='Train MAE', color='blue')
    plt.plot(val_mae, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SAE
    plt.subplot(2, 2, 3)
    train_sae = [m['sae'] for m in history['train_metrics']]
    val_sae = [m['sae'] for m in history['val_metrics']]
    plt.plot(train_sae, label='Train SAE', color='blue')
    plt.plot(val_sae, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('SAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1
    plt.subplot(2, 2, 4)
    train_f1 = [m['f1'] for m in history['train_metrics']]
    val_f1 = [m['f1'] for m in history['val_metrics']]
    plt.plot(train_f1, label='Train F1', color='blue')
    plt.plot(val_f1, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_metrics.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def save_training_history(history, test_metrics, appliance_name, model_name, model_params, train_params, save_dir):
    """Save training history to JSON"""
    config = {
        'appliance': appliance_name,
        'model_params': model_params,
        'train_params': train_params,
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best': history['best']
        }
    }

    with open(os.path.join(save_dir, f'{model_name}_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)


def test_attention_liquidnn_on_all_redd_appliances(
    model_type='attention', window_size=100, hidden_size=256,
    num_layers=3, dt=0.1, num_heads=4, dropout=0.1,
    epochs=20, lr=0.001, patience=10,
    augmentation='none', aug_probability=0.5
):
    """
    Test Attention-Enhanced LNN on all REDD appliances

    Args:
        model_type: 'attention' or 'advanced_attention'
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        num_layers: Number of layers (for advanced model)
        dt: Time step for liquid dynamics
        num_heads: Number of attention heads
        dropout: Dropout rate
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Dictionary of results for all appliances
    """
    # Load data
    print("Loading REDD data...")
    data_dict = load_redd_specific_splits()

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "attention_liquid" if model_type == 'attention' else "advanced_attention_liquid"
    base_save_dir = f"models/{model_name}_redd_specific_test_{timestamp}"

    # Results storage
    all_results = {}

    # Test on each appliance
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} on {appliance_name}")
        print(f"{'='*60}\n")

        # Create appliance-specific directory
        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        # Get MatNILM-specific augmentation probability for this appliance
        if augmentation != 'none':
            appliance_aug_prob = get_matnilm_augmentation_probability(appliance_name)
            print(f"Using MatNILM augmentation probability: {appliance_aug_prob} for {appliance_name}")
        else:
            appliance_aug_prob = aug_probability  # Use provided value for no augmentation

        try:
            model, history, test_metrics = train_attention_liquidnn_on_specific_redd_appliance(
                data_dict,
                appliance_name=appliance_name,
                model_type=model_type,
                window_size=window_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dt=dt,
                num_heads=num_heads,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                patience=patience,
                seed=42,
                save_dir=appliance_dir,
                augmentation=augmentation,
                aug_probability=appliance_aug_prob
            )

            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(appliance_dir,
                                              f"{model_name}_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'best_epoch': history.get('best', {}).get('epoch')
                }
                print(f"✅ Successfully tested {model_name} on {appliance_name}")
            else:
                print(f"❌ Failed to test {model_name} on {appliance_name}")

        except Exception as e:
            print(f"Error testing {model_name} on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary = {
        'timestamp': timestamp,
        'model_type': model_name,
        'model_params': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dt': dt,
            'num_heads': num_heads,
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

    print(f"\n🎉 {model_name} testing completed on all REDD appliances!")
    print(f"Results saved to {base_save_dir}")

    return all_results


if __name__ == "__main__":
    print("Testing Attention-Enhanced Liquid Neural Networks on REDD dataset...")
    print("=" * 70)

    # Check if data files exist
    required_files = [
        'data/redd/train_small.pkl',
        'data/redd/val_small.pkl',
        'data/redd/test_small.pkl'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            print("Please ensure the REDD dataset pickle files are in the data/redd/ directory.")
            sys.exit(1)

    # =================================================================
    # OPTION 1: Test WITHOUT Data Augmentation (Baseline)
    # =================================================================
    print("\n" + "=" * 70)
    print("TESTING: Attention LNN WITHOUT Data Augmentation (Baseline)")
    print("=" * 70)
    results_baseline = test_attention_liquidnn_on_all_redd_appliances(
        model_type='attention',
        window_size=100,
        hidden_size=256,
        num_layers=1,  # Not used for single-layer
        dt=0.1,
        num_heads=4,
        dropout=0.1,
        epochs=20,
        lr=0.001,
        patience=10,
        augmentation='none',  # NO augmentation
        aug_probability=0.0
    )

    # Print summary
    print(f"\n📊 Summary of Attention LNN (Baseline - No Augmentation):")
    print(f"Total appliances tested: {len(results_baseline)}")
    for appliance, result in results_baseline.items():
        print(f"\n  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")

    # =================================================================
    # OPTION 2: Test WITH Data Augmentation (MatNILM-exact)
    # =================================================================
    print("\n" + "=" * 70)
    print("TESTING: Attention LNN WITH MatNILM-Exact Data Augmentation")
    print("=" * 70)
    print("Augmentation mode: 'mixed' (4 modes with equal 25% probability)")
    print("  - 25% none (no augmentation)")
    print("  - 25% vertical scaling only")
    print("  - 25% horizontal scaling only")
    print("  - 25% both vertical + horizontal")
    print("Augmentation probabilities (MatNILM-specific per appliance):")
    print("  - Dishwasher: 0.3")
    print("  - Fridge: 0.6")
    print("  - Microwave: 0.3")
    print("  - Washer Dryer: 0.3")
    results_augmented = test_attention_liquidnn_on_all_redd_appliances(
        model_type='attention',
        window_size=100,
        hidden_size=256,
        num_layers=1,
        dt=0.1,
        num_heads=4,
        dropout=0.1,
        epochs=20,
        lr=0.001,
        patience=10,
        augmentation='mixed',  # MatNILM-exact augmentation
        aug_probability=0.5    # This will be overridden by appliance-specific values
    )

    # Print summary
    print(f"\n📊 Summary of Attention LNN (With Augmentation):")
    print(f"Total appliances tested: {len(results_augmented)}")
    for appliance, result in results_augmented.items():
        print(f"\n  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")

    print("\n✅ All attention model testing completed!")
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Augmented")
    print("=" * 70)
    for appliance in results_baseline.keys():
        baseline_f1 = results_baseline[appliance]['final_metrics']['f1']
        aug_f1 = results_augmented[appliance]['final_metrics']['f1']
        improvement = ((aug_f1 - baseline_f1) / baseline_f1) * 100
        print(f"{appliance}:")
        print(f"  Baseline F1: {baseline_f1:.4f}")
        print(f"  Augmented F1: {aug_f1:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
