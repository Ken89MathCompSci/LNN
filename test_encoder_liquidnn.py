"""
Test Encoder-Based Liquid Neural Networks on REDD Dataset
With MatNILM-Exact Data Augmentation Support

This script tests three new encoder architectures:
1. CNN Encoder + Liquid Decoder
2. Transformer Encoder + Liquid Decoder
3. Bidirectional Encoder + Liquid Decoder

Each encoder extracts features in different ways before processing with liquid dynamics.

NEW: Full MatNILM-exact data augmentation support for all encoder models!
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
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

# Add Source Code to path
sys.path.append('Source Code')

# Import models
from models import (
    CNNEncoderLiquidNetworkModel,
    TransformerEncoderLiquidNetworkModel,
    BidirectionalEncoderLiquidNetworkModel
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
    return matnilm_probs.get(appliance_name, 0.3)


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


def load_redd_specific_splits():
    """Load REDD data with specific splits"""
    print("Loading REDD data with specific splits...")

    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]

    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]

    with open('data/redd/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    return {'train': train_data, 'val': val_data, 'test': test_data}


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(mains, appliance, window_size=100, target_size=1):
    """Create sequences for sequence-to-point prediction"""
    X, y = [], []
    for i in range(len(mains) - window_size - target_size + 1):
        X.append(mains[i:i+window_size])
        if target_size == 1:
            y.append(appliance[i+window_size])
        else:
            y.append(mains[i+window_size:i+window_size+target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def train_encoder_liquidnn_on_specific_redd_appliance(
    data_dict, appliance_name, model_type='cnn_encoder',
    window_size=100, hidden_size=256, dt=0.1,
    num_conv_layers=3, kernel_size=5,
    num_encoder_layers=2, num_heads=4,
    dropout=0.1, epochs=20, lr=0.001, patience=10,
    seed=42, save_dir='models/encoder_liquidnn_redd_specific',
    augmentation='none', aug_probability=0.5
):
    """
    Train Encoder-Based Liquid Neural Network on specific REDD appliance

    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        model_type: 'cnn_encoder', 'transformer_encoder', or 'bidirectional_encoder'
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        dt: Time step for liquid dynamics
        num_conv_layers: Number of CNN layers (for CNN encoder)
        kernel_size: Kernel size for CNN (for CNN encoder)
        num_encoder_layers: Number of Transformer layers (for Transformer encoder)
        num_heads: Number of attention heads (for Transformer encoder)
        dropout: Dropout rate
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        seed: Random seed
        save_dir: Directory to save model and results
        augmentation: Data augmentation mode (MatNILM-exact):
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract data for this appliance
    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    # Create sequences
    print(f"Creating sequences for {appliance_name}...")
    X_train, y_train = create_sequences(
        train_data['main'].values,
        train_data[appliance_name].values,
        window_size=window_size
    )
    X_val, y_val = create_sequences(
        val_data['main'].values,
        val_data[appliance_name].values,
        window_size=window_size
    )
    X_test, y_test = create_sequences(
        test_data['main'].values,
        test_data[appliance_name].values,
        window_size=window_size
    )

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}, Test sequences: {X_test.shape}")

    # Normalize input data
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Normalize target data
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Create datasets
    batch_size = 128
    train_dataset = SimpleDataset(X_train, y_train.reshape(-1, 1))
    val_dataset = SimpleDataset(X_val, y_val.reshape(-1, 1))
    test_dataset = SimpleDataset(X_test, y_test.reshape(-1, 1))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = 1
    output_size = 1

    if model_type == 'cnn_encoder':
        model = CNNEncoderLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        model_name = "cnn_encoder_liquid"
    elif model_type == 'transformer_encoder':
        model = TransformerEncoderLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        model_name = "transformer_encoder_liquid"
    elif model_type == 'bidirectional_encoder':
        model = BidirectionalEncoderLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt,
            dropout=dropout
        )
        model_name = "bidirectional_encoder_liquid"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {num_params:,}")

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = None
    best_metrics = None
    counter = 0
    best_model_path = None

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
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Calculate training metrics
        train_targets = np.concatenate(train_targets) * y_std + y_mean
        train_outputs = np.concatenate(train_outputs) * y_std + y_mean
        train_threshold = get_threshold_for_appliance(appliance_name)
        train_metrics = calculate_nilm_metrics(train_targets, train_outputs, threshold=train_threshold)
        history['train_f1'].append(train_metrics['f1'])

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                val_targets.append(targets.cpu().numpy())
                val_outputs.append(outputs.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Calculate validation metrics
        val_targets = np.concatenate(val_targets) * y_std + y_mean
        val_outputs = np.concatenate(val_outputs) * y_std + y_mean
        metrics = calculate_nilm_metrics(val_targets, val_outputs, threshold=train_threshold)
        history['val_f1'].append(metrics['f1'])

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Val MAE: {metrics['mae']:.2f} | Val F1: {metrics['f1']:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_metrics = metrics
            counter = 0

            # Save best model
            best_model_path = os.path.join(save_dir, f"{model_name}_{appliance_name.replace(' ', '_')}_best.pth")

            model_params = {
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': hidden_size,
                'dt': dt,
                'model_type': model_type
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience,
                'window_size': window_size
            }

            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"✓ Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for testing
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {best_epoch}")

    # Test phase
    model.eval()
    test_loss = 0.0
    test_targets = []
    test_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            test_targets.append(targets.cpu().numpy())
            test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # Calculate test metrics
    test_targets = np.concatenate(test_targets) * y_std + y_mean
    test_outputs = np.concatenate(test_outputs) * y_std + y_mean
    test_metrics = calculate_nilm_metrics(test_targets, test_outputs, threshold=train_threshold)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS for {appliance_name} ({model_name})")
    print(f"{'='*60}")
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.2f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test SAE: {test_metrics['sae']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.replace("_", " ").title()} - Training History')
    axes[0].legend()
    axes[0].grid(True)

    # F1 plot
    axes[1].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1].plot(history['val_f1'], label='Val F1', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title(f'{model_name.replace("_", " ").title()} - F1 Score')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_{appliance_name.replace(" ", "_")}_history.png'))
    plt.close()

    # Plot predictions
    num_samples = min(500, len(test_targets))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].plot(test_targets[:num_samples], label='Actual', alpha=0.7)
    axes[0].plot(test_outputs[:num_samples], label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title(f'{model_name.replace("_", " ").title()} - {appliance_name} - Test Predictions')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(test_targets[:num_samples], test_outputs[:num_samples], alpha=0.5)
    axes[1].plot([test_targets.min(), test_targets.max()],
                 [test_targets.min(), test_targets.max()], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Actual Power (W)')
    axes[1].set_ylabel('Predicted Power (W)')
    axes[1].set_title('Prediction Scatter Plot')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_{appliance_name.replace(" ", "_")}_predictions.png'))
    plt.close()

    # Save history to JSON
    history_data = {
        'appliance': appliance_name,
        'model_type': model_type,
        'model_name': model_name,
        'num_parameters': num_params,
        'window_size': window_size,
        'best_epoch': best_epoch,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_f1': history['train_f1'],
        'val_f1': history['val_f1'],
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'test_loss': avg_test_loss
    }

    with open(os.path.join(save_dir, f'{model_name}_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(history_data, f, indent=4)

    return model, history, test_metrics


def test_encoder_liquidnn_on_all_appliances(
    model_type='cnn_encoder',
    window_size=100, hidden_size=256, dt=0.1,
    num_conv_layers=3, kernel_size=5,
    num_encoder_layers=2, num_heads=4,
    dropout=0.1, epochs=20, lr=0.001, patience=10,
    augmentation='none', aug_probability=0.5
):
    """
    Test Encoder-Based LNN on all REDD appliances

    Args:
        model_type: 'cnn_encoder', 'transformer_encoder', or 'bidirectional_encoder'
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        dt: Time step for liquid dynamics
        num_conv_layers: Number of CNN layers (for CNN encoder)
        kernel_size: Kernel size for CNN (for CNN encoder)
        num_encoder_layers: Number of Transformer layers (for Transformer encoder)
        num_heads: Number of attention heads (for Transformer encoder)
        dropout: Dropout rate
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        augmentation: Data augmentation mode (MatNILM-exact):
            'none': No augmentation (default)
            'vertical': Vertical scaling only
            'horizontal': Horizontal scaling only
            'both': Both vertical and horizontal
            'mixed': Randomly choose augmentation (MatNILM: equal 25% for each mode)
        aug_probability: Base augmentation probability (overridden by appliance-specific values)

    Returns:
        Dictionary of results for all appliances
    """
    # Load data
    data_dict = load_redd_specific_splits()

    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f'models/encoder_liquidnn_test_{timestamp}'
    os.makedirs(base_save_dir, exist_ok=True)

    # Model name mapping
    model_names = {
        'cnn_encoder': 'CNN Encoder + Liquid',
        'transformer_encoder': 'Transformer Encoder + Liquid',
        'bidirectional_encoder': 'Bidirectional Encoder + Liquid'
    }
    model_display_name = model_names.get(model_type, model_type)

    print(f"\n{'='*70}")
    print(f"TESTING: {model_display_name}")
    print(f"{'='*70}\n")

    # Test on all appliances
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']
    all_results = {}

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing {model_display_name} on {appliance_name}")
        print(f"{'='*60}\n")

        # Create appliance-specific directory
        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        # Get MatNILM-specific augmentation probability for this appliance
        if augmentation != 'none':
            appliance_aug_prob = get_matnilm_augmentation_probability(appliance_name)
            print(f"Using MatNILM augmentation probability: {appliance_aug_prob} for {appliance_name}")
        else:
            appliance_aug_prob = aug_probability

        try:
            model, history, test_metrics = train_encoder_liquidnn_on_specific_redd_appliance(
                data_dict,
                appliance_name=appliance_name,
                model_type=model_type,
                window_size=window_size,
                hidden_size=hidden_size,
                dt=dt,
                num_conv_layers=num_conv_layers,
                kernel_size=kernel_size,
                num_encoder_layers=num_encoder_layers,
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
                    'model_path': os.path.join(appliance_dir, f"{model_type}_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"✅ Successfully tested {model_display_name} on {appliance_name}")
            else:
                print(f"❌ Failed to test {model_display_name} on {appliance_name}")

        except Exception as e:
            print(f"❌ Error testing {model_display_name} on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary = {
        'timestamp': timestamp,
        'model_type': model_type,
        'model_name': model_display_name,
        'window_size': window_size,
        'hidden_size': hidden_size,
        'dt': dt,
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n🎉 {model_display_name} testing completed!")
    print(f"Results saved to: {base_save_dir}")

    return all_results


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description='Test Encoder-Based Liquid Neural Networks')
    parser.add_argument('--augmentation', type=str, default='none',
                       choices=['none', 'vertical', 'horizontal', 'both', 'mixed'],
                       help='Data augmentation mode (default: none)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'cnn', 'transformer', 'bidirectional'],
                       help='Which model to test (default: all)')
    args = parser.parse_args()

    # Verify data files exist
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
    # Test all three encoder types
    # =================================================================

    print("\n" + "=" * 70)
    print("ENCODER-BASED LIQUID NEURAL NETWORKS - COMPREHENSIVE TEST")
    print("=" * 70)
    print(f"Augmentation: {args.augmentation}")
    print(f"Epochs: {args.epochs}")
    print(f"Models: {args.model}")
    print("=" * 70)

    all_model_results = {}

    # Determine which models to test
    models_to_test = []
    if args.model == 'all':
        models_to_test = ['cnn', 'transformer', 'bidirectional']
    else:
        models_to_test = [args.model]

    # 1. CNN Encoder + Liquid
    if 'cnn' in models_to_test:
        print("\n\n" + "=" * 70)
        print("TEST: CNN Encoder + Liquid Neural Network")
        print("=" * 70)
        results_cnn = test_encoder_liquidnn_on_all_appliances(
            model_type='cnn_encoder',
            window_size=100,
            hidden_size=256,
            dt=0.1,
            num_conv_layers=3,
            kernel_size=5,
            dropout=0.1,
            epochs=args.epochs,
            lr=0.001,
            patience=10,
            augmentation=args.augmentation
        )
        all_model_results['cnn_encoder'] = results_cnn

    # 2. Transformer Encoder + Liquid
    if 'transformer' in models_to_test:
        print("\n\n" + "=" * 70)
        print("TEST: Transformer Encoder + Liquid Neural Network")
        print("=" * 70)
        results_transformer = test_encoder_liquidnn_on_all_appliances(
            model_type='transformer_encoder',
            window_size=100,
            hidden_size=256,
            dt=0.1,
            num_encoder_layers=2,
            num_heads=4,
            dropout=0.1,
            epochs=args.epochs,
            lr=0.001,
            patience=10,
            augmentation=args.augmentation
        )
        all_model_results['transformer_encoder'] = results_transformer

    # 3. Bidirectional Encoder + Liquid
    if 'bidirectional' in models_to_test:
        print("\n\n" + "=" * 70)
        print("TEST: Bidirectional Encoder + Liquid Neural Network")
        print("=" * 70)
        results_bidirectional = test_encoder_liquidnn_on_all_appliances(
            model_type='bidirectional_encoder',
            window_size=100,
            hidden_size=256,
            dt=0.1,
            dropout=0.1,
            epochs=args.epochs,
            lr=0.001,
            patience=10,
            augmentation=args.augmentation
        )
        all_model_results['bidirectional_encoder'] = results_bidirectional

    # =================================================================
    # Comparison Summary
    # =================================================================
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON: All Encoder-Based Models")
    print("=" * 70)

    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance in appliances:
        print(f"\n{appliance.upper()}:")
        print("-" * 60)

        for model_name, results in all_model_results.items():
            if appliance in results:
                f1 = results[appliance]['final_metrics']['f1']
                mae = results[appliance]['final_metrics']['mae']
                display_name = {
                    'cnn_encoder': 'CNN Encoder + Liquid',
                    'transformer_encoder': 'Transformer Encoder + Liquid',
                    'bidirectional_encoder': 'Bidirectional Encoder + Liquid'
                }[model_name]
                print(f"  {display_name:35s} | F1: {f1:.4f} | MAE: {mae:.2f}")

    print("\n" + "=" * 70)
    print("✅ ALL ENCODER-BASED LIQUID NEURAL NETWORK TESTS COMPLETED!")
    print("=" * 70)
