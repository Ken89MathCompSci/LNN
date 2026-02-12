"""
Comprehensive Comparison of ALL Liquid Neural Network Models on REDD Dataset

This script tests and compares 6 different LNN architectures:
1. Standard LNN
2. Advanced LNN
3. Attention LNN
4. CNN Encoder + Liquid
5. Transformer Encoder + Liquid
6. Bidirectional Encoder + Liquid

Each model is tested with and without MatNILM-exact data augmentation for comprehensive comparison.
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
import pandas as pd

# Add Source Code to path
sys.path.append('Source Code')

# Import all models
from models import (
    LiquidNetworkModel,
    AdvancedLiquidNetworkModel,
    AttentionLiquidNetworkModel,
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
    """Get MatNILM-specific augmentation probability for each appliance"""
    matnilm_probs = {
        'dish washer': 0.3,
        'fridge': 0.6,
        'microwave': 0.3,
        'washer dryer': 0.3
    }
    return matnilm_probs.get(appliance_name, 0.3)


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


# ============================================================================
# DATA AUGMENTATION (MatNILM-Exact Implementation)
# ============================================================================

def vertical_scale(x):
    """Vertically scale signal amplitude using truncated normal distribution"""
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)[0]
    return x * scale


def horizontal_scale(signal, target_length=None):
    """Horizontally scale (time-stretch/compress) signal using interpolation"""
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
        diffSize = target_length - len(ynew)
        ynew = np.pad(ynew, (0, diffSize), 'constant')
        return ynew


def apply_augmentation_to_batch(X_batch, mode='none'):
    """Apply data augmentation to a batch of input sequences (MatNILM-aligned)"""
    if mode == 'none':
        return X_batch

    X_aug = X_batch.copy()
    batch_size, seq_len, _ = X_batch.shape

    for i in range(batch_size):
        signal = X_batch[i, :, 0]

        if mode == 'mixed':
            aug_choice = np.random.choice(['none', 'vertical', 'horizontal', 'both'],
                                         p=[0.25, 0.25, 0.25, 0.25])
        else:
            aug_choice = mode

        if aug_choice == 'vertical':
            signal = vertical_scale(signal)
        elif aug_choice == 'horizontal':
            signal = horizontal_scale(signal, target_length=seq_len)
        elif aug_choice == 'both':
            signal = horizontal_scale(signal, target_length=seq_len)
            signal = vertical_scale(signal)

        X_aug[i, :, 0] = signal

    return X_aug


def test_single_model(
    data_dict, appliance_name, model_type,
    window_size=100, hidden_size=256, dt=0.1,
    epochs=20, lr=0.001, patience=10,
    augmentation='none', aug_probability=0.5,
    save_dir='models/comparison'
):
    """
    Test a single model type on a specific appliance

    Returns:
        Dictionary with test metrics and model info
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract data
    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    # Create sequences
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

    # Normalize
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Create dataloaders
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_train, y_train.reshape(-1, 1)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_val, y_val.reshape(-1, 1)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_test, y_test.reshape(-1, 1)),
        batch_size=batch_size, shuffle=False
    )

    # Create model
    input_size, output_size = 1, 1

    if model_type == 'standard_lnn':
        model = LiquidNetworkModel(input_size, hidden_size, output_size, dt=dt)
        model_name = "Standard LNN"
    elif model_type == 'advanced_lnn':
        model = AdvancedLiquidNetworkModel(input_size, hidden_size, output_size, num_layers=2, dt=dt)
        model_name = "Advanced LNN"
    elif model_type == 'attention_lnn':
        model = AttentionLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt, num_heads=4)
        model_name = "Attention LNN"
    elif model_type == 'cnn_encoder':
        model = CNNEncoderLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt, num_conv_layers=3)
        model_name = "CNN Encoder + Liquid"
    elif model_type == 'transformer_encoder':
        model = TransformerEncoderLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt, num_encoder_layers=2, num_heads=4)
        model_name = "Transformer Encoder + Liquid"
    elif model_type == 'bidirectional_encoder':
        model = BidirectionalEncoderLiquidNetworkModel(input_size, hidden_size, output_size, dt=dt)
        model_name = "Bidirectional Encoder + Liquid"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"  Model: {model_name} | Parameters: {num_params:,}")
    if augmentation != 'none':
        print(f"  Augmentation: {augmentation} (prob={aug_probability})")

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0

    # Training loop (simplified)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Apply augmentation
            if augmentation != 'none' and np.random.rand() < aug_probability:
                inputs_np = inputs.cpu().numpy()
                inputs_aug = apply_augmentation_to_batch(inputs_np, mode=augmentation)
                inputs = torch.FloatTensor(inputs_aug)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Test
    model.eval()
    test_targets, test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_targets.append(targets.cpu().numpy())
            test_outputs.append(outputs.cpu().numpy())

    test_targets = np.concatenate(test_targets) * y_std + y_mean
    test_outputs = np.concatenate(test_outputs) * y_std + y_mean
    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(test_targets, test_outputs, threshold=threshold)

    return {
        'model_name': model_name,
        'model_type': model_type,
        'num_params': num_params,
        'metrics': test_metrics
    }


def run_comprehensive_comparison(augmentation='mixed', epochs=20):
    """Run comprehensive comparison of all models"""

    # Load data
    data_dict = load_redd_specific_splits()

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f'models/comprehensive_comparison_{timestamp}'
    os.makedirs(base_save_dir, exist_ok=True)

    # Model types to test
    model_types = [
        'standard_lnn',
        'advanced_lnn',
        'attention_lnn',
        'cnn_encoder',
        'transformer_encoder',
        'bidirectional_encoder'
    ]

    # Appliances
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    # Results storage
    all_results = {appliance: {} for appliance in appliances}

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON - ALL 6 LNN ARCHITECTURES")
    print("=" * 80)
    print(f"Augmentation: {augmentation}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    # Test each model on each appliance
    for appliance_name in appliances:
        print(f"\n{'='*80}")
        print(f"TESTING ON: {appliance_name.upper()}")
        print(f"{'='*80}\n")

        # Get augmentation probability
        aug_prob = get_matnilm_augmentation_probability(appliance_name) if augmentation != 'none' else 0.0

        for model_type in model_types:
            try:
                result = test_single_model(
                    data_dict, appliance_name, model_type,
                    window_size=100, hidden_size=256, dt=0.1,
                    epochs=epochs, lr=0.001, patience=10,
                    augmentation=augmentation, aug_probability=aug_prob,
                    save_dir=base_save_dir
                )

                all_results[appliance_name][model_type] = result

                print(f"    ✅ {result['model_name']:35s} | F1: {result['metrics']['f1']:.4f} | MAE: {result['metrics']['mae']:.2f}")

            except Exception as e:
                print(f"    ❌ {model_type}: {str(e)}")
                continue

    # Create comparison table
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON - F1 SCORES")
    print("=" * 80)
    print(f"{'Model':<35s} | {'Dishwasher':>10s} | {'Fridge':>10s} | {'Microwave':>10s} | {'Washer':>10s} | {'Avg':>10s}")
    print("-" * 80)

    for model_type in model_types:
        model_name = all_results[appliances[0]][model_type]['model_name'] if model_type in all_results[appliances[0]] else model_type

        scores = []
        row = f"{model_name:<35s}"

        for appliance in appliances:
            if model_type in all_results[appliance]:
                f1 = all_results[appliance][model_type]['metrics']['f1']
                row += f" | {f1:>10.4f}"
                scores.append(f1)
            else:
                row += f" | {'N/A':>10s}"

        avg_f1 = np.mean(scores) if scores else 0.0
        row += f" | {avg_f1:>10.4f}"
        print(row)

    # Save results
    results_file = os.path.join(base_save_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'augmentation': augmentation,
            'epochs': epochs,
            'results': {k: {m: {**v, 'metrics': {mk: float(mv) for mk, mv in v['metrics'].items()}}
                            for m, v in models.items()}
                       for k, models in all_results.items()}
        }, f, indent=4)

    print(f"\n✅ Results saved to: {results_file}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description='Comprehensive comparison of all LNN models')
    parser.add_argument('--augmentation', type=str, default='none',
                       choices=['none', 'vertical', 'horizontal', 'both', 'mixed'],
                       help='Data augmentation mode (default: none, use "mixed" for MatNILM-exact)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    args = parser.parse_args()

    # Verify data files
    required_files = [
        'data/redd/train_small.pkl',
        'data/redd/val_small.pkl',
        'data/redd/test_small.pkl'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            sys.exit(1)

    # Run comparison
    aug_desc = "WITH MatNILM-Exact Augmentation" if args.augmentation == 'mixed' else f"WITH {args.augmentation} augmentation" if args.augmentation != 'none' else "WITHOUT Augmentation (Baseline)"

    print("\n" + "=" * 80)
    print(f"RUNNING COMPREHENSIVE COMPARISON {aug_desc}")
    print("=" * 80)
    print(f"Augmentation: {args.augmentation}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)

    results = run_comprehensive_comparison(augmentation=args.augmentation, epochs=args.epochs)

    print("\n✅ COMPREHENSIVE COMPARISON COMPLETED!")
