"""
Test Attention-Enhanced Liquid Neural Networks on REDD Dataset
Compares standard LNN vs attention-enhanced LNN models
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from test_liquidnn_redd_specific_splits import (
    load_redd_specific_splits,
    create_sliding_windows,
    calculate_metrics
)

# Import the new attention models
sys.path.append('Source Code')
from models import (
    LiquidNetworkModel,
    AdvancedLiquidNetworkModel,
    AttentionLiquidNetworkModel,
    AdvancedAttentionLiquidNetworkModel
)


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, patience=10, device='cuda'):
    """Train a model with early stopping"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return model, history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    metrics = calculate_metrics(targets, predictions)
    return metrics, predictions, targets


def run_comparison(appliance_name='dish washer', window_size=100, hidden_size=256,
                   num_layers=3, lr=0.001, epochs=20, seed=42):
    """
    Compare four models:
    1. Standard LiquidNetworkModel
    2. AdvancedLiquidNetworkModel
    3. AttentionLiquidNetworkModel
    4. AdvancedAttentionLiquidNetworkModel
    """

    print("=" * 80)
    print(f"Attention-Enhanced LNN Comparison on {appliance_name}")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    print("\nLoading REDD dataset...")
    data_dict = load_redd_specific_splits()

    # Prepare data
    train_x = data_dict['train'][appliance_name]['aggregate']
    train_y = data_dict['train'][appliance_name]['appliance']
    val_x = data_dict['val'][appliance_name]['aggregate']
    val_y = data_dict['val'][appliance_name]['appliance']
    test_x = data_dict['test'][appliance_name]['aggregate']
    test_y = data_dict['test'][appliance_name]['appliance']

    print(f"\nData shapes:")
    print(f"  Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")

    # Create sliding windows
    print(f"\nCreating sliding windows (window_size={window_size})...")
    train_windows_x, train_windows_y = create_sliding_windows(train_x, train_y, window_size)
    val_windows_x, val_windows_y = create_sliding_windows(val_x, val_y, window_size)
    test_windows_x, test_windows_y = create_sliding_windows(test_x, test_y, window_size)

    # Create data loaders
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_windows_x),
        torch.FloatTensor(train_windows_y)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_windows_x),
        torch.FloatTensor(val_windows_y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_windows_x),
        torch.FloatTensor(test_windows_y)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 1
    output_size = 1

    # Results storage
    results = {}

    # Model configurations
    models_config = [
        {
            'name': 'Standard LNN',
            'model': LiquidNetworkModel(input_size, hidden_size, output_size, dt=0.1),
            'color': 'blue'
        },
        {
            'name': 'Advanced LNN',
            'model': AdvancedLiquidNetworkModel(input_size, hidden_size, output_size,
                                                num_layers=num_layers, dt=0.1),
            'color': 'green'
        },
        {
            'name': 'Attention LNN',
            'model': AttentionLiquidNetworkModel(input_size, hidden_size, output_size,
                                                 dt=0.1, num_heads=4, dropout=0.1),
            'color': 'orange'
        },
        {
            'name': 'Advanced Attention LNN',
            'model': AdvancedAttentionLiquidNetworkModel(input_size, hidden_size, output_size,
                                                         num_layers=num_layers, dt=0.1,
                                                         num_heads=4, dropout=0.1),
            'color': 'red'
        }
    ]

    # Train and evaluate each model
    for config in models_config:
        print("\n" + "=" * 80)
        print(f"Training: {config['name']}")
        print("=" * 80)

        model = config['model'].to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Train
        print("\nTraining...")
        trained_model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, patience=10, device=device
        )

        # Evaluate
        print("\nEvaluating on test set...")
        test_metrics, predictions, targets = evaluate_model(trained_model, test_loader, device)

        # Store results
        results[config['name']] = {
            'metrics': test_metrics,
            'history': history,
            'num_params': num_params,
            'predictions': predictions,
            'targets': targets
        }

        print(f"\n✅ {config['name']} Test Results:")
        print(f"   F1 Score:   {test_metrics['f1']:.4f}")
        print(f"   MAE:        {test_metrics['mae']:.4f}")
        print(f"   SAE:        {test_metrics['sae']:.4f}")
        print(f"   Precision:  {test_metrics['precision']:.4f}")
        print(f"   Recall:     {test_metrics['recall']:.4f}")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<25} {'F1 Score':<12} {'MAE':<12} {'SAE':<12} {'Params':<12}")
    print("-" * 80)

    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)

    for rank, (name, result) in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else "  "
        metrics = result['metrics']
        params = result['num_params']
        print(f"{marker} {name:<23} {metrics['f1']:<12.4f} {metrics['mae']:<12.4f} "
              f"{metrics['sae']:<12.4f} {params:<12,}")

    # Calculate improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    baseline_f1 = results['Standard LNN']['metrics']['f1']

    print(f"\nBaseline (Standard LNN) F1: {baseline_f1:.4f}\n")

    for name, result in results.items():
        if name != 'Standard LNN':
            f1 = result['metrics']['f1']
            improvement = ((f1 - baseline_f1) / baseline_f1) * 100
            sign = "+" if improvement > 0 else ""
            print(f"{name:<25} F1: {f1:.4f}  ({sign}{improvement:>6.2f}% vs baseline)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/attention_comparison_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    import json
    summary = {
        'timestamp': timestamp,
        'appliance': appliance_name,
        'configuration': {
            'window_size': window_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'seed': seed
        },
        'results': {
            name: {
                'metrics': result['metrics'],
                'num_params': result['num_params']
            }
            for name, result in results.items()
        }
    }

    with open(os.path.join(save_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Results saved to: {save_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    print("Attention-Enhanced Liquid Neural Network Comparison")
    print("=" * 80)

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

    # Configuration
    appliance = 'dish washer'  # Change to: 'fridge', 'microwave', or 'washer dryer'

    print(f"\nTesting appliance: {appliance}")
    print("\nThis will train and compare 4 models:")
    print("  1. Standard LNN (baseline)")
    print("  2. Advanced LNN (multi-layer)")
    print("  3. Attention LNN (single-layer + attention)")
    print("  4. Advanced Attention LNN (multi-layer + attention)")
    print("\nConfiguration:")
    print("  - Hidden size: 256")
    print("  - Num layers: 3 (for multi-layer models)")
    print("  - Learning rate: 0.001")
    print("  - Epochs: 20 (with early stopping)")
    print("\nExpected runtime: ~30-45 minutes with GPU")
    print("=" * 80)

    # Run comparison
    results = run_comparison(
        appliance_name=appliance,
        window_size=100,
        hidden_size=256,
        num_layers=3,
        lr=0.001,
        epochs=20,
        seed=42
    )

    print(f"\n✅ Comparison completed!")
    print(f"\nBest model: {max(results.items(), key=lambda x: x[1]['metrics']['f1'])[0]}")
    print(f"Best F1 score: {max(result['metrics']['f1'] for result in results.values()):.4f}")
