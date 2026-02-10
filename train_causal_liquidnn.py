"""
Training script for Causal Liquid Neural Network on REDD dataset
Incorporates causal learning principles to improve F1 scores
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

# Add Source Code to path
sys.path.append('Source Code')

# Import causal models
from causal_liquidnn import (
    CausalLiquidNetworkModel,
    AdvancedCausalLiquidNetworkModel,
    CausalEventLoss,
    detect_causal_events,
    compute_granger_causality_score
)

# Import utilities
from utils import calculate_nilm_metrics, save_model


class REDDCausalDataset(torch.utils.data.Dataset):
    """
    Dataset class for REDD data with causal event annotations
    """
    def __init__(self, X, y, detect_events=True, event_threshold=10.0):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        # Detect causal events in targets
        self.events = None
        if detect_events:
            self.events = detect_causal_events(self.y.squeeze(), threshold=event_threshold)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.events is not None:
            return self.X[idx], self.y[idx], self.events[idx]
        else:
            return self.X[idx], self.y[idx], torch.tensor(0.0)


def load_redd_specific_splits():
    """Load REDD data with specific splits"""
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


def get_threshold_for_appliance(appliance_name):
    """Get appropriate power threshold for on/off detection"""
    if appliance_name == 'washer dryer':
        return 0.5
    else:
        return 10.0


def train_causal_liquidnn(data_dict, appliance_name, window_size=100,
                          hidden_size=128, num_layers=2, dt=0.1, advanced=True,
                          epochs=20, lr=0.001, patience=10,
                          use_causal_loss=True, event_weight_scale=2.0,
                          seed=42, save_dir='models/causal_liquidnn'):
    """
    Train Causal Liquid Neural Network on REDD appliance data

    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        num_layers: Number of liquid layers (for advanced model)
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced causal model
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        use_causal_loss: Whether to use causal event-weighted loss
        event_weight_scale: Scale factor for causal events
        seed: Random seed for reproducibility
        save_dir: Directory to save model and results

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
    X_train, y_train = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val, y_val = create_sequences(val_data, window_size=window_size, target_size=1)
    X_test, y_test = create_sequences(test_data, window_size=window_size, target_size=1)

    # Use appliance power as target
    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    print(f"Training sequences: {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences: {X_test.shape} -> {y_test.shape}")

    # Compute Granger causality (aggregate -> appliance)
    print(f"\nComputing Granger causality score...")
    aggregate_sample = train_data['main'].values[:1000]
    appliance_sample = train_data[appliance_name].values[:1000]
    granger_score = compute_granger_causality_score(aggregate_sample, appliance_sample, max_lag=5)
    print(f"Granger causality (aggregate -> {appliance_name}): {granger_score:.4f}")

    # Create datasets with causal event detection
    event_threshold = get_threshold_for_appliance(appliance_name)
    train_dataset = REDDCausalDataset(X_train, y_train, detect_events=True, event_threshold=event_threshold)
    val_dataset = REDDCausalDataset(X_val, y_val, detect_events=True, event_threshold=event_threshold)
    test_dataset = REDDCausalDataset(X_test, y_test, detect_events=False)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create Causal Liquid Neural Network model
    input_size = 1
    output_size = 1

    if advanced:
        model = AdvancedCausalLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dt=dt
        )
        model_name = "advanced_causal_liquid"
    else:
        model = CausalLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt
        )
        model_name = "causal_liquid"

    model = model.to(device)

    # Loss function (causal event-weighted)
    if use_causal_loss:
        criterion = CausalEventLoss(base_loss='mse', event_weight_scale=event_weight_scale)
        print(f"Using Causal Event Loss (event_weight_scale={event_weight_scale})")
    else:
        criterion = torch.nn.MSELoss()
        print("Using standard MSE loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'avg_event_weights': [],
        'granger_causality': granger_score,
        'best': None
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = None
    best_metrics = None
    counter = 0
    best_model_path = None

    print(f"Starting {model_name} training for {appliance_name}...")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []
        epoch_event_weights = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, events in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass (returns predictions and event weights)
            if advanced:
                outputs, all_event_weights = model(inputs)
                event_weights = all_event_weights[-1]  # Use last layer's events
            else:
                outputs, event_weights = model(inputs)

            # Calculate causal loss
            if use_causal_loss:
                loss = criterion(outputs, targets, event_weights)
            else:
                loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            train_targets.append(targets.detach().cpu().numpy())
            train_outputs.append(outputs.detach().cpu().numpy())
            epoch_event_weights.append(event_weights.mean().item())

        avg_train_loss = train_loss / len(train_loader)
        avg_event_weight = np.mean(epoch_event_weights)
        history['train_loss'].append(avg_train_loss)
        history['avg_event_weights'].append(avg_event_weight)

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
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                if advanced:
                    outputs, all_event_weights = model(inputs)
                    event_weights = all_event_weights[-1]
                else:
                    outputs, event_weights = model(inputs)

                if use_causal_loss:
                    loss = criterion(outputs, targets, event_weights)
                else:
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

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
              f"Val F1: {metrics['f1']:.4f}, Avg Event Weight: {avg_event_weight:.4f}")

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
                'num_layers': num_layers if advanced else 1,
                'dt': dt,
                'advanced': advanced,
                'causal': True
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience,
                'use_causal_loss': use_causal_loss,
                'event_weight_scale': event_weight_scale,
                'seed': seed
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
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if advanced:
                outputs, _ = model(inputs)
            else:
                outputs, _ = model(inputs)

            if use_causal_loss:
                loss = criterion(outputs, targets, None)
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # Calculate test metrics
    all_test_targets = np.concatenate(all_test_targets)
    all_test_outputs = np.concatenate(all_test_outputs)
    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")

    # Save history
    history['best'] = {
        'epoch': int(best_epoch) if best_epoch is not None else None,
        'val_loss': float(best_val_loss) if best_epoch is not None else None,
        'val_metrics': {k: float(v) for k, v in best_metrics.items()} if best_metrics is not None else None,
        'model_path': best_model_path
    }

    config = {
        'appliance': appliance_name,
        'window_size': window_size,
        'model_params': {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers if advanced else 1,
            'dt': dt,
            'advanced': advanced,
            'causal': True
        },
        'train_params': {
            'lr': lr,
            'epochs': epochs,
            'patience': patience,
            'use_causal_loss': use_causal_loss,
            'event_weight_scale': event_weight_scale,
            'seed': seed
        },
        'final_metrics': {
            'test_loss': avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'granger_causality': granger_score,
            'best': history['best']
        }
    }

    with open(os.path.join(save_dir, f'{model_name}_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Plot training curves
    plot_causal_training_history(history, appliance_name, model_name, save_dir)

    return model, history, test_metrics


def plot_causal_training_history(history, appliance_name, model_name, save_dir):
    """Plot training history including causal event weights"""
    plt.figure(figsize=(15, 10))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 Score
    plt.subplot(2, 3, 2)
    train_f1 = [m['f1'] for m in history['train_metrics']]
    val_f1 = [m['f1'] for m in history['val_metrics']]
    plt.plot(train_f1, label='Train F1', color='blue')
    plt.plot(val_f1, label='Val F1', color='red')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # MAE
    plt.subplot(2, 3, 3)
    train_mae = [m['mae'] for m in history['train_metrics']]
    val_mae = [m['mae'] for m in history['val_metrics']]
    plt.plot(train_mae, label='Train MAE', color='blue')
    plt.plot(val_mae, label='Val MAE', color='red')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Causal Event Weights
    plt.subplot(2, 3, 4)
    plt.plot(history['avg_event_weights'], label='Avg Event Weight', color='green')
    plt.title(f'Causal Event Weights - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Event Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision & Recall
    plt.subplot(2, 3, 5)
    train_precision = [m['precision'] for m in history['train_metrics']]
    val_precision = [m['precision'] for m in history['val_metrics']]
    train_recall = [m['recall'] for m in history['train_metrics']]
    val_recall = [m['recall'] for m in history['val_metrics']]
    plt.plot(train_precision, label='Train Precision', color='blue', linestyle='--')
    plt.plot(val_precision, label='Val Precision', color='red', linestyle='--')
    plt.plot(train_recall, label='Train Recall', color='blue', linestyle=':')
    plt.plot(val_recall, label='Val Recall', color='red', linestyle=':')
    plt.title(f'Precision & Recall - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SAE
    plt.subplot(2, 3, 6)
    train_sae = [m['sae'] for m in history['train_metrics']]
    val_sae = [m['sae'] for m in history['val_metrics']]
    plt.plot(train_sae, label='Train SAE', color='blue')
    plt.plot(val_sae, label='Val SAE', color='red')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('SAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_training.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Training Causal Liquid Neural Network on REDD Dataset")
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

    # Load data
    data_dict = load_redd_specific_splits()

    # Configuration
    appliance = 'dish washer'  # Change this to test different appliances
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/causal_liquid_test_{timestamp}"

    print(f"\nTraining on appliance: {appliance}")
    print(f"Save directory: {save_dir}")

    # Train causal liquid network
    model, history, test_metrics = train_causal_liquidnn(
        data_dict,
        appliance_name=appliance,
        window_size=100,
        hidden_size=128,
        num_layers=2,
        dt=0.1,
        advanced=True,
        epochs=20,
        lr=0.001,
        patience=10,
        use_causal_loss=True,
        event_weight_scale=2.0,
        seed=42,
        save_dir=save_dir
    )

    print(f"\n✅ Training completed!")
    print(f"\nFinal Test Metrics for {appliance}:")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  SAE: {test_metrics['sae']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
