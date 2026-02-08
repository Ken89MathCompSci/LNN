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

# Add Source Code to path
sys.path.append('Source Code')

# Import from our modules
from models import LiquidNetworkModel, AdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


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

def train_liquidnn_on_specific_redd_appliance(data_dict, appliance_name, window_size=100,
                                            hidden_size=128, num_layers=2, dt=0.1, advanced=True,
                                            epochs=50, lr=0.001, patience=10, save_dir='models/liquidnn_redd_specific'):
    """
    Train Liquid Neural Network model on a specific REDD appliance with the exact splits you specified
    
    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size for liquid network
        num_layers: Number of liquid layers (for advanced model)
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced liquid network model
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
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
    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]
    
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
    
    # Create Liquid Neural Network model
    input_size = 1  # Single feature (aggregate power)
    output_size = 1  # Single target (appliance power)
    
    if advanced:
        model = AdvancedLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dt=dt
        )
        model_name = "advanced_liquid"
    else:
        model = LiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt
        )
        model_name = "liquid"
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_path = None
    
    print(f"Starting {model_name} training for {appliance_name}...")
    
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
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
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
        # Calculate NILM metrics
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
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
            best_model_path = os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_best.pth")
            
            # Save model with metadata
            model_params = {
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers if advanced else 1,
                'dt': dt,
                'advanced': advanced
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
    
    # Calculate test metrics
    all_test_targets = np.concatenate(all_test_targets)
    all_test_outputs = np.concatenate(all_test_outputs)
    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)
    
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    val_mae = [metrics['mae'] for metrics in history['val_metrics']]
    plt.plot(val_mae, label='Validation MAE')
    plt.title(f'Validation MAE - {appliance_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_training_history.png"))
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
    plt.suptitle(f'{model_name.title()} Predictions vs Actual - {appliance_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_redd_{appliance_name.replace(' ', '_')}_predictions.png"))
    plt.close()
    
    # Save training history to JSON
    config = {
        'appliance': appliance_name,
        'window_size': window_size,
        'model_params': {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers if advanced else 1,
            'dt': dt,
            'advanced': advanced
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
            'test_metrics': {k: float(v) for k, v in test_metrics.items()}
        }
    }
    
    with open(os.path.join(save_dir, f'{model_name}_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    return model, history, test_metrics

def test_liquidnn_on_all_redd_appliances(window_size=100, hidden_size=128, num_layers=2, dt=0.1,
                                       advanced=True, epochs=50, lr=0.001, patience=10):
    """
    Test Liquid Neural Network model on all specified REDD appliances with the exact splits you mentioned
    
    Args:
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size for liquid network
        num_layers: Number of liquid layers (for advanced model)
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced liquid network model
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
    
    Returns:
        Dictionary of results for all appliances
    """
    # Load data with specific splits
    print("Loading REDD data with specific splits...")
    data_dict = load_redd_specific_splits()
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "advanced_liquid" if advanced else "liquid"
    base_save_dir = f"models/{model_type}_redd_specific_test_{timestamp}"
    
    # Dictionary to store results
    all_results = {}
    
    # Test Liquid Neural Network model on each specified appliance
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']
    
    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing {model_type} on {appliance_name}")
        print(f"{'='*60}\n")
        
        # Create appliance-specific save directory
        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)
        
        try:
            # Train and evaluate Liquid Neural Network model
            model, history, test_metrics = train_liquidnn_on_specific_redd_appliance(
                data_dict,
                appliance_name=appliance_name,
                window_size=window_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dt=dt,
                advanced=advanced,
                epochs=epochs,
                lr=lr,
                patience=patience,
                save_dir=appliance_dir
            )
            
            if model is not None:
                # Store results
                all_results[appliance_name] = {
                    'model_path': os.path.join(appliance_dir, f"{model_type}_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"✅ Successfully tested {model_type} on {appliance_name}")
            else:
                print(f"❌ Failed to test {model_type} on {appliance_name}")
        
        except Exception as e:
            print(f"Error testing {model_type} on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary of results
    summary = {
        'timestamp': timestamp,
        'model_type': model_type,
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
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dt': dt,
            'advanced': advanced
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
    
    print(f"\n🎉 {model_type} testing completed on all REDD appliances with specific splits!")
    print(f"Results saved to {base_save_dir}")
    
    return all_results

if __name__ == "__main__":
    print("Testing Liquid Neural Network algorithm on REDD dataset with specific splits...")
    
    # Check if the required pickle files exist
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
    
    # Run comprehensive test on all specified appliances with standard Liquid Neural Network
    print("Running comprehensive standard Liquid Neural Network test on all REDD appliances with specific splits...")
    results = test_liquidnn_on_all_redd_appliances(
        window_size=100,
        hidden_size=128,
        num_layers=2,
        dt=0.1,
        advanced=False,
        epochs=50,
        lr=0.001,
        patience=10
    )
    
    # Print summary for standard Liquid Neural Network
    print(f"\n📊 Summary of standard Liquid Neural Network testing on REDD dataset with specific splits:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")
    
    # Run comprehensive test on all specified appliances with advanced Liquid Neural Network
    print("\nRunning comprehensive advanced Liquid Neural Network test on all REDD appliances with specific splits...")
    results = test_liquidnn_on_all_redd_appliances(
        window_size=100,
        hidden_size=128,
        num_layers=2,
        dt=0.1,
        advanced=True,
        epochs=50,
        lr=0.001,
        patience=10
    )
    
    # Print summary for advanced Liquid Neural Network
    print(f"\n📊 Summary of advanced Liquid Neural Network testing on REDD dataset with specific splits:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")