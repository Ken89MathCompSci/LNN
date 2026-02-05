import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

# Add Source Code to path
sys.path.append('Source Code')

# Import from our modules
from data_loader_redd import explore_available_meters, load_and_preprocess_redd
from models import GRUModel
from utils import calculate_nilm_metrics, save_model

def train_gru_on_redd(building_number=1, meter_number=1, window_size=100,
                     hidden_size=128, num_layers=2, epochs=20, lr=0.001,
                     patience=5, save_dir='models/gru_redd'):
    """
    Train GRU model on REDD dataset for a specific building and meter

    Args:
        building_number: Building number in REDD dataset
        meter_number: Meter number within the building
        window_size: Size of input sequence window
        hidden_size: Size of GRU hidden state
        num_layers: Number of GRU layers
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

    # Load and preprocess REDD data
    print(f"Loading REDD data for Building {building_number}, Meter {meter_number}...")
    try:
        data_dict = load_and_preprocess_redd(
            "redd.h5",
            building_number=building_number,
            meter_number=meter_number,
            window_size=window_size,
            target_size=1
        )
    except Exception as e:
        print(f"Error loading REDD data: {str(e)}")
        return None, None, None

    print(f"✅ Data loaded successfully!")
    print(f"Meter: {data_dict['meter_name']}")
    print(f"Training batches: {len(data_dict['train_loader'])}")
    print(f"Validation batches: {len(data_dict['val_loader'])}")
    print(f"Test batches: {len(data_dict['test_loader'])}")

    # Extract data loaders
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']

    # Create GRU model
    input_size = data_dict['input_size']
    output_size = data_dict['output_size']

    model = GRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        bidirectional=True
    )

    model = model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    print(f"Starting GRU training on REDD data...")

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
        metrics = calculate_nilm_metrics(all_targets, all_outputs, scaler=data_dict.get('appliance_scaler'))
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
            best_model_path = os.path.join(save_dir, f"gru_redd_best.pth")

            # Save model with metadata
            model_params = {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'output_size': output_size,
                'bidirectional': True
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience,
                'window_size': window_size,
                'building_number': building_number,
                'meter_number': meter_number
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
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, scaler=data_dict.get('appliance_scaler'))

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    val_mae = [metrics['mae'] for metrics in history['val_metrics']]
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gru_redd_training_history.png"))
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
    plt.suptitle(f'GRU Predictions vs Actual - {data_dict["meter_name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gru_redd_predictions.png"))
    plt.close()

    # Save training history to JSON
    config = {
        'building_number': building_number,
        'meter_number': meter_number,
        'meter_name': data_dict['meter_name'],
        'window_size': window_size,
        'model_params': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'bidirectional': True
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

    with open(os.path.join(save_dir, 'gru_redd_history.json'), 'w') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics

def test_gru_on_all_redd_meters(window_size=100, hidden_size=128, num_layers=2,
                               epochs=10, lr=0.001, patience=3):
    """
    Test GRU model on all available meters in the REDD dataset

    Args:
        window_size: Size of input sequence window
        hidden_size: Size of GRU hidden state
        num_layers: Number of GRU layers
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Dictionary of results for all meters
    """
    # Explore available meters
    print("Exploring available meters in REDD dataset...")
    meters_info = explore_available_meters("redd.h5")

    if not meters_info:
        print("No meters found in REDD dataset.")
        return {}

    print("Available meters in REDD dataset:")
    for building_num, meters_dict in meters_info.items():
        print(f"  Building {building_num}:")
        for meter_num, meter_name in meters_dict.items():
            print(f"    Meter {meter_num}: {meter_name}")

    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/gru_redd_test_{timestamp}"

    # Dictionary to store results
    all_results = {}

    # Test GRU model on each meter
    for building_num, meters_dict in meters_info.items():
        for meter_num, meter_name in meters_dict.items():
            print(f"\n{'='*60}")
            print(f"Testing GRU on Building {building_num}, Meter {meter_num}: {meter_name}")
            print(f"{'='*60}\n")

            # Create building-specific save directory
            building_dir = os.path.join(base_save_dir, f"building{building_num}")
            os.makedirs(building_dir, exist_ok=True)

            # Create meter-specific save directory
            meter_dir = os.path.join(building_dir, f"meter{meter_num}_{meter_name.replace(' ', '_')}")
            os.makedirs(meter_dir, exist_ok=True)

            try:
                # Train and evaluate GRU model
                model, history, test_metrics = train_gru_on_redd(
                    building_number=building_num,
                    meter_number=meter_num,
                    window_size=window_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    epochs=epochs,
                    lr=lr,
                    patience=patience,
                    save_dir=meter_dir
                )

                if model is not None:
                    # Store results
                    all_results[f"building{building_num}_meter{meter_num}"] = {
                        'meter_name': meter_name,
                        'model_path': os.path.join(meter_dir, "gru_redd_best.pth"),
                        'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                    }
                    print(f"✅ Successfully tested GRU on {meter_name}")
                else:
                    print(f"❌ Failed to test GRU on {meter_name}")

            except Exception as e:
                print(f"Error testing GRU on {meter_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Save summary of results
    summary = {
        'timestamp': timestamp,
        'window_size': window_size,
        'model_params': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': True
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

    print(f"\n🎉 GRU testing completed on all REDD meters!")
    print(f"Results saved to {base_save_dir}")

    return all_results

if __name__ == "__main__":
    print("Testing GRU algorithm on REDD dataset...")

    # Check if redd.h5 file exists
    if not os.path.exists("redd.h5"):
        print("❌ Error: redd.h5 file not found!")
        print("Please ensure the REDD dataset file is in the current directory.")
        print("You can download the REDD dataset from: https://redd.csail.mit.edu/")
        sys.exit(1)

    # Run comprehensive test on all available meters
    print("Running comprehensive GRU test on all REDD meters...")
    results = test_gru_on_all_redd_meters(
        window_size=100,
        hidden_size=128,
        num_layers=2,
        epochs=10,
        lr=0.001,
        patience=3
    )

    # Print summary
    print(f"\n📊 Summary of GRU testing on REDD dataset:")
    print(f"Total meters tested: {len(results)}")
    for key, result in results.items():
        print(f"  {key}: {result['meter_name']}")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")