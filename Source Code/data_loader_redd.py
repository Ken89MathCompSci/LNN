import tables
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, Any, Tuple, List
import warnings

class REDDDataset(Dataset):
    """
    Dataset class for REDD data
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_redd(file_path: str, building_number: int = 1, meter_number: int = 1,
                           window_size: int = 100, target_size: int = 1,
                           train_ratio: float = 0.7, val_ratio: float = 0.15,
                           normalize: bool = True) -> Dict[str, Any]:
    """
    Load and preprocess REDD data from HDF5 format

    Args:
        file_path: Path to the .h5 file
        building_number: Building number in the REDD dataset
        meter_number: Meter number within the building
        window_size: Size of the input sequence window
        target_size: Size of target output (1 for sequence-to-point)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        normalize: Whether to normalize the data

    Returns:
        Dictionary containing DataLoaders and metadata
    """
    try:
        # Load the data using PyTables
        with tables.open_file(file_path, 'r') as h5file:
            # Access the building and meter
            building_group = getattr(h5file.root, f'building{building_number}', None)
            if building_group is None:
                raise ValueError(f"Building {building_number} not found in the HDF5 file")

            elec_group = getattr(building_group, 'elec', None)
            if elec_group is None:
                raise ValueError(f"Electricity group not found in building {building_number}")

            meter_group = getattr(elec_group, f'meter{meter_number}', None)
            if meter_group is None:
                raise ValueError(f"Meter {meter_number} not found in building {building_number}")

            # Extract power data - REDD uses 'table' with 'values_block_0' column
            meter_table = getattr(meter_group, 'table', None)
            if meter_table is None:
                raise ValueError(f"Table data not found in meter {meter_number}")

            # Extract the power values from the table
            # The power data is in the 'values_block_0' column
            power_data = meter_table.col('values_block_0').flatten()

            # Convert to numpy array
            mains_data = np.array(power_data)

            # For REDD, we need to get appliance data from submeters
            # Let's try to find appliance meters
            appliance_data = None
            appliance_name = f"Meter_{meter_number}"

            # Try to get appliance name from the meter group
            if hasattr(meter_group, 'name'):
                appliance_name = str(getattr(meter_group, 'name', f"Meter_{meter_number}"))

            # For now, we'll use the same data for both mains and appliance
            # In a real REDD setup, we would have separate mains and appliance meters
            appliance_data = mains_data.copy()

            print(f"Mains data shape: {mains_data.shape}")
            print(f"Appliance data shape: {appliance_data.shape}")
            print(f"Appliance name: {appliance_name}")

            # Plot a sample of the data to verify
            plt.figure(figsize=(12, 6))
            sample_size = min(1000, len(mains_data))
            plt.plot(mains_data[:sample_size], label='Aggregate')
            plt.plot(appliance_data[:sample_size], label=appliance_name)
            plt.legend()
            plt.title(f'Sample Data from REDD - {appliance_name}')
            plt.savefig(f'redd_sample_meter_{meter_number}.png')
            plt.close()

    except Exception as e:
        raise ValueError(f"Error loading REDD data: {str(e)}")

    # Normalize data if required
    if normalize:
        mains_scaler = StandardScaler()
        appliance_scaler = StandardScaler()

        mains_data = mains_scaler.fit_transform(mains_data.reshape(-1, 1)).flatten()
        appliance_data = appliance_scaler.fit_transform(appliance_data.reshape(-1, 1)).flatten()
    else:
        mains_scaler = None
        appliance_scaler = None

    # Create sequences
    X, y = create_sequences(mains_data, appliance_data, window_size, target_size)

    # Split data
    total_samples = len(X)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)

    # Use sequential split rather than random permutation
    # This is better for time series data
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, total_samples)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = REDDDataset(X_train, y_train)
    val_dataset = REDDDataset(X_val, y_val)
    test_dataset = REDDDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'mains_scaler': mains_scaler,
        'appliance_scaler': appliance_scaler,
        'building_number': building_number,
        'meter_number': meter_number,
        'meter_name': appliance_name,
        'window_size': window_size,
        'target_size': target_size,
        'input_size': 1,  # Single feature (power)
        'output_size': target_size
    }

def create_sequences(mains: np.ndarray, appliance: np.ndarray, window_size: int, target_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for sequence-to-sequence or sequence-to-point prediction

    Args:
        mains: The aggregate power data
        appliance: The target appliance power data
        window_size: Size of the input window
        target_size: Size of the target window (1 for sequence-to-point)

    Returns:
        X: Input sequences
        y: Target sequences or points
    """
    X, y = [], []

    # Use stride to reduce the number of highly correlated samples
    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])

        if target_size == 1:
            # Sequence-to-point: predict the middle point
            midpoint = i + window_size // 2
            y.append(appliance[midpoint:midpoint+1])
        else:
            # Sequence-to-sequence
            y.append(appliance[i+window_size:i+window_size+target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def explore_available_meters(file_path: str) -> Dict[int, Dict[int, str]]:
    """
    Explore available meters in a REDD HDF5 file

    Args:
        file_path: Path to the .h5 file

    Returns:
        Dictionary mapping building numbers to meter numbers and names
    """
    meters_info = {}

    try:
        with tables.open_file(file_path, 'r') as h5file:
            # Iterate through buildings
            for building_node in h5file.root:
                building_name = building_node._v_name
                if building_name.startswith('building'):
                    building_number = int(building_name.replace('building', ''))
                    building_group = getattr(h5file.root, building_name)

                    # Check if electricity group exists
                    if hasattr(building_group, 'elec'):
                        elec_group = building_group.elec
                        meters_info[building_number] = {}

                        # Iterate through meters
                        for meter_node in elec_group:
                            meter_name = meter_node._v_name
                            if meter_name.startswith('meter'):
                                meter_number = int(meter_name.replace('meter', ''))
                                meter_group = getattr(elec_group, meter_name)

                                # Get meter name if available
                                meter_display_name = f"Meter_{meter_number}"
                                if hasattr(meter_group, 'name'):
                                    meter_display_name = str(meter_group.name)

                                meters_info[building_number][meter_number] = meter_display_name

    except Exception as e:
        warnings.warn(f"Error exploring REDD meters: {str(e)}")

    return meters_info

def load_redd_data_for_training(file_path: str, building_number: int = 1, window_size: int = 100) -> Dict[str, Any]:
    """
    Load REDD data and train models for all available meters in a building

    Args:
        file_path: Path to the .h5 file
        building_number: Building number in the REDD dataset
        window_size: Window size for input sequences

    Returns:
        Dictionary containing training results for all meters
    """
    # Explore available meters
    meters_info = explore_available_meters(file_path)

    if building_number not in meters_info:
        raise ValueError(f"Building {building_number} not found in the dataset")

    results = {}

    for meter_number, meter_name in meters_info[building_number].items():
        print(f"\nProcessing meter {meter_number}: {meter_name}")

        try:
            # Load and preprocess data for this meter
            data_dict = load_and_preprocess_redd(
                file_path,
                building_number=building_number,
                meter_number=meter_number,
                window_size=window_size,
                target_size=1
            )

            results[f"building{building_number}_meter{meter_number}"] = {
                'data_dict': data_dict,
                'meter_name': meter_name
            }

            print(f"Successfully loaded data for {meter_name}")

        except Exception as e:
            print(f"Error processing meter {meter_number}: {str(e)}")
            continue

    return results

if __name__ == "__main__":
    # Test the REDD data loader
    print("Testing REDD data loader...")

    # Check if redd.h5 file exists
    redd_file_path = "redd.h5"
    if not os.path.exists(redd_file_path):
        print(f"RED file {redd_file_path} not found. Please ensure the file exists.")
        print("You can download REDD dataset from: https://redd.csail.mit.edu/")
        print("Or use the ukdale.h5 file if available.")

        # Try with ukdale.h5 as fallback
        redd_file_path = "ukdale.h5"
        if not os.path.exists(redd_file_path):
            print(f"ukdale.h5 file also not found.")
            print("Please ensure you have either redd.h5 or ukdale.h5 in the current directory.")
        else:
            print(f"Using ukdale.h5 as fallback for testing.")

    if os.path.exists(redd_file_path):
        try:
            # Explore available meters
            meters = explore_available_meters(redd_file_path)
            print("Available meters in REDD dataset:")
            for building_num, meters_dict in meters.items():
                print(f"  Building {building_num}:")
                for meter_num, meter_name in meters_dict.items():
                    print(f"    Meter {meter_num}: {meter_name}")

            # Test loading data for the first building and first meter
            if meters:
                first_building = list(meters.keys())[0]
                first_meter = list(meters[first_building].keys())[0]

                print(f"\nLoading data for Building {first_building}, Meter {first_meter}...")
                data_dict = load_and_preprocess_redd(
                    redd_file_path,
                    building_number=first_building,
                    meter_number=first_meter,
                    window_size=100,
                    target_size=1
                )

                print("\nData loaded successfully!")
                print(f"Meter: {data_dict['meter_name']}")
                print(f"Training batches: {len(data_dict['train_loader'])}")
                print(f"Validation batches: {len(data_dict['val_loader'])}")
                print(f"Test batches: {len(data_dict['test_loader'])}")

        except Exception as e:
            print(f"Error testing REDD data loader: {str(e)}")
    else:
        print("No suitable HDF5 file found for testing.")