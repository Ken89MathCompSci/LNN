import tables
import numpy as np
from scipy.io import savemat
import os

def load_meter_data(h5file, building_name, meter_id, max_rows=None):
    """Load data from a specific meter in a specific building"""
    try:
        meter = getattr(h5file.root, building_name).elec[f'meter{meter_id}']
        table = meter.table
        nrows = table.nrows
        if max_rows is not None and max_rows < nrows:
            nrows = max_rows
        data = table[:nrows]
        # Convert to numpy
        index = data['index']
        values = data['values_block_0']
        # Flatten if needed
        if values.ndim > 1:
            values = values.flatten()
        return index, values
    except Exception as e:
        print(f"  Error accessing {building_name}/meter{meter_id}: {e}")
        return None, None

print("Preprocessing UK-DALE House 5 to create ukdale5.mat...")

building_name = 'building5'
max_rows = 200000  # Limit for speed, set None for full

with tables.open_file('Source Code/ukdale/ukdale.h5', 'r') as h5file:
    # Check if building exists
    if not hasattr(h5file.root, building_name):
        print(f"Error: {building_name} not found in the HDF5 file")
        exit(1)

    print(f"Processing {building_name} (House 5)")

    # Load aggregate (meter1) - should be available in all houses
    print("Loading aggregate (meter1)...")
    idx_agg, val_agg = load_meter_data(h5file, building_name, 1, max_rows)
    if idx_agg is None:
        print("Error: Could not load aggregate meter")
        exit(1)

    n_samples = len(idx_agg)
    print(f"Aggregate samples: {n_samples}")

    # UK-DALE House 5 is well-instrumented with many appliances
    # Common appliances in House 5: Fridge, Freezer, Microwave, Washer Dryer, Dishwasher, Television, etc.
    # We'll try to find appliances that match our target set or similar ones
    appliance_meter_mapping = {
        2: "Fridge",           # Usually available
        3: "Freezer",          # House 5 has separate freezer
        4: "Microwave",        # Usually available
        5: "Washer_Dryer",     # Usually available
        6: "Dishwasher",       # Usually available
        7: "Television",       # Usually available
        8: "Kettle",           # May be available
        # House 5 has many more appliances, but we'll focus on common ones
    }

    appliance_data = []
    appliance_names = []

    for meter_id, appliance_name in appliance_meter_mapping.items():
        print(f"Loading meter {meter_id} ({appliance_name})...")
        try:
            idx_app, val_app = load_meter_data(h5file, building_name, meter_id, max_rows)
            if idx_app is not None and len(idx_app) == n_samples:
                appliance_data.append(val_app)
                appliance_names.append(appliance_name)
                print(f"  Successfully loaded {appliance_name} with {len(val_app)} samples")
            else:
                print(f"  Skipping {appliance_name} - sample count mismatch or not available")
        except Exception as e:
            print(f"  Error loading meter {meter_id} ({appliance_name}): {e}")

    if len(appliance_data) < 3:
        print(f"Warning: Only found {len(appliance_data)} appliances. This may not be enough for training.")

    if appliance_data:
        # Create input: [time, id (0 for agg), power]
        input_data = np.column_stack((idx_agg.astype(float), np.zeros(n_samples), val_agg))

        # Create output: [time, id (0), app1, app2, ...]
        output_data = np.column_stack((idx_agg.astype(float), np.zeros(n_samples), *appliance_data))

        # Save to mat
        os.makedirs('Source Code/preprocessed_datasets/ukdale', exist_ok=True)
        savemat('Source Code/preprocessed_datasets/ukdale/ukdale5.mat', {
            'input': input_data,
            'output': output_data,
            'labelOut': np.array(['time', 'id'] + appliance_names, dtype=object)
        })
        print("Saved ukdale5.mat successfully!")
        print(f"Input shape: {input_data.shape}, Output shape: {output_data.shape}")
        print(f"Available appliances: {appliance_names}")

        # House 5 specific info
        print("\nHouse 5 Notes:")
        print("- House 5 is well-instrumented with many appliances")
        print("- Often used as a benchmark in NILM research")
        print("- May have separate fridge and freezer meters")
    else:
        print("No appliance data loaded for House 5")
