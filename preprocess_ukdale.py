import tables
import numpy as np
from scipy.io import savemat
import os

def load_meter_data(h5file, meter_id, max_rows=None):
    meter = getattr(h5file.root.building1.elec, f'meter{meter_id}')
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

print("Preprocessing UK-DALE to create ukdale1.mat...")

max_rows = 200000  # Limit for speed, set None for full

with tables.open_file('Source Code/ukdale/ukdale.h5', 'r') as h5file:
    # Load aggregate (meter1)
    print("Loading aggregate (meter1)...")
    idx_agg, val_agg = load_meter_data(h5file, 1, max_rows)
    n_samples = len(idx_agg)
    print(f"Aggregate samples: {n_samples}")

    # Load the correct appliances for UK-DALE house 1
    # Based on training logs: 0=Fridge, 1=Dishwasher, 2=Microwave, 3=Washer_Dryer, 4=Kettle
    # These typically correspond to meter IDs: Fridge(2), Dishwasher(3), Microwave(4), Washer_Dryer(5), Kettle(6)
    appliance_meter_mapping = {
        2: "Fridge",
        3: "Dishwasher",
        4: "Microwave",
        5: "Washer_Dryer",
        6: "Kettle"
    }

    appliance_data = []
    appliance_names = []
    for meter_id, appliance_name in appliance_meter_mapping.items():
        print(f"Loading meter {meter_id} ({appliance_name})...")
        try:
            idx_app, val_app = load_meter_data(h5file, meter_id, max_rows)
            if len(idx_app) == n_samples:
                appliance_data.append(val_app)
                appliance_names.append(appliance_name)
                print(f"  Successfully loaded {appliance_name} with {len(val_app)} samples")
            else:
                print(f"  Skipping {appliance_name} - sample count mismatch ({len(idx_app)} vs {n_samples})")
        except Exception as e:
            print(f"  Error loading meter {meter_id} ({appliance_name}): {e}")

    if appliance_data:
        # Create input: [time, id (0 for agg), power]
        input_data = np.column_stack((idx_agg.astype(float), np.zeros(n_samples), val_agg))

        # Create output: [time, id (0), app1, app2, ...]
        output_data = np.column_stack((idx_agg.astype(float), np.zeros(n_samples), *appliance_data))

        # Save to mat
        os.makedirs('Source Code/preprocessed_datasets/ukdale', exist_ok=True)
        savemat('Source Code/preprocessed_datasets/ukdale/ukdale1.mat', {
            'input': input_data,
            'output': output_data,
            'labelOut': np.array(['time', 'id'] + appliance_names, dtype=object)  # Row vector
        })
        print("Saved ukdale1.mat successfully!")
        print(f"Input shape: {input_data.shape}, Output shape: {output_data.shape}")
    else:
        print("No appliance data loaded")
