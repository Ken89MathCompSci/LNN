import scipy.io
import numpy as np
import os

# Load the data
folder_path = "Source Code/preprocessed_datasets/ukdale"
file_path = os.path.join(folder_path, "ukdale2.mat")

print(f"Loading file: {file_path}")
data = scipy.io.loadmat(file_path)

print("Keys in the .mat file:")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  {key}: {type(data[key])}")
        if hasattr(data[key], 'shape'):
            print(f"    Shape: {data[key].shape}")
        elif isinstance(data[key], np.ndarray):
            print(f"    Shape: {data[key].shape}")
        else:
            print(f"    Value: {data[key]}")

# Check input and output specifically
if 'input' in data:
    print(f"\n'input' details:")
    print(f"  Type: {type(data['input'])}")
    print(f"  Shape: {data['input'].shape}")
    print(f"  Dtype: {data['input'].dtype}")
    if data['input'].shape[0] > 0:
        print(f"  First row sample: {data['input'][0]}")

if 'output' in data:
    print(f"\n'output' details:")
    print(f"  Type: {type(data['output'])}")
    print(f"  Shape: {data['output'].shape}")
    print(f"  Dtype: {data['output'].dtype}")
    if len(data['output'].shape) > 1 and data['output'].shape[0] > 0:
        print(f"  First row sample: {data['output'][0]}")
    elif len(data['output'].shape) == 1 and data['output'].shape[0] > 0:
        print(f"  First 5 values: {data['output'][:5]}")

if 'labelOut' in data:
    print(f"\n'labelOut' details:")
    print(f"  Type: {type(data['labelOut'])}")
    print(f"  Shape: {data['labelOut'].shape if hasattr(data['labelOut'], 'shape') else 'No shape'}")
    print(f"  Content: {data['labelOut']}")
