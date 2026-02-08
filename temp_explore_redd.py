import pickle
import pandas as pd

file_path = "data/redd/test_small.pkl"

try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Successfully loaded {file_path}")
    print(f"Type of data: {type(data)}")
    
    # Attempt to print a portion of the data, depending on its type
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        for key, value in data.items():
            print(f"--- Key: {key} ---")
            print(f"Type of value: {type(value)}")
            if hasattr(value, '__len__'):
                print(f"Length of value: {len(value)}")
            if isinstance(value, dict):
                print(f"Sub-keys: {value.keys()}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"First element type: {type(value[0])}")
                print(f"First element: {value[0]}")
            elif isinstance(value, pd.DataFrame):
                print(f"DataFrame head:\n{value.head()}\n")
            elif hasattr(value, 'shape'): # For numpy arrays
                print(f"Shape of value: {value.shape}")
                print(f"First few elements: \n{value if value.size < 10 else value.flatten()[:5]}")
            else:
                print(f"Value: {value}")
            print("\n")
    elif isinstance(data, list):
        print(f"Length of list: {len(data)}")
        if len(data) > 0:
            print(f"Type of first element: {type(data[0])}")
            print(f"First element: {data[0]}")
            if hasattr(data[0], 'shape'):
                print(f"Shape of first element: {data[0].shape}")
            if isinstance(data[0], pd.DataFrame):
                print(f"DataFrame head:\n{data[0].head()}\n")
    elif isinstance(data, pd.DataFrame):
        print(f"DataFrame head:\n{data.head()}\n")
    elif hasattr(data, 'shape'): # For numpy arrays
        print(f"Shape of data: {data.shape}")
        print(f"First few elements: \n{data if data.size < 10 else data.flatten()[:5]}")
    else:
        print(f"Data: {data}")

except Exception as e:
    print(f"Error loading or inspecting the pkl file: {e}")