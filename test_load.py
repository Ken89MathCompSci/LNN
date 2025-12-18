import sys
sys.path.append('.')
from Source_Code.data_loader import load_and_preprocess_ukdale

print("Testing load...")
try:
    data = load_and_preprocess_ukdale('preprocessed_datasets/ukdale/ukdale1.mat', 0, window_size=100, target_size=1)
    print("Loaded successfully!")
    print(f"Train loader: {len(data['train_loader'])} batches")
except Exception as e:
    print(f"Error: {e}")