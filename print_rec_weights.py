import torch
import torch.nn as nn

# Import the LiquidTimeLayer from models.py
import sys
sys.path.append('Source Code')
from models import LiquidTimeLayer

# Create a small example with hidden_size=4
hidden_size = 4
dt = 0.1

# Create the layer
layer = LiquidTimeLayer(input_size=1, hidden_size=hidden_size, dt=dt)

# Print the recurrent weights matrix
print("Recurrent Weights Matrix (rec_weights):")
print("=" * 50)
print(layer.rec_weights.data)
print("\nShape:", layer.rec_weights.shape)
print("Data type:", layer.rec_weights.dtype)
