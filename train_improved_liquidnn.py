"""
Improved Liquid Neural Network training with proven NILM techniques
Focus on data augmentation, better normalization, and training improvements
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

from liquidnn import LiquidNetworkModel, AdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


from scipy.interpolate import interp1d
from scipy.stats import truncnorm


class DataAugmentation:
    """
    Advanced data augmentation for NILM
    Uses vertical and horizontal scaling with truncated normal distribution
    """
    def __init__(self, vert_scale=True, horiz_scale=True, noise=True):
        """
        Args:
            vert_scale: Enable vertical (amplitude) scaling
            horiz_scale: Enable horizontal (time) scaling
            noise: Enable Gaussian noise
        """
        self.vert_scale = vert_scale
        self.horiz_scale = horiz_scale
        self.noise = noise

    def vertScale(self, x):
        """
        Vertical scaling using truncated normal distribution
        More realistic than uniform distribution
        """
        mu, sigma = 1, 0.2
        lower, upper = mu - 2*sigma, mu + 2*sigma
        tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        scale = tn.rvs(1)[0]
        return x * scale

    def horiScale(self, x):
        """
        Horizontal scaling using interpolation
        Stretches or compresses signal in time
        """
        mu, sigma = 1, 0.2
        lower, upper = mu - 2*sigma, mu + 2*sigma
        tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        scale = tn.rvs(1)[0]

        # Original length
        orig_length = x.shape[0]

        # Convert to numpy for interpolation
        if torch.is_tensor(x):
            y = x.numpy().reshape(-1)
            is_tensor = True
        else:
            y = x.reshape(-1)
            is_tensor = False

        # Interpolate
        x_orig = np.arange(0, len(y))
        f = interp1d(x_orig, y, kind='linear', fill_value='extrapolate')

        # New time points
        x_new = np.arange(0, len(y) - 1, scale)
        y_new = f(x_new)

        # Crop or pad to original length
        if len(y_new) > orig_length:
            y_new = y_new[:orig_length]
        else:
            diff_size = orig_length - len(y_new)
            y_new = np.pad(y_new, (0, diff_size), 'constant')

        return torch.from_numpy(y_new).float() if is_tensor else y_new

    def add_noise(self, x):
        """Add Gaussian noise"""
        noise_std = 0.05 * x.std()
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def augment(self, x, y=None, probability=0.5):
        """
        Apply augmentation with given probability

        Args:
            x: Input sequence (batch_size, seq_len, input_size) or (seq_len, input_size)
            y: Target (batch_size, 1) or (1,) - optional
            probability: Probability of applying each augmentation

        Returns:
            Augmented (x, y) or x
        """
        # Check if batch or single sequence
        if x.dim() == 3:
            # Batch mode - apply to each sample
            x_aug = x.clone()
            for i in range(x.shape[0]):
                x_aug[i] = self._augment_single(x[i], probability)
            return (x_aug, y) if y is not None else x_aug
        else:
            # Single sequence mode
            x_aug = self._augment_single(x, probability)
            return (x_aug, y) if y is not None else x_aug

    def _augment_single(self, x, probability):
        """Apply augmentation to single sequence"""
        x_aug = x.clone()

        # Choose augmentation mode
        mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])[0]

        if mode == 0:
            # Original - no augmentation
            pass
        elif mode == 1 and self.vert_scale:
            # Vertical scaling only
            x_aug = self.vertScale(x_aug)
        elif mode == 2 and self.horiz_scale:
            # Horizontal scaling only
            # Apply to each feature separately
            for feat_idx in range(x_aug.shape[1]):
                x_aug[:, feat_idx] = self.horiScale(x_aug[:, feat_idx])
        elif mode == 3 and self.vert_scale and self.horiz_scale:
            # Both vertical and horizontal
            for feat_idx in range(x_aug.shape[1]):
                x_aug[:, feat_idx] = self.horiScale(x_aug[:, feat_idx])
            x_aug = self.vertScale(x_aug)

        # Add noise with probability
        if self.noise and np.random.random() < probability:
            x_aug = self.add_noise(x_aug)

        return x_aug


class ImprovedREDDDataset(torch.utils.data.Dataset):
    """
    Improved dataset with normalization and augmentation
    """
    def __init__(self, X, y, augment=False, augmentation=None, normalize=True,
                 mean=None, std=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.augmentation = augmentation
        self.normalize = normalize

        # Compute or use provided normalization stats
        if normalize:
            if mean is None or std is None:
                self.mean = self.X.mean()
                self.std = self.X.std()
            else:
                self.mean = mean
                self.std = std

            # Normalize input
            self.X = (self.X - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0
            self.std = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        # Apply augmentation during training
        if self.augment and self.augmentation is not None:
            x, y = self.augmentation.augment(x, y, probability=0.5)

        return x, y

    def get_stats(self):
        """Return normalization statistics"""
        return {'mean': self.mean, 'std': self.std}


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


def train_improved_liquidnn(data_dict, appliance_name, window_size=100,
                           hidden_size=128, num_layers=2, dt=0.1, advanced=True,
                           epochs=50, lr=0.001, patience=15,
                           use_augmentation=True, use_lr_scheduler=True,
                           gradient_clip=1.0, seed=42,
                           save_dir='models/improved_liquidnn'):
    """
    Train improved Liquid Neural Network with data augmentation and better training

    Args:
        data_dict: Dictionary containing train, val, test data
        appliance_name: Name of the appliance to train on
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size (increased from 128)
        num_layers: Number of liquid layers
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced model
        epochs: Number of training epochs (increased from 20)
        lr: Initial learning rate
        patience: Early stopping patience (increased from 10)
        use_augmentation: Whether to use data augmentation
        use_lr_scheduler: Whether to use learning rate scheduling
        gradient_clip: Gradient clipping value (None to disable)
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
    X_train, y_train_agg = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val, y_val_agg = create_sequences(val_data, window_size=window_size, target_size=1)
    X_test, y_test_agg = create_sequences(test_data, window_size=window_size, target_size=1)

    # Use appliance power as target
    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    print(f"Training sequences: {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences: {X_test.shape} -> {y_test.shape}")

    # Initialize data augmentation (advanced approach with truncated normal)
    augmentation = DataAugmentation(
        vert_scale=True,   # Vertical (amplitude) scaling
        horiz_scale=True,  # Horizontal (time) scaling
        noise=True         # Gaussian noise
    ) if use_augmentation else None

    # Create datasets with normalization
    train_dataset = ImprovedREDDDataset(
        X_train, y_train,
        augment=use_augmentation,
        augmentation=augmentation,
        normalize=True
    )

    # Use training stats for validation and test
    train_stats = train_dataset.get_stats()
    val_dataset = ImprovedREDDDataset(
        X_val, y_val,
        augment=False,
        augmentation=None,
        normalize=True,
        mean=train_stats['mean'],
        std=train_stats['std']
    )
    test_dataset = ImprovedREDDDataset(
        X_test, y_test,
        augment=False,
        augmentation=None,
        normalize=True,
        mean=train_stats['mean'],
        std=train_stats['std']
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    # Initialize model
    input_size = 1
    output_size = 1

    if advanced:
        model = AdvancedLiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dt=dt
        ).to(device)
        model_name = "advanced_improved_liquid"
    else:
        model = LiquidNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt
        ).to(device)
        model_name = "improved_liquid"

    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'learning_rates': [],
        'best': {}
    }

    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = None
    counter = 0

    print(f"Starting {model_name} training for {appliance_name}...")
    print(f"Augmentation: {use_augmentation}, LR Scheduler: {use_lr_scheduler}, Gradient Clip: {gradient_clip}")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            train_targets.append(targets.detach().cpu().numpy())
            train_outputs.append(outputs.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

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
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
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

        # Store learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Val F1: {metrics['f1']:.4f}, LR: {current_lr:.6f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)

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
                'advanced': advanced
            }

            train_params = {
                'lr': lr,
                'epochs': epochs,
                'patience': patience,
                'use_augmentation': use_augmentation,
                'use_lr_scheduler': use_lr_scheduler,
                'gradient_clip': gradient_clip,
                'seed': seed,
                'normalization_stats': train_stats
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

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # Calculate test metrics
    all_test_targets = np.concatenate(all_test_targets)
    all_test_outputs = np.concatenate(all_test_outputs)
    test_threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=test_threshold)

    print(f"\nTest Results for {appliance_name}:")
    print(f"  Test Loss: {avg_test_loss:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test SAE: {test_metrics['sae']:.4f}")

    # Store best epoch info
    history['best'] = {
        'epoch': best_epoch,
        'val_loss': best_val_loss,
        'val_metrics': best_metrics
    }

    # Save training history and final results
    config = {
        'model_params': {
            'window_size': window_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers if advanced else 1,
            'dt': dt,
            'advanced': advanced
        },
        'train_params': {
            'lr': lr,
            'epochs': epochs,
            'patience': patience,
            'use_augmentation': use_augmentation,
            'use_lr_scheduler': use_lr_scheduler,
            'gradient_clip': gradient_clip,
            'seed': seed,
            'normalization_stats': {
                'mean': float(train_stats['mean']),
                'std': float(train_stats['std'])
            }
        },
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_loss': avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best': history['best']
        }
    }

    with open(os.path.join(save_dir, f'{model_name}_redd_{appliance_name.replace(" ", "_")}_history.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1 Score plot
    train_f1s = [m['f1'] for m in history['train_metrics']]
    val_f1s = [m['f1'] for m in history['val_metrics']]
    axes[0, 1].plot(train_f1s, label='Train F1')
    axes[0, 1].plot(val_f1s, label='Val F1')
    axes[0, 1].axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate plot
    if use_lr_scheduler:
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

    # MAE plot
    train_maes = [m['mae'] for m in history['train_metrics']]
    val_maes = [m['mae'] for m in history['val_metrics']]
    axes[1, 1].plot(train_maes, label='Train MAE')
    axes[1, 1].plot(val_maes, label='Val MAE')
    axes[1, 1].axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Mean Absolute Error Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_redd_{appliance_name.replace(" ", "_")}_training.png'))
    plt.close()

    return model, history, test_metrics


if __name__ == "__main__":
    # Load data
    data_dict = load_redd_specific_splits()

    # Train on dish washer
    appliance = 'dish washer'

    print(f"\n{'='*70}")
    print(f"Training Improved Liquid Neural Network on {appliance}")
    print(f"{'='*70}\n")

    model, history, test_metrics = train_improved_liquidnn(
        data_dict,
        appliance_name=appliance,
        window_size=100,
        hidden_size=256,  # Increased from 128
        num_layers=3,     # Increased from 2
        dt=0.1,
        advanced=True,
        epochs=50,        # Increased from 20
        lr=0.001,
        patience=15,      # Increased from 10
        use_augmentation=True,
        use_lr_scheduler=True,
        gradient_clip=1.0,
        seed=42,
        save_dir=f'models/improved_liquidnn_{appliance.replace(" ", "_")}'
    )

    print(f"\n✅ Training completed!")
    print(f"Final Test F1: {test_metrics['f1']:.4f}")
