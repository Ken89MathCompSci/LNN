"""
Test improved Liquid Neural Network on all REDD appliances
Uses data augmentation, better normalization, and improved training
"""

import sys
import os
from datetime import datetime
from train_improved_liquidnn import (
    train_improved_liquidnn,
    load_redd_specific_splits
)


def test_improved_liquidnn_on_all_appliances(window_size=100, hidden_size=256, num_layers=3,
                                            dt=0.1, advanced=True, epochs=50, lr=0.001,
                                            patience=15, use_augmentation=True,
                                            use_lr_scheduler=True, gradient_clip=1.0):
    """
    Test Improved Liquid Neural Network on all REDD appliances

    Args:
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size (increased to 256)
        num_layers: Number of liquid layers (increased to 3)
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced model
        epochs: Number of training epochs (increased to 50)
        lr: Learning rate
        patience: Early stopping patience (increased to 15)
        use_augmentation: Whether to use data augmentation
        use_lr_scheduler: Whether to use learning rate scheduling
        gradient_clip: Gradient clipping value

    Returns:
        Dictionary of results for all appliances
    """
    print("=" * 70)
    print("Testing Improved Liquid Neural Network on All REDD Appliances")
    print("=" * 70)
    print(f"\nImprovements:")
    print(f"  ✅ Data Augmentation (noise, scaling, time shift)")
    print(f"  ✅ Better Normalization (per-dataset statistics)")
    print(f"  ✅ Increased Model Capacity ({hidden_size} hidden size, {num_layers} layers)")
    print(f"  ✅ Learning Rate Scheduling (ReduceLROnPlateau)")
    print(f"  ✅ Gradient Clipping ({gradient_clip})")
    print(f"  ✅ Longer Training ({epochs} epochs, patience {patience})")
    print(f"  ✅ Weight Decay (L2 regularization)")
    print("=" * 70)

    # Load data
    data_dict = load_redd_specific_splits()

    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "advanced_improved_liquid" if advanced else "improved_liquid"
    base_save_dir = f"models/{model_type}_redd_test_{timestamp}"

    # Dictionary to store results
    all_results = {}

    # Test on each appliance
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*70}")
        print(f"Testing {model_type} on {appliance_name}")
        print(f"{'='*70}\n")

        # Create appliance-specific save directory
        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            # Train and evaluate Improved Liquid Neural Network
            model, history, test_metrics = train_improved_liquidnn(
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
                use_augmentation=use_augmentation,
                use_lr_scheduler=use_lr_scheduler,
                gradient_clip=gradient_clip,
                seed=42,
                save_dir=appliance_dir
            )

            if model is not None:
                # Store results
                all_results[appliance_name] = {
                    'model_path': os.path.join(appliance_dir,
                                              f"{model_type}_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'best_epoch': history.get('best', {}).get('epoch')
                }
                print(f"✅ Successfully tested {model_type} on {appliance_name}")
            else:
                print(f"❌ Failed to test {model_type} on {appliance_name}")

        except Exception as e:
            print(f"Error testing {model_type} on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    import json
    summary = {
        'timestamp': timestamp,
        'model_type': model_type,
        'improvements': [
            'Data augmentation (noise, scaling, time shift)',
            'Better normalization (dataset statistics)',
            'Increased model capacity (256 hidden, 3 layers)',
            'Learning rate scheduling (ReduceLROnPlateau)',
            'Gradient clipping (1.0)',
            'Longer training (50 epochs, patience 15)',
            'Weight decay (L2 regularization)'
        ],
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
        'model_params': {
            'window_size': window_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dt': dt,
            'advanced': advanced
        },
        'train_params': {
            'epochs': epochs,
            'lr': lr,
            'patience': patience,
            'use_augmentation': use_augmentation,
            'use_lr_scheduler': use_lr_scheduler,
            'gradient_clip': gradient_clip
        },
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n🎉 {model_type} testing completed on all REDD appliances!")
    print(f"Results saved to {base_save_dir}")

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"📊 FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Appliance':<15} {'F1 Score':<12} {'MAE':<12} {'SAE':<12}")
    print(f"{'-'*50}")
    for appliance, result in all_results.items():
        metrics = result['final_metrics']
        print(f"{appliance:<15} {metrics['f1']:<12.4f} {metrics['mae']:<12.4f} {metrics['sae']:<12.4f}")

    return all_results


if __name__ == "__main__":
    print("Comprehensive Improved Liquid Neural Network Testing on REDD Dataset")
    print("=" * 70)

    # Check if data files exist
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

    # Test advanced improved model with all enhancements
    results = test_improved_liquidnn_on_all_appliances(
        window_size=100,
        hidden_size=256,         # Increased from 128
        num_layers=3,            # Increased from 2
        dt=0.1,
        advanced=True,
        epochs=50,               # Increased from 20
        lr=0.001,
        patience=15,             # Increased from 10
        use_augmentation=True,   # NEW: Data augmentation
        use_lr_scheduler=True,   # NEW: LR scheduling
        gradient_clip=1.0        # NEW: Gradient clipping
    )

    print("\n" + "=" * 70)
    print("TESTING COMPLETED!")
    print("=" * 70)

    # Print expected improvements
    print("\n📈 Expected Improvements vs Standard LNN:")
    print("  Standard LNN (dish washer): F1 = 0.42")
    print("  Improved LNN (expected):    F1 = 0.45-0.55 (+7-31%)")
    print("\nKey improvements:")
    print("  - Data augmentation makes model more robust")
    print("  - Better normalization improves convergence")
    print("  - Increased capacity captures complex patterns")
    print("  - LR scheduling finds better optima")
    print("  - Gradient clipping stabilizes training")
