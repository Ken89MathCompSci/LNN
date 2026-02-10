"""
Comprehensive testing script for Causal Liquid Neural Network on all REDD appliances
Tests all 4 appliances with causal learning
"""

import sys
import os
from datetime import datetime
from train_causal_liquidnn import (
    train_causal_liquidnn,
    load_redd_specific_splits
)


def test_causal_liquidnn_on_all_appliances(window_size=100, hidden_size=128, num_layers=2,
                                          dt=0.1, advanced=True, epochs=20, lr=0.001,
                                          patience=10, use_causal_loss=True,
                                          event_weight_scale=2.0):
    """
    Test Causal Liquid Neural Network on all REDD appliances

    Args:
        window_size: Size of input sequence window
        hidden_size: Hidden dimension size
        num_layers: Number of liquid layers (for advanced model)
        dt: Time step for liquid dynamics
        advanced: Whether to use advanced causal model
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        use_causal_loss: Whether to use causal event-weighted loss
        event_weight_scale: Scale factor for causal events

    Returns:
        Dictionary of results for all appliances
    """
    print("=" * 70)
    print("Testing Causal Liquid Neural Network on All REDD Appliances")
    print("=" * 70)

    # Load data
    data_dict = load_redd_specific_splits()

    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "advanced_causal_liquid" if advanced else "causal_liquid"
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
            # Train and evaluate Causal Liquid Neural Network
            model, history, test_metrics = train_causal_liquidnn(
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
                use_causal_loss=use_causal_loss,
                event_weight_scale=event_weight_scale,
                seed=42,
                save_dir=appliance_dir
            )

            if model is not None:
                # Store results
                all_results[appliance_name] = {
                    'model_path': os.path.join(appliance_dir,
                                              f"{model_type}_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'granger_causality': history.get('granger_causality', 0.0),
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
            'use_causal_loss': use_causal_loss,
            'event_weight_scale': event_weight_scale
        },
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n🎉 {model_type} testing completed on all REDD appliances!")
    print(f"Results saved to {base_save_dir}")

    return all_results


if __name__ == "__main__":
    print("Comprehensive Causal Liquid Neural Network Testing on REDD Dataset")
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

    # Test both standard and advanced causal models
    print("\n" + "=" * 70)
    print("PART 1: Standard Causal Liquid Neural Network")
    print("=" * 70)

    results_standard = test_causal_liquidnn_on_all_appliances(
        window_size=100,
        hidden_size=128,
        num_layers=2,
        dt=0.1,
        advanced=False,
        epochs=20,
        lr=0.001,
        patience=10,
        use_causal_loss=True,
        event_weight_scale=2.0
    )

    # Print summary
    print(f"\n📊 Summary of Standard Causal LNN:")
    print(f"Total appliances tested: {len(results_standard)}")
    for appliance, result in results_standard.items():
        print(f"\n  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")
        print(f"    Granger Causality: {result['granger_causality']:.4f}")

    print("\n" + "=" * 70)
    print("PART 2: Advanced Causal Liquid Neural Network")
    print("=" * 70)

    results_advanced = test_causal_liquidnn_on_all_appliances(
        window_size=100,
        hidden_size=128,
        num_layers=2,
        dt=0.1,
        advanced=True,
        epochs=20,
        lr=0.001,
        patience=10,
        use_causal_loss=True,
        event_weight_scale=2.0
    )

    # Print summary
    print(f"\n📊 Summary of Advanced Causal LNN:")
    print(f"Total appliances tested: {len(results_advanced)}")
    for appliance, result in results_advanced.items():
        print(f"\n  {appliance}:")
        print(f"    Test MAE: {result['final_metrics']['mae']:.4f}")
        print(f"    Test F1: {result['final_metrics']['f1']:.4f}")
        print(f"    Test SAE: {result['final_metrics']['sae']:.4f}")
        print(f"    Granger Causality: {result['granger_causality']:.4f}")

    print("\n" + "=" * 70)
    print("ALL TESTING COMPLETED!")
    print("=" * 70)
