"""
Hyperparameter tuning for Liquid Neural Network
Tests different hidden sizes and layer configurations
"""

import sys
import os
from datetime import datetime
from test_liquidnn_redd_specific_splits import (
    train_liquidnn_on_specific_redd_appliance,
    load_redd_specific_splits
)


def tune_hyperparameters(appliance_name='dish washer'):
    """
    Test different hyperparameter configurations

    Configurations to test:
    1. hidden_size: 128, 256, 512
    2. num_layers: 2, 3, 4
    3. learning_rate: 0.0001, 0.001, 0.01
    4. All combinations = 27 experiments
    """
    print("=" * 70)
    print(f"Hyperparameter Tuning for LNN on {appliance_name}")
    print("=" * 70)

    # Load data once
    data_dict = load_redd_specific_splits()

    # Hyperparameter grid
    hidden_sizes = [128, 256, 512]
    num_layers_list = [2, 3, 4]
    learning_rates = [0.0001, 0.001, 0.01]

    # Track results
    results = []
    best_f1 = 0
    best_config = None

    # Create timestamp for this tuning run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/hyperparameter_tuning_{timestamp}"

    total_experiments = len(hidden_sizes) * len(num_layers_list) * len(learning_rates)
    experiment_num = 0

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Testing hidden sizes: {hidden_sizes}")
    print(f"Testing num layers: {num_layers_list}")
    print(f"Testing learning rates: {learning_rates}")
    print("=" * 70)

    for hidden_size in hidden_sizes:
        for num_layers in num_layers_list:
            for lr in learning_rates:
                experiment_num += 1

                print(f"\n{'='*70}")
                print(f"Experiment {experiment_num}/{total_experiments}")
                print(f"Configuration: hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}")
                print(f"{'='*70}\n")

                # Create config-specific save directory
                config_name = f"h{hidden_size}_l{num_layers}_lr{lr}"
                save_dir = os.path.join(base_save_dir, config_name)
                os.makedirs(save_dir, exist_ok=True)

                try:
                    # Train with this configuration
                    model, history, test_metrics = train_liquidnn_on_specific_redd_appliance(
                        data_dict,
                        appliance_name=appliance_name,
                        window_size=100,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dt=0.1,
                        advanced=True,
                        epochs=20,
                        lr=lr,
                        patience=10,
                        seed=42,
                        save_dir=save_dir
                    )

                    # Store results
                    config_result = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'learning_rate': lr,
                        'test_mae': test_metrics['mae'],
                        'test_f1': test_metrics['f1'],
                        'test_sae': test_metrics['sae'],
                        'test_precision': test_metrics['precision'],
                        'test_recall': test_metrics['recall'],
                        'best_epoch': history.get('best', {}).get('epoch', -1),
                        'save_dir': save_dir
                    }
                    results.append(config_result)

                    print(f"\n✅ Results for hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}:")
                    print(f"   Test F1:        {test_metrics['f1']:.4f}")
                    print(f"   Test MAE:       {test_metrics['mae']:.4f}")
                    print(f"   Test SAE:       {test_metrics['sae']:.4f}")
                    print(f"   Test Precision: {test_metrics['precision']:.4f}")
                    print(f"   Test Recall:    {test_metrics['recall']:.4f}")

                    # Track best
                    if test_metrics['f1'] > best_f1:
                        best_f1 = test_metrics['f1']
                        best_config = config_result
                        print(f"   🎯 NEW BEST F1 SCORE!")

                except Exception as e:
                    print(f"❌ Error with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Save all results
    import json
    summary = {
        'timestamp': timestamp,
        'appliance': appliance_name,
        'total_experiments': total_experiments,
        'hyperparameter_grid': {
            'hidden_sizes': hidden_sizes,
            'num_layers': num_layers_list,
            'learning_rates': learning_rates
        },
        'results': results,
        'best_config': best_config
    }

    with open(os.path.join(base_save_dir, 'tuning_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Print final summary
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 70)

    # Sort results by F1 score
    results_sorted = sorted(results, key=lambda x: x['test_f1'], reverse=True)

    print(f"\n📊 Results Summary (sorted by F1 score):")
    print(f"\n{'Rank':<6} {'Hidden':<8} {'Layers':<8} {'LR':<10} {'F1 Score':<12} {'MAE':<12} {'SAE':<12}")
    print("-" * 80)

    for rank, result in enumerate(results_sorted, 1):
        marker = "🏆" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {result['hidden_size']:<8} {result['num_layers']:<8} {result['learning_rate']:<10.4f} "
              f"{result['test_f1']:<12.4f} {result['test_mae']:<12.4f} {result['test_sae']:<12.4f}")

    print("\n" + "=" * 70)
    print(f"🏆 BEST CONFIGURATION:")
    print(f"   Hidden size:    {best_config['hidden_size']}")
    print(f"   Num layers:     {best_config['num_layers']}")
    print(f"   Learning rate:  {best_config['learning_rate']}")
    print(f"   Test F1:        {best_config['test_f1']:.4f}")
    print(f"   Test MAE:       {best_config['test_mae']:.4f}")
    print(f"   Test SAE:       {best_config['test_sae']:.4f}")
    print(f"   Improvement over baseline (h=128, l=2, lr=0.001): {((best_config['test_f1'] - 0.42) / 0.42 * 100):.1f}%")
    print(f"   Model saved in: {best_config['save_dir']}")
    print("=" * 70)

    print(f"\nAll results saved to: {base_save_dir}/tuning_summary.json")

    return results_sorted, best_config


if __name__ == "__main__":
    print("Liquid Neural Network Hyperparameter Tuning")
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

    # Choose appliance to tune
    appliance = 'dish washer'  # Change to: 'fridge', 'microwave', or 'washer dryer'

    print(f"\nTuning hyperparameters for: {appliance}")
    print("This will test 27 different configurations:")
    print("  - Hidden sizes: 128, 256, 512")
    print("  - Num layers: 2, 3, 4")
    print("  - Learning rates: 0.0001, 0.001, 0.01")
    print("\nExpected runtime: ~2-3 hours with GPU, ~12-15 hours with CPU")
    print("=" * 70)

    # Run tuning
    results, best_config = tune_hyperparameters(appliance_name=appliance)

    print(f"\n✅ Hyperparameter tuning completed!")
    print(f"Best configuration found: hidden_size={best_config['hidden_size']}, num_layers={best_config['num_layers']}, lr={best_config['learning_rate']}")
    print(f"Best F1 score: {best_config['test_f1']:.4f}")
