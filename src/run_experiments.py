"""
Main experiment runner for systematic evaluation of RNN architectures.
This script runs all required experiments according to the homework specification.
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append('src')

from models import get_model, count_parameters
from train import train_model, create_dataloaders, get_optimizer
from evaluate import evaluate_model, print_evaluation_results
from utils import (set_seed, get_device, save_metrics, plot_training_history,
                   plot_sequence_length_comparison, plot_architecture_comparison,
                   plot_optimizer_comparison, create_results_summary, print_system_info)


def run_experiment(model_type, activation, optimizer_name, seq_length, 
                   use_grad_clip, X_train, y_train, X_test, y_test, 
                   vocab_size, device, epochs=10, batch_size=32):
    """
    Run a single experiment with specific configuration.
    
    Args:
        model_type (str): Type of model ('rnn', 'lstm', 'bilstm')
        activation (str): Activation function
        optimizer_name (str): Optimizer name
        seq_length (int): Sequence length
        use_grad_clip (bool): Whether to use gradient clipping
        X_train, y_train, X_test, y_test: Data
        vocab_size (int): Vocabulary size
        device: Device to use
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        dict: Experiment results
    """
    # Filter data to sequence length
    X_train_seq = X_train[:, :seq_length]
    X_test_seq = X_test[:, :seq_length]
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train_seq, y_train, X_test_seq, y_test, batch_size=batch_size
    )
    
    # Create model
    model = get_model(
        model_type, 
        vocab_size, 
        embedding_dim=100, 
        hidden_dim=64,
        n_layers=2, 
        dropout=0.3, 
        activation=activation
    )
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, lr=0.001)
    
    # Train model
    print(f"\nTraining {model_type.upper()} with {activation} activation, "
          f"{optimizer_name} optimizer, seq_len={seq_length}, "
          f"grad_clip={'Yes' if use_grad_clip else 'No'}")
    
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs=epochs, use_grad_clip=use_grad_clip, 
        clip_value=1.0, verbose=False
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    metrics = evaluate_model(model, val_loader, device)
    
    # Calculate average epoch time
    avg_epoch_time = np.mean(history['epoch_times'])
    
    # Prepare results
    result = {
        'Model': f"{model_type}_{activation}_{optimizer_name}_{seq_length}",
        'Architecture': model_type.upper(),
        'Activation': activation,
        'Optimizer': optimizer_name,
        'Seq Length': seq_length,
        'Grad Clipping': 'Yes' if use_grad_clip else 'No',
        'Accuracy': metrics['accuracy'],
        'F1': metrics['f1_score'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Epoch Time (s)': avg_epoch_time,
        'Total Training Time (s)': training_time,
        'Parameters': count_parameters(model)
    }
    
    print(f"  → Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
          f"Avg Epoch Time: {avg_epoch_time:.2f}s")
    
    return result, history


def run_all_experiments(X_train, y_train, X_test, y_test, vocab_size, 
                       device, epochs=10, output_dir='results'):
    """
    Run all required experiments systematically.
    
    Args:
        X_train, y_train, X_test, y_test: Data
        vocab_size (int): Vocabulary size
        device: Device to use
        epochs (int): Number of epochs per experiment
        output_dir (str): Output directory for results
        
    Returns:
        list: List of all experiment results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    
    results = []
    histories = {}
    
    # Define experiment configurations
    architectures = ['rnn', 'lstm', 'bilstm']
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['adam', 'sgd', 'rmsprop']
    seq_lengths = [25, 50, 100]
    grad_clip_options = [False, True]
    
    total_experiments = 0
    
    # Count total experiments (we'll run a subset to cover all variations)
    print("\n" + "=" * 80)
    print("EXPERIMENT PLAN")
    print("=" * 80)
    print("Running systematic experiments to evaluate:")
    print(f"  - Architectures: {architectures}")
    print(f"  - Activations: {activations}")
    print(f"  - Optimizers: {optimizers}")
    print(f"  - Sequence Lengths: {seq_lengths}")
    print(f"  - Gradient Clipping: {grad_clip_options}")
    print("=" * 80)
    
    # Experiment 1: Test all architectures (with fixed parameters)
    print("\n### EXPERIMENT SET 1: Architecture Comparison ###")
    for arch in architectures:
        result, history = run_experiment(
            arch, 'relu', 'adam', 50, True,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    # Experiment 2: Test all activations (with LSTM)
    print("\n### EXPERIMENT SET 2: Activation Function Comparison ###")
    for activation in activations:
        result, history = run_experiment(
            'lstm', activation, 'adam', 50, True,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    # Experiment 3: Test all optimizers (with LSTM)
    print("\n### EXPERIMENT SET 3: Optimizer Comparison ###")
    for optimizer in optimizers:
        result, history = run_experiment(
            'lstm', 'relu', optimizer, 50, True,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    # Experiment 4: Test all sequence lengths (with LSTM)
    print("\n### EXPERIMENT SET 4: Sequence Length Comparison ###")
    for seq_len in seq_lengths:
        result, history = run_experiment(
            'lstm', 'relu', 'adam', seq_len, True,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    # Experiment 5: Test gradient clipping (with LSTM)
    print("\n### EXPERIMENT SET 5: Gradient Clipping Comparison ###")
    for use_clip in grad_clip_options:
        result, history = run_experiment(
            'lstm', 'relu', 'adam', 50, use_clip,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    # Additional comprehensive experiments (combining variations)
    print("\n### EXPERIMENT SET 6: Additional Combinations ###")
    additional_configs = [
        ('bilstm', 'tanh', 'adam', 100, True),
        ('bilstm', 'relu', 'rmsprop', 50, True),
        ('rnn', 'tanh', 'sgd', 25, False),
        ('lstm', 'sigmoid', 'adam', 100, True),
    ]
    
    for arch, act, opt, seq, clip in additional_configs:
        result, history = run_experiment(
            arch, act, opt, seq, clip,
            X_train, y_train, X_test, y_test, vocab_size, device, epochs
        )
        results.append(result)
        histories[result['Model']] = history
        total_experiments += 1
    
    print(f"\n{'=' * 80}")
    print(f"COMPLETED {total_experiments} EXPERIMENTS")
    print(f"{'=' * 80}\n")
    
    return results, histories


def main():
    """Main function to run all experiments."""
    
    # Print system info
    print_system_info()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data_dir = Path('data')
    
    try:
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        print(f"Data loaded successfully!")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Sequence length: {X_train.shape[1]}")
        
        vocab_size = int(X_train.max()) + 1
        print(f"  Vocabulary size: {vocab_size}")
        
    except FileNotFoundError:
        print("\nPreprocessed data not found!")
        print("Please run: python src/preprocess.py")
        print("Or place preprocessed .npy files in the data/ directory")
        return
    
    # Run all experiments
    print("\nStarting experiments...")
    results, histories = run_all_experiments(
        X_train, y_train, X_test, y_test, vocab_size, 
        device, epochs=5, output_dir='results'  # Using 5 epochs for faster testing
    )
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    save_metrics(results, 'results/metrics.csv')
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot sequence length comparison
    plot_sequence_length_comparison(results_df, 'results/plots/seq_length_comparison.png')
    
    # Plot architecture comparison
    plot_architecture_comparison(results_df, 'results/plots/architecture_comparison.png')
    
    # Plot optimizer comparison
    plot_optimizer_comparison(results_df, 'results/plots/optimizer_comparison.png')
    
    # Plot training history for best and worst models
    best_idx = results_df['Accuracy'].idxmax()
    worst_idx = results_df['Accuracy'].idxmin()
    
    best_model = results_df.loc[best_idx, 'Model']
    worst_model = results_df.loc[worst_idx, 'Model']
    
    plot_training_history(
        [histories[best_model], histories[worst_model]],
        [f"Best: {best_model}", f"Worst: {worst_model}"],
        'results/plots/training_history.png'
    )
    
    # Create results summary
    create_results_summary(results_df, 'results/summary.txt')
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments run: {len(results)}")
    print(f"\nBest performing model:")
    best_result = results_df.loc[best_idx]
    print(f"  Model: {best_result['Model']}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}")
    print(f"  F1-Score: {best_result['F1']:.4f}")
    print(f"  Architecture: {best_result['Architecture']}")
    print(f"  Optimizer: {best_result['Optimizer']}")
    print(f"  Sequence Length: {best_result['Seq Length']}")
    
    print(f"\nAll results saved to: results/")
    print(f"  - Metrics: results/metrics.csv")
    print(f"  - Plots: results/plots/")
    print(f"  - Summary: results/summary.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
