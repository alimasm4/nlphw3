"""
Utility functions for the sentiment classification project.
"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the available device (CPU or CUDA).
    
    Returns:
        torch.device: Available device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def save_metrics(results, filepath):
    """
    Save evaluation metrics to CSV file.
    
    Args:
        results (list): List of dictionaries containing metrics
        filepath (str): Path to save the CSV file
    """
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def plot_training_history(history_list, model_names, save_path):
    """
    Plot training loss vs epochs for multiple models.
    
    Args:
        history_list (list): List of training histories
        model_names (list): List of model names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for history, name in zip(history_list, model_names):
        plt.plot(history['train_loss'], label=f'{name} - Train', linewidth=2)
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label=f'{name} - Val', 
                    linewidth=2, linestyle='--')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_sequence_length_comparison(results_df, save_path):
    """
    Plot accuracy and F1-score vs sequence length.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        save_path (str): Path to save the plot
    """
    # Filter results by sequence length
    seq_lengths = sorted(results_df['Seq Length'].unique())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by sequence length and calculate mean
    grouped = results_df.groupby('Seq Length')[['Accuracy', 'F1']].mean()
    
    # Plot Accuracy
    ax1.bar(grouped.index, grouped['Accuracy'], color='steelblue', alpha=0.8, width=8)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.set_xticks(seq_lengths)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot F1-Score
    ax2.bar(grouped.index, grouped['F1'], color='coral', alpha=0.8, width=8)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score vs Sequence Length', fontsize=14, fontweight='bold')
    ax2.set_xticks(seq_lengths)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sequence length comparison plot saved to {save_path}")


def plot_architecture_comparison(results_df, save_path):
    """
    Plot comparison of different architectures.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Accuracy', 'F1', 'Epoch Time (s)']
    titles = ['Accuracy by Architecture', 'F1-Score by Architecture', 'Training Time by Architecture']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        # Group by model architecture (first word of Model name)
        results_df['Architecture'] = results_df['Model'].str.split('_').str[0]
        grouped = results_df.groupby('Architecture')[metric].mean().sort_values()
        
        ax.barh(grouped.index, grouped.values, color=color, alpha=0.8)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture comparison plot saved to {save_path}")


def plot_optimizer_comparison(results_df, save_path):
    """
    Plot comparison of different optimizers.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by optimizer
    grouped = results_df.groupby('Optimizer')[['Accuracy', 'F1']].mean()
    
    x = np.arange(len(grouped.index))
    width = 0.35
    
    ax1.bar(x - width/2, grouped['Accuracy'], width, label='Accuracy', 
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, grouped['F1'], width, label='F1-Score', 
            color='coral', alpha=0.8)
    ax1.set_xlabel('Optimizer', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance by Optimizer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped.index)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Training time comparison
    time_grouped = results_df.groupby('Optimizer')['Epoch Time (s)'].mean()
    ax2.bar(time_grouped.index, time_grouped.values, color='mediumseagreen', alpha=0.8)
    ax2.set_xlabel('Optimizer', fontsize=12)
    ax2.set_ylabel('Average Epoch Time (s)', fontsize=12)
    ax2.set_title('Training Time by Optimizer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimizer comparison plot saved to {save_path}")


def create_results_summary(results_df, save_path):
    """
    Create a comprehensive results summary with statistics.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        save_path (str): Path to save the summary
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SENTIMENT CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Best performing model
        best_acc_idx = results_df['Accuracy'].idxmax()
        best_f1_idx = results_df['F1'].idxmax()
        
        f.write("BEST MODELS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Accuracy: {results_df.loc[best_acc_idx, 'Model']}\n")
        f.write(f"  - Accuracy: {results_df.loc[best_acc_idx, 'Accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {results_df.loc[best_acc_idx, 'F1']:.4f}\n")
        f.write(f"  - Optimizer: {results_df.loc[best_acc_idx, 'Optimizer']}\n")
        f.write(f"  - Sequence Length: {results_df.loc[best_acc_idx, 'Seq Length']}\n\n")
        
        f.write(f"Best F1-Score: {results_df.loc[best_f1_idx, 'Model']}\n")
        f.write(f"  - Accuracy: {results_df.loc[best_f1_idx, 'Accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {results_df.loc[best_f1_idx, 'F1']:.4f}\n")
        f.write(f"  - Optimizer: {results_df.loc[best_f1_idx, 'Optimizer']}\n")
        f.write(f"  - Sequence Length: {results_df.loc[best_f1_idx, 'Seq Length']}\n\n")
        
        # Statistics by architecture
        f.write("\nPERFORMANCE BY ARCHITECTURE:\n")
        f.write("-" * 80 + "\n")
        results_df['Architecture'] = results_df['Model'].str.split('_').str[0]
        arch_stats = results_df.groupby('Architecture')[['Accuracy', 'F1', 'Epoch Time (s)']].agg(['mean', 'std'])
        f.write(arch_stats.to_string())
        f.write("\n\n")
        
        # Statistics by optimizer
        f.write("\nPERFORMANCE BY OPTIMIZER:\n")
        f.write("-" * 80 + "\n")
        opt_stats = results_df.groupby('Optimizer')[['Accuracy', 'F1', 'Epoch Time (s)']].agg(['mean', 'std'])
        f.write(opt_stats.to_string())
        f.write("\n\n")
        
        # Statistics by sequence length
        f.write("\nPERFORMANCE BY SEQUENCE LENGTH:\n")
        f.write("-" * 80 + "\n")
        seq_stats = results_df.groupby('Seq Length')[['Accuracy', 'F1', 'Epoch Time (s)']].agg(['mean', 'std'])
        f.write(seq_stats.to_string())
        f.write("\n\n")
        
    print(f"Results summary saved to {save_path}")


def print_system_info():
    """Print system and PyTorch information."""
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 80 + "\n")
