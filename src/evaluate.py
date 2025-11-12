"""
Evaluation module for sentiment classification models.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def predict(model, dataloader, device):
    """
    Make predictions on data.
    
    Args:
        model (nn.Module): Model to use
        dataloader (DataLoader): Data loader
        device: Device to use
        
    Returns:
        tuple: (predictions, probabilities, true_labels)
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Predicting"):
            batch_x = batch_x.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Get predictions
            probabilities = outputs.cpu().numpy()
            predictions = (outputs >= 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(batch_y.numpy())
    
    return (np.array(all_predictions).flatten(), 
            np.array(all_probabilities).flatten(),
            np.array(all_labels))


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and compute metrics.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device: Device to use
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    predictions, probabilities, true_labels = predict(model, dataloader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }
    
    return metrics


def print_evaluation_results(metrics):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 60)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        metrics['true_labels'], 
        metrics['predictions'],
        target_names=['Negative', 'Positive']
    )
    print(report)


def plot_confusion_matrix(true_labels, predictions, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels
        save_path (str): Path to save plot (optional)
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(true_labels, probabilities, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        true_labels (np.ndarray): True labels
        probabilities (np.ndarray): Predicted probabilities
        save_path (str): Path to save plot (optional)
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_predictions(model, dataloader, device, preprocessor, n_samples=5):
    """
    Analyze and display sample predictions.
    
    Args:
        model (nn.Module): Model to use
        dataloader (DataLoader): Data loader
        device: Device to use
        preprocessor: Text preprocessor
        n_samples (int): Number of samples to show
    """
    model.eval()
    
    samples_shown = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            if samples_shown >= n_samples:
                break
            
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions = (outputs >= 0.5).float()
            
            for i in range(min(len(batch_x), n_samples - samples_shown)):
                sequence = batch_x[i].cpu().numpy()
                true_label = batch_y[i].item()
                pred_label = predictions[i].item()
                prob = outputs[i].item()
                
                # Convert sequence back to text
                words = []
                for idx in sequence:
                    if idx > 1:  # Skip PAD and UNK
                        words.append(preprocessor.idx2word.get(idx, '<UNK>'))
                
                text = ' '.join(words[:50])  # Show first 50 words
                
                print("\n" + "-" * 60)
                print(f"Sample {samples_shown + 1}:")
                print(f"Text: {text}...")
                print(f"True Label: {'Positive' if true_label == 1 else 'Negative'}")
                print(f"Predicted: {'Positive' if pred_label == 1 else 'Negative'}")
                print(f"Confidence: {prob:.4f}")
                print(f"Correct: {'✓' if true_label == pred_label else '✗'}")
                
                samples_shown += 1
                
                if samples_shown >= n_samples:
                    break


def compare_models(model_results):
    """
    Compare multiple models and display results.
    
    Args:
        model_results (dict): Dictionary of model names and their metrics
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for model_name, metrics in model_results.items():
        print(f"{model_name:<30} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f}")
    
    print("=" * 80)
    
    # Find best models
    best_accuracy = max(model_results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(model_results.items(), key=lambda x: x[1]['f1_score'])
    
    print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")


if __name__ == "__main__":
    from models import get_model
    from train import create_dataloaders
    from utils import set_seed, get_device
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create dummy data
    print("Creating dummy data for testing...")
    vocab_size = 10000
    seq_len = 50
    n_samples = 200
    
    X_test = np.random.randint(0, vocab_size, (n_samples, seq_len))
    y_test = np.random.randint(0, 2, n_samples)
    
    # Create dataloader
    test_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(X_test), 
        torch.LongTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = get_model('lstm', vocab_size, activation='relu')
    model = model.to(device)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(metrics['true_labels'], metrics['predictions'])
    
    print("\nEvaluation complete!")
