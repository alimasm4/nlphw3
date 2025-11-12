"""
Training module for sentiment classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from tqdm import tqdm


def get_optimizer(model, optimizer_name='adam', lr=0.001):
    """
    Get optimizer by name.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer ('adam', 'sgd', 'rmsprop')
        lr (float): Learning rate
        
    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, use_grad_clip=False, clip_value=1.0):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        use_grad_clip (bool): Whether to use gradient clipping
        clip_value (float): Gradient clipping value
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if enabled
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        
        # Calculate accuracy
        predictions = (outputs >= 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test data.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        tuple: (average_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1).float()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def train_model(model, train_loader, val_loader, optimizer, criterion, 
                device, epochs=10, use_grad_clip=False, clip_value=1.0,
                verbose=True):
    """
    Train model for multiple epochs.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epochs (int): Number of epochs
        use_grad_clip (bool): Whether to use gradient clipping
        clip_value (float): Gradient clipping value
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_grad_clip, clip_value
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                  f"Time: {epoch_time:.2f}s")
    
    return history


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation data
        y_val (np.ndarray): Validation labels
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.LongTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    from models import get_model, count_parameters
    from utils import set_seed, get_device
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create dummy data
    print("Creating dummy data for testing...")
    vocab_size = 10000
    seq_len = 50
    n_samples = 1000
    
    X_train = np.random.randint(0, vocab_size, (n_samples, seq_len))
    y_train = np.random.randint(0, 2, n_samples)
    X_val = np.random.randint(0, vocab_size, (200, seq_len))
    y_val = np.random.randint(0, 2, 200)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Create model
    print("\nCreating model...")
    model = get_model('lstm', vocab_size, activation='relu')
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, 'adam', lr=0.001)
    
    # Train
    print("\nTraining model...")
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs=3, use_grad_clip=True, verbose=True
    )
    
    print("\nTraining complete!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
