"""
RNN model architectures for sentiment classification.
"""

import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    """
    Base RNN model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, 
                 output_dim=1, n_layers=2, dropout=0.3, activation='relu'):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (1 for binary classification)
            n_layers (int): Number of RNN layers
            dropout (float): Dropout rate
            activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(BaseRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        output, hidden = self.rnn(embedded)
        
        # Take the last output
        last_output = output[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply dropout and activation
        last_output = self.dropout(last_output)
        last_output = self.activation(last_output)
        
        # Fully connected layer
        out = self.fc(last_output)  # (batch_size, output_dim)
        out = self.sigmoid(out)
        
        return out


class LSTMModel(nn.Module):
    """
    LSTM model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, 
                 output_dim=1, n_layers=2, dropout=0.3, activation='relu'):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (1 for binary classification)
            n_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take the last output
        last_output = output[:, -1, :]
        
        # Apply dropout and activation
        last_output = self.dropout(last_output)
        last_output = self.activation(last_output)
        
        # Fully connected layer
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, 
                 output_dim=1, n_layers=2, dropout=0.3, activation='relu'):
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (1 for binary classification)
            n_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(BiLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Note: hidden_dim * 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # Bidirectional LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take the last output (from both directions)
        last_output = output[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Apply dropout and activation
        last_output = self.dropout(last_output)
        last_output = self.activation(last_output)
        
        # Fully connected layer
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out


def get_model(model_type, vocab_size, embedding_dim=100, hidden_dim=64, 
              output_dim=1, n_layers=2, dropout=0.3, activation='relu'):
    """
    Factory function to get model by type.
    
    Args:
        model_type (str): Type of model ('rnn', 'lstm', 'bilstm')
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of embeddings
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        n_layers (int): Number of layers
        dropout (float): Dropout rate
        activation (str): Activation function
        
    Returns:
        nn.Module: Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'rnn':
        return BaseRNN(vocab_size, embedding_dim, hidden_dim, output_dim, 
                       n_layers, dropout, activation)
    elif model_type == 'lstm':
        return LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, 
                        n_layers, dropout, activation)
    elif model_type == 'bilstm':
        return BiLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, 
                          n_layers, dropout, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    print("Model Architectures - Example Usage\n")
    
    vocab_size = 10000
    
    # Test RNN
    print("=" * 60)
    print("Base RNN Model")
    print("=" * 60)
    rnn_model = get_model('rnn', vocab_size, activation='relu')
    print(rnn_model)
    print(f"Parameters: {count_parameters(rnn_model):,}")
    
    # Test LSTM
    print("\n" + "=" * 60)
    print("LSTM Model")
    print("=" * 60)
    lstm_model = get_model('lstm', vocab_size, activation='relu')
    print(lstm_model)
    print(f"Parameters: {count_parameters(lstm_model):,}")
    
    # Test BiLSTM
    print("\n" + "=" * 60)
    print("Bidirectional LSTM Model")
    print("=" * 60)
    bilstm_model = get_model('bilstm', vocab_size, activation='relu')
    print(bilstm_model)
    print(f"Parameters: {count_parameters(bilstm_model):,}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    batch_size = 32
    seq_len = 50
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test each model
    with torch.no_grad():
        rnn_output = rnn_model(dummy_input)
        lstm_output = lstm_model(dummy_input)
        bilstm_output = bilstm_model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"RNN output shape: {rnn_output.shape}")
    print(f"LSTM output shape: {lstm_output.shape}")
    print(f"BiLSTM output shape: {bilstm_output.shape}")
    
    print("\nAll tests passed!")
