"""
Data preprocessing module for IMDb sentiment classification.
"""

import re
import string
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import nltk
from collections import Counter
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class IMDbPreprocessor:
    """
    Preprocessor for IMDb movie review dataset.
    """
    
    def __init__(self, max_words=10000, max_len=100):
        """
        Initialize the preprocessor.
        
        Args:
            max_words (int): Maximum number of words in vocabulary
            max_len (int): Maximum sequence length
        """
        self.max_words = max_words
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return nltk.word_tokenize(text)
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts.
        
        Args:
            texts (list): List of text strings
        """
        print("Building vocabulary...")
        
        # Tokenize all texts
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            all_tokens.extend(tokens)
        
        # Count word frequencies
        word_counts = Counter(all_tokens)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_words - 2)  # Reserve 2 for PAD and UNK
        
        # Build word2idx mapping (0 for PAD, 1 for UNK)
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        
        # Build idx2word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of token IDs.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of token IDs
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        # Convert tokens to IDs
        sequence = []
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx['<UNK>'])
        
        return sequence
    
    def pad_sequence(self, sequence, max_len=None):
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            sequence (list): Input sequence
            max_len (int): Maximum length (uses self.max_len if None)
            
        Returns:
            list: Padded/truncated sequence
        """
        if max_len is None:
            max_len = self.max_len
        
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [0] * (max_len - len(sequence))
    
    def preprocess_texts(self, texts, max_len=None):
        """
        Preprocess a list of texts.
        
        Args:
            texts (list): List of text strings
            max_len (int): Maximum sequence length
            
        Returns:
            np.ndarray: Array of sequences
        """
        sequences = []
        for text in tqdm(texts, desc="Processing texts"):
            seq = self.text_to_sequence(text)
            padded = self.pad_sequence(seq, max_len)
            sequences.append(padded)
        
        return np.array(sequences)
    
    def save(self, filepath):
        """
        Save preprocessor to file.
        
        Args:
            filepath (str): Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_words': self.max_words,
                'max_len': self.max_len,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load preprocessor from file.
        
        Args:
            filepath (str): Path to load file
            
        Returns:
            IMDbPreprocessor: Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(data['max_words'], data['max_len'])
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.vocab_size = data['vocab_size']
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def load_imdb_data(data_dir='data'):
    """
    Load IMDb dataset from text files.
    This is a placeholder - you'll need to download the dataset.
    
    Args:
        data_dir (str): Directory containing data
        
    Returns:
        tuple: (train_texts, train_labels, test_texts, test_labels)
    """
    print("Note: This function expects IMDb dataset in the following structure:")
    print(f"  {data_dir}/train/pos/  (positive training reviews)")
    print(f"  {data_dir}/train/neg/  (negative training reviews)")
    print(f"  {data_dir}/test/pos/   (positive test reviews)")
    print(f"  {data_dir}/test/neg/   (negative test reviews)")
    print("\nAlternatively, you can use PyTorch's built-in IMDb dataset loader.")
    
    data_path = Path(data_dir)
    
    # Check if data exists
    if not data_path.exists():
        print(f"\nData directory not found. Will use PyTorch's built-in loader.")
        return load_imdb_pytorch()
    
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    # Load training data
    train_pos_dir = data_path / 'train' / 'pos'
    train_neg_dir = data_path / 'train' / 'neg'
    
    if train_pos_dir.exists() and train_neg_dir.exists():
        print("Loading training data...")
        # Positive reviews
        for filepath in tqdm(list(train_pos_dir.glob('*.txt')), desc="Train positive"):
            with open(filepath, 'r', encoding='utf-8') as f:
                train_texts.append(f.read())
                train_labels.append(1)
        
        # Negative reviews
        for filepath in tqdm(list(train_neg_dir.glob('*.txt')), desc="Train negative"):
            with open(filepath, 'r', encoding='utf-8') as f:
                train_texts.append(f.read())
                train_labels.append(0)
    
    # Load test data
    test_pos_dir = data_path / 'test' / 'pos'
    test_neg_dir = data_path / 'test' / 'neg'
    
    if test_pos_dir.exists() and test_neg_dir.exists():
        print("Loading test data...")
        # Positive reviews
        for filepath in tqdm(list(test_pos_dir.glob('*.txt')), desc="Test positive"):
            with open(filepath, 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
                test_labels.append(1)
        
        # Negative reviews
        for filepath in tqdm(list(test_neg_dir.glob('*.txt')), desc="Test negative"):
            with open(filepath, 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
                test_labels.append(0)
    
    if not train_texts:
        print("No data found in directory structure. Using PyTorch loader...")
        return load_imdb_pytorch()
    
    return train_texts, np.array(train_labels), test_texts, np.array(test_labels)


def load_imdb_pytorch():
    """
    Load IMDb dataset using PyTorch/TorchText (alternative method).
    
    Returns:
        tuple: (train_texts, train_labels, test_texts, test_labels)
    """
    try:
        from torchtext.datasets import IMDB
        import torch
        
        print("Loading IMDb dataset using PyTorch...")
        
        # Load training data
        train_iter = IMDB(split='train')
        train_texts, train_labels = [], []
        
        for label, text in tqdm(train_iter, desc="Loading train", total=25000):
            train_texts.append(text)
            train_labels.append(1 if label == 2 else 0)  # 2=pos, 1=neg in torchtext
        
        # Load test data
        test_iter = IMDB(split='test')
        test_texts, test_labels = [], []
        
        for label, text in tqdm(test_iter, desc="Loading test", total=25000):
            test_texts.append(text)
            test_labels.append(1 if label == 2 else 0)
        
        return train_texts, np.array(train_labels), test_texts, np.array(test_labels)
    
    except Exception as e:
        print(f"Error loading with PyTorch: {e}")
        print("\nPlease download the IMDb dataset manually from:")
        print("https://ai.stanford.edu/~amaas/data/sentiment/")
        raise


if __name__ == "__main__":
    # Example usage
    print("IMDb Preprocessor - Example Usage\n")
    
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    
    # Create preprocessor
    preprocessor = IMDbPreprocessor(max_words=10000, max_len=100)
    
    # Build vocabulary on training data
    preprocessor.build_vocabulary(train_texts)
    
    # Preprocess training data
    X_train = preprocessor.preprocess_texts(train_texts, max_len=100)
    
    # Preprocess test data
    X_test = preprocessor.preprocess_texts(test_texts, max_len=100)
    
    print(f"\nPreprocessed data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Save preprocessor
    preprocessor.save('data/preprocessor.pkl')
    
    # Save preprocessed data
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', train_labels)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', test_labels)
    
    print("\nPreprocessing complete!")
