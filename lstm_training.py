"""
LSTM-based Semantic Similarity Training Script for SNLI Dataset

This script implements LSTM-based sequence classification for the Stanford
Natural Language Inference (SNLI) task using GloVe embeddings. It includes:
- Data preprocessing and text cleaning
- Vocabulary building
- GloVe embedding loading
- LSTM model training with validation
- Model evaluation
- Compatibility with data_exploration.py for data loading
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import sys
import re
import os
import pickle
import gc
import zipfile
import subprocess
from datetime import datetime
from string import punctuation
from collections import Counter

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import random

# NLTK
import nltk as nlp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# Sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Try to import from data_exploration for compatibility
try:
    from data_exploration import load_data, clean_data
    USE_DATA_EXPLORATION = True
except ImportError:
    USE_DATA_EXPLORATION = False
    print("Note: data_exploration.py not found. Using standalone data loading.")


def install_package(package_name):
    """Install a package using pip if not already installed."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def setup_nltk():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    nlp.download('punkt', quiet=True)
    nlp.download('wordnet', quiet=True)
    nlp.download('stopwords', quiet=True)
    nlp.download('popular', quiet=True)


def download_glove(glove_file='glove.6B.zip', data_dir='./data'):
    """
    Download GloVe embeddings.
    
    Args:
        glove_file: Name of the GloVe zip file
        data_dir: Directory to save the embeddings
    """
    glove_url = f"http://nlp.stanford.edu/data/{glove_file}"
    glove_path = os.path.join(data_dir, glove_file)
    
    if not os.path.exists(os.path.join(data_dir, "glove.6B.300d.txt")):
        print("Downloading word embedding...")
        try:
            import wget
            downloaded_glove = wget.download(glove_url)
        except ImportError:
            install_package('wget')
            import wget
            downloaded_glove = wget.download(glove_url)
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        print("\nExtracting...")
        with zipfile.ZipFile(downloaded_glove, 'r') as zip_ref:
            zip_ref.extractall(path=data_dir)
        
        # Clean up zip file
        if os.path.exists(downloaded_glove):
            os.remove(downloaded_glove)
        
        print("Done!")
    else:
        print("GloVe embeddings already downloaded.")


def get_word_map(count):
    """
    Create a word count map from a series of counts.
    
    Args:
        count: Series or list of counts
        
    Returns:
        dict: Word count map
    """
    word_map = {}
    for num in count:
        if num in word_map:
            word_map[num] += 1
        else:
            word_map[num] = 1
    return word_map


def clean_text(text, stop_words=False, lemmatization=False):
    """
    Clean and preprocess text.
    
    Args:
        text: Input text string
        stop_words: Whether to remove stop words
        lemmatization: Whether to apply lemmatization
        
    Returns:
        str: Cleaned text
    """
    text = str(text).lower().split()
    
    if stop_words:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
    
    text = " ".join(text)
    text = re.sub("[^A-Za-z']+", ' ', str(text)).replace("'", '')
    text = re.sub(r"\bum*\b", "", text)
    text = re.sub(r"\buh*\b", "", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
    
    text = text.translate(str.maketrans('', '', punctuation))
    return text.strip()


def pair_generator(df):
    """
    Generate sentence pairs and labels from DataFrame.
    
    Args:
        df: DataFrame with sentence1, sentence2, and gold_label columns
        
    Returns:
        tuple: (sentence_pairs, sentence_labels)
    """
    sentence_pair = []
    sentence_label = []
    for _, row in df.iterrows():
        sentence_pair.append((row['sentence1'], row['sentence2']))
        sentence_label.append(row['gold_label'])
    return sentence_pair, sentence_label


class Vocabulary:
    """Vocabulary class to build word-to-index mappings."""
    
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        """Add all words from a sentence to the vocabulary."""
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.word2count[word] = 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataSetLoader(Dataset):
    """PyTorch Dataset class for sentence pairs."""
    
    def __init__(self, sentence_pair, labels):
        self.sentence_pair = sentence_pair
        self.labels = labels

    def __len__(self):
        return len(self.sentence_pair)

    def __getitem__(self, index):
        return self.sentence_pair[index], self.labels[index]


def get_pair_indices(sentence_pairs, vocab):
    """
    Convert sentence pairs to index sequences.
    
    Args:
        sentence_pairs: List of (premise, hypothesis) tuples
        vocab: Vocabulary object
        
    Returns:
        list: List of (premise_indices, hypothesis_indices) tuples
    """
    indices_pairs = []
    for sentence_pair in sentence_pairs:
        premise = sentence_pair[0]
        premise_indices = [vocab.word2index.get(w, 0) for w in word_tokenize(premise)]
        hypothesis = sentence_pair[1]
        hypothesis_indices = [vocab.word2index.get(w, 0) for w in word_tokenize(hypothesis)]
        indices_pairs.append((premise_indices, hypothesis_indices))
    return indices_pairs


def load_embeddings(path):
    """
    Load GloVe embeddings from file.
    
    Args:
        path: Path to GloVe embedding file
        
    Returns:
        dict: Dictionary mapping words to embedding vectors
    """
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    
    print(f"Loading embeddings from {path}...")
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = get_coefs(*line.strip().split(' '))
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def create_embedding_matrix(vocab, embeddings_index, embedding_size=300):
    """
    Create embedding matrix from vocabulary and GloVe embeddings.
    
    Args:
        vocab: Vocabulary object
        embeddings_index: Dictionary of GloVe embeddings
        embedding_size: Size of embedding vectors
        
    Returns:
        numpy.ndarray: Embedding weight matrix
    """
    vocab_size = len(vocab.word2index)
    weights = 1 * np.random.randn(vocab_size + 1, embedding_size)
    embedded_count = 0
    
    for word, lang_word_index in vocab.word2index.items():
        if word in embeddings_index:
            weights[lang_word_index] = embeddings_index[word]
            embedded_count += 1
    
    print(f"Embedded {embedded_count}/{vocab_size} words from vocabulary.")
    return weights


class LSTM(nn.Module):
    """LSTM model for semantic similarity classification."""
    
    def __init__(self, vocab_size, hidden_size, target_size, stacked_layers, weights_matrix, bidirectional=True):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            target_size: Number of output classes
            stacked_layers: Number of LSTM layers
            weights_matrix: Pre-trained embedding weights
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.target_size = target_size
        self.stacked_layers = stacked_layers
        
        num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.stacked_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=bidirectional
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Calculate input size for FC layer
        fc_input_size = 2 * 2 * hidden_size if bidirectional else 2 * hidden_size
        self.FC_concat1 = nn.Linear(fc_input_size, 128)
        self.FC_concat2 = nn.Linear(128, 64)
        self.FC_concat3 = nn.Linear(64, 32)

        for lin in [self.FC_concat1, self.FC_concat2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.output = nn.Linear(32, self.target_size)

        self.out = nn.Sequential(
            self.FC_concat1,
            self.relu,
            self.dropout,
            self.FC_concat2,
            self.relu,
            self.FC_concat3,
            self.relu,
            self.dropout,
            self.output
        )

    def forward_once(self, seq, hidden, seq_len, device):
        """Process a single sequence through LSTM."""
        embedd_seq = self.embedding(seq)
        packed_seq = pack_padded_sequence(
            embedd_seq, lengths=seq_len, batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(packed_seq, hidden)
        return hidden

    def forward(self, input, premise_len, hypothesis_len, device):
        """
        Forward pass through the model.
        
        Args:
            input: Tuple of (premise, hypothesis) tensors
            premise_len: List of premise lengths
            hypothesis_len: List of hypothesis lengths
            device: Device to run on
        """
        premise = input[0]
        hypothesis = input[1]
        batch_size = premise.size(0)

        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.stacked_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.stacked_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)

        premise_hidden = self.forward_once(premise, (h0, c0), premise_len, device)
        hypothesis_hidden = self.forward_once(hypothesis, (h0, c0), hypothesis_len, device)
        
        # Concatenate features
        combined_outputs = torch.cat((
            premise_hidden,
            hypothesis_hidden,
            torch.abs(premise_hidden - hypothesis_hidden),
            premise_hidden * hypothesis_hidden
        ), dim=2)

        return self.out(combined_outputs[-1])


def multi_acc(y_pred, y_test):
    """Calculate multi-class accuracy."""
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    """
    Train the LSTM model.
    
    Args:
        model: LSTM model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of training epochs
        
    Returns:
        dict: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    total_step = len(train_loader)

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        
        for val in train_loader:
            sentence_pairs, labels = map(list, zip(*val))

            premise_seq = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
            hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
            batch = len(premise_seq)

            premise_len = list(map(len, premise_seq))
            hypothesis_len = list(map(len, hypothesis_seq))

            temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
            premise_seq = temp[:batch, :]
            hypothesis_seq = temp[batch:, :]
            labels = torch.tensor(labels).long().to(device)

            model.zero_grad()
            prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len, device)

            loss = criterion(prediction, labels)
            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        
        with torch.no_grad():
            for val in val_loader:
                sentence_pairs, labels = map(list, zip(*val))

                premise_seq = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
                hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
                batch = len(premise_seq)

                premise_len = list(map(len, premise_seq))
                hypothesis_len = list(map(len, hypothesis_seq))

                temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
                premise_seq = temp[:batch, :]
                hypothesis_seq = temp[batch:, :]

                premise_seq = premise_seq.to(device)
                hypothesis_seq = hypothesis_seq.to(device)
                labels = torch.tensor(labels).long().to(device)

                prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len, device)
                
                loss = criterion(prediction, labels)
                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")
        print(f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return history


def test_model(model, test_loader, device, tag2idx):
    """
    Test the trained model.
    
    Args:
        model: Trained LSTM model
        test_loader: Test DataLoader
        device: Device to run on
        tag2idx: Label to index mapping
        
    Returns:
        tuple: (confusion_matrix, predictions, labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for val in test_loader:
            sentence_pairs, labels = map(list, zip(*val))

            premise_seq = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
            hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
            batch = len(premise_seq)

            premise_len = list(map(len, premise_seq))
            hypothesis_len = list(map(len, hypothesis_seq))

            temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
            premise_seq = temp[:batch, :]
            hypothesis_seq = temp[batch:, :]
            labels = torch.tensor(labels).long().to(device)

            prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len, device)
            predictions = torch.log_softmax(prediction, dim=1).argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    confusion_mat = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Create reverse mapping for labels
    idx2tag = {v: k for k, v in tag2idx.items()}
    label_names = [idx2tag[i] for i in sorted(idx2tag.keys())]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))
    
    return confusion_mat, all_predictions, all_labels


def save_model(model, save_path='saved_models'):
    """Save the trained model."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    model_path = os.path.join(save_path, 'lstm_snli_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main function to run the complete LSTM training pipeline."""
    print("="*60)
    print("LSTM SEMANTIC SIMILARITY TRAINING")
    print("="*60)

    # Setup
    setup_nltk()
    
    # Configuration
    dataset_folder = 'stanford-natural-language-inference-corpus'
    data_dir = './data'
    glove_path = os.path.join(data_dir, 'glove.6B.300d.txt')
    
    # Hyperparameters
    BATCH_SIZE = 32
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    STACKED_LAYERS = 2
    EPOCHS = 10
    
    # Download GloVe embeddings
    print("\n" + "="*60)
    print("DOWNLOADING GLOVE EMBEDDINGS")
    print("="*60)
    download_glove(data_dir=data_dir)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    if USE_DATA_EXPLORATION:
        print("Using data loading functions from data_exploration.py")
        train_raw, test_raw, eval_raw = load_data(dataset_folder)
        train_raw, eval_raw, test_raw = clean_data(train_raw, eval_raw, test_raw)
    else:
        if not os.path.exists(dataset_folder):
            print(f"Error: Dataset folder '{dataset_folder}' not found.")
            return
        
        train_raw = pd.read_csv(os.path.join(dataset_folder, "snli_1.0_train.csv"))
        eval_raw = pd.read_csv(os.path.join(dataset_folder, "snli_1.0_dev.csv"))
    
    # Preprocess data
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Filter and clean
    train_df = train_raw[['gold_label', 'sentence1', 'sentence2']].copy()
    val_df = eval_raw[['gold_label', 'sentence1', 'sentence2']].copy()
    
    train_df = train_df[train_df['gold_label'] != '-']
    val_df = val_df[val_df['gold_label'] != '-']
    
    train_df['sentence1'] = train_df['sentence1'].astype(str)
    train_df['sentence2'] = train_df['sentence2'].astype(str)
    val_df['sentence1'] = val_df['sentence1'].astype(str)
    val_df['sentence2'] = val_df['sentence2'].astype(str)
    
    # Filter empty sentences
    train_df = train_df[
        (train_df['sentence1'].str.split().str.len() > 0) & 
        (train_df['sentence2'].str.split().str.len() > 0)
    ]
    val_df = val_df[
        (val_df['sentence1'].str.split().str.len() > 0) & 
        (val_df['sentence2'].str.split().str.len() > 0)
    ]
    
    # Clean text
    print("Cleaning text...")
    train_df['sentence1'] = train_df['sentence1'].apply(lambda text: clean_text(text))
    train_df['sentence2'] = train_df['sentence2'].apply(lambda text: clean_text(text))
    val_df['sentence1'] = val_df['sentence1'].apply(lambda text: clean_text(text))
    val_df['sentence2'] = val_df['sentence2'].apply(lambda text: clean_text(text))
    
    # Filter again after cleaning
    train_df = train_df[
        (train_df['sentence1'].str.split().str.len() > 0) & 
        (train_df['sentence2'].str.split().str.len() > 0)
    ]
    val_df = val_df[
        (val_df['sentence1'].str.split().str.len() > 0) & 
        (val_df['sentence2'].str.split().str.len() > 0)
    ]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Generate sentence pairs
    print("\nGenerating sentence pairs...")
    df = pd.concat([train_df, val_df])
    sentence_pairs, _ = pair_generator(df)
    train_sentence_pairs, train_sentence_labels = pair_generator(train_df)
    val_sentence_pairs, val_sentence_labels = pair_generator(val_df)
    
    # Encode labels
    labels = set(train_sentence_labels)
    tag2idx = {word: i for i, word in enumerate(sorted(labels))}
    print(f"Labels: {tag2idx}")
    
    train_labels = [tag2idx[t] for t in train_sentence_labels]
    val_labels = [tag2idx[t] for t in val_sentence_labels]
    
    # Build vocabulary
    print("\n" + "="*60)
    print("BUILDING VOCABULARY")
    print("="*60)
    vocab = Vocabulary()
    for sentence_pair in sentence_pairs:
        premise = sentence_pair[0]
        hypothesis = sentence_pair[1]
        vocab.addSentence(premise)
        vocab.addSentence(hypothesis)
    
    print(f"Vocabulary size: {len(vocab.word2index)}")
    
    # Create datasets
    train_data = DataSetLoader(get_pair_indices(train_sentence_pairs, vocab), train_labels)
    val_data = DataSetLoader(get_pair_indices(val_sentence_pairs, vocab), val_labels)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=lambda x: x)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=lambda x: x)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Load embeddings
    print("\n" + "="*60)
    print("LOADING GLOVE EMBEDDINGS")
    print("="*60)
    embeddings_index = load_embeddings(glove_path)
    weights = create_embedding_matrix(vocab, embeddings_index, EMBEDDING_SIZE)
    del embeddings_index
    gc.collect()
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    VOCAB_SIZE = len(vocab.word2index)
    TARGET_SIZE = len(tag2idx)
    
    lstm_model = LSTM(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        target_size=TARGET_SIZE,
        stacked_layers=STACKED_LAYERS,
        weights_matrix=weights,
        bidirectional=True
    )
    lstm_model.to(device)
    print(lstm_model)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    history = train_model(
        lstm_model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS
    )
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    save_model(lstm_model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


