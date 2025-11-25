"""
BERT-based Semantic Similarity Training Script for SNLI Dataset

This script implements BERT-based sequence classification for the Stanford
Natural Language Inference (SNLI) task. It includes:
- Data preprocessing and tokenization using BERT tokenizer
- BERT model training with validation
- Model evaluation and prediction
- Compatibility with data_exploration.py for data loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import json
import os
import sys
import subprocess
import gc

# PyTorch and Transformers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    BertConfig, 
    get_linear_schedule_with_warmup
)
import random

# Keras preprocessing for padding
try:
    from keras_preprocessing.sequence import pad_sequences
except ImportError:
    print("Installing keras_preprocessing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras_preprocessing"])
    from keras_preprocessing.sequence import pad_sequences

# Sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, hamming_loss

# Try to import from data_exploration for compatibility
try:
    from data_exploration import load_data, clean_data
    USE_DATA_EXPLORATION = True
except ImportError:
    USE_DATA_EXPLORATION = False
    print("Note: data_exploration.py not found. Using standalone data loading.")


# Constants
SPECIAL_TOKENS = {
    'CLS': ['[CLS]'],
    'SEP': ['[SEP]']
}

SNLI_FILE_NAMES = {
    "train": "snli_1.0_train.csv",
    "validation": "snli_1.0_dev.csv",
    "test": "snli_1.0_test.csv"
}

DATASET_LABELS = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2
}


def install_package(package_name):
    """Install a package using pip if not already installed."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def setup_environment():
    """Install required packages if not already installed."""
    required_packages = ['sentencepiece', 'transformers', 'torch']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            install_package(package)


class SNLIDataset:
    """
    Class to handle SNLI datasets and preprocess them for BERT model training/testing.
    Compatible with data_exploration.py data loading functions.
    """

    def __init__(
        self, 
        tokenizer, 
        dataset_folder='stanford-natural-language-inference-corpus',
        preprocessed_folder='snli-preprocessed',
        batch_size=32,
        max_len_tokens=256,
        dataset_labels=None,
        download_dataset=False,
        use_padding=True,
        train_sample_frac=0.3
    ):
        """
        Initialize SNLI Dataset handler.
        
        Args:
            tokenizer: BERT tokenizer instance
            dataset_folder: Path to dataset folder
            preprocessed_folder: Path to save preprocessed data
            batch_size: Batch size for DataLoader
            max_len_tokens: Maximum sequence length
            dataset_labels: Dictionary mapping labels to integers
            download_dataset: Whether to download dataset
            use_padding: Whether to pad sequences
            train_sample_frac: Fraction of training data to use (for faster training)
        """
        self.dataset_folder = dataset_folder
        self.preprocessed_folder = preprocessed_folder
        self.batch_size = batch_size
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.max_len_tokens = max_len_tokens
        self.train_sample_frac = train_sample_frac

        self.dataset_labels = dataset_labels if dataset_labels else DATASET_LABELS

        if download_dataset:
            self.download_dataset()

        # Load data - try to use data_exploration functions if available
        if USE_DATA_EXPLORATION:
            print("Using data loading functions from data_exploration.py")
            train_raw, test_raw, eval_raw = load_data(dataset_folder)
            train_raw, eval_raw, test_raw = clean_data(train_raw, eval_raw, test_raw)
            
            # The clean_data function returns DataFrames with pairID as index
            # This matches the expected format (same as reading with index_col=1)
            # No additional processing needed
        else:
            # Standalone data loading
            train_data = os.path.join(self.dataset_folder, SNLI_FILE_NAMES["train"])
            validation_data = os.path.join(self.dataset_folder, SNLI_FILE_NAMES["validation"])
            test_data = os.path.join(self.dataset_folder, SNLI_FILE_NAMES["test"])

            train_raw = pd.read_csv(train_data, index_col=1)
            eval_raw = pd.read_csv(validation_data, index_col=1)
            test_raw = pd.read_csv(test_data, index_col=1)

        # Sample training data if specified
        if self.train_sample_frac < 1.0:
            print(f"Sampling {self.train_sample_frac*100}% of training data")
            self.train_raw = train_raw.sample(frac=self.train_sample_frac)
        else:
            self.train_raw = train_raw

        self.eval_raw = eval_raw
        self.test_raw = test_raw

    def download_dataset(self):
        """Download SNLI dataset if not present."""
        print("Downloading SNLI Dataset in CSV format")

        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder, exist_ok=True)

        try:
            import opendatasets as op
            url = "https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus"
            if not os.path.exists(os.path.join(self.dataset_folder, SNLI_FILE_NAMES["train"])):
                op.download(url, data_dir=self.dataset_folder)
                print(f"Dataset downloaded successfully in {self.dataset_folder}")
            else:
                print("Dataset already present")
        except ImportError:
            print("opendatasets not installed. Please install it or download the dataset manually.")
            raise

    def preprocess_dataset_util(self, dataset_df):
        """
        Preprocess dataset: tokenize sentences and create input IDs.
        
        Args:
            dataset_df: DataFrame with sentence1, sentence2, and gold_label columns
            
        Returns:
            tuple: (sentence_tokens, input_ids, token_lengths, processed_labels)
        """
        tokenizer = self.tokenizer
        labels_dict = self.dataset_labels

        def tokenize_sentence(tokenizer, input_sentence):
            """Tokenize a single sentence."""
            if pd.isna(input_sentence):
                return []
            return tokenizer.tokenize(str(input_sentence))

        sentence_A = dataset_df.sentence1.to_numpy()
        sentence_B = dataset_df.sentence2.to_numpy()
        labels = dataset_df.gold_label.to_numpy()

        sentence_A_tokens = []
        sentence_B_tokens = []
        processed_labels = []

        for i, j, k in zip(sentence_A, sentence_B, labels):
            try:
                if k == '-' or pd.isna(k):
                    continue

                t1 = tokenize_sentence(tokenizer, i)
                t2 = tokenize_sentence(tokenizer, j)

                if len(t1) == 0 or len(t2) == 0:
                    continue

                sentence_A_tokens.append(t1)
                sentence_B_tokens.append(t2)

                label = labels_dict.get(k, -1)
                if label == -1:
                    continue
                processed_labels.append(label)

            except Exception as e:
                print(f"Error processing sentence pair: {e}")
                continue

        sentence_tokens = []
        input_ids = []
        token_lengths = []

        CLS_TOKEN = SPECIAL_TOKENS['CLS']
        SEP_TOKEN = SPECIAL_TOKENS['SEP']

        for i, j in zip(sentence_A_tokens, sentence_B_tokens):
            sentence = CLS_TOKEN + i + SEP_TOKEN + j + SEP_TOKEN

            token_ids = tokenizer.convert_tokens_to_ids(sentence)
            sentence_tokens.append(sentence)

            token_lengths.append(len(token_ids))
            input_ids.append(token_ids)

        # Use dtype=object to handle variable-length sequences
        return (
            np.array(sentence_tokens, dtype=object),
            np.array(input_ids, dtype=object),
            np.array(token_lengths),
            np.array(processed_labels)
        )

    def preprocess_dataset(self, d_partition="train"):
        """
        Preprocess a dataset partition (train/validation/test).
        
        Args:
            d_partition: Partition name ("train", "validation", or "test")
            
        Returns:
            tuple: (tokens, input_ids, token_lengths, labels)
        """
        print(f"Preprocessing {d_partition} data")

        if d_partition.lower() not in ["train", "validation", "test"]:
            raise ValueError("d_partition must be train, validation or test")

        if not os.path.exists(self.preprocessed_folder):
            os.makedirs(self.preprocessed_folder, exist_ok=True)

        file_name_base = os.path.join(self.preprocessed_folder, d_partition + "_")

        # Check if preprocessed data exists
        if os.path.exists(file_name_base + "tokens.npy"):
            print("Retrieving preprocessed data from .npy files")
            tokens = np.load(file_name_base + "tokens.npy", allow_pickle=True)
            ids = np.load(file_name_base + "token-ids.npy", allow_pickle=True)
            lengths = np.load(file_name_base + "token-lengths.npy", allow_pickle=True)
            labels = np.load(file_name_base + "labels.npy", allow_pickle=True)
        else:
            # Select appropriate dataset
            if d_partition.lower() == "train":
                dataset_df = self.train_raw
            elif d_partition.lower() == "validation":
                dataset_df = self.eval_raw
            else:
                dataset_df = self.test_raw

            tokens, ids, lengths, labels = self.preprocess_dataset_util(dataset_df)

            # Save preprocessed data
            print(f"Saving preprocessed {d_partition} data")
            np.save(file_name_base + "tokens.npy", tokens)
            np.save(file_name_base + "token-ids.npy", ids)
            np.save(file_name_base + "token-lengths.npy", lengths)
            np.save(file_name_base + "labels.npy", labels)

        return (tokens, ids, lengths, labels)

    def pad_and_create_attention_masks(self, input_ids):
        """
        Pad input IDs and create attention masks.
        
        Args:
            input_ids: List of token ID sequences
            
        Returns:
            tuple: (padded_input_ids, attention_masks)
        """
        max_len_tokens = self.max_len_tokens
        if self.use_padding:
            input_ids = pad_sequences(
                input_ids, 
                maxlen=max_len_tokens, 
                dtype="long", 
                value=0, 
                truncating="post", 
                padding="post"
            )

        attention_masks = []
        for sentence in input_ids:
            attention_mask = [int(token_id > 0) for token_id in sentence]
            attention_masks.append(attention_mask)

        return input_ids, attention_masks

    def convert_data_to_tensor_dataset(self, tokens, attention_masks, labels):
        """
        Convert preprocessed data to PyTorch TensorDataset and DataLoader.
        
        Args:
            tokens: Token IDs
            attention_masks: Attention masks
            labels: Labels
            
        Returns:
            tuple: (data, sampler, dataloader)
        """
        batch_size = self.batch_size

        tokens = torch.tensor(tokens)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels, dtype=torch.long)

        data = TensorDataset(tokens, attention_masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return (data, sampler, dataloader)


def multi_acc(y_pred, y_test):
    """
    Calculate multi-class accuracy.
    
    Args:
        y_pred: Model predictions (logits)
        y_test: True labels
        
    Returns:
        float: Accuracy score
    """
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train_model(model, train_loader, val_loader, optimizer, device, epochs=5):
    """
    Train the BERT model.
    
    Args:
        model: BERT model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer instance
        device: Device (cuda or cpu)
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

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pair_token_ids = batch[0].to(device)
            mask_ids = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(
                pair_token_ids,
                token_type_ids=None,
                attention_mask=mask_ids,
                labels=labels
            )
            loss = outputs.loss
            prediction = outputs.logits

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
            for batch_idx, batch in enumerate(val_loader):
                pair_token_ids = batch[0].to(device)
                mask_ids = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model(
                    pair_token_ids,
                    token_type_ids=None,
                    attention_mask=mask_ids,
                    labels=labels
                )
                loss = outputs.loss
                prediction = outputs.logits

                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f'Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | '
              f'val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        print(f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return history


def test_model(model, test_loader, tokenizer, device):
    """
    Test the trained model and generate predictions.
    
    Args:
        model: Trained BERT model
        test_loader: Test DataLoader
        tokenizer: BERT tokenizer
        device: Device (cuda or cpu)
        
    Returns:
        tuple: (confusion_matrix, prediction_dataframe)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    sentences_and_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            pair_token_ids = batch[0].to(device)
            mask_ids = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(
                pair_token_ids,
                token_type_ids=None,
                attention_mask=mask_ids
            )

            logits = outputs.logits
            predictions = torch.log_softmax(logits, dim=1).argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

            for sentence, prediction, true_label in zip(pair_token_ids, predictions, labels):
                decoded_sentence = tokenizer.decode(sentence)
                sentences_and_predictions.append((
                    decoded_sentence, 
                    prediction.item(), 
                    true_label.item()
                ))

    confusion_mat = confusion_matrix(all_labels, all_predictions)
    prediction_df = pd.DataFrame(
        sentences_and_predictions, 
        columns=["sentence", "prediction", "true_label"]
    )

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print classification report
    label_names = list(DATASET_LABELS.keys())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))

    return confusion_mat, prediction_df


def save_model(model, save_path='saved_models'):
    """Save the trained model."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    model_path = os.path.join(save_path, 'bert_snli_model')
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path='saved_models/bert_snli_model'):
    """Load a saved model."""
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model


def main():
    """Main function to run the complete BERT training pipeline."""
    print("="*60)
    print("BERT SEMANTIC SIMILARITY TRAINING")
    print("="*60)

    # Setup environment
    setup_environment()

    # Configuration
    dataset_folder = 'stanford-natural-language-inference-corpus'
    preprocessed_folder = 'snli-preprocessed'
    saved_model_location = 'saved_models'
    max_len_tokens = 256
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    train_sample_frac = 0.3  # Use 30% of training data for faster training

    # Check if dataset exists
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        print("Please ensure the dataset is downloaded or run data_exploration.py first.")
        return

    # Initialize tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize dataset handler
    print("\nInitializing dataset handler...")
    dataset = SNLIDataset(
        tokenizer=tokenizer,
        dataset_folder=dataset_folder,
        preprocessed_folder=preprocessed_folder,
        batch_size=batch_size,
        max_len_tokens=max_len_tokens,
        dataset_labels=DATASET_LABELS,
        download_dataset=False,
        use_padding=True,
        train_sample_frac=train_sample_frac
    )

    # Preprocess datasets
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    train_tokens, train_input_ids, train_token_lengths, train_labels = dataset.preprocess_dataset(d_partition="train")
    validation_tokens, validation_input_ids, validation_token_lengths, validation_labels = dataset.preprocess_dataset(d_partition="validation")
    test_tokens, test_input_ids, test_token_lengths, test_labels = dataset.preprocess_dataset(d_partition="test")

    # Pad inputs and create attention masks
    print("\nPadding inputs and creating attention masks...")
    train_input_ids, train_attention_masks = dataset.pad_and_create_attention_masks(train_input_ids)
    validation_input_ids, validation_attention_masks = dataset.pad_and_create_attention_masks(validation_input_ids)
    test_input_ids, test_attention_masks = dataset.pad_and_create_attention_masks(test_input_ids)

    # Convert to PyTorch datasets
    print("Converting dataset to PyTorch TensorDataset...")
    train_data, train_sampler, train_dataloader = dataset.convert_data_to_tensor_dataset(
        train_input_ids, train_attention_masks, train_labels
    )
    validation_data, validation_sampler, validation_dataloader = dataset.convert_data_to_tensor_dataset(
        validation_input_ids, validation_attention_masks, validation_labels
    )
    test_data, test_sampler, test_dataloader = dataset.convert_data_to_tensor_dataset(
        test_input_ids, test_attention_masks, test_labels
    )

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize model
    print("\nInitializing BERT model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0.0)

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    history = train_model(model, train_dataloader, validation_dataloader, optimizer, device, epochs=epochs)

    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    save_model(model, saved_model_location)

    # Test model
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    confusion_mat, prediction_df = test_model(model, test_dataloader, tokenizer, device)

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Save predictions
    prediction_df.to_csv('predictions_BERT.csv', index=False)
    print("\nPredictions saved to 'predictions_BERT.csv'")

    # Plot training history
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

