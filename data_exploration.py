"""
Data Exploration Script for Stanford Natural Language Inference (SNLI) Dataset

This script performs comprehensive data exploration including:
- Data loading and basic statistics
- Missing value analysis
- Label distribution visualization
- Sentence length analysis
- Word frequency analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report, confusion_matrix
import shutil
import sys
import os
import subprocess

# NLTK
import nltk as nlp
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# Download NLTK data
print("Downloading NLTK data...")
nlp.download('stopwords', quiet=True)
nlp.download('popular', quiet=True)
stop_words = stopwords.words('english')


def install_package(package_name):
    """Install a package using pip if not already installed."""
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def download_data():
    """
    Download GloVe embeddings and SNLI dataset.
    Note: This function requires opendatasets package and Kaggle credentials.
    """
    # Install opendatasets if needed
    try:
        import opendatasets as op
    except ImportError:
        print("Installing opendatasets...")
        install_package("opendatasets")
        import opendatasets as op
    
    # Download GloVe embeddings
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_file = "glove.6B.zip"
    
    if not os.path.exists(glove_file):
        print("Downloading GloVe embeddings...")
        subprocess.run(["wget", glove_url], check=True)
        subprocess.run(["unzip", "-o", glove_file], check=True)
    else:
        print("GloVe embeddings already downloaded.")
    
    # Download SNLI dataset
    snli_url = "https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus"
    dataset_folder = "stanford-natural-language-inference-corpus"
    
    if not os.path.exists(dataset_folder):
        print("Downloading SNLI dataset...")
        op.download(snli_url)
    else:
        print("SNLI dataset already downloaded.")


def load_data(dataset_folder="stanford-natural-language-inference-corpus"):
    """
    Load the SNLI dataset from CSV files.
    
    Args:
        dataset_folder: Path to the dataset folder
        
    Returns:
        tuple: (train_raw, test_raw, eval_raw) DataFrames
    """
    train_raw = pd.read_csv(os.path.join(dataset_folder, "snli_1.0_train.csv"))
    test_raw = pd.read_csv(os.path.join(dataset_folder, "snli_1.0_test.csv"))
    eval_raw = pd.read_csv(os.path.join(dataset_folder, "snli_1.0_dev.csv"))
    
    return train_raw, test_raw, eval_raw


def analyze_data_shape_and_missing(train_raw, eval_raw, test_raw):
    """
    Analyze and print data shape and missing values for all datasets.
    
    Args:
        train_raw: Training dataset DataFrame
        eval_raw: Evaluation dataset DataFrame
        test_raw: Test dataset DataFrame
    """
    print("\n" + "="*60)
    print("DATA SHAPE AND MISSING VALUES ANALYSIS")
    print("="*60)
    
    datasets = {
        'Train set': train_raw,
        'Evaluation set': eval_raw,
        'Test set': test_raw
    }
    
    for name, df in datasets.items():
        print(f'\n{name}: {df.shape}')
        print(df.isnull().sum())


def analyze_label_distribution(train_raw):
    """
    Analyze and visualize the distribution of gold labels.
    
    Args:
        train_raw: Training dataset DataFrame
    """
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Calculate proportions
    ratio_gold = train_raw['gold_label'].value_counts(normalize=True).sort_index(ascending=False).reset_index().set_index('index')
    ratio_gold['gold_label'] = ratio_gold['gold_label'].apply(lambda x: round(x, 3))
    
    print("\nLabel proportions:")
    print(ratio_gold)
    
    # Visualize the proportion
    colors = sns.color_palette('pastel')
    plt.figure(figsize=(8, 6))
    plt.pie(ratio_gold['gold_label'], labels=ratio_gold.index, colors=colors, 
            autopct='%.1f%%', startangle=90)
    plt.title('Proportion of the gold labels')
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved label distribution plot to 'label_distribution.png'")
    plt.close()


def analyze_sentence_lengths(train_raw):
    """
    Analyze and visualize the distribution of sentence lengths.
    
    Args:
        train_raw: Training dataset DataFrame
    """
    print("\n" + "="*60)
    print("SENTENCE LENGTH ANALYSIS")
    print("="*60)
    
    # Sentence 1
    train_sent1 = train_raw['sentence1'].str.count(' ')
    train_sent1 = train_sent1.apply(lambda x: int(x))
    print('\nSentence 1 Statistics:')
    print(round(train_sent1.describe(), 2))
    
    # Sentence 2
    train_sent2 = train_raw['sentence2'].dropna().str.count(' ')
    train_sent2 = train_sent2.apply(lambda x: int(x))
    print('\nSentence 2 Statistics:')
    print(round(train_sent2.describe(), 2))
    
    # Visualize the distribution
    train_sentences = pd.DataFrame({
        'sentence1': train_sent1,
        'sentence2': train_sent2
    })
    
    colors = sns.color_palette('pastel')
    plt.figure(figsize=(10, 6))
    box = sns.boxplot(data=train_sentences, palette=colors)
    box.set_ylabel('Words in a sentence')
    box.set_title('Distribution of the lengths of sentences')
    plt.tight_layout()
    plt.savefig('sentence_length_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved sentence length distribution plot to 'sentence_length_distribution.png'")
    plt.close()
    
    # Examples of min/max lengths
    print("\nSentence 1 Examples:")
    example1_min = train_sent1[train_sent1 == train_sent1.min()].sample(1)
    print(f'Min word count: {train_sent1.min()}')
    print(f'Example: {train_raw["sentence1"].loc[example1_min.index].values[0]}')
    
    example1_max = train_sent1[train_sent1 == train_sent1.max()].sample(1)
    print(f'\nMax word count: {train_sent1.max()}')
    print(f'Example: {train_raw["sentence1"].loc[example1_max.index].values[0]}')
    
    print("\nSentence 2 Examples:")
    example2_min = train_sent2[train_sent2 == train_sent2.min()].sample(1)
    print(f'Min word count: {train_sent2.min()}')
    print(f'Example: {train_raw["sentence2"].loc[example2_min.index].values[0]}')
    
    example2_max = train_sent2[train_sent2 == train_sent2.max()].sample(1)
    print(f'\nMax word count: {train_sent2.max()}')
    print(f'Example: {train_raw["sentence2"].loc[example2_max.index].values[0]}')


def clean_data(train_raw, eval_raw, test_raw):
    """
    Clean the datasets by removing rows with "-" gold labels and null values.
    
    Args:
        train_raw: Training dataset DataFrame
        eval_raw: Evaluation dataset DataFrame
        test_raw: Test dataset DataFrame
        
    Returns:
        tuple: (train, eval, test) cleaned DataFrames
    """
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    # Omit rows having the gold label "-" and irrelevant columns
    # Filter first, then set index if pairID exists
    train_filtered = train_raw[train_raw['gold_label'] != '-'].copy()
    eval_filtered = eval_raw[eval_raw['gold_label'] != '-'].copy()
    test_filtered = test_raw[test_raw['gold_label'] != '-'].copy()
    
    # Select only needed columns
    train = train_filtered[['gold_label', 'sentence1', 'sentence2']].copy()
    eval = eval_filtered[['gold_label', 'sentence1', 'sentence2']].copy()
    test = test_filtered[['gold_label', 'sentence1', 'sentence2']].copy()
    
    # Set index to pairID if it exists, otherwise use default index
    if 'pairID' in train_filtered.columns:
        train = train.set_index(train_filtered['pairID'])
    if 'pairID' in eval_filtered.columns:
        eval = eval.set_index(eval_filtered['pairID'])
    if 'pairID' in test_filtered.columns:
        test = test.set_index(test_filtered['pairID'])
    
    # Omit null indexes
    train.dropna(subset=['sentence1', 'sentence2'], inplace=True)
    eval.dropna(subset=['sentence1', 'sentence2'], inplace=True)
    test.dropna(subset=['sentence1', 'sentence2'], inplace=True)
    
    # Recheck the number of null values
    print("\nNull values after cleaning:")
    print("Train set:")
    print(train.isnull().sum())
    print("\nEvaluation set:")
    print(eval.isnull().sum())
    print("\nTest set:")
    print(test.isnull().sum())
    
    print(f"\nFinal dataset sizes:")
    print(f"Train: {len(train)}")
    print(f"Eval: {len(eval)}")
    print(f"Test: {len(test)}")
    
    return train, eval, test


def tokenize(sentence):
    """
    Tokenize and preprocess a sentence.
    
    Args:
        sentence: Input sentence string
        
    Returns:
        list: List of preprocessed tokens
    """
    # Tokenization
    new_tokens = word_tokenize(sentence)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens = [t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    
    # Lemmatization (become, becomes, becoming, became --> become)
    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
    return new_tokens


def analyze_word_frequency(df, dataset_name, save_prefix=""):
    """
    Analyze and visualize word frequencies in sentences.
    
    Args:
        df: DataFrame with 'sentence1' and 'sentence2' columns
        dataset_name: Name of the dataset (for titles)
        save_prefix: Prefix for saved plot files
    """
    print(f"\n" + "="*60)
    print(f"WORD FREQUENCY ANALYSIS - {dataset_name.upper()}")
    print("="*60)
    
    # Connect all sentences in the preprocessed dataset
    train_sentence1 = " ".join(df['sentence1'])
    token_s1 = tokenize(train_sentence1)
    
    train_sentence2 = " ".join(df['sentence2'])
    token_s2 = tokenize(train_sentence2)
    
    # Count the words
    count_s1 = Counter(token_s1)
    word_freq_s1 = pd.DataFrame(count_s1.items(), columns=['Word', 'Frequency']).sort_values(
        by='Frequency', ascending=False
    )
    
    count_s2 = Counter(token_s2)
    word_freq_s2 = pd.DataFrame(count_s2.items(), columns=['Word', 'Frequency']).sort_values(
        by='Frequency', ascending=False
    )
    
    print(f"\nUnique words in Sentence 1: {len(word_freq_s1)}")
    print(f"Unique words in Sentence 2: {len(word_freq_s2)}")
    
    # Create visualization
    nb_ranking = 20
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Determine xlim based on dataset size
    if len(df) > 100000:
        xlim_max = 100000
    else:
        xlim_max = 5000
    
    sns.barplot(x='Frequency', y='Word', data=word_freq_s1.head(nb_ranking), ax=ax1)
    ax1.set_xlim(0, xlim_max)
    ax1.set_title(f'Top {nb_ranking} frequent words in Sentence 1 for {dataset_name} data: n = {len(word_freq_s1)}')
    
    sns.barplot(x='Frequency', y='Word', data=word_freq_s2.head(nb_ranking), ax=ax2)
    ax2.set_xlim(0, xlim_max)
    ax2.set_title(f'Top {nb_ranking} frequent words in Sentence 2 for {dataset_name} data: n = {len(word_freq_s2)}')
    
    plt.tight_layout()
    filename = f'{save_prefix}word_frequency_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved word frequency plot to '{filename}'")
    plt.close()


def main():
    """
    Main function to run the complete data exploration pipeline.
    """
    print("="*60)
    print("SNLI DATASET EXPLORATION")
    print("="*60)
    
    # Step 1: Download data (uncomment if needed)
    # download_data()
    
    # Step 2: Load data
    print("\nLoading data...")
    dataset_folder = "stanford-natural-language-inference-corpus"
    
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        print("Please run download_data() first or ensure the dataset is in the correct location.")
        return
    
    train_raw, test_raw, eval_raw = load_data(dataset_folder)
    
    # Display basic info
    print("\nDataset loaded successfully!")
    print(f"Train set shape: {train_raw.shape}")
    print(f"Eval set shape: {eval_raw.shape}")
    print(f"Test set shape: {test_raw.shape}")
    
    print("\nFirst few rows of training data:")
    print(train_raw.head(3))
    
    # Step 3: Analyze data shape and missing values
    analyze_data_shape_and_missing(train_raw, eval_raw, test_raw)
    
    # Step 4: Analyze label distribution
    analyze_label_distribution(train_raw)
    
    # Step 5: Analyze sentence lengths
    analyze_sentence_lengths(train_raw)
    
    # Step 6: Clean data
    train, eval, test = clean_data(train_raw, eval_raw, test_raw)
    
    # Step 7: Word frequency analysis
    print("\n" + "="*60)
    print("NOTE: Word frequency analysis may take some time for large datasets.")
    print("The analysis will process all sentences in each dataset.")
    print("="*60)
    
    analyze_word_frequency(train, "Training", "train_")
    analyze_word_frequency(eval, "Validation", "eval_")
    analyze_word_frequency(test, "Test", "test_")
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("- label_distribution.png")
    print("- sentence_length_distribution.png")
    print("- train_word_frequency_training.png")
    print("- eval_word_frequency_validation.png")
    print("- test_word_frequency_test.png")


if __name__ == "__main__":
    main()

