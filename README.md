# Semantic Similarity Project - SNLI Dataset

This project implements semantic similarity classification for the Stanford Natural Language Inference (SNLI) dataset using two different approaches:
- **BERT-based** sequence classification
- **LSTM-based** classification with GloVe embeddings

The project includes comprehensive data exploration, preprocessing, model training, and evaluation pipelines.

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 📁 Project Structure

```
InfoRet_SemanticSimilarity/
├── data_exploration.py          # Data exploration and analysis script
├── bert_training.py              # BERT model training script
├── lstm_training.py              # LSTM model training script
├── main.py                       # Main orchestration script
├── README.md                      # This file
├── stanford-natural-language-inference-corpus/  # Dataset folder (created after download)
├── data/                          # GloVe embeddings folder (created after download)
├── snli-preprocessed/            # Preprocessed data cache (created during training)
└── saved_models/                 # Saved model checkpoints (created after training)
```

## 🔧 Requirements

### Python Version
- Python 3.7 or higher

### Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision
pip install transformers
pip install nltk
pip install keras-preprocessing
pip install opendatasets
pip install wget
```

### Optional but Recommended
- CUDA-enabled GPU for faster training (PyTorch will automatically use GPU if available)

## 📦 Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torch transformers nltk keras-preprocessing opendatasets wget
   ```

3. **Download NLTK data** (will be done automatically, but you can pre-download):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('popular')
   ```

## 📊 Dataset Setup

### Option 1: Automatic Download (Recommended)

The dataset will be downloaded automatically when you run the data exploration script:

```bash
python main.py --explore
```

### Option 2: Manual Download

1. **SNLI Dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus)
   - Extract to `stanford-natural-language-inference-corpus/` folder
   - You'll need Kaggle credentials (username and API key)

2. **GloVe Embeddings** (for LSTM):
   - Will be downloaded automatically when running LSTM training
   - Or download manually from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
   - Extract to `data/` folder

## 🚀 Usage

### Quick Start

Run everything in sequence:
```bash
python main.py --all
```

This will:
1. Run data exploration
2. Train BERT model
3. Train LSTM model

### Individual Components

#### Data Exploration Only
```bash
python main.py --explore
```
or
```bash
python data_exploration.py
```

**Outputs:**
- `label_distribution.png` - Distribution of labels
- `sentence_length_distribution.png` - Sentence length statistics
- `train_word_frequency_training.png` - Word frequency analysis
- `eval_word_frequency_validation.png` - Validation word frequencies
- `test_word_frequency_test.png` - Test word frequencies

#### BERT Training Only
```bash
python main.py --bert
```
or
```bash
python bert_training.py
```

**Configuration** (in `bert_training.py`):
- `max_len_tokens = 256` - Maximum sequence length
- `batch_size = 32` - Batch size
- `epochs = 5` - Number of training epochs
- `learning_rate = 2e-5` - Learning rate
- `train_sample_frac = 0.3` - Fraction of training data to use (0.3 = 30%)

**Outputs:**
- `saved_models/bert_snli_model/` - Saved BERT model
- `predictions_BERT.csv` - Test predictions

#### LSTM Training Only
```bash
python main.py --lstm
```
or
```bash
python lstm_training.py
```

**Configuration** (in `lstm_training.py`):
- `BATCH_SIZE = 32` - Batch size
- `EMBEDDING_SIZE = 300` - GloVe embedding dimension
- `HIDDEN_SIZE = 64` - LSTM hidden state size
- `LEARNING_RATE = 0.001` - Learning rate
- `STACKED_LAYERS = 2` - Number of LSTM layers
- `EPOCHS = 10` - Number of training epochs

**Outputs:**
- `saved_models/lstm_snli_model.pth` - Saved LSTM model
- `data/glove.6B.300d.txt` - GloVe embeddings (downloaded automatically)

### Combined Execution

Run multiple components:
```bash
python main.py --explore --bert
python main.py --bert --lstm
```

### Skip Checks

Skip dependency and dataset checks:
```bash
python main.py --all --skip-checks
```

## 📄 File Descriptions

### `data_exploration.py`
Comprehensive data exploration script that:
- Loads and analyzes the SNLI dataset
- Computes statistics (missing values, label distribution, sentence lengths)
- Visualizes data distributions
- Performs word frequency analysis
- Cleans and preprocesses data

**Key Functions:**
- `load_data()` - Load CSV files
- `clean_data()` - Remove invalid labels and null values
- `analyze_label_distribution()` - Analyze and visualize labels
- `analyze_sentence_lengths()` - Sentence length statistics
- `analyze_word_frequency()` - Word frequency analysis

### `bert_training.py`
BERT-based sequence classification for SNLI:
- Uses BERT tokenizer for preprocessing
- Implements BERT fine-tuning for 3-class classification
- Includes training, validation, and testing
- Saves models and predictions

**Key Classes:**
- `SNLIDataset` - Dataset handler with BERT tokenization
- Compatible with `data_exploration.py` for data loading

**Key Functions:**
- `train_model()` - Training loop with validation
- `test_model()` - Model evaluation
- `save_model()` / `load_model()` - Model persistence

### `lstm_training.py`
LSTM-based classification with GloVe embeddings:
- Text cleaning and preprocessing
- Vocabulary building
- GloVe embedding integration
- Bidirectional LSTM architecture
- Training and evaluation

**Key Classes:**
- `Vocabulary` - Word-to-index mapping
- `DataSetLoader` - PyTorch Dataset wrapper
- `LSTM` - LSTM model architecture

**Key Functions:**
- `clean_text()` - Text preprocessing
- `load_embeddings()` - Load GloVe vectors
- `create_embedding_matrix()` - Create embedding weights
- `train_model()` - Training loop

### `main.py`
Orchestration script that:
- Coordinates all components
- Checks dependencies and dataset availability
- Provides command-line interface
- Tracks execution time and results

## ⚙️ Configuration

### BERT Model Configuration

Edit `bert_training.py`:
```python
max_len_tokens = 256      # Maximum sequence length
batch_size = 32           # Batch size
epochs = 5                # Training epochs
learning_rate = 2e-5      # Learning rate
train_sample_frac = 0.3   # Use 30% of training data
```

### LSTM Model Configuration

Edit `lstm_training.py`:
```python
BATCH_SIZE = 32           # Batch size
EMBEDDING_SIZE = 300      # GloVe embedding dimension
HIDDEN_SIZE = 64          # LSTM hidden size
LEARNING_RATE = 0.001     # Learning rate
STACKED_LAYERS = 2        # Number of LSTM layers
EPOCHS = 10               # Training epochs
```

## 🔍 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install missing packages:
```bash
pip install torch transformers nltk
```

### Dataset Not Found

**Problem**: `Error: Dataset folder 'stanford-natural-language-inference-corpus' not found`

**Solution**: 
1. Run data exploration first: `python main.py --explore`
2. Or download manually from Kaggle and extract to the project folder

### CUDA/GPU Issues

**Problem**: CUDA out of memory or GPU not detected

**Solution**:
- Reduce batch size in configuration
- Use CPU mode (automatically falls back if CUDA unavailable)
- Reduce `train_sample_frac` to use less data

### NLTK Data Download Issues

**Problem**: NLTK data download fails

**Solution**: Download manually:
```python
import nltk
nltk.download('punkt', download_dir='/path/to/nltk_data')
nltk.download('wordnet', download_dir='/path/to/nltk_data')
nltk.download('stopwords', download_dir='/path/to/nltk_data')
```

### Kaggle Authentication

**Problem**: Cannot download dataset from Kaggle

**Solution**:
1. Get Kaggle API credentials from https://www.kaggle.com/account
2. Place `kaggle.json` in `~/.kaggle/` folder
3. Or download dataset manually and extract

### Memory Issues

**Problem**: Out of memory during training

**Solution**:
- Reduce batch size
- Use data sampling (reduce `train_sample_frac`)
- Process data in smaller chunks
- Use CPU instead of GPU (slower but uses less memory)

## 📊 Expected Results

### Data Exploration
- Training set: ~550K samples
- Validation set: ~10K samples  
- Test set: ~10K samples
- Label distribution: ~33% each (contradiction, entailment, neutral)

### Model Performance
- **BERT**: Typically achieves 70-75% accuracy on validation set
- **LSTM**: Typically achieves 80-82% accuracy on validation set

*Note: Actual performance depends on hyperparameters, data sampling, and training duration*

## 📝 Notes

- The BERT script uses 30% of training data by default for faster training. Adjust `train_sample_frac` for full dataset training.
- Preprocessed data is cached in `snli-preprocessed/` to avoid reprocessing.
- Models are saved automatically after training.
- All scripts are compatible and can share data loading functions.

## 📚 References

- **SNLI Dataset**: [Stanford Natural Language Inference Corpus](https://nlp.stanford.edu/projects/snli/)
- **GloVe Embeddings**: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **PyTorch**: [PyTorch Documentation](https://pytorch.org/docs/)
- **Transformers**: [Hugging Face Transformers](https://huggingface.co/transformers/)

## 👤 Author

Lorenz Rossi (Student ID: 982595)

## 📄 License

This project is for educational purposes as part of an Information Retrieval course.
