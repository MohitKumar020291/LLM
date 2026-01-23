# LLM: GPT-2 Implementation with BPE Tokenizer

A complete PyTorch implementation of GPT-2 (Generative Pre-trained Transformer 2) from scratch, including a Byte Pair Encoding (BPE) tokenizer for efficient subword tokenization.

## 📋 Overview

This project implements the core components of GPT-2, a state-of-the-art language model. It demonstrates:
- Transformer architecture with multi-head self-attention
- Efficient tokenization using BPE algorithm
- Full training pipeline with loss evaluation and checkpointing
- Text generation capabilities

## 📁 Project Structure

```
LLM/
├── GPT2/                          # GPT-2 model implementation
│   ├── GPT2.py                    # Model architecture (attention, transformer blocks)
│   ├── train.py                   # Training loop with evaluation and checkpointing
│   └── Cache/                     # Cached model weights and corpus data
├── Tokenizer/                     # Byte Pair Encoding tokenizer
│   ├── BPE.py                     # BPE tokenizer implementation
│   ├── train.py                   # Tokenizer training script
│   ├── run.sh                     # Shell script for training
│   ├── Experiment/                # Tokenizer experiments and results
│   └── Cache/                     # Cached tokenizer vocabulary
├── outputs/                       # Training outputs organized by date/time
├── utils.py                       # Utility functions (web scraping, corpus loading)
├── run.sh                         # Main entry point script
└── README.md                      # This file
```

## 🎯 Key Features

### GPT-2 Model Architecture (`GPT2/`)
- **Transformer Blocks**: Attention layers with feed-forward networks and residual connections
- **Self-Attention Heads**: 
  - Single attention head with Q, K, V projections
  - Multi-head attention combining multiple heads in parallel
  - Causal masking for autoregressive generation
- **Positional Encoding**: Token position awareness within context windows
- **Dropout Regularization**: Prevents overfitting with configurable dropout rates
- **Training Pipeline**: 
  - Gradient-based optimization with learning rate scheduling
  - Loss estimation on train/validation splits
  - Model checkpointing and resumption
  - Batch processing with configurable block sizes

### BPE Tokenizer (`Tokenizer/`)
- **Byte Pair Encoding Algorithm**: Hierarchical subword tokenization
- **Vocabulary Management**: Customizable vocabulary size and merging strategies
- **Corpus Processing**: 
  - Support for local file loading
  - Web page scraping and text extraction
  - Batch encoding and decoding
- **Efficient Encoding**: Reduces vocabulary size compared to character-level tokenization

### Utilities (`utils.py`)
- Web page scraping with BeautifulSoup
- Corpus loading from local files
- URL management for distributed training data
- Text preprocessing utilities

## 🚀 Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- Additional dependencies: Hydra, OmegaConf, BeautifulSoup4, Requests, Regex

### Setup

```bash
# Clone or navigate to the project
cd LLM

# Install dependencies
pip install torch hydra-core omegaconf beautifulsoup4 requests regex
```

## 💻 Usage

### Training the BPE Tokenizer

Train the tokenizer on a corpus:
```bash
cd Tokenizer
./run.sh TRAIN_BPE="true" \
         RUN_TYPE="cli" \
         CORPUS_PATH="corpus.txt" \
         VOCAB_SIZE=500
```

Or with web-sourced corpus:
```bash
./run.sh TRAIN_BPE="true" \
         RUN_TYPE="cli" \
         CORPUS_URLS="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
         VOCAB_SIZE=500
```

### Training the GPT-2 Model

Using Python API:
```python
from GPT2.train import train
from GPT2.GPT2 import GPT2

# Initialize model
model = GPT2(
    vocab_size=256,     # This vocab size means no new tokens
    n_embed=384,        # Embedding dimension
    num_heads=6,        # Number of attention heads
    block_size=256,     # Context window size
    num_layers=6        # Number of transformer layers
)

# Train the model
train(
    model,
    warmup_iters=500,
    max_iters=5000,
    batch_size=64,
    block_size=256,
    eval_interval=500,
    learning_rate=3e-4
)
```

Using Hydra configuration:
```bash
python3 -m GPT2.train \
    +train_model="true" \
    +generate="false" \
    +max_iters=5000 \
    +batch_size=64
```

### Using the BPE Tokenizer

```python
from Tokenizer.BPE import Tokenizer

# Initialize and train tokenizer
tokenizer = Tokenizer(corpus="corpus.txt", vocab_size=256)
tokenizer.train()

# Encode text
tokens = tokenizer.encode("Hello world")

# Decode tokens
text = tokenizer.decode(tokens)
```

### Generating Text

```python
from GPT2.GPT2 import GPT2
from Tokenizer.BPE import Tokenizer

# Load model and tokenizer
model = GPT2.load_checkpoint("path/to/checkpoint.pth")
tokenizer = Tokenizer.load("path/to/tokenizer.pkl")

# Generate text
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt)
generated = model.generate(tokens, max_new_tokens=100)
output = tokenizer.decode(generated)
print(output)
```

## 🏗️ Architecture Details

### Attention Mechanism

**Head** (Single Attention Head):
- Computes scaled dot-product attention
- Implements causal masking to prevent attending to future tokens
- Formula: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

**MultiHead** (Multi-Head Attention):
- Runs multiple attention heads in parallel
- Concatenates outputs and projects to embedding dimension
- Enables learning of different representation subspaces
- Formula: $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

### Transformer Block
- Multi-head attention layer with residual connection
- Position-wise feed-forward network (2 linear layers with ReLU)
- Layer normalization before each sub-layer
- Dropout for regularization

### Training Strategy
- Cosine learning rate scheduling with warmup
- Gradient accumulation for larger effective batch sizes
- Loss evaluation on both training and validation splits
- Early stopping and model checkpointing

## 📊 Output Organization

Training outputs:
```
GPT2/
└── Cache/
    └── gpt2.pth
```

Each training run stores:
- Model checkpoints (`.pth` files)
- Training logs and metrics
- Configuration files used for training

## 🔧 Configuration

Hydra configuration allows flexible parameter tuning. Common parameters:
- `n_embed`: Embedding dimension (default: 384)
- `num_heads`: Number of attention heads (default: 6)
- `num_layers`: Number of transformer blocks (default: 6)
- `block_size`: Context window size (default: 256)
- `batch_size`: Training batch size (default: 64)
- `learning_rate`: Initial learning rate (default: 3e-4)
- `max_iters`: Maximum training iterations (default: 2000)
- `eval_interval`: Evaluation frequency (default: 500)

## 📚 References

This implementation is inspired by:
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Byte Pair Encoding](https://arxiv.org/abs/1508.07909)

## 📝 License

This project is provided for educational and research purposes.

### Feed-Forward Network
- Projects embeddings to 4x hidden dimension
- ReLU activation with dropout

### Model Configuration
- Configurable embedding dimension (`n_embed`)
- Adjustable number of attention heads (`num_heads`)
- Tunable context length (`block_size`)
- Support for dropout regularization

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Batch Size**: Configurable (default: 64)
- **Block Size**: Context window length (default: 256)
- **Evaluation**: Separate train/val loss tracking at regular intervals

## Output

Training runs generate timestamped outputs in the `outputs/` directory, organized by date and time for easy experiment tracking.

## Requirements

- Python 3.8+
- PyTorch
- Hydra
- OmegaConf
- BeautifulSoup4
- Requests
- Regex

## Notes

- Models are cached in `GPT2/Cache/` for quick loading
- Corpus files are stored in `Tokenizer/Cache/`

python3 -m GPT2.train_shakespear +train_model="true" training_corpus_urls="https://www.gutenberg.org/cache/epub/100/pg100.txt" +tokenizer_path="Tokenizer/Cache/Tokenizers/tokenizer_fs.pkl" +model_path="GPT2/Cache/gpt_fs.pth"