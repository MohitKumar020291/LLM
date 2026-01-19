# LLM - GPT-2 Implementation with BPE Tokenizer

A PyTorch implementation of GPT-2 from scratch, featuring a Byte Pair Encoding (BPE) tokenizer for text preprocessing.

## Project Structure

```
├── GPT2/                 # GPT-2 model implementation
│   ├── GPT2.py          # Core model architecture (attention heads, transformers)
│   ├── train.py         # Training script with loss evaluation
│   └── Cache/           # Cached model checkpoints and corpus
├── Tokenizer/           # Byte Pair Encoding tokenizer
│   ├── BPE.py           # BPE tokenizer implementation
│   ├── train.py         # Tokenizer training script
│   ├── run.sh           # Shell script for running tokenizer
│   └── Corpus/          # Training corpus for tokenizer
├── utils.py             # Utility functions (web scraping, corpus loading)
├── outputs/             # Training outputs and logs
└── README.md            # This file
```

## Features

### GPT-2 Model (`GPT2/`)
- **Self-Attention Heads**: Single and multi-head attention mechanisms with causal masking
- **Transformer Blocks**: Feed-forward networks and residual connections
- **Multi-Head Attention**: Parallel attention heads for richer feature representation
- **Training Pipeline**: Loss estimation, batch processing, and model evaluation

### BPE Tokenizer (`Tokenizer/`)
- **Byte Pair Encoding**: Efficient subword tokenization algorithm
- **Vocabulary Management**: Customizable vocabulary size
- **Corpus Processing**: Handles text encoding/decoding with learned merge operations

### Utilities (`utils.py`)
- Web page scraping and text extraction
- Corpus loading from local files
- URL management for training data

## Installation

1. Ensure you have Python 3.8+ and PyTorch installed
2. Install required dependencies:
```bash
pip install torch hydra-core omegaconf beautifulsoup4 requests regex
```

## Usage

### Training the BPE Tokenizer

Train a BPE tokenizer with corpus URLs:
```bash
./Tokenizer/run.sh TRAIN_BPE="true" RUN_TYPE="cli" CORPUS_PATH="Tokenizer/corpus.txt" CORPUS_URLS="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" VOCAB_SIZE=500
```

### Training the GPT-2 Model

```python
from GPT2.train import train
from GPT2.GPT2 import GPT2

# Initialize model
model = GPT2(vocab_size=256, n_embed=384, num_heads=6, block_size=256)

# Train the model
train(
    model,
    max_iter=5000,
    batch_size=64,
    block_size=256,
    eval_interval=500,
    learning_rate=3e-4
)
```

Train GPT-2 using the bigram approach:
```bash
python3 -m GPT2.train +train_model="true" +generate="false"
```

### Using the BPE Tokenizer

```python
from Tokenizer.BPE import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer(corpus="your_corpus.txt", vocab_size=256)

# Train and encode text
encoded = tokenizer.train_encode(["hello world"])
```

## Key Components

### Attention Mechanism
- **Head**: Single self-attention head with query, key, and value projections
- **MultiHead**: Combines multiple attention heads with projection layer
- Includes dropout regularization and causal masking

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