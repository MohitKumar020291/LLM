# LLM: GPT-2 Implementation with BPE Tokenizer

A from-scratch PyTorch implementation of GPT-2 with a Byte Pair Encoding tokenizer.

## Quick Start

### Installation

```bash
pip -r requirements.txt
```

### Train a BPE Tokenizer

```bash
cd Tokenizer
./run.sh TRAIN_BPE="true" RUN_TYPE="cli" CORPUS_PATH="corpus.txt" VOCAB_SIZE=500
```

Or train on web content:
```bash
./run.sh TRAIN_BPE="true" RUN_TYPE="cli" \
  CORPUS_URLS="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
  VOCAB_SIZE=500
```

### Train the GPT-2 Model

```bash
python3 -m GPT2.train +train_model="true" +max_iters=5000 +batch_size=64
```

Or with Python API:
```python
from GPT2.GPT2 import GPT2
from GPT2.train import train

model = GPT2(vocab_size=256, n_embed=384, num_heads=6, block_size=256, num_layers=6)
train(model, max_iters=5000, batch_size=64, learning_rate=3e-4)
```

### Generate Text

```python
from GPT2.GPT2 import GPT2
from Tokenizer.BPE import Tokenizer

model = GPT2.load_checkpoint("path/to/checkpoint.pth")
tokenizer = Tokenizer.load("path/to/tokenizer.pkl")

tokens = tokenizer.encode("Once upon a time")
generated = model.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(generated))
```

## Project Structure

- `GPT2/` - Model implementation and training scripts
- `Tokenizer/` - BPE tokenizer training and utilities
- `outputs/` - Training checkpoints and logs
- `utils.py` - Helper functions for corpus loading and web scraping

## Key Parameters

- `n_embed` - Embedding dimension (default: 384)
- `num_heads` - Attention heads (default: 6)
- `num_layers` - Transformer blocks (default: 6)
- `block_size` - Context window (default: 256)
- `batch_size` - Training batch size (default: 64)
- `learning_rate` - Initial LR (default: 3e-4)
- `max_iters` - Training iterations (default: 2000)

## Notes

- Models cached in `GPT2/Cache/`
- Tokenizers cached in `Tokenizer/Cache/`
- Timestamped outputs saved to `outputs/`