from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from utils import read_web_page

# 1. Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 2. Set up pre-tokenization (e.g., split by whitespace)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Define the trainer with desired vocab size and special tokens
trainer = trainers.BpeTrainer(
    vocab_size=1000,
    show_progress=True
)

# 4. Prepare your text data (as an iterator/list)
# texts = ["your first sentence here", "another text sample..."]
# For demonstration, let's use some sample data:
corpus: str = read_web_page(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# 5. Train the tokenizer
tokenizer.train_from_iterator([corpus], trainer)

# 6. Save the tokenizer
tokenizer.save("my_bpe_tokenizer.json")

# 7. Test the tokenizer
encoded = tokenizer.encode("Let's test the trained BPE tokenizer.")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

import pickle as pkl
my_tokenizer = None
with open("Tokenizer/Cache/Tokenizers/tokenizer.pkl", 'rb') as file:
    my_tokenizer = pkl.load(file)

ids = my_tokenizer.encode("Let's test the trained BPE tokenizer.")
tokens = [my_tokenizer.vocab[id] for id in ids]
print("Tokens:", tokens)
print("IDs", ids)