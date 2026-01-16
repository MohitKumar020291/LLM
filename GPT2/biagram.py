import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import hydra
from omegaconf import DictConfig
from Tokenizer.BPE import Tokenizer
from GPT2.train import Train
from utils import read_web_page


class Head(nn.Module):
    """
    A single self-attention (input is same for generating q, k, v) head
    """
    def __init__(self, n_embed: int, head_size: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, head_size: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
    
    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class BiagramLanguageModel(nn.Module, Train):
    def __init__(
            self,
            vocab_size: int, 
            corpus: str, 
            tokenizer: Tokenizer, 
            n_embed: int = 768, 
            train_size: float = 0.9, 
            block_size: int = 32,
            head_size: int = None,
            num_heads: int = 4
        ):
        nn.Module.__init__(self)
        Train.__init__(self, corpus=corpus, tokenizer=tokenizer, train_size=train_size)
        self.block_size = block_size
        self.embeddings = nn.Embedding(vocab_size, n_embed) # each token directly maps to logits of next token - but I also smell this as a problem
        self.position_embedding = nn.Embedding(block_size, n_embed)
        head_size = head_size or n_embed
        self.sa_head = MultiHead(n_embed, num_heads=num_heads, head_size=head_size//num_heads, block_size=block_size)
        self.mlp = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, flatten: bool = True):
        """
        Docstring for forward
        
        :param self: Description
        :param idx: Shape is (B, T)
        :type idx: torch.Tensor
        :param targets: Shape is (B, T)
        :type targets: torch.Tensor
        """
        idx = idx.squeeze(1) if idx.ndim == 3 else idx
        B, T = idx.shape
        # targets = targets.squeeze(1).view(B*T) if targets and targets.ndim == 3 else targets
        token_emb = self.embeddings(idx) # (B, T, n_embed)
        position_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, n_embed)
        # print(token_emb.shape, position_emb.shape)
        x = token_emb + position_emb # (B, T, n_embed)
        x = self.sa_head(x)
        logits = self.mlp(x) # (B, T, vocab_size)
        # logits_bow = wei @ logits

        loss = None
        if targets is not None:
            if targets.ndim == 3:
                targets = targets.squeeze(1)
            targets = targets.view(-1)
            logits_flatten = logits.view(B*T, -1)
            loss = F.cross_entropy(logits_flatten, targets)
        
        if flatten:
            logits = logits.view(B*T, -1)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, flatten: bool = False):
        """
        Docstring for generate
        
        :param self: Description
        :param idx: Shape is (B, T)
        :type idx: torch.Tensor
        :param max_new_tokens: Description
        :type max_new_tokens: int
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # print("idx_cond shape:", idx_cond.shape)
            logits, loss = self.forward(idx_cond, flatten=False) # (B, T, C)
            logits = logits[:, -1, :] # (B, C)
            probs = nn.functional.softmax(logits, dim=-1) # (B, C)
            next_token = torch.multinomial(probs, num_samples=1).view(-1, 1) # (B, 1)
            idx = torch.cat((idx, next_token), dim=1) # (B, T+1)
            # print(idx)
        return idx
    
@torch.no_grad()
def estimate_loss(model, eval_iters, block_size=32):
    out = {}
    model.eval() #crucial
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = model.get_batch(split, block_size=block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses
    model.train()
    return out

def train(
    model, 
    max_iter: int = 2000, 
    batch_size: int = 32, 
    block_size: int = 32, 
    eval_interval: int = 200, 
    eval_iters: int = 100,
    learning_rate: float = 1e-3
):
    print("Starting training...")
    loss = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model, eval_iters=eval_iters)
            print("step %d: train loss %.4f, val loss %.4f" % (iter, losses['train'].mean(), losses['val'].mean()))
        x_batches, y_batches = model.get_batch(split="train", batch_size=batch_size, block_size=block_size) #(B,T)
        for x_batch, y_batch in zip(x_batches, y_batches):
            logits, loss = model(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    train_model = cfg.train_model
    generate = cfg.generate
    with open("Tokenizer/Cache/tokenizer.pkl", 'rb') as file:
        tokenizer: Tokenizer = pkl.load(file)

    print(train_model)
    if train_model:
        vocab_size = len(tokenizer.vocab) #currently the size if 1000
        corpus = ""
        pages = read_web_page(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        corpus += "\n".join(pages)
        train_size = 0.9
        block_size = 32
        model = BiagramLanguageModel(vocab_size=vocab_size, corpus=corpus, tokenizer=tokenizer, train_size=train_size, block_size=block_size)
        print("Model initialized with vocab size:", vocab_size)

        try:
            print("Beginning training...")
            train(model=model, block_size=block_size)
            with open("GPT2/Cache/biagram_model.pkl", "wb") as f:
                pkl.dump(model, f)
        except Exception as e:
            print("An error occurred during training:", e)
    else:
        with open("GPT2/Cache/biagram_model.pkl", "rb") as f:
            model: BiagramLanguageModel = pkl.load(f)
        print("Model loaded from disk.")

    # generating from the model
    if generate:
        context = torch.zeros((1, 1), dtype=torch.long) # (B, T)
        generated_tokens = model.generate(idx=context, max_new_tokens=100)
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        print("Generated text:\n", "".join(generated_text.splitlines()))  # Print first 10 lines of generated text

if __name__ == "__main__":
    main()