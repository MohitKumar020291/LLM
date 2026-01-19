import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import os

from Tokenizer.BPE import Tokenizer


class Head(nn.Module):
    """
    A single self-attention (input is same for generating q, k, v) head.
    """
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * q.size(-1) ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, head_size: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)  #residual made me do this
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout: float=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), #that 4 is a practice
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), #residual made me do this
            nn.Dropout(dropout)
        ) # this allows the embeddings to understand what they just learned through multi head attention
    
    def forward(self, x):
        """
        Docstring for forward
        
        :param x: shape of x is B,T,n_embed
        """
        return self.net(x) #B,T,n_embed

class Block(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, block_size: int):
        super().__init__()
        head_size = n_embed // num_heads
        self.attn = MultiHead(n_embed=n_embed, num_heads=num_heads, head_size=head_size, block_size=block_size) 
        self.ffwd = FeedForward(n_embed=n_embed)
        self.l1 = nn.LayerNorm(n_embed)
        self.l2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.l1(x)) #B,T,n_embed
        x = x + self.ffwd(self.l2(x)) #B,T,n_embed
        return x

class GPT2(nn.Module):
    def __init__(
            self,
            vocab_size: int, 
            corpus: str, 
            tokenizer: Tokenizer, 
            n_embed: int = 768, 
            train_size: float = 0.9, 
            block_size: int = 32,
            head_size: int = None,
            num_heads: int = 4,
            n_layers: int = 8,
            use_previous_tokens: bool = True
        ):
        nn.Module.__init__(self)

        previous_corpus_path = "GPT2/Cache/prev_corpus.txt"
        previous_tokens_path = "GPT2/Cache/prev_corpus.pkl"
        self.train_size = train_size
        if os.path.exists(previous_corpus_path) and use_previous_tokens and os.path.exists(previous_tokens_path):
            with open(previous_tokens_path, 'rb') as file:
                self.corpus_tokens = pkl.load(file)
        else:
            print(os.listdir("GPT2"))
            print(os.listdir("GPT2/Cache"))
            with open(previous_corpus_path, 'w') as fp:
                pass
            with open(previous_corpus_path, 'w') as file:
                file.write(corpus)
            print("Tokenizing the corpus and saving tokens for future use...")
            self.corpus_tokens = tokenizer.encode(corpus) if isinstance(corpus, str) else corpus # no need to encode if tokens are provided already!
            with open(previous_tokens_path, 'wb') as file:
                pkl.dump(self.corpus_tokens, file)

        n = int(len(self.corpus_tokens)*train_size)
        self.train_corpus = self.corpus_tokens[:n]
        self.val_corpus = self.corpus_tokens[n:]

        # This is a .pt file containing pre-trained embeddings - Might become unuseful
        self.embedding_path = "./Embeddings/embeddings.pt"
        self.block_size = block_size
        self.embeddings = nn.Embedding(vocab_size, n_embed) # each token directly maps to logits of next token - but I also smell this as a problem
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed=n_embed, num_heads=num_heads, block_size=block_size) for _ in range(n_layers)])
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
        x = self.blocks(x) # (B, T, n_embed)
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
    
    def get_batch(self, split, batch_size: int = 32, block_size: int = 8):
        """
        This function generates a batch of data for training/validation of tokens not embeddings.
        :param split: type of split, either 'train' or 'val'
        :type split: str
        :param batch_size: Number of sequences to generate
        :type batch_size: int
        :param block_size: Context length of each sequence
        :type block_size: int
        """
        data = self.train_corpus if split == "train" else self.val_corpus
        ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices, we subtract block_size to avoid overflow
        x = torch.stack([torch.tensor([data[i:i+block_size]]) for i in ix])
        y = torch.stack([torch.tensor([data[i+1:i+block_size+1]]) for i in ix])
        return x, y