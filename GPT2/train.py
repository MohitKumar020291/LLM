# reading the trained tokenizer
# reading the corpus again - as we might want to read from multiple sources

from Tokenizer import train
import hydra
from omegaconf import DictConfig
from utils import read_web_pages
import pickle as pkl
import torch
import torch.nn as nn
from Tokenizer.BPE import Tokenizer
from typing import Union, List


class Train:
    def __init__(self, corpus: Union[List[int], str], tokenizer: Tokenizer, train_size: float):
        self.train_size = train_size
        self.corpus_tokens = tokenizer.encode(corpus) if isinstance(corpus, str) else corpus # no need to encode if tokens are provided already!

        n = int(len(self.corpus_tokens)*train_size)
        self.train_corpus = self.corpus_tokens[:n]
        self.val_corpus = self.corpus_tokens[n:]

        # This is a .pt file containing pre-trained embeddings - Might become unuseful
        self.embedding_path = "./Embeddings/embeddings.pt"

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

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    corpus_urls = cfg.corpus_urls
    tokenizer_cache = "Tokenizer/Cache/tokenizer.pkl"
    # retreiving tokenizer
    try:
        with open(tokenizer_cache, 'rb') as file:
            tokenizer: Tokenizer = pkl.load(file)
        # Now 'data' contains the original Python object
    except FileNotFoundError:
        print(f"Error: The file '{tokenizer_cache}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    corpus = ""
    pages = read_web_pages(urls=corpus_urls)
    corpus += "\n".join(pages)
    
    # GPT2 training
    try:
        float(cfg.train_size)
    except:
        train_size = 0.9
    trainer = Train(corpus=corpus, train_size=train_size, tokenizer=tokenizer)
    print(trainer.get_batch(split="train", batch_size=2, block_size=8))



# if __name__ == "__main__":
#     main()