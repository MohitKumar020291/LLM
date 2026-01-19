# This is a script from training a BPE tokenizer
# It includes reading different corpus

import hydra
from omegaconf import DictConfig
from Tokenizer.BPE import Tokenizer
from typing import Union, List
import requests
from bs4 import BeautifulSoup
import os
import pickle as pkl
from utils import get_corpus, read_web_pages


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    try:
        corpus_urls = cfg.corpus_urls
    except:
        corpus_urls = None
    # corpus = get_corpus(corpus_path=corpus_path)
    corpus = "" # I am doing this willingly
    pages = read_web_pages(urls=corpus_urls)
    corpus += "\n".join(pages)
    print("Read corpus:\n", corpus[:500], "\n")  # Print first 500 characters of the corpus for verification

    vocab_size = int(cfg.vocab_size)
    print("Training tokenizer with vocab size:", vocab_size)
    tokenizer = Tokenizer(corpus=corpus, vocab_size=vocab_size)
    tokenizer.train()
    print("Trained tokenizer vocabulary size:", len(tokenizer.vocab))

    # Example encoding and decoding
    sample_text = "Hello, world!"
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded[0])
    print("Decoded:", decoded)
    assert decoded == sample_text, "Decoded text does not match the original!"

    # Save the tokenizer merges and vocab if needed
    with open("Tokenizer/Cache/tokenizer.pkl", "wb") as f:
        pkl.dump(tokenizer, f)
    print("Tokenizer saved to tokenizer.pkl")

# if __name__ == "__main__":
#     main()