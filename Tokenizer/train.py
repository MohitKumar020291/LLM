# This is a script from training a BPE tokenizer
# It includes reading different corpus

import hydra
from omegaconf import DictConfig
from Tokenizer.BPE import Tokenizer

import os
import pickle as pkl
from utils import read_web_pages, get_corpus, get_hf_tokenizer


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    try:
        corpus_urls = cfg.corpus_urls
    except:
        corpus_urls = None

    try:
        train_hf = cfg.train_hf
    except:
        train_hf = False
    
    try:
        corpus_path = cfg.corpus_path
    except:
        corpus_path = None
    
    tokenizers_path = "Tokenizer/Cache/Tokenizers"
    if not os.path.exists(tokenizers_path):
        os.makedirs(tokenizers_path)
    
    try:
        # for fs "Tokenizer/Cache/Tokenizers/fs_tokenizer.pkl" 
        new_tokenizer_path = cfg.new_tokenizer_path
        try:
            path, ext = os.path.splitext(new_tokenizer_path)
        except:
            raise e
        if not ext:
            new_tokenizer_path = path + ".pkl"
            print(f"filetype of tokenizer is not .pkl, so adding .pkl, new path name is {new_tokenizer_path}")
        new_tokenizer_corpus_path = new_tokenizer_path + ".txt"
        print(new_tokenizer_path)
    except:
        new_tokenizer_path = None

    if not new_tokenizer_path:
        tokenizers = os.listdir(tokenizers_path)
        new_tokenizer_base_path = os.path.join(tokenizers_path, tokenizers[-1].split('.')[0])
        new_tokenizer_corpus_path = new_tokenizer_base_path + str(len(tokenizers)+1) + ".txt"
        new_tokenizer_path = new_tokenizer_base_path + str(len(tokenizers)+1) + ".pkl"

    print("Tokenizer will be saved on", new_tokenizer_path)
    corpus = get_corpus(corpus_path=corpus_path)
    # corpus = "" # I am doing this willingly
    corpus = read_web_pages(urls=corpus_urls)[0]
    # corpus += "\n".join(pages)
    print("Read corpus:\n", corpus[:500], "\n")  # Print first 500 characters of the corpus for verification

    try:
        with open(new_tokenizer_corpus_path, "w") as file:
            file.write(corpus)
    except Exception as e:
        raise e

    try:
        # if the corpus is empty, raise an error
        vocab_size = int(cfg.vocab_size)
    except:
        if not corpus:
            # raise ValueError("Corpus is empty. Please provide a valid corpus_path or corpus_urls in the config file.")
            vocab_size = 256
    print("Training tokenizer with vocab size:", vocab_size)
    if train_hf:
        tokenizer_save_path = new_tokenizer_path
        tokenizer = get_hf_tokenizer(vocab_size=vocab_size, corpus=[corpus], tokenizer_save_path=tokenizer_save_path)
        sample_text = "Hello, world!"
        encoded = tokenizer.encode("Let's test the trained BPE tokenizer.")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
    else:
        tokenizer = Tokenizer(corpus=corpus, vocab_size=vocab_size, corpus_path=new_tokenizer_corpus_path)
        tokenizer.train()
        print("Trained tokenizer vocabulary size:", len(tokenizer.vocab))

        # Save the tokenizer merges and vocab if needed
        try:
            with open(new_tokenizer_path, "wb") as f:
                pkl.dump(tokenizer, f)
            print(f"Tokenizer saved to {new_tokenizer_path}") #This is important to be caught by the shell
        except Exception as e:
            raise e

        # Example encoding and decoding
        sample_text = "Hello, world!"
        encoded = tokenizer.encode(sample_text)
        print("Encoded:", encoded)
        decoded = tokenizer.decode(encoded[0] if isinstance(encoded[0], list) and len(encoded) > 0 else encoded)
        print("Decoded:", decoded)
        assert decoded == sample_text, "Decoded text does not match the original!"

if __name__ == "__main__":
    main()