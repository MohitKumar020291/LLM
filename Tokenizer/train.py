# This is a script from training a BPE tokenizer
# It includes reading different corpus

import hydra
from omegaconf import DictConfig, OmegaConf
from Tokenizer.BPE import Tokenizer

import os
import pickle as pkl
from utils import read_web_pages, get_corpus, get_hf_tokenizer


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    corpus_urls = list(OmegaConf.select(cfg, "corpus_urls", default=None))
    train_hf = OmegaConf.select(cfg, "train_hf", default=False)
    corpus_path = OmegaConf.select(cfg, "corpus_path", default=None)
    hf_tokenizer_save_dir = OmegaConf.select(cfg, "hf_tokenizer_save_dir", default="Tokenizer/Cache/Tokenizers/HF_tokenizer")
    vocab_size = OmegaConf.select(cfg, "vocab_size", default=256)

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
    print(len(corpus))
    # corpus += "\n".join(pages)
    print("Read corpus:\n", corpus[:500], "\n")  # Print first 500 characters of the corpus for verification

    try:
        with open(new_tokenizer_corpus_path, "w") as file:
            file.write(corpus)
    except Exception as e:
        raise e

    print("Training tokenizer with vocab size:", vocab_size)
    if train_hf:
        tokenizer_save_path = new_tokenizer_path
        tokenizer = get_hf_tokenizer(vocab_size=vocab_size, corpus=[corpus], tokenizer_save_path=tokenizer_save_path, hf_tokenizer_save_dir=hf_tokenizer_save_dir)
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