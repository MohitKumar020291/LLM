import hydra
from omegaconf import DictConfig
import pickle as pkl
import torch
from torch.utils.data import DataLoader

from utils import get_hf_tokenizer, read_web_page
from GPT2.GPT2 import GPT2, Data
from GPT2.train_shakespear import train


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):

    idx = 100 #full shakespear
    corpus_url = f"https://www.gutenberg.org/cache/epub/{idx}/pg{idx}.txt"
    try:
        train_model = cfg.train_model
    except:
        train_model = False    
    
    try:
        train_tokenizer = cfg.train_tokenizer
    except:
        train_tokenizer = False

    tokenizer_save_path = "Tokenizer/Cache/Tokenizers/fs_tokenizer.pkl"
    vocab_size = 8_000
    if train_tokenizer:    
        # train tokenizer
        tokenizer = get_hf_tokenizer(vocab_size=vocab_size, corpus_url=corpus_url, tokenizer_save_path=tokenizer_save_path)
    else:
        with open(tokenizer_save_path, 'rb') as file:
            tokenizer = pkl.load(file)

    gpt_2_model = GPT2(vocab_size=vocab_size)

    train_size = 0.9
    block_size = 256
    batch_size = 64
    corpus = read_web_page(url=corpus_url)[0]
    model = GPT2(vocab_size=vocab_size, block_size=block_size).to("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Data(
        corpus=corpus, 
        tokenizer=tokenizer, 
        train_size=train_size,
        split="train",
        block_size=block_size,
        device=next(model.parameters()).device
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Data(
        corpus=corpus, 
        tokenizer=tokenizer, 
        train_size=train_size,
        split="val",
        block_size=block_size,
        device=next(model.parameters()).device
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Model initialized on device:", next(model.parameters()).device)



if __name__ == "__main__":
    main()  