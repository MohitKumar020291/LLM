import os
import requests
import pickle as pkl
import yaml
from bs4 import BeautifulSoup
from typing import Union, List
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from GPT2.GPT2_causal import GPT2Causal
from GPT2.conf.GPT2Config import GPT2Config

class URLs:
    urls: Union[List[str], str, None]

def get_corpus(corpus_path: Union[None, str] = None) -> str:
    if not corpus_path:
        return ""
    print("Reading corpus_path", corpus_path)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError()
    
    with open(corpus_path) as f:
        corpus: str = f.read()

    return corpus

def read_web_page(url: str) -> str:
    # checking if a page is reachable
    print("Reading web page", url)
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if page.status_code != 200:
            raise Exception(f"Page not reachable, status code: {page.status_code}")
        soup = BeautifulSoup(page.text, 'html.parser') #we create this soup object to leverage BeautifulSoup's text extraction capabilities
        print(soup.get_text()[:100])
        return soup.get_text()
    except Exception as e:
        raise e

def read_web_pages(urls: URLs) -> str:
    if not urls:
        return []
    if isinstance(urls, str):
        page = read_web_page(url=urls)
        return [page]
    pages = []
    if isinstance(urls, list):
        print(True)
        for url in urls:
            print(url)
            pages.append(read_web_page(url=url))
            print(len(pages))
    return pages

def get_hf_tokenizer(
        vocab_size: int, 
        tokenizer_save_path: str, 
        corpus: list[str] = None, 
        corpus_url: str = None, 
        hf_tokenizer_save_dir: str = "Tokenizer/Cache/Tokenizers/HF_tokenizer"
    ) -> Tokenizer:
    corpus = corpus
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # Example vocabulary size
        show_progress=True
    )
    corpus += read_web_page(url=corpus_url) if corpus_url else ""
    tokenizer.train_from_iterator(corpus, trainer)

    # expected to be pickle file
    with open(tokenizer_save_path, 'wb') as file:
        pkl.dump(tokenizer, file)
        print("hf tokenizer saved successfully")

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.add_special_tokens({
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        })
    hf_tokenizer.save_pretrained(hf_tokenizer_save_dir)
    return tokenizer

def save_as_hf_model(model_path, model_save_dir, tokenizer_path):
    cfg_dict = yaml.safe_load(open("GPT2/conf/config.yaml"))
    config = GPT2Config(**cfg_dict)

    model = GPT2Causal(config)
    state = torch.load(model_path)
    model.gpt2.load_state_dict(state["model_state_dict"], strict=True)

    model.save_pretrained(model_save_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(model_save_dir)