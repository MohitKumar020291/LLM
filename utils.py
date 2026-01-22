import os
import requests
import pickle as pkl
from bs4 import BeautifulSoup
from typing import Union, List
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

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

def read_web_page(url: str):
    # checking if a page is reachable
    print("Reading web page", url)
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if page.status_code != 200:
            raise Exception(f"Page not reachable, status code: {page.status_code}")
        soup = BeautifulSoup(page.text, 'html.parser') #we create this soup object to leverage BeautifulSoup's text extraction capabilities
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
        for url in urls:
            pages.extend(read_web_page(url=url))
    return pages

def get_hf_tokenizer(vocab_size: int, tokenizer_save_path: str, corpus: list[str] = None, corpus_url: str = None) -> Tokenizer:
    corpus = corpus
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # Example vocabulary size
        show_progress=True
    )
    corpus += read_web_page(url=corpus_url) if corpus_url else ""
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save("my_bpe_tokenizer.json")

    # expected to be pickle file
    with open(tokenizer_save_path, 'wb') as file:
        pkl.dump(tokenizer, file)
        print("hf tokenizer saved successfully")
    return tokenizer