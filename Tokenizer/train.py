# This is a script from training a BPE tokenizer
# It includes reading different corpus

import hydra
from omegaconf import DictConfig
from Tokenizer.BPE import Tokenizer
from typing import Union, List
import requests
from bs4 import BeautifulSoup
import os

class Page:
    """A better representation of html content"""

class URLs:
    urls: Union[List[str], str, None]

def get_corpus(corpus_path: str) -> str:
    print("Reading corpus_path", corpus_path)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError()
    
    with open(corpus_path) as f:
        corpus: str = f.read()

    return corpus

def read_web_page(url: str) -> Page:
    # checking if a page is reachable
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if page.status_code != 200:
            raise Exception(f"Page not reachable, status code: {page.status_code}")
        soup = BeautifulSoup(page.text, 'html') #we create this soup object to leverage BeautifulSoup's text extraction capabilities
        print(soup)
    except Exception as e:
        raise e

def read_web_pages(urls: URLs) -> str:
    if urls is None:
        return [""]
    if isinstance(urls, str):
        return [read_web_page(url=urls)]
    pages = []
    if isinstance(urls, list):
        for url in urls:
            pages.extend(read_web_page(url=url))
    return pages

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    corpus_path = cfg.corpus_path
    try:
        corpus_urls = cfg.corpus_urls
    except:
        corpus_urls = None
    pages = read_web_pages(urls=corpus_urls)
    corpus = get_corpus(corpus_path=corpus_path)
    corpus += "\n".join(pages)
    tokenizer = Tokenizer(corpus=corpus)
    tokenizer.train()

if __name__ == "__main__":
    main()