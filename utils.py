import os
import requests
from bs4 import BeautifulSoup
from typing import Union, List

class URLs:
    urls: Union[List[str], str, None]

def get_corpus(corpus_path: str) -> str:
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
    if urls is None:
        print(True)
        return []
    if isinstance(urls, str):
        return [read_web_page(url=urls)]
    pages = []
    if isinstance(urls, list):
        for url in urls:
            pages.extend(read_web_page(url=url))
    return pages