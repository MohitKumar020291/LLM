from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.tools import tool
from pydantic import BaseModel
from omegaconf import DictConfig
from typing import Optional, Union, List
import hydra
from functools import singledispatch

class Tokenize(BaseModel):
    class Config:
        arbitrary_types_allowed=True
    tokens: torch.Tensor

class TokensLocalInference(BaseModel):
    # sole purpose of the class is to call method reverse string via Local models
    class Config:
        arbitrary_types_allowed=True
    tokens: Union[List[int], torch.Tensor]

class TokensApiInference(BaseModel):
    # sole purpose of the class is to call method reverse string via apis
    class Config:
        arbitrary_types_allowed=True
    tokens: Union[List[int], torch.Tensor]  

class Tokenizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, string: Union[str, list[str]]) -> torch.Tensor:
        return self.tokenizer.encode(string, return_tensors="pt")
        
    def decode(self, tokens: torch.Tensor) -> list[str]:
        decoded_string = []
        for token in tokens[0]:
            decoded_string.append(self.tokenizer.decode(token_ids=token))
        return decoded_string

    def merge_same(self, tokens: list[str], c_idx=0):
        for idx, (idx_str, idxx_str) in enumerate(zip(tokens[:-1], tokens[1:])):
            if idx_str == idxx_str and idx > c_idx:
                c_idx = idx
                encoded_tokens = self.encode(string=idx_str+idxx_str) #a list of integers
                decoded_tokens = self.decode(tokens=encoded_tokens) #a list of strings
                tokens_copy = tokens[:idx]
                tokens_copy.extend(decoded_tokens)
                tokens_copy.extend(tokens[idx+2:])
                break
            c_idx = idx if idx > c_idx else c_idx

        if c_idx >= len(tokens) - 2:
            return tokens
        return self.merge_same(tokens=tokens, c_idx=c_idx)

    def tokenize(self, string: str):
        encoded_tokens = self.tokenizer.encode(string, return_tensors="pt")
        return self.merge_same(encoded_tokens)

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    @singledispatch
    def reverse_string(self, tokens: TokensLocalInference):
        return self.model(tokens.tokens)

    @reverse_string.register
    def reverse_string(self, tokens: TokensApiInference):
        # make a client # key thing is the tokenizer should be of tokens
        # call the method
        ...

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: Optional[DictConfig] = None):
    model_name = cfg.model_name if cfg and 'model_name' in cfg else "distilgpt2"
    tokenizer = Tokenizer(model_name=model_name)

    string_to_reverse = "babbbcdefghiiiiigiiighiiiiiiiigggghhihihihg"

    tokenizer = Tokenizer(model_name=model_name)
    encodings = tokenizer.encode(string=string_to_reverse)
    decodings = tokenizer.decode(tokens=encodings)
    print(encodings)
    print(decodings)
    merged_tokens = tokenizer.merge_same(decodings)
    print(merged_tokens)

if __name__ == "__main__":
    main()