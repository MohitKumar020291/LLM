import regex
from typing import Tuple, List, Dict, Union

class Tokenizer:
    def __init__(self, corpus, vocab_size: int = 276):
        self.corpus = corpus
        self.vocab = {i: bytes([i]) for i in range(256)} # this version is cleaner as we work
        self.vocab_size = vocab_size

    def train_encode(self, string: list[str]) -> list[list[int]]:
        encodings = []
        for s in string:
            encodings.append(list(map(int, s.encode('utf-8'))))
        return encodings

    def get_stats(self, encodings: Union[List[int], List[List[int]]]) -> list[int]:
        counts = dict()
        if isinstance(encodings[0], int):
            _type = "encoding"
            encodings = [encodings]
        for enc in encodings:
            try:
                for pair in zip(enc, enc[1:]):
                    counts[pair] = counts.get(pair, 0) + 1
            except:
                # for enc in encodings:
                #     if isinstance(enc, int):
                #         print(enc)
                #         print(encodings)
                #         break
                ...
        return counts

    def merge_most_common_pair_andrej(self, encodings: Union[List[List[int]], List[List[int]]], pair: tuple, idx: int) -> List[List[int]]:
        new_encodings = []
        if isinstance(encodings[0], int):
            _type = "encoding"
            encodings = [encodings]
        for enc in encodings:
            i = 0
            new_enc = []
            while i < len(enc):
                # for last encoding we cannot check next two encodings
                if i < len(enc) - 1 and enc[i] == pair[0] and enc[i+1] == pair[1]:
                    new_enc.append(idx)
                    i += 2
                else:
                    new_enc.append(enc[i])
                    i += 1
            new_encodings.append(new_enc)
        if _type == "encoding":
            return new_encodings[0]
        return new_encodings

    def merge(self, encodings: list[list[int]]) -> Tuple[List[List[int]], Dict]:
        idx = 256
        num_merges = self.vocab_size - idx
        merges = dict()
        while num_merges > 0:
            stats = self.get_stats(encodings=encodings)
            pair = max(stats, key=stats.get)
            # print("merging", pair, "to", idx, "with count", stats[pair], "new number of encodings", len(encodings) - stats[pair])
            merges[pair] = idx
            encodings = self.merge_most_common_pair_andrej(encodings=encodings, pair=pair, idx=idx)
            idx += 1
            num_merges -= 1

        return encodings, merges

    def decode(self, encodings: list[int]) -> str:
        tokens: str = b"".join(self.vocab[idx] for idx in encodings) # this is string
        text = tokens.decode("utf-8", errors="ignore")
        return text

    def encode(self, string: str) -> list[int]:
        try:
            self.merges
        except AttributeError:
            raise AttributeError("Tokenizer not trained yet. Please call train() before encoding.")
        try:
            self.new_encodings
        except AttributeError:
            raise AttributeError("Tokenizer not trained yet. Please call train() before encoding.")
        encodings = list(string.encode("utf-8"))
        while len(encodings) >= 2:
            stats = self.get_stats(encodings=encodings) # count of pairs
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) #each key of stats; then merges[key]
            if pair not in self.merges: # if not any of p in stats is present in merges - all inf
                break
            idx = self.merges[pair]
            encodings = self.merge_most_common_pair_andrej(encodings=encodings, pair=pair, idx=idx)
        return encodings
    
    def regex_splitting(self, corpus: str, contractions: str = None) -> list[str]:
        """
        This ensures that there are merges on illegal merges like character + whitespace
        
        :param self: Description
        :param corpus: Description
        :type corpus: str
        :return: Description
        :rtype: list[str]
        """
        contractions = contractions or "'s|'t|'re|'ve|'m|'ll|'d"
        white_space = "\s+"
        white_space_with_letters = " [A-Za-z]+" #tells starting of a word
        numbers = "[0-9]+"
        punctuations_symbols = "[^\w\s]+"
        merged_patterns = f"{contractions}|{white_space_with_letters}|{white_space}|{numbers}|{punctuations_symbols}"
        pattern = regex.compile(merged_patterns)
        chunks = pattern.findall(string=corpus)
        return chunks

    def train(self, contractions: str = None):
        # working directly with the corpus might give less meaningful pairs
        chunks = self.regex_splitting(corpus=self.corpus, contractions=contractions)
        encodings = self.train_encode(string=chunks)
        self.new_encodings, self.merges = self.merge(encodings)
        for (e0, e1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[e0] + self.vocab[e1]

        # new_encodings were still chunked encodings
        self.concatenated_encoding = [] # These tokens are actually wrong
        for enc in self.new_encodings:
            self.concatenated_encoding.extend(enc)