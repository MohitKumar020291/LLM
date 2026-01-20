import torch 
from torch.utils.data import Dataset 
from typing import Union 


class AddNumData(Dataset):
    def __init__(self, max_num=100):
        self.max = max_num
        self.special = {'=': 10, '+': 11}  # digits 0-9, +, =

    def __len__(self):
        return 10_000_000  # infinite-style

    def encode(self, ch):
        if ch in self.special:
            return self.special[ch]
        return int(ch)

    def __getitem__(self, idx):
        a = torch.randint(0, self.max, (1,)).item()
        b = torch.randint(0, self.max, (1,)).item()
        c = a + b

        s = f"{a}+{b}={c}"
        tokens = list(s)

        x, y = [], []
        mask = True

        for i in range(len(tokens) - 1):
            xi = self.encode(tokens[i])
            yi = self.encode(tokens[i+1])

            if mask:
                yi = -1  # ignore problem part

            if tokens[i] == '=':
                mask = False  # start learning result

            x.append(xi)
            y.append(yi)

        return torch.tensor(x), torch.tensor(y)
