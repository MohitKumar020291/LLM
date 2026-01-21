import torch 
from torch.utils.data import Dataset 
from functools import singledispatchmethod
from typing import Union


class AddNumData(Dataset):
    def __init__(self, max_num=100, device: torch.device = torch.device(type='cpu'), split_ab: bool = False):
        self.max_num = max_num
        self.max_size = len(str(2 * (max_num - 1)))
        self.device = device
        self.split_ab = split_ab
        if self.split_ab:
            self.special = {'=': 10, '+': 11}  # digits 0-9, +, =
        else:
            self.special = {'=': self.max_num, '+': self.max_num+1}

    def __len__(self):
        return 10_000_000

    @singledispatchmethod
    def encode(self, ch):
        """Default implementation if no specific type is matched."""
        raise NotImplementedError(f"Unsupported type: {type(ch)}")

    @encode.register
    def _(self, ch: str):
        if ch in self.special:
            return self.special[ch]
        return int(ch)

    @encode.register
    def _(self, ch: list):
        encodings = []
        for ch in ch:
            encodings.append(self.encode(ch))
        return encodings

    def decode(self, string: str):
        """
            Default implementation if no specific type is matched.
            :param string: this should look like 0781110010080
            :return : special character's idx changed to character and paddings are removed
        """
        plus_size = len(str(self.special.get('+')))
        equal_size = len(str(self.special.get('=')))
        # print(string, self.max_size+plus_size, 2*self.max_size+plus_size)
        a = int(string[:self.max_size])
        b = int(string[self.max_size+plus_size : 2*self.max_size+plus_size])
        c = int(string[2*self.max_size+plus_size+equal_size:][::-1]) # the predicted valye should be in the reverse order
        # print(string, a, b, c)
        return f"{a}+{b}={c}"

    @staticmethod
    def pad(*args):
        if len(args) == 5:
            a, b, c, max_size, return_string = args
            a = str(a).zfill(max_size)
            b = str(b).zfill(max_size)
            c = str(c)[::-1].ljust(max_size, "0")
            if return_string:
                return f"{a}+{b}={c}"
            return a, b, c

        elif len(args) in (2, 4):
            num, max_size, *rest = args
            c = rest[0] if rest else False
            pad_flag = rest[1] if len(rest) > 1 else False

            s = str(num)
            if c:
                s = s[::-1]
                return s.ljust(max_size, "0")
            return s.zfill(max_size) if pad_flag else s

        else:
            raise TypeError("Invalid arguments")

    def __getitem__(self, _):
        a = torch.randint(0, self.max_num, (1,)).item()
        b = torch.randint(0, self.max_num, (1,)).item()
        c = a + b

        # We have to pad the numbers to the max
        a = AddNumData.pad(a, self.max_size, False, True)
        b = AddNumData.pad(b, self.max_size, False, True)
        c = AddNumData.pad(c, self.max_size, True, True) #have been reversed

        tokens = []
        for idx, num in enumerate([a, b, c]):
            if idx == 0:
                if self.split_ab:
                    tokens.extend(list(num))
                else:
                    tokens.append(num)
                tokens.append('+')
            elif idx == 1:
                if self.split_ab:
                    tokens.extend(list(num))
                else:
                    tokens.append(num)
                tokens.append('=')
            else:
                tokens.extend(list(num))

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

        return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)