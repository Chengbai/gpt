# Date: 2024-07-21

import torch

import tiktoken
from tokenizer.base_tokenizer import BaseTokenizer


class Tiktoken(BaseTokenizer):
    def __init__(self, doc: str, device: torch.device = torch.device("cpu")) -> None:
        assert doc
        assert device is not None
        super().__init__()

        self.enc = tiktoken.get_encoding("gpt2")
        tokens = self.enc.encode(doc)
        self.vocabulary = sorted(list(set(tokens)))
        self.ttoi = {t: i for i, t in enumerate(self.vocabulary)}
        self.itot = {i: t for i, t in enumerate(self.vocabulary)}
        self.device = device

    def encode(self, input: str) -> torch.tensor:
        tokens = self.enc.encode(input)

        return torch.tensor(
            [self.ttoi[t] for t in tokens], dtype=torch.long, device=self.device
        )

    def decode(self, input: torch.tensor) -> str:
        return "".join(self.enc.decode([self.itot[i] for i in input.tolist()]))
