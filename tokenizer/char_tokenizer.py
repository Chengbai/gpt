# Date: 2024-07-21

import torch

from tokenizer.base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self, doc: str, device: torch.device = torch.device("cpu")) -> None:
        assert doc
        assert device is not None
        super().__init__()

        self.vocabulary = sorted(list(set(doc)))
        self.stoi = {c: i for i, c in enumerate(self.vocabulary)}
        self.itos = {i: c for i, c in enumerate(self.vocabulary)}
        self.device = device

    def encode(self, input: str) -> torch.tensor:
        return torch.tensor(
            [self.stoi[c] for c in input], dtype=torch.long, device=self.device
        )

    def decode(self, input: torch.tensor) -> str:
        return "".join([self.itos[i] for i in input.tolist()])
