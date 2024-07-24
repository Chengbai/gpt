# Date: 2024-07-21

import torch

from torch.utils.data import Dataset, DataLoader
from tokenizer import BaseTokenizer


class TSDataset(Dataset):
    def __init__(self, doc: str, tokenizer: BaseTokenizer, sequence_len: int):
        assert doc
        assert tokenizer is not None
        assert sequence_len > 0
        self.doc = doc
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(doc)
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.tokens) - self.sequence_len - 1

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Extract the index-th feature and label.
            - x: feature tensor w/ shape of [C]
            - y: label tensor w/ shape of [1]
        """
        index = index % len(self)
        tokens = self.tokens[index : (index + self.sequence_len + 1)]
        assert (
            len(tokens) == self.sequence_len + 1
        ), f"tokens: {len(tokens)}, expected length: {self.sequence_len+1}"

        x = tokens[:-1]
        y = tokens[1:]
        return x, y
