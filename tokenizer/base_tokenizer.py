# Date: 2024-07-21

import torch


class BaseTokenizer:
    def __init__(self):
        self.vocabulary = None

    # Virtual method
    def encode(self, input: str) -> torch.tensor:
        raise Exception("`encode` is not implemented")

    # Virtual method
    def decode(self, input: torch.tensor) -> str:
        raise Exception("`decode` is not implemented")
