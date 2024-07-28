import logging
import torch

from collections import Counter
from config import Config
from data_source import DirectoryTxtDataSource
from tokenizer.base_tokenizer import BaseTokenizer

from tqdm import tqdm
from typing import Dict, Iterable, List, Tuple, Self

logging.basicConfig(level=logging.INFO)


# BPE
def bpe(data: List[int], vocab_size: int) -> Dict[Tuple[int, int], int]:
    assert data
    assert vocab_size > 256

    def _get_most_frequent_pair(data: List[int]) -> Tuple[int, int]:
        counter = {}
        most_frequent_pair = None
        for i in range(len(data) - 1):
            key = (data[i], data[i + 1])
            if key not in counter:
                counter[key] = 1
            else:
                counter[key] += 1

            if not most_frequent_pair or counter[most_frequent_pair] < counter[key]:
                most_frequent_pair = key
        return most_frequent_pair

    def _bpe_encode(
        data: List[int], pair: Tuple[int, int], pair_encode: int
    ) -> List[int]:
        i = 0
        bpe_encoded_data = []
        while i < len(data) - 1:
            if data[i] == pair[0] and data[i + 1] == pair[1]:
                bpe_encoded_data.append(pair_encode)
                i += 2
            else:
                bpe_encoded_data.append(data[i])
                i += 1
        if i < len(data):
            bpe_encoded_data.append(data[i])
        return bpe_encoded_data

    extra_encode = 256
    extra_encodes = {}
    for i in tqdm(range(vocab_size - 256)):
        pair = _get_most_frequent_pair(data)
        logging.debug(f"original: {data}")
        data = _bpe_encode(data, pair, extra_encode)
        logging.debug(f"bpe: {data}")
        extra_encodes[pair] = extra_encode
        extra_encode += 1
    return extra_encodes


class BasicTokenizer(BaseTokenizer):
    def __init__(self, extra_encodes: Dict[Tuple[int, int], int]):
        super().__init__()
        assert extra_encodes is not None
        self.extra_encodes = extra_encodes
        self.decodes = self._init_decode_map()

    def _init_decode_map(self):
        decode_map = {}
        for pair, encode in self.extra_encodes.items():
            decode = []
            for ele in pair:
                if ele in decode_map:
                    decode.extend(decode_map[ele])
                else:
                    decode.append(ele)
            decode_map[encode] = decode
        return decode_map

    # Virtual method
    def encode(self, input: str) -> torch.tensor:
        assert input
        data = input.encode(encoding="utf-8")
        encoded_data = []
        for pair, encode in self.extra_encodes.items():
            i = 0
            while i < len(data) - 1:
                if data[i] == pair[0] and data[i + 1] == pair[1]:
                    encoded_data.append(encode)
                    i += 2
                else:
                    encoded_data.append(data[i])
                    i += 1
            if i < len(data):
                encoded_data.append(data[-1])
            data, encoded_data = encoded_data, []
        return torch.tensor(data, dtype=torch.long)

    # Virtual method
    def decode(self, input: torch.tensor) -> str:
        assert input is not None
        data = input.tolist()
        result = []
        for item in data:
            if item in self.decodes:
                result.extend(self.decodes[item])
            else:
                result.append(item)
        return bytes(result).decode("utf-8", errors="replace")

    @staticmethod
    def build_from_string(input: str, vocab_size: int) -> Self:
        assert str
        assert vocab_size > 256
        encoded_data = input.encode(encoding="utf-8")
        logging.debug(
            f"encoded_data: {[encoded_data[i] for i in range(len(encoded_data))]}"
        )

        extra_encodes = bpe(data=encoded_data, vocab_size=vocab_size)
        logging.debug(f"extra_encodes: {extra_encodes}")
        return BasicTokenizer(extra_encodes=extra_encodes)


if __name__ == "__main__":
    # Train tokenizer
    test_str = """Had same issue in OSX when updating tensoflow to 1.13 using conda.

    Solution 1: /gcamargo worked but 3x slower per training epoch.
    Solution 2: /sjcoding worked and removed serious warining but also 3x slower in training.
    Solution 3: that restored performance was: Install pip in new conda env and use pip to install tensorflow. Using conda-forge also worked but version of tf is old.
    Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
    # """
    # test_str = "To remove all dimensions of size 1, use a.squeeze().tolist()."

    # Load txt
    # root_dir = "/Users/chengbai/src/pytorch"
    # data_source = DirectoryTxtDataSource(
    #     dir_path=root_dir, file_patterns=Config.FILE_PATTERNS
    # )
    # dir_txt = data_source.txt
    # logging.info(f"dir_txt: {len(dir_txt)}, example: {dir_txt[:50]}")
    # test_str = dir_txt

    tokenizer = BasicTokenizer.build_from_string(test_str, vocab_size=276)

    # Usage tokenizer
    input = "To remove all dimensions of size 1, use a.squeeze().tolist()."
    encode_data = tokenizer.encode(input)
    decode_data = tokenizer.decode(encode_data)
    logging.info(f"input: {len(input)}, {input}")
    logging.info(
        f"encode_data: {len(encode_data)}, compress rate: {len(input)/len(encode_data)}"
    )
    logging.info(f"decode_data: {len(decode_data)}, {decode_data}")

    print(tokenizer.decode(torch.tensor([128])))
