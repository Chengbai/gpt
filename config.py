# Date: 2024-07-21


class Config:
    # Model settings
    EPOCHS = 2
    BATCH_SIZE = 50
    SEQUENCE_LEN = 512
    EMB_DIM = 768
    HC = 4
    ATT_BLOCKS = 4
    DROPOUT = 0.2
    GRADIENT_ACCUMULATION_STEPS = 1


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
