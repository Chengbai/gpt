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

    WEIGHT_DECAY = 1e-1
    BETA1 = 0.9
    BETA2 = 0.95

    DECAY_LR = True  # whether to decay the learning rate
    WARMUP_ITERS = 2000  # how many steps to warm up for
    LR_DECAY_ITERS = 600000  # should be ~= max_iters per Chinchilla
    LEARNING_RATE = 6e-4
    MIN_LR = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    FILE_PATTERNS = ["*.py", "*.cc", "*.hh"]

    GEN_SEQUENCE_LENGTH = 200

    EVAL_STEP_INTERVAL = 500
    EVAL_BATCHES = 200
    EVAL_SEQUENCE_LENGTH = 200


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
