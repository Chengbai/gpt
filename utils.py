# Date: 2024-07-21

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import math
from pathlib import Path
from config import Config
from typing import List


def load_text_from_file(file_path: Path) -> List[str]:
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_model(
    version: str,
    model_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    train_loss: torch.tensor,
    eval_loss: torch.tensor,
):
    model_meta = {
        "version": version,
        "epoch": epoch,
        "global_step": global_step,
        "batch_size": Config.BATCH_SIZE,
        "sequence_len": Config.SEQUENCE_LEN,
        "emb_dim": Config.EMB_DIM,
        "head_count": Config.HC,
        "att_blocks": Config.ATT_BLOCKS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
    torch.save(
        model_meta,
        model_path,
    )
    print(f"Save model to: {model_path}")


@torch.no_grad()
def viz_model(model: nn.Module, eval_dataloader: DataLoader, writer: SummaryWriter):
    model.eval()

    for _, batch_data in enumerate(eval_dataloader):
        x_batch, y_batch = batch_data
        writer.add_graph(model, (x_batch, y_batch))
        writer.flush()
        break
    model.train()


def get_prefered_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() or torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int):
    # 1) linear warmup for warmup_iters steps
    if it < Config.WARMUP_ITERS:
        return Config.LEARNING_RATE * it / Config.WARMUP_ITERS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > Config.LR_DECAY_ITERS:
        return Config.MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - Config.WARMUP_ITERS) / (
        Config.LR_DECAY_ITERS - Config.WARMUP_ITERS
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return Config.MIN_LR + coeff * (Config.LEARNING_RATE - Config.MIN_LR)
