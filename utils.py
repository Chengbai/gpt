# Date: 2024-07-21

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from pathlib import Path
from config import Config
from typing import List


def load_text_from_file(file_path: Path) -> List[str]:
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_model(
    model_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: torch.tensor,
    eval_loss: torch.tensor,
):
    model_meta = {
        "epoch": epoch,
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
