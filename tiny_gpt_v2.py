import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Following modules are in this code base.
from config import Config
from dataset import TSDataset
from gpt_model import TinyGPT
from tokenizer import Tiktoken
from utils import load_text_from_file, get_prefered_device, save_model, viz_model

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = get_prefered_device()


# Load .txt
tinyshakespear_file = Path("tinyshakespeare.txt")
tinyshakespeare_text = load_text_from_file(tinyshakespear_file)
len(tinyshakespeare_text), tinyshakespeare_text[:50]

# Create Tokenizer
tokenizer = Tiktoken(tinyshakespeare_text, device=device)
print(tokenizer.encode("First Citizen:\nBefore"))
print(tokenizer.decode(tokenizer.encode("First Citizen:\nBefore")))
print(tokenizer.encode(tinyshakespeare_text)[:100])


# Dataset
train_val_split = int(0.9 * len(tinyshakespeare_text))
train_doc, val_doc = (
    tinyshakespeare_text[:train_val_split],
    tinyshakespeare_text[train_val_split:],
)
train_dataset = TSDataset(
    doc=train_doc, tokenizer=tokenizer, sequence_len=Config.SEQUENCE_LEN
)
eval_dataset = TSDataset(
    doc=val_doc, tokenizer=tokenizer, sequence_len=Config.SEQUENCE_LEN
)
print(len(train_dataset), len(eval_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# Model
model = TinyGPT(
    num_embeddings=len(tokenizer.vocabulary),
    embedding_dim=Config.EMB_DIM,
    sequence_length=Config.SEQUENCE_LEN,
    att_blocks=Config.ATT_BLOCKS,
    multi_head_count=Config.HC,
    dropout=Config.DROPOUT,
    device=device,
)
model = model.to(device=device)
print(f"model parameters: {model.get_num_params()}")

# x = torch.randint(low=0, high=len(tokenizer.vocabulary)-1, size = (BATCH_SIZE, T))
# pred, loss = model(x)
# print(pred.size(), pred[0][0])

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer = model.configure_optimizers(
    Config.WEIGHT_DECAY, Config.LEARNING_RATE, (Config.BETA1, Config.BETA2), device
)


# Before Model Training
model.generate(
    tokenizer=tokenizer, max_len=100, device=device, temperature=0.8, top_k=200
)
# Model Training
print(model)
min_eval_loss = float("inf")
with SummaryWriter() as writer:
    # Viz model
    viz_model(model=model, eval_dataloader=eval_dataloader, writer=writer)

    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_loss, eval_loss = model.train_model(
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        writer=writer,
        eval_step_interval=200,
    )
    writer.flush()

    if eval_loss < min_eval_loss:
        # update the running min eval loss
        min_eval_loss = eval_loss

        # save best model
        model_path = "tiny_gpy_best_.chkpt"
        save_model(
            model_path=model_path,
            model=model,
            optimizer=optimizer,
            epoch=Config.EPOCHS,
            train_loss=train_loss,
            eval_loss=eval_loss,
        )

    # Post model Training
    model.generate(tokenizer=tokenizer, max_len=500, device=device)
    model_path = f"tiny_gpy_final_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.chkpt"
    eval_loss = model.eval_model(eval_dataloader=eval_dataloader)
    save_model(
        model_path=model_path,
        model=model,
        optimizer=optimizer,
        epoch=Config.EPOCHS,
        train_loss=train_loss,
        eval_loss=eval_loss,
    )
