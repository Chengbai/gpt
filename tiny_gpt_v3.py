# Date: 2024-07-24
# Cheng Bai

import torch
from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter

from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Following modules are in this code base.
from config import Config
from dataset import TSDataset
from data_source import DirectoryTxtDataSource
from gpt_model import TinyGPT
from tokenizer import CharTokenizer
from utils import load_text_from_file, get_prefered_device, save_model, viz_model


def run():
    # Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = get_prefered_device()

    # Load txt
    root_dir = "/Users/test/"
    data_source = DirectoryTxtDataSource(
        dir_path=root_dir, file_patterns=Config.FILE_PATTERNS
    )
    dir_txt = data_source.txt
    print(len(dir_txt), dir_txt[:50])

    # Create Tokenizer
    tokenizer = CharTokenizer(dir_txt, device=device)
    print(f"Vocabulary: {len(tokenizer.vocabulary)}")
    print(tokenizer.encode("First Citizen:\nBefore"))
    print(tokenizer.decode(tokenizer.encode("First Citizen:\nBefore")))
    print(tokenizer.encode(dir_txt)[:100])

    # Dataset
    train_val_split = int(0.9 * len(dir_txt))
    train_doc, val_doc = (
        dir_txt[:train_val_split],
        dir_txt[train_val_split:],
    )
    train_dataset = TSDataset(
        doc=train_doc, tokenizer=tokenizer, sequence_len=Config.SEQUENCE_LEN
    )
    eval_dataset = TSDataset(
        doc=val_doc, tokenizer=tokenizer, sequence_len=Config.SEQUENCE_LEN
    )
    print(len(train_dataset), len(eval_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

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
    print(f"model parameters: {model.get_num_params(non_embedding=False)}")
    model = torch.compile(model)
    # x = torch.randint(low=0, high=len(tokenizer.vocabulary)-1, size = (BATCH_SIZE, T))
    # pred, loss = model(x)
    # print(pred.size(), pred[0][0])

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer = model.configure_optimizers(
        Config.WEIGHT_DECAY, Config.LEARNING_RATE, (Config.BETA1, Config.BETA2), device
    )

    # Before Model Training
    model.generate(
        tokenizer=tokenizer,
        max_len=Config.GEN_SEQUENCE_LENGTH,
        device=device,
        temperature=0.8,
        top_k=200,
    )
    # Model Training
    print(model)

    min_eval_loss = float("inf")
    with SummaryWriter() as writer:
        # Viz model
        # viz_model(model=model, eval_dataloader=eval_dataloader, writer=writer)

        # Train model
        train_loss, eval_loss = model.train_model(
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            writer=writer,
            eval_step_interval=Config.EVAL_STEP_INTERVAL,
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
        model.generate(
            tokenizer=tokenizer, max_len=Config.GEN_SEQUENCE_LENGTH, device=device
        )
        model_path = (
            f"tiny_gpy_final_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.chkpt"
        )
        eval_loss = model.eval_model(eval_dataloader=eval_dataloader)
        save_model(
            model_path=model_path,
            model=model,
            optimizer=optimizer,
            epoch=Config.EPOCHS,
            train_loss=train_loss,
            eval_loss=eval_loss,
        )


if __name__ == "__main__":
    run()
