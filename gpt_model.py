# Date: 2024-07-21

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import inspect
from contextlib import nullcontext
from datetime import datetime, timezone
from tqdm import tqdm
from typing import Tuple

from attention import SelfAttentionBlck
from config import Config
from tokenizer import BaseTokenizer
from utils import save_model, get_lr


class TinyGPT(nn.Module):
    """
    TinyGPT is a language model.
      - use multi-heads self-attention transformer encoding as backbone
      - Inputs
        - x: (B, T): encoded string token

      - Outputs:
        - y: (B, 1): string token to be decoded to string
      - Configs:
        - num_embeddings: vocabulary size
        - embedding_dim: embedding dimention
        - sequence_len: the length of the time sequence / context
        - mult_heads: number of the multi-heads
        - att_blocks: number of the attention blocks
        - multi_head_count: the multi-head count

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sequence_length: int,
        att_blocks: int,
        multi_head_count: int,
        dropout: float,
        device: torch.device = torch.device("cpu"),
    ):
        assert num_embeddings > 0
        assert embedding_dim > 0
        assert sequence_length > 0
        assert att_blocks > 0
        assert multi_head_count > 0
        assert device is not None

        super().__init__()
        # token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,  # device=device
        )
        # position embeddings
        self.position_embeddings = nn.Embedding(
            num_embeddings=sequence_length,
            embedding_dim=embedding_dim,  # device=device
        )
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        self.backbone = nn.Sequential(
            *[
                SelfAttentionBlck(
                    emb=embedding_dim, heads=multi_head_count, dropout=dropout
                )
                for _ in range(att_blocks)
            ]
        )
        self.lm_head = nn.Linear(embedding_dim, num_embeddings, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.token_embeddings.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> torch.tensor:
        """
        x: (B, T)
        """
        B, T = x.size()
        token_emb = self.token_embeddings(x)  # (B, T) -> (B, T, Emb)
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)  # shape (t)
        pos_emb = self.position_embeddings(pos)  # position embeddings of shape (T, Emb)
        pos_emb = torch.unsqueeze(pos_emb, dim=0)  # (1, T, Emb)
        x = self.dropout(token_emb + pos_emb)
        x = self.backbone(x)  # (B, T, Emb) -> (B, T, Emb)
        x = self.lm_head(x)  # (B, T, Emb) -> (B, T, VOCAB)

        if y is None:
            loss = None
        else:
            # loss = F.cross_entropy(x[:, -1, :], y)
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1)

        return x, loss

    def get_amp_context(self):
        dtype = "bfloat16"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]

        return nullcontext()
        # ctx = (
        #     nullcontext()
        #     if self.device == "cpu"
        #     else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        # )
        # return ctx

    def train_model(
        self,
        tokenizer: BaseTokenizer,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        writer: SummaryWriter,
        eval_step_interval: int = 1_000,
    ) -> Tuple[torch.tensor, torch.tensor]:
        assert tokenizer is not None
        assert optimizer is not None
        assert train_dataloader is not None
        assert eval_dataloader is not None
        assert writer is not None

        amp_context = self.get_amp_context()
        print(f"steps per Epoch: {len(train_dataloader)}")
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/tiny_gpt"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            global_step = 0
            for epoch in range(Config.EPOCHS):
                for idx, batch_data in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
                ):
                    prof.step()

                    lr = get_lr(idx)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    x_batch, y_batch = batch_data
                    # print(x_batch.size(), y_batch.size())
                    with amp_context:
                        _, train_loss = self(x=x_batch, y=y_batch)

                    if idx % eval_step_interval == 0:
                        eval_loss = self.eval_model(eval_dataloader=eval_dataloader)
                        print(f"train loss: {train_loss}, eval loss: {eval_loss}")
                        # writer.add_scalar(
                        #     tag="Loss/train",
                        #     scalar_value=train_loss,
                        #     global_step=global_step,
                        # )
                        # writer.add_scalar(
                        #     tag="Loss/eval", scalar_value=eval_loss, global_step=global_step
                        # )
                        writer.add_scalars(
                            main_tag="tiny_gpt",
                            tag_scalar_dict={
                                "Loss/train": train_loss,
                                "Loss/eval": eval_loss,
                            },
                            global_step=global_step,
                        )
                        self.generate(
                            tokenizer=tokenizer,
                            max_len=Config.GEN_SEQUENCE_LENGTH,
                            device=self.device,
                        )

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    global_step += 1

                # Save epoch model
                model_path = f"tiny_gpy_{epoch}_{datetime.now(timezone.utc).strftime('%m_%d_%Y_%H_%M_%S')}.chkpt"
                save_model(
                    model_path=model_path,
                    model=self,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    eval_loss=eval_loss,
                )

        return train_loss, eval_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "mps"  # "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def eval_model(self, eval_dataloader) -> torch.tensor:
        assert eval_dataloader is not None

        amp_context = self.get_amp_context()

        self.eval()
        losses = []
        for idx, batch_data in tqdm(
            enumerate(eval_dataloader), total=len(eval_dataloader)
        ):
            if idx > Config.EVAL_SEQUENCE_LENGTH:
                break

            x_batch, y_batch = batch_data
            # print(x_batch.size(), y_batch.size())
            with amp_context:
                _, loss = self(x=x_batch, y=y_batch)
            losses.append(loss)
        avg_loss = torch.mean(torch.tensor(losses))
        self.train()
        return avg_loss

    @torch.no_grad()
    def generate(
        self,
        tokenizer: BaseTokenizer,
        max_len: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> str:
        assert tokenizer is not None
        assert max_len > 0
        assert device is not None
        self.eval()

        pred = []
        # context = torch.randint(
        #     low=0,
        #     high=len(tokenizer.vocabulary) - 1,
        #     size=(1, Config.SEQUENCE_LEN),
        #     device=device,
        # )
        start = "\n"
        context = torch.unsqueeze(tokenizer.encode(start), dim=0)
        for _ in tqdm(range(max_len)):
            logits, _ = self(
                context[:, -Config.SEQUENCE_LEN :]
            )  # logits: (1, t, VOCAB)
            logits = (
                logits[:, -1, :] / temperature
            )  # logits : (1, VOCAB) and is the last one in the sequence
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size()[-1]))
                logits[logits < v[:, -1]] = -float("inf")  # (1, VOCAB)

            prob = F.softmax(logits, dim=-1)  # prob: (1, VOCAB)
            # print(f"prob: {prob.size()}")
            predict_encod = torch.multinomial(prob, num_samples=1)  # (1, 1)
            context = torch.cat([context, predict_encod], dim=1)
            # print(f"predict_encod: {predict_encod}")

            pred.append(int(predict_encod))
            # print(pred)
        print("\n=====================================================")
        print(tokenizer.decode(torch.tensor(pred, device=device)))
        print("=====================================================\n")

        self.train()
