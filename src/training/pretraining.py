import os
import numpy as np
import torch
import torch.nn.functional as F
import math

from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from src.config import ModelConfig, TrainConfig
from src.model import LLM
from src.tokenizer import EOS_TOKEN
from src.datasets.pretraining import *


def dataset_iterator(cfg, tokenizer):
    ds = MultiDataset(
        Wikipedia(cfg.hf_token, 11, shuffle=True),
        Wikireading(cfg.hf_token, 2, shuffle=True),
        CulturaX(cfg.hf_token, 1, shuffle=True),
        TaigaProza(cfg.hf_token, 3, shuffle=True),
        MathD0rj(cfg.hf_token, shuffle=True),
        TaigaStihi(cfg.hf_token, 1, shuffle=True),
        ZeroAgency(cfg.hf_token, shuffle=True),
        Kolmogorov(cfg.hf_token, shuffle=True),
        shuffle=True,
        weights=[30, 27.4, 20, 10, 6, 3, 3, 0.6]
    )
    eos = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    shards = []
    for shard in ds:
        shards.append(shard)
        if len(shards) >= cfg.tokenizer_batch_size:
            enc = tokenizer(shards, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for ids in enc:
                yield np.asarray(ids + [eos], dtype=np.int32)
            shards = []

def batch_train(batch, context, model, device, it, optimizer, token_buffer=[]):
    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0
    for _ in range(batch):
        while len(token_buffer) < context + 1:
            token_buffer.extend(next(it).tolist())

        chunk = token_buffer[:context + 1]
        token_buffer = token_buffer[context:]

        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:],  dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss = loss / batch
        loss.backward()
        batch_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return batch_loss, token_buffer

def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    os.makedirs("checkpoints", exist_ok=True)

    model = LLM(
        vocab_size=model_cfg.vocab_size, 
        d_model=model_cfg.hidden_layer,
        seq_len=model_cfg.context, 
        n_heads=model_cfg.heads, 
        n_kv_heads=model_cfg.kv_heads,
        n_layers=model_cfg.layers
    )
    model.to(dtype=getattr(torch, train_cfg.dtype))
    model.to(device=train_cfg.device)
    model.train()
    
    if hasattr(torch, "compile") and train_cfg.device == "cuda":
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.pretrain_lr)
    w = train_cfg.warmup_steps
    t = train_cfg.sched_steps
    sched = LambdaLR(optimizer, lambda step: (
        step / w if step < w
        else train_cfg.min_lr_rate + 0.5 * (1 - train_cfg.min_lr_rate) * (1 + math.cos(math.pi * min((step - w) / (t - w), 1)))
    ))

    data_iter = dataset_iterator(train_cfg, tokenizer)

    token_buffer = []

    step = 0
    loss = 0.0

    while True:
        try:
            l, b = batch_train(
                batch=train_cfg.batch_size,
                context=model_cfg.context,
                model=model,
                device=train_cfg.device,
                it=data_iter,
                optimizer=optimizer,
                token_buffer=token_buffer
            )
            loss += l
            token_buffer = b
        except StopIteration:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, f"checkpoints/step_{step}.pt")
            break

        step += 1

        sched.step()

        if step % train_cfg.log_every == 0:
            avg_loss = loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.2f} | lr {sched.get_last_lr()[0]:.2e}")
            loss = 0.0

        if step % train_cfg.save_every == 0:
            ckpt_path = f"checkpoints/step_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")
