import math
import torch
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

from src.readers import EOSBinaryReader, DirReader
from src.configs import ModelConfig, TrainConfig
from src.model import build_model


def dataset_iterator(path, eos, pad, seq_len, dtype=np.int32):
    reader = DirReader(EOSBinaryReader, path, "*.bin", eos=eos, dtype=dtype)

    for dialog in reader:
        L = min(len(dialog) - 1, seq_len)
        start = len(dialog) - 1 - L

        x = dialog[start : start + L]
        y = dialog[start + 1 : start + L + 1]

        if seq_len > L:
            x = x.tolist() + [pad] * (seq_len - L)
            y = y.tolist() + [-100] * (seq_len - L)

        yield x, y

def batch_train(batch, model, device, it, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = []
    y = []
    for _ in range(batch):
        ids, labels = next(it)
        x.append(ids)
        y.append(labels)
    
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    ckpts_path = Path(train_cfg.checkpoints_dir) / "sft"
    ckpts_path.mkdir(parents=True, exist_ok=True)

    model = build_model(
        vocab=model_cfg.vocab_size,
        hidden_layer=model_cfg.hidden_layer,
        intermediate_size=model_cfg.intermediate_size,
        context=model_cfg.context,
        heads=model_cfg.heads,
        kv_heads=model_cfg.kv_heads,
        n_layers=model_cfg.layers,
        dtype=getattr(torch, train_cfg.dtype),
        device=train_cfg.device,
        compile_=train_cfg.compile,
        init_ckpt=train_cfg.sft_init_ckpt
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.sft_lr)
    
    w = train_cfg.warmup_steps
    t = train_cfg.sched_steps
    
    sched = LambdaLR(optimizer, lambda step: (
        step / w if step < w
        else train_cfg.min_lr_rate + 0.5 * (1 - train_cfg.min_lr_rate) * (1 + math.cos(math.pi * min((step - w) / (t - w), 1)))
    ))
    
    ds_path = Path(train_cfg.dataset_dir) / "sft"
    data_iter = dataset_iterator(
        path=ds_path, 
        eos=train_cfg.eos_token, 
        pad=train_cfg.pad_token,
        seq_len=model_cfg.context
    )

    step = 0
    loss = 0.0

    while True:
        try:
            loss += batch_train(train_cfg.batch_size, model, train_cfg.device, data_iter, optimizer)
        except StopIteration:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"completed at step: {step}")
            break

        step += 1

        sched.step()

        if step % train_cfg.log_every == 0:
            avg_loss = loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f} | lr {sched.get_last_lr()[0]:.2e}")
            loss = 0.0

        if step % train_cfg.save_every == 0:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"saved checkpoint: step_{step}.pt")
            