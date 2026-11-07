import torch
import torch.nn.functional as F
import math

from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

from src.configs import ModelConfig, TrainConfig
from src.model import build_model
from src.readers import DirReader, BlockBinaryReader


def batch_train(batch, model, device, it, optimizer):
    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0
    for _ in range(batch):
        chunk = next(it)

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
    return batch_loss

def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    ckpts_path = Path(train_cfg.checkpoints_dir) / "pretrain"
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
        compile_=train_cfg.compile
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.pretrain_lr)
    w = train_cfg.warmup_steps
    t = train_cfg.sched_steps
    sched = LambdaLR(optimizer, lambda step: (
        step / w if step < w
        else train_cfg.min_lr_rate + 0.5 * (1 - train_cfg.min_lr_rate) * (1 + math.cos(math.pi * min((step - w) / (t - w), 1)))
    ))

    ds_path = Path(train_cfg.dataset_dir) / "pretrain"
    data_iter = iter(
        DirReader(
            reader=BlockBinaryReader,
            dir=ds_path,
            pat="*.bin",
            seq_len=model_cfg.context+1,
        )
    )

    step = 0
    loss = 0.0

    while True:
        try:
            loss += batch_train(
                batch=train_cfg.batch_size,
                model=model,
                device=train_cfg.device,
                it=data_iter,
                optimizer=optimizer
            )
        except StopIteration:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"completed at step: {step}")
            break

        step += 1

        sched.step()

        if step % train_cfg.log_every == 0:
            avg_loss = loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.2f} | lr {sched.get_last_lr()[0]:.2e}")
            loss = 0.0

        if step % train_cfg.save_every == 0:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"saved checkpoint: step_{step}.pt")
