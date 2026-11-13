import torch
import torch.nn.functional as F
import webdataset as wds
import math

from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

from src.model import build_model


def dataset_iterator(path, context):
    path = Path(path)
    dataset = wds.WebDataset(map(str, path.rglob("*.tar"))).decode()

    arr = []
    
    for sample in dataset:
        arr.extend(sample["ids.npy"].tolist())
        while len(arr) > context:
            yield arr[:context], arr[1:context + 1]
            del arr[:context]

def batch_train(batch, model, device, it, optimizer):
    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0
    for _ in range(batch):
        x, y = next(it)

        x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(y,  dtype=torch.long, device=device).unsqueeze(0)

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

def run_pretrain(model_cfg, train_cfg):    
    ckpts_path = Path(train_cfg.checkpoints_dir)
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
        init_ckpt=train_cfg.init_ckpt
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    w = train_cfg.warmup_steps
    t = train_cfg.sched_steps
    sched = LambdaLR(optimizer, lambda step: (
        step / w if step < w
        else train_cfg.min_lr_rate + 0.5 * (1 - train_cfg.min_lr_rate) * (1 + math.cos(math.pi * min((step - w) / (t - w), 1)))
    ))

    data_iter = iter(dataset_iterator(train_cfg.dataset_dir, model_cfg.context))

    step = 0
    loss = 0.0

    while True:
        sched.step()

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

        if step % train_cfg.log_every == 0:
            avg_loss = loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.2f} | lr {sched.get_last_lr()[0]:.2e}")
            loss = 0.0

        if step % train_cfg.save_every == 0:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"saved checkpoint: step_{step}.pt")
