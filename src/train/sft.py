import math
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.model import build_model


def make_collate_fn(pad_id, eos_id, seq_len):
    def collate(batch):
        seqs, labs = [], []
        for ex in batch:
            ids = ex["input_ids"]
            m = ex["assistant_mask"]
            if len(ids) > seq_len + 1:
                ids = ids[-seq_len:] + [eos_id]
                m = m[-seq_len:] + [True]
            lab = [t if keep else -100 for t, keep in zip(ids, m)]

            seqs.append(ids)
            labs.append(lab)

        L = max(len(s) for s in seqs)
        B = len(seqs)
        x = torch.full((B, L), pad_id, dtype=torch.long)
        y = torch.full((B, L), -100, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            x[i, :len(s)] = torch.tensor(s)
            y[i, :len(s)] = torch.tensor(l)

        return x[:, :-1], y[:, 1:]

    return collate

def step_train(model, device, it, optimizer):
    optimizer.zero_grad(set_to_none=True)

    x, y = next(it)
    
    x = x.to(device)
    y = y.to(device)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def run_sft(model_cfg, train_cfg):
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
    
    ds = load_from_disk(train_cfg.dataset_dir)
    loader = iter(
        DataLoader(
            dataset=ds, 
            batch_size=train_cfg.batch_size, 
            shuffle=True, 
            num_workers=4, 
            collate_fn=make_collate_fn(
                train_cfg.pad_id, 
                train_cfg.eos_id, 
                model_cfg.context
            )
        )
    )

    step = 0
    loss = 0.0

    while True:
        sched.step()

        try:
            loss += step_train(model, train_cfg.device, loader, optimizer)
        except StopIteration:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"completed at step: {step}")
            break

        step += 1

        if step % train_cfg.log_every == 0:
            avg_loss = loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f} | lr {sched.get_last_lr()[0]:.2e}")
            loss = 0.0

        if step % train_cfg.save_every == 0:
            torch.save(model.state_dict(), ckpts_path / f"step_{step}.pt")
            print(f"saved checkpoint: step_{step}.pt")
            