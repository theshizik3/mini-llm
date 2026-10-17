import os
import random
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from src.config import ModelConfig, TrainConfig
from src.model import LLM
from src.datasets.sft import Ru0x7o, RuAttnSigns, MultiSFTDataset
from src.tokenizer import EOS_TOKEN, PAD_TOKEN


def dataset_iterator(cfg, tokenizer, context):
    ds = MultiSFTDataset(
        Ru0x7o(cfg.hf_token, True),
        RuAttnSigns(cfg.hf_token, True),
        shuffle=True
    )
    eos = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    for sample in ds:
        prompt = tokenizer.apply_chat_template(sample[:-1], add_generation_prompt=True, tokenize=False)
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        c_ids = tokenizer.encode(sample[-1]["content"], add_special_tokens=False)
        ids = p_ids + c_ids + [eos]
    
        n_prompt = len(p_ids)
        if len(ids) > context:
            drop = len(ids) - context
            n_prompt = max(n_prompt - drop, 0)
            ids = ids[drop:]
    
        labels = [-100] * len(ids)
        for p in range(max(n_prompt - 1, 0), len(ids) - 1):
            labels[p] = ids[p + 1]
        yield ids, labels

def batch_train(batch, model, device, it, optimizer):
    optimizer.zero_grad(set_to_none=True)
    batch_ids = []
    batch_labels = []
    for _ in range(batch):
        ids, labels = next(it)
        batch_ids.append(ids)
        batch_labels.append(labels)
    T = max(len(x) for x in batch_ids)
    x = torch.full((len(batch_ids), T), PAD_TOKEN, dtype=torch.long, device=device)
    y = torch.full((len(batch_labels), T), -100, dtype=torch.long, device=device)
    for i, (ids, labels) in enumerate(zip(batch_ids, batch_labels)):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer)
    data_iter = dataset_iterator(train_cfg, tokenizer, model_cfg.context)

    model = LLM(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.hidden_layer,
        seq_len=model_cfg.context,
        n_heads=model_cfg.heads,
        n_kv_heads=model_cfg.kv_heads,
        n_layers=model_cfg.layers,
    )
    model.to(dtype=getattr(torch, train_cfg.dtype))
    ckpt = torch.load(train_cfg.init_ckpt, map_location=train_cfg.device)
    model.load_state_dict(ckpt["model"])
    model.head.weight = model.embed.weight
    model.to(device=train_cfg.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    step = 0
    loss = 0.0

    while True:
        try:
            loss += batch_train(train_cfg.batch_size, model, train_cfg.device, data_iter, optimizer)
        except StopIteration:
            torch.save(model.state_dict(), "checkpoints/sft_last.pt")
            break

        step += 1
            
        if step % train_cfg.log_every == 0:
            avg_loss = running_loss / train_cfg.log_every
            print(f"step {step} | loss {avg_loss:.4f}")
            running_loss = 0.0

        if step % train_cfg.save_every == 0:
            ckpt_path = f"checkpoints/sft_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")
            