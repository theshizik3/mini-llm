from src.model.llm import LLM

import torch


def build_model(
        vocab, hidden_layer, intermediate_size, 
        context, heads, kv_heads, n_layers, 
        dtype, device, compile_, init_ckpt = ""
    ):
    model = LLM(
        vocab_size=vocab, 
        d_model=hidden_layer,
        intermediate_size=intermediate_size,
        seq_len=context, 
        n_heads=heads, 
        n_kv_heads=kv_heads,
        n_layers=n_layers
    )
    model.to(dtype=dtype)
    model.to(device=device)
    model.train()
    
    if compile_:
        model = torch.compile(model)
    
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location=device))
        model.head.weight = model.embed.weight
    
    return model
