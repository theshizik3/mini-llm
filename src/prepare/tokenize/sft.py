from src.readers import ColumnParquetReader, DirReader
from src.configs import TokenizeConfig
from src.tokenizer import EOS_TOKEN

from transformers import AutoTokenizer
from pathlib import Path

import numpy as np


def tokenize_sft():
    cfg = TokenizeConfig()
    
    input_path = Path(cfg.input_dir) / "sft"
    output_path = Path(cfg.output_dir) / "sft"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    it = iter(
        DirReader(
            reader=ColumnParquetReader,
            dir=input_path,
            pat="*.parquet",
            shuffle=cfg.shuffle,
            batch_size=512,
            column="conversation"
        )
    )
    
    eos = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    
    shard_n = 0
    shard_len = 0
    
    while True:
        with open(output_path / f"train_{shard_n:02d}.bin", 'a') as f:
            while shard_len < cfg.shard_size:
                try:
                    row = next(it)
                except StopIteration:
                    break
                messages = []
                for m in row:
                    role = m.get("role")
                    content = m.get("content")
                    messages.append({"role": role, "content": content})
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                ids.append(eos)
                
                arr = np.asarray(ids, dtype=np.int32)
                arr.tofile(f)
                
                shard_len += len(ids)
            else:
                shard_len = 0
                shard_n += 1
                continue
            break
