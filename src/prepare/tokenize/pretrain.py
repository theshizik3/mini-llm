from transformers import AutoTokenizer
from pathlib import Path
from humanfriendly import parse_size

import webdataset as wds
import numpy as np
import duckdb


def run_tokenize_pretrain(cfg):
    output_path = Path(cfg.output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    eos = tokenizer.eos_token_id
    
    cur = duckdb.sql("SELECT text FROM read_parquet(?)", params=[cfg.input_dir+"/**/*.parquet"])
    sample_idx = 0
    
    offset = 0
    
    with wds.ShardWriter(str(output_path / "train_%02d.tar"), maxsize=parse_size(cfg.shard_size, binary=True)) as sink:
        while True:
            batch = cur.offset(offset).limit(50_000).fetchall()
            if not batch:
                break
            offset += len(batch)
            for row in batch:
                ids = tokenizer(row[0], add_special_tokens=False, return_attention_mask=False)["input_ids"]
                ids.append(eos)

                arr = np.asarray(ids, dtype=np.int32)
                sink.write({
                    "__key__": f"sample_{sample_idx:09d}",
                    "ids.npy": arr,
                })
                sample_idx += 1
