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
    bos = tokenizer.bos_token_id
    
    rel = duckdb.sql("SELECT text, src FROM read_parquet(?)", params=[cfg.input_dir+"/**/*.parquet"])
    sample_idx = 0
    
    with wds.ShardWriter(str(output_path / "train_%d.tar"), maxsize=parse_size(cfg.shard_size, binary=True)) as sink:
        for batch in rel.fetch_record_batch(50_000):
            texts = batch.column("text").to_pylist()
            srcs = batch.column("src").to_pylist()

            encs = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for ids, src in zip(encs, srcs):
                ids = [bos] + ids + [eos]
                arr = np.asarray(ids, dtype=np.int32)
                sink.write({
                    "__key__": f"sample_{sample_idx:09d}",
                    "ids.npy": arr,
                    "src.txt": src,
                })
                sample_idx += 1
