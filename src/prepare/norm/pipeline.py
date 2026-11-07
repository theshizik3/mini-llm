from src.configs import NormConfig
from src.readers import ColumnParquetReader, DirReader
from llm_utils.chat_format.transform import transform_messages

from pathlib import Path
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq


def norm():
    cfg = NormConfig()
    
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    
    for ds in cfg.datasets:
        match ds.reader:
            case "ColumnParquetReader":
                reader = ColumnParquetReader
            case _:
                raise TypeError("Reader not found")
        
        n_shard = 0
        
        ds_output_dir = output_dir / ds.type / ds.name
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        
        it = iter(
            DirReader(
                reader=reader,
                dir=input_dir / ds.name,
                pat=ds.pat,
                shuffle=ds.shuffle,
                **ds.reader_args
            )
        )
        
        match ds.type:
            case "sft":
                t = pa.list_(pa.struct([
                    ('role', pa.string()),
                    ('content', pa.string())
                ]))
                column = "conversation"
            case "pretrain":
                t = pa.string()
                column = "text"
            case _:
                raise TypeError("Invalid dataset type")
        
        while True:
            chunk = tuple(islice(it, cfg.shard_rows))
            if not chunk:
                break
            
            if ds.sharegpt:
                chunk = tuple(transform_messages({"conversations": conv}, frm="sharegpt", to="chatml") for conv in chunk)
            
            filename = ds_output_dir / f"train_{n_shard:02d}.parquet"
            table = pa.Table.from_pydict({column: chunk}, schema=pa.schema([(column, t)]))
            pq.write_table(table, filename)
                
            n_shard += 1
