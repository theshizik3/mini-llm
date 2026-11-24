from pathlib import Path

import duckdb


def run_dedup(cfg):
    con = duckdb.connect(
        config={
            "temp_directory": "/tmp/duckdb_tmp",
            "max_temp_directory_size": "20GB",
            "memory_limit": "8GB",
            "preserve_insertion_order": False
        }
    )
    
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open("src/sql/dedup_datasets.sql", 'r') as f:
        q = f.read()
    
    rel = con.sql(q, params=[str(Path(cfg.input_dir) / "**/*.parquet")])
    rel.write_parquet(
        file_name=cfg.output_dir,
        per_thread_output=True,
        filename_pattern="train_{i}"
    )
