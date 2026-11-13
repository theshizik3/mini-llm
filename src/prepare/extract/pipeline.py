from jinja2 import Template
from pathlib import Path

import duckdb


def run_extract(cfg):
    con = duckdb.connect(
        config={
            "temp_directory": "/tmp/duckdb_tmp",
            "max_temp_directory_size": "20GB",
            "memory_limit": "4GB",
            "preserve_insertion_order": False
        }
    )
    
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open("src/sql/extract_datasets.sql.jinja", 'r') as f:
        t = Template(f.read())
    q = t.render(datasets=cfg.datasets)
    
    params = []
    for ds in cfg.datasets:
        params.append(f"^{ds.column}$")
        path = Path(cfg.input_dir) / ds.name / "**/*"
        params.append(str(path))
        params.append(ds.limit)
    
    rel = con.sql(q, params=params)

    rel.write_parquet(
        file_name=cfg.output_dir,
        per_thread_output=True,
        row_group_size_bytes=cfg.shard_size
    )
