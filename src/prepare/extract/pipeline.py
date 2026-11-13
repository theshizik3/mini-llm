from jinja2 import Template

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
    con.execute(
        "CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?)",
        parameters=[cfg.hf_token]
    )
    
    with open("src/sql/extract_hf_datasets.sql.jinja", 'r') as f:
        t = Template(f.read())
    q = t.render(ds_n=len(cfg.datasets))
    
    params = []
    for ds in cfg.datasets:
        params.append(f"^{ds.column}$")
        path = f"hf://datasets/{ds.name}@~parquet/{ds.pat}"
        params.append(path)
        params.append(ds.limit)
    
    rel = con.sql(q, params=params)

    rel.write_parquet(
        file_name=cfg.output_dir,
        per_thread_output=True,
        row_group_size_bytes=cfg.shard_size
    )
