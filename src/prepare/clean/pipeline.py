import duckdb
import kenlm
import pyarrow as pa

from pathlib import Path
from huggingface_hub import hf_hub_download


def make_perplexity_calc(model):
    def calc_perplexity(texts):
        ppls = []
        for s in texts.to_pylist():
            ppls.append(model.perplexity(s))
        return pa.array(ppls, type=pa.float32())
    return calc_perplexity

def run_clean(cfg):
    con = duckdb.connect(
        config={
            "temp_directory": "/tmp/duckdb_tmp",
            "max_temp_directory_size": "20GB",
            "memory_limit": "8GB",
            "preserve_insertion_order": False
        }
    )
    
    model_path = hf_hub_download(repo_id="JonaWhisper/kenlm-models", filename="ru.binary")
    model = kenlm.Model(model_path)
    
    calc_perplexity = make_perplexity_calc(model)
    
    con.create_function(
        "perplexity",
        calc_perplexity,
        parameters=[str],
        return_type=float,
        type="arrow"
    )
    
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open("src/sql/clean_datasets.sql", 'r') as f:
        q = f.read()
    
    rel = con.sql(q, params=[str(Path(cfg.input_dir) / "**/*.parquet"), cfg.min_ppl, cfg.max_ppl])
    rel.write_parquet(
        file_name=cfg.output_dir,
        per_thread_output=True,
        filename_pattern="train_{i}"
    )
