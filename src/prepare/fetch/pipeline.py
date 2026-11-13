import globre

from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path


def run_fetch(cfg):
    output_dir = Path(cfg.output_dir)
    for ds in cfg.datasets:
        pat = globre.compile(ds.pat)
        files = [f for f in list_repo_files(repo_id=ds.name, repo_type="dataset") if pat.match(f)][:ds.n_files]
        for f in files:
            hf_hub_download(
                repo_id=ds.name,
                filename=f,
                revision="main",
                repo_type="dataset",
                token=cfg.hf_token,
                local_dir=output_dir / ds.name,
            )
