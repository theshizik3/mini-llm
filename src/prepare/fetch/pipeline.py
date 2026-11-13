import requests
import globre

from huggingface_hub import hf_hub_download
from pathlib import Path


def run_fetch(cfg):
    output_dir = Path(cfg.output_dir)
    for ds in cfg.datasets:
        pat = globre.compile(ds.pat)
        r = requests.get(
            f"https://huggingface.co/api/datasets/{ds.name}/tree/main?recursive=true",
            headers={"Authorization": f"Bearer {cfg.hf_token}"},
            timeout=5,
        )
        r.raise_for_status()
        files = [f["path"] for f in r.json() if f.get("type") == "file" and pat.match(f["path"])][:ds.n_files]
        for f in files:
            hf_hub_download(
                repo_id=ds.name,
                filename=f,
                revision="main",
                repo_type="dataset",
                token=cfg.hf_token,
                local_dir=output_dir / ds.name,
            )
