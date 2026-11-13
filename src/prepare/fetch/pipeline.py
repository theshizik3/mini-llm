import requests

from huggingface_hub import hf_hub_download
from pathlib import Path


def run_fetch(cfg):
    output_dir = Path(cfg.output_dir)
    for ds in cfg.datasets:
        r = requests.get(
            f"https://huggingface.co/api/datasets/{ds.name}/parquet",
            headers={"Authorization": f"Bearer {cfg.hf_token}"},
            timeout=5,
        )
        r.raise_for_status()
        files = [f["file"] for f in r.json()["parquet_files"] if (not ds.config or f["config"] == ds.config)][:ds.n_files]
        for f in files:
            hf_hub_download(
                repo_id=ds.name,
                filename=f,
                revision="refs/convert/parquet",
                repo_type="dataset",
                token=cfg.hf_token,
                local_dir=output_dir / ds.name,
            )
