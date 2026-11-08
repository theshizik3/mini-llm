from src.configs import FetchConfig

from pathlib import Path
from huggingface_hub import HfApi

import re
import random
import requests


HF_ENDPOINT = "https://huggingface.co"

def fetch():
    cfg = FetchConfig()
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {"Authorization": f"Bearer {cfg.hf_token}"}
    
    for ds in cfg.datasets:
        shards = HfApi(token=cfg.hf_token).list_repo_files(ds.name, repo_type="dataset")
        shards.sort()
        rx = re.compile(ds.pat)
        shards = [f for f in shards if rx.search(f)]
        if ds.files:
            if ds.random:
                random.shuffle(shards)
            shards = shards[:ds.files]
        
        for shard in shards:
            url = f"{HF_ENDPOINT}/datasets/{ds.name}/resolve/main/{shard}"
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(output_dir / ds.name / shard, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2**20):
                        f.write(chunk)
