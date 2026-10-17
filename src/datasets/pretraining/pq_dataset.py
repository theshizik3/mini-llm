import io
import re
import pyarrow.parquet as pq
import httpx
import random

from abc import ABC
from huggingface_hub import HfApi


HF_ENDPOINT = "https://huggingface.co"

class ParquetDataset(ABC):
    repo: str
    pat: str
    column: str
    
    def __init__(self, token, max_files=None, shuffle=False):
        self.client = httpx.Client(follow_redirects=True)
        self.headers = {"Authorization": f"Bearer {token}"}
        
        shards = HfApi(token=token).list_repo_files(self.repo, repo_type="dataset")
        shards = [f for f in shards if f.endswith(".parquet")]
        shards.sort()
        if self.pat:
            rx = re.compile(self.pat)
            shards = [f for f in shards if rx.search(f)]
        if shuffle:
            random.shuffle(shards)
        self.shards = shards
        if max_files:
            self.max_files = max_files
        else:
            self.max_files = len(self)

    def __len__(self):
        return len(self.shards)
    
    def __iter__(self):
        for i in range(len(self.shards)):
            if i >= self.max_files:
                break
            buf = io.BytesIO()
            try:
                url = f"{HF_ENDPOINT}/datasets/{self.repo}/resolve/main/{self.shards[i]}"
                with self.client.stream("GET", url, headers=self.headers) as r:
                    r.raise_for_status()
                    for chunk in r.iter_bytes(chunk_size=2**20):
                        buf.write(chunk)
                        
                buf.seek(0)
                
                pf = pq.ParquetFile(buf)
                try:
                    for batch in pf.iter_batches(batch_size=2**16, columns=[self.column]):
                        for s in batch.column(self.column).to_pylist():
                            if isinstance(s, str) and s.strip():
                                yield s
                finally:
                    pf.close()
            finally:
                buf.close()
