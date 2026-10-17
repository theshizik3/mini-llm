from abc import ABC, abstractmethod
from huggingface_hub import HfApi
from typing import Tuple

import pyarrow.parquet as pq
import re
import random
import httpx
import io


HF_ENDPOINT = "https://huggingface.co"

class SFTDataset(ABC):
    repo: str
    pat: str = ""
    columns: Tuple[str]

    def __init__(self, token, shuffle=False, max_samples=None):
        self.client = httpx.Client(follow_redirects=True)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

        shards = HfApi(token=token or None).list_repo_files(self.repo, repo_type="dataset")
        shards = [f for f in shards if f.endswith(".parquet")]
        shards.sort()
        if self.pat:
            rx = re.compile(self.pat)
            shards = [f for f in shards if rx.search(f)]
        if shuffle:
            random.shuffle(shards)

        self.samples = []
        for f in shards:
            self._read(f)
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _read(self, f):
        buf = io.BytesIO()
        try:
            url = f"{HF_ENDPOINT}/datasets/{self.repo}/resolve/main/{f}"
            with self.client.stream("GET", url, headers=self.headers) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    buf.write(chunk)
            buf.seek(0)
            pf = pq.ParquetFile(buf)
            try:
                for batch in pf.iter_batches(batch_size=2**16, columns=self.columns):
                    cols = [batch.column(name).to_pylist() for name in self.columns]
                    for row in zip(*cols):
                        sample = self.handler(row)
                        if sample:
                            self.samples.append(sample)
            finally:
                pf.close()
        finally:
            buf.close()

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @abstractmethod
    def handler(self, row):
        pass