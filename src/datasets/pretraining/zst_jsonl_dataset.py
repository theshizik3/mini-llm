import io
import re
import httpx
import random
import zstandard as zstd
import json

from abc import ABC
from huggingface_hub import HfApi


HF_ENDPOINT = "https://huggingface.co"

class ZstJsonlDataset(ABC):
    repo: str
    pat: str = ""
    column: str
    
    def __init__(self, token, max_files=None, shuffle=False):
        self.client = httpx.Client(follow_redirects=True)
        self.headers = {"Authorization": f"Bearer {token}"}
        
        shards = HfApi(token=token).list_repo_files(self.repo, repo_type="dataset")
        shards = [f for f in shards if f.endswith(".jsonl.zst")]
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
                
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(buf) as reader:
                    text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                    for line in text_reader:
                        data = json.loads(line)
                        yield data[self.column]
            finally:
                buf.close()
