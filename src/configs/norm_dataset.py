from pydantic import BaseModel
from typing import Literal, Dict, Any


class NormDatasetConfig(BaseModel):
    name: str
    type: Literal["sft", "pretrain"]
    pat: str = "*"
    max_rows: int = 0
    reader: str
    reader_args: Dict[str, Any]
    shuffle: bool = False
    sharegpt: bool = False
