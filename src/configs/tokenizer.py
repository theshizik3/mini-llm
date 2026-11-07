from pydantic_settings import SettingsConfigDict
from typing import List

from src.configs.base import BaseYAMLSettings


class TokenizerConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/tokenizer.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    path: str = "./tokenizer"
    dataset_path: str
    
    vocab: int
    batch_size: int 
    train_seq_len: int
