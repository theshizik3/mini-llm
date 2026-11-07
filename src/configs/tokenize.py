from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class TokenizeConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/data/tokenize.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    input_dir: str
    output_dir: str
    shard_size: int
    tokenizer: str
    shuffle: bool = False
