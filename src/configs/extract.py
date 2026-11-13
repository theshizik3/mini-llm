from src.configs.extract_ds import ExtractDSConfig
from src.configs.base import BaseYAMLSettings

from pydantic_settings import SettingsConfigDict
from typing import List


class ExtractConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    shard_size: str
    output_dir: str
    datasets: List[ExtractDSConfig]
    hf_token: str
