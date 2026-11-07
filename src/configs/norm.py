from src.configs.norm_dataset import NormDatasetConfig
from src.configs.base import BaseYAMLSettings

from pydantic_settings import SettingsConfigDict
from typing import List


class NormConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/data/norm.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    shard_rows: int
    input_dir: str
    output_dir: str
    datasets: List[NormDatasetConfig]
