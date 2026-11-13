from src.configs.fetch_ds import FetchDSConfig
from src.configs.base import BaseYAMLSettings

from pydantic_settings import SettingsConfigDict
from typing import List


class FetchConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    output_dir: str
    datasets: List[FetchDSConfig]
    hf_token: str
