from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class DedupConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/model.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    input_dir: str
    output_dir: str
