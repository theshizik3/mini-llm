from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class ModelConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/model.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    vocab_size: int
    hidden_layer: int
    intermediate_size: int
    context: int
    heads: int
    kv_heads: int
    layers: int
