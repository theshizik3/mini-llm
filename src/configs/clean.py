from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class CleanConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    input_dir: str
    output_dir: str
    
    min_ppl: float
    max_ppl: float
    
    hf_token: str
