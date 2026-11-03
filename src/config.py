from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class BaseYAMLSettings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (init_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))

class ModelConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/model.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    vocab_size: int
    hidden_layer: int
    context: int
    heads: int
    kv_heads: int
    layers: int

class TrainConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="configs/train.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    batch_size: int
    tokenizer_batch_size: int
    pretrain_lr: float
    sft_lr: float
    device: str
    hf_token: str
    tokenizer: str
    dtype: str
    log_every: int = 10
    save_every: int = 1000
    tokenizer_len: float = 10_000_000_000
    init_ckpt: str = ""
    warmup_steps: int = 500
    sched_steps: int = 500_000
    min_lr_rate: float = 0.1
    
class DatasetConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="configs/dataset/config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    output_dir: str
    shard_size: int
    tokenizer: str
