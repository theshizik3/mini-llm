from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class TrainConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file="configs/train.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    batch_size: int
    
    pretrain_lr: float
    sft_lr: float
    min_lr_rate: float = 0.1
    
    warmup_steps: int = 500
    sched_steps: int = 500_000
    
    device: str
    dtype: str
    compile: bool = False
    
    log_every: int = 10
    save_every: int = 1000
    
    sft_init_ckpt: str
    dataset_dir: str
    checkpoints_dir: str = "./checkpoints"
    
    eos_token: int
    pad_token: int
