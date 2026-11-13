from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class TrainConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    batch_size: int
    
    lr: float
    min_lr_rate: float = 0.1
    
    warmup_steps: int = 500
    sched_steps: int = 500_000
    
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    
    log_every: int = 10
    save_every: int = 1000
    
    dataset_dir: str
    init_ckpt: str = ""
    checkpoints_dir: str = "./checkpoints"
    
    eos_id: int | None = None
    pad_id: int | None = None
