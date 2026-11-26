from pydantic_settings import SettingsConfigDict
from src.configs.base import BaseYAMLSettings


class TrainConfig(BaseYAMLSettings):
    model_config = SettingsConfigDict(
        yaml_file_encoding="utf-8",
        extra="ignore"
    )
    batch_size: int
    
    lr: float
    min_lr_rate: float
    
    warmup_steps: int
    sched_steps: int
    
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    
    save_every: int = 1000
    
    logs_dir: str = "./logs"
    dataset_dir: str
    checkpoints_dir: str = "./checkpoints"
    
    init_ckpt: str = ""
    
    eot_id: int | None = None
    pad_id: int | None = None
