import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import run_sft, run_pretrain
from src.configs import ModelConfig, TrainConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sft")
    subparsers.add_parser("pretrain")

    args = parser.parse_args()

    model_cfg = ModelConfig()

    match args.command:
        case "sft":
            train_cfg = TrainConfig.from_yaml("configs/sft/train.yaml")
            run_sft(model_cfg, train_cfg)
        case "pretrain":
            train_cfg = TrainConfig.from_yaml("configs/pretrain/train.yaml")
            run_pretrain(model_cfg, train_cfg)

if __name__ == "__main__":
    main()
