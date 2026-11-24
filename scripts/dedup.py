import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import run_dedup
from src.configs import DedupConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("sft")
    subparsers.add_parser("pretrain")

    args = parser.parse_args()
    
    match args.command:
        case "sft":
            config = DedupConfig.from_yaml("configs/sft/dedup.yaml")
        case "pretrain":
            config = DedupConfig.from_yaml("configs/pretrain/dedup.yaml")
    
    run_dedup(config)
    
if __name__ == "__main__":
    main()
