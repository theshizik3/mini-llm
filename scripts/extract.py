import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import run_extract
from src.configs import ExtractConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("sft")
    subparsers.add_parser("pretrain")

    args = parser.parse_args()
    
    match args.command:
        case "sft":
            config = ExtractConfig.from_yaml("configs/sft/extract.yaml")
        case "pretrain":
            config = ExtractConfig.from_yaml("configs/pretrain/extract.yaml")
    
    run_extract(config)
    
if __name__ == "__main__":
    main()
