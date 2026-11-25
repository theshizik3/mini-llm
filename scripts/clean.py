import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import run_clean
from src.configs import CleanConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("pretrain")

    args = parser.parse_args()
    
    match args.command:
        case "pretrain":
            config = CleanConfig.from_yaml("configs/pretrain/clean.yaml")
    
    run_clean(config)
    
if __name__ == "__main__":
    main()
