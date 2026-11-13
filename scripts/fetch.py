import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import run_fetch
from src.configs import FetchConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sft")
    subparsers.add_parser("pretrain")

    args = parser.parse_args()

    match args.command:
        case "sft":
            config = FetchConfig.from_yaml("configs/sft/fetch.yaml")
        case "pretrain":
            config = FetchConfig.from_yaml("configs/pretrain/fetch.yaml")
    run_fetch(config)

if __name__ == "__main__":
    main()
