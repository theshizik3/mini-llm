import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import run_tokenize_sft, run_tokenize_pretrain
from src.configs import TokenizeConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sft")
    subparsers.add_parser("pretrain")

    args = parser.parse_args()

    match args.command:
        case "sft":
            config = TokenizeConfig.from_yaml("configs/sft/tokenize.yaml")
            run_tokenize_sft(config)
        case "pretrain":
            config = TokenizeConfig.from_yaml("configs/pretrain/tokenize.yaml")
            run_tokenize_pretrain(config)

if __name__ == "__main__":
    main()
