import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import tokenize_sft, tokenize_pretrain

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sft", action="store_true")
    
    args = parser.parse_args()
    
    if args.sft:
        tokenize_sft()
    else:
        tokenize_pretrain()

if __name__ == "__main__":
    main()
