import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import run_train_tokenizer
from src.configs import TokenizerConfig


def main():
    config = TokenizerConfig()
    run_train_tokenizer(config)

if __name__ == "__main__":
    main()
