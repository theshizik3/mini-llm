import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train.pretrain import train


def main():
    train()

if __name__ == "__main__":
    main()
