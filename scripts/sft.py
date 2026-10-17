import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.sft import train


if __name__ == "__main__":
    train()
