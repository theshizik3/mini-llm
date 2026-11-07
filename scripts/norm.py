import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare import norm


def main():
    norm()

if __name__ == "__main__":
    main()
