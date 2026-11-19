"""Run the project pipeline.

Usage:
    python run.py data             Download NHTSA recalls and complaints into data/raw/
    python run.py eval-chunking    Compare RAG chunking strategies (see src/evaluate_chunking.py)
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1]

    if target == "data":
        from src import download_data
        download_data.main()
    elif target == "eval-chunking":
        from src import evaluate_chunking
        evaluate_chunking.main(sys.argv[2:])
    else:
        print(f"Unknown target: {target}")
        print(__doc__)


if __name__ == "__main__":
    main()