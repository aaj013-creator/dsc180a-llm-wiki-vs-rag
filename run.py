"""Run the project pipeline.

Usage:
    python run.py data    Download NHTSA recalls and complaints into data/raw/
"""

import sys

from src import download_data


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1]

    if target == "data":
        download_data.main()
    else:
        print(f"Unknown target: {target}")
        print(__doc__)


if __name__ == "__main__":
    main()