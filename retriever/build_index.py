"""
retriever/build_index.py
========================
Standalone script to build (or rebuild) the FAISS index from catalog.json.

Usage (always run from project root):
  python retriever/build_index.py
  python retriever/build_index.py --force
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path so imports work from any CWD
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from retriever.retriever import build_index  # noqa: E402  (after sys.path fix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from catalog.json")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    args = parser.parse_args()

    build_index(force=args.force)
    print("Done.")
