"""Pre-download models and datasets. Run on a node WITH internet (the login node).

GPU nodes on Grace have no internet, so jobs run with HF_HUB_OFFLINE=1 and read
everything from the cache ($HF_HOME) filled by this script.

  python scripts/download_assets.py
  python scripts/download_assets.py --models Qwen/Qwen2.5-1.5B
"""

import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download

from codealign.data import KODCODE, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"])
args = ap.parse_args()

for split in ("train", "eval"):
    print(f"dataset {split}: {len(load_split(split))} problems")
print(f"dataset {KODCODE}: {len(load_dataset(KODCODE, split='train'))} rows")  # ~2.6 GB, for large SFT
for model in args.models:
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
    print(f"model {model}: {path}")
