from datasets import load_from_disk
import argparse
import os
from pathlib import Path


def create_splits(args):
    ds = load_from_disk(args.source)
    ds = ds.shuffle(seed=42)
    val = ds.select(range(args.val))
    ds = ds.select(range(args.val, len(ds)))
    splits = []
    for i in range(args.subsets):
        end_idx = args.start_n * (2**i)
        split = ds.select(range(end_idx))
        splits.append(split)

    dir = Path(args.source) / "splits"
    os.makedirs(dir, exist_ok=True)
    val.save_to_disk(dir / "val")
    print(f"Saved validation set with {len(val)} samples to {dir / 'val'}")
    for i, split in enumerate(splits):
        _split_n = args.start_n * (2**i)
        split.save_to_disk(dir / f"train_{_split_n}")
        print(
            f"Saved train split with {len(split)} samples to {dir / f'train_{_split_n}'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("--start_n", type=int, default=100)
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--val", type=int, default=1000)

    args = parser.parse_args()

    create_splits(args)
