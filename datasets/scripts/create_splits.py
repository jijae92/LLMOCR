#!/usr/bin/env python3
"""
Create train/val/test splits from cleaned dataset.

Usage:
    python create_splits.py --input processed/synthdog_ko_small --train_ratio 0.8 --val_ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install tqdm")
    exit(1)


def create_splits(
    input_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Create train/val/test splits from annotations."""
    input_dir = Path(input_dir)

    # Read annotations
    annotations_file = input_dir / "annotations.jsonl"
    if not annotations_file.exists():
        print(f"Error: No annotations.jsonl found in {input_dir}")
        return

    print(f"Reading annotations from {annotations_file}...")
    annotations = []
    with open(annotations_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    print(f"Total annotations: {len(annotations)}")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(annotations)

    # Calculate split sizes
    total = len(annotations)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    print(f"\nSplit sizes:")
    print(f"  Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"  Val: {val_size} ({val_ratio*100:.1f}%)")
    print(f"  Test: {test_size} ({(1-train_ratio-val_ratio)*100:.1f}%)")

    # Create splits
    train_data = annotations[:train_size]
    val_data = annotations[train_size:train_size + val_size]
    test_data = annotations[train_size + val_size:]

    # Add split field
    for item in train_data:
        item["split"] = "train"
    for item in val_data:
        item["split"] = "val"
    for item in test_data:
        item["split"] = "test"

    # Save splits
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    for split_name, split_data in splits.items():
        output_file = input_dir / f"{split_name}.jsonl"
        print(f"\nSaving {len(split_data)} samples to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Generate split statistics
    stats = generate_split_stats(splits)

    # Save statistics
    stats_file = input_dir / "split_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Splits created successfully")
    print(f"  Train: {input_dir / 'train.jsonl'}")
    print(f"  Val: {input_dir / 'val.jsonl'}")
    print(f"  Test: {input_dir / 'test.jsonl'}")
    print(f"  Stats: {stats_file}")


def generate_split_stats(splits: Dict[str, List[Dict]]) -> Dict:
    """Generate statistics for each split."""
    stats = {}

    for split_name, split_data in splits.items():
        if not split_data:
            continue

        # Calculate statistics
        lengths = [item["metadata"]["length"] for item in split_data if "metadata" in item]
        ko_ratios = [item["metadata"]["ko_ratio"] for item in split_data if "metadata" in item]

        # Difficulty distribution
        difficulties = [item["metadata"]["difficulty"] for item in split_data if "metadata" in item]
        difficulty_dist = {
            "easy": difficulties.count("easy"),
            "medium": difficulties.count("medium"),
            "hard": difficulties.count("hard"),
        }

        stats[split_name] = {
            "total_samples": len(split_data),
            "avg_text_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_text_length": min(lengths) if lengths else 0,
            "max_text_length": max(lengths) if lengths else 0,
            "avg_ko_ratio": sum(ko_ratios) / len(ko_ratios) if ko_ratios else 0,
            "difficulty_distribution": difficulty_dist,
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from cleaned dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with annotations.jsonl"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be < 1.0")
        return

    create_splits(
        input_dir=Path(args.input),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
