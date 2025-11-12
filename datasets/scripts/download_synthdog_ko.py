#!/usr/bin/env python3
"""
Download SynthDoG-ko dataset from Hugging Face.

Usage:
    python download_synthdog_ko.py --output_dir ./raw/synthdog_ko --limit 10000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install datasets pillow tqdm")
    exit(1)


def download_synthdog_ko(
    output_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
    start_idx: int = 0,
):
    """Download SynthDoG-ko dataset from Hugging Face."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Downloading SynthDoG-ko dataset (split: {split})...")
    print(f"Output directory: {output_path}")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(
            "naver-clova-ix/synthdog-ko",
            split=split,
            streaming=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative approach with streaming...")
        dataset = load_dataset(
            "naver-clova-ix/synthdog-ko",
            split=split,
            streaming=True
        )

    # Process samples
    annotations = []
    count = 0

    # Determine how many samples to process
    if limit:
        total = min(limit, len(dataset) if hasattr(dataset, '__len__') else limit)
    else:
        total = len(dataset) if hasattr(dataset, '__len__') else None

    print(f"Processing samples (start: {start_idx}, limit: {limit})...")

    iterator = iter(dataset)

    # Skip to start index
    for _ in range(start_idx):
        next(iterator)

    # Process samples with progress bar
    import time
    start_time = time.time()
    pbar = tqdm(total=total, desc="Downloading", ncols=100)

    try:
        for idx, sample in enumerate(iterator):
            if limit and count >= limit:
                break

            actual_idx = start_idx + idx

            # Get image and text
            image = sample.get('image')
            text = sample.get('text', '') or sample.get('ground_truth', '')

            if image is None or not text:
                continue

            # Save image
            image_filename = f"synthdog_ko_{actual_idx:08d}.jpg"
            image_path = images_dir / image_filename

            try:
                if isinstance(image, Image.Image):
                    image.save(image_path, "JPEG", quality=95)
                else:
                    # Handle other image formats
                    img = Image.open(image) if hasattr(image, 'read') else image
                    img.save(image_path, "JPEG", quality=95)
            except Exception as e:
                print(f"\nError saving image {actual_idx}: {e}")
                continue

            # Create annotation entry
            annotation = {
                "image_path": f"images/{image_filename}",
                "text": text.strip(),
                "source": "synthdog_ko",
                "original_idx": actual_idx,
            }

            annotations.append(annotation)
            count += 1
            pbar.update(1)

            # Print progress info for GUI parsing (every 10 samples)
            if count % 10 == 0 or count == total:
                elapsed = time.time() - start_time
                progress_pct = (count / total * 100) if total else 0
                speed = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / speed if speed > 0 and total else 0
                print(f"PROGRESS:{count}/{total}:{progress_pct:.1f}%:{eta:.0f}s", flush=True)

    except StopIteration:
        pass
    finally:
        pbar.close()

    # Save annotations
    annotations_file = output_path / "annotations.jsonl"
    print(f"\nSaving {len(annotations)} annotations to {annotations_file}...")

    with open(annotations_file, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')

    # Save metadata
    metadata = {
        "dataset": "synthdog-ko",
        "split": split,
        "source": "https://huggingface.co/datasets/naver-clova-ix/synthdog-ko",
        "total_samples": len(annotations),
        "start_idx": start_idx,
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Successfully downloaded {len(annotations)} samples")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_file}")
    print(f"  Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SynthDoG-ko dataset from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw/synthdog_ko",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to download"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for downloading (useful for resuming)"
    )

    args = parser.parse_args()

    download_synthdog_ko(
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        start_idx=args.start_idx,
    )


if __name__ == "__main__":
    main()
