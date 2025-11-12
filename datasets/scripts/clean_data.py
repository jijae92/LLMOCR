#!/usr/bin/env python3
"""
Clean and process OCR datasets.

Features:
- Remove corrupted images
- Filter by text length
- Tag difficulty based on character distribution
- Validate image quality

Usage:
    python clean_data.py --source raw/synthdog_ko --output processed/synthdog_ko_small
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import cv2
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install opencv-python pillow numpy tqdm")
    exit(1)


class DataCleaner:
    """Clean and process OCR dataset."""

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 1000,
        min_dimension: int = 32,
        blur_threshold: float = 100.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_dimension = min_dimension
        self.blur_threshold = blur_threshold

        self.stats = {
            "total": 0,
            "corrupted_image": 0,
            "invalid_dimensions": 0,
            "text_too_short": 0,
            "text_too_long": 0,
            "too_blurry": 0,
            "empty_text": 0,
            "valid": 0,
        }

    def is_image_valid(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if image is valid and meets quality criteria."""
        if not image_path.exists():
            return False, "file_not_found"

        try:
            # Try to open with PIL
            img = Image.open(image_path)
            img.verify()

            # Re-open for actual processing
            img = Image.open(image_path)
            width, height = img.size

            # Check dimensions
            if width < self.min_dimension or height < self.min_dimension:
                return False, "invalid_dimensions"

            # Convert to array for quality checks
            img_array = np.array(img)

            # Check if image is completely black or white
            if img_array.std() < 5:
                return False, "too_uniform"

            # Check blur (Laplacian variance)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.blur_threshold:
                return False, "too_blurry"

            return True, None

        except Exception as e:
            return False, "corrupted_image"

    def analyze_text_difficulty(self, text: str) -> Dict:
        """Analyze text and determine difficulty level."""
        if not text:
            return {
                "difficulty": "unknown",
                "ko_ratio": 0.0,
                "en_ratio": 0.0,
                "symbol_ratio": 0.0,
                "length": 0,
            }

        # Count character types
        korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
        english_pattern = re.compile(r'[a-zA-Z]')
        digit_pattern = re.compile(r'[0-9]')
        whitespace_pattern = re.compile(r'\s')

        korean_chars = len(korean_pattern.findall(text))
        english_chars = len(english_pattern.findall(text))
        digit_chars = len(digit_pattern.findall(text))
        whitespace_chars = len(whitespace_pattern.findall(text))

        # Total non-whitespace characters
        total_chars = len(text) - whitespace_chars

        if total_chars == 0:
            return {
                "difficulty": "unknown",
                "ko_ratio": 0.0,
                "en_ratio": 0.0,
                "symbol_ratio": 0.0,
                "length": len(text),
            }

        # Calculate ratios
        ko_ratio = korean_chars / total_chars
        en_ratio = english_chars / total_chars
        digit_ratio = digit_chars / total_chars
        symbol_ratio = 1.0 - (ko_ratio + en_ratio + digit_ratio)

        # Determine difficulty
        if ko_ratio > 0.9 and total_chars < 50:
            difficulty = "easy"
        elif ko_ratio > 0.7:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return {
            "difficulty": difficulty,
            "ko_ratio": round(ko_ratio, 3),
            "en_ratio": round(en_ratio, 3),
            "digit_ratio": round(digit_ratio, 3),
            "symbol_ratio": round(symbol_ratio, 3),
            "length": len(text),
        }

    def clean_dataset(
        self,
        source_dir: Path,
        output_dir: Path,
        copy_images: bool = False,
    ):
        """Clean dataset and create processed version."""
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find annotations file
        annotations_file = source_dir / "annotations.jsonl"
        if not annotations_file.exists():
            print(f"Error: No annotations.jsonl found in {source_dir}")
            return

        # Read annotations
        print(f"Reading annotations from {annotations_file}...")
        annotations = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))

        print(f"Total annotations: {len(annotations)}")

        # Process each annotation
        cleaned_annotations = []

        for ann in tqdm(annotations, desc="Cleaning"):
            self.stats["total"] += 1

            text = ann.get("text", "").strip()

            # Check text validity
            if not text:
                self.stats["empty_text"] += 1
                continue

            if len(text) < self.min_length:
                self.stats["text_too_short"] += 1
                continue

            if len(text) > self.max_length:
                self.stats["text_too_long"] += 1
                continue

            # Check image validity
            image_rel_path = ann.get("image_path", "")
            image_path = source_dir / image_rel_path

            is_valid, reason = self.is_image_valid(image_path)

            if not is_valid:
                self.stats[reason] += 1
                continue

            # Analyze text difficulty
            text_analysis = self.analyze_text_difficulty(text)

            # Create cleaned annotation
            cleaned_ann = {
                "image_path": image_rel_path,
                "text": text,
                "source": ann.get("source", "unknown"),
                "metadata": text_analysis,
            }

            if "original_idx" in ann:
                cleaned_ann["original_idx"] = ann["original_idx"]

            cleaned_annotations.append(cleaned_ann)
            self.stats["valid"] += 1

        # Copy/link images if requested
        if copy_images:
            print("\nCopying images...")
            images_src = source_dir / "images"
            images_dst = output_dir / "images"
            images_dst.mkdir(exist_ok=True)

            for ann in tqdm(cleaned_annotations, desc="Copying"):
                src_path = source_dir / ann["image_path"]
                dst_path = output_dir / ann["image_path"]
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                if not dst_path.exists():
                    # Use symlink to save space (or copy if needed)
                    try:
                        dst_path.symlink_to(src_path.absolute())
                    except OSError:
                        # Fallback to copy
                        import shutil
                        shutil.copy2(src_path, dst_path)

        # Save cleaned annotations
        output_file = output_dir / "annotations.jsonl"
        print(f"\nSaving {len(cleaned_annotations)} cleaned annotations...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for ann in cleaned_annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + '\n')

        # Save statistics
        stats_file = output_dir / "cleaning_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        # Print summary
        self._print_summary(output_dir)

    def _print_summary(self, output_dir: Path):
        """Print cleaning summary."""
        print("\n" + "="*70)
        print("CLEANING SUMMARY")
        print("="*70)
        print(f"Total samples: {self.stats['total']}")
        print(f"Valid samples: {self.stats['valid']} ({self.stats['valid']/self.stats['total']*100:.1f}%)")
        print("\nFiltered out:")
        print(f"  - Corrupted images: {self.stats['corrupted_image']}")
        print(f"  - Invalid dimensions: {self.stats['invalid_dimensions']}")
        print(f"  - Too blurry: {self.stats['too_blurry']}")
        print(f"  - Empty text: {self.stats['empty_text']}")
        print(f"  - Text too short: {self.stats['text_too_short']}")
        print(f"  - Text too long: {self.stats['text_too_long']}")
        print(f"\nOutput: {output_dir / 'annotations.jsonl'}")
        print(f"Stats: {output_dir / 'cleaning_stats.json'}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Clean and process OCR dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory with raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for cleaned data"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum text length"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum text length"
    )
    parser.add_argument(
        "--min_dimension",
        type=int,
        default=32,
        help="Minimum image dimension"
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=100.0,
        help="Blur detection threshold (Laplacian variance)"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy/link images to output directory"
    )

    args = parser.parse_args()

    cleaner = DataCleaner(
        min_length=args.min_length,
        max_length=args.max_length,
        min_dimension=args.min_dimension,
        blur_threshold=args.blur_threshold,
    )

    cleaner.clean_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
