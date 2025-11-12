#!/usr/bin/env python3
"""
Error analysis tools for OCR models.

Features:
- Find highest CER samples
- Generate thumbnails with prediction/ground truth diff
- Character-level diff visualization
- Error pattern analysis
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import difflib

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import Levenshtein
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install pillow numpy python-Levenshtein")
    exit(1)


@dataclass
class ErrorSample:
    """Sample with error information."""
    image_path: Path
    prediction: str
    ground_truth: str
    cer: float
    wer: float
    error_types: Dict[str, int]  # substitution, insertion, deletion
    char_diffs: List[Tuple[str, str, str]]  # (type, pred_char, gt_char)


class ErrorAnalyzer:
    """Analyze OCR errors and generate visualizations."""

    def __init__(self, output_dir: str = "reports/error_analysis"):
        """
        Initialize error analyzer.

        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_cer(self, prediction: str, ground_truth: str) -> float:
        """Calculate character error rate."""
        distance = Levenshtein.distance(prediction, ground_truth)
        return distance / len(ground_truth) if ground_truth else 0.0

    def calculate_wer(self, prediction: str, ground_truth: str) -> float:
        """Calculate word error rate."""
        pred_words = prediction.split()
        gt_words = ground_truth.split()
        distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
        return distance / len(gt_words) if gt_words else 0.0

    def analyze_character_diffs(
        self, prediction: str, ground_truth: str
    ) -> Tuple[List[Tuple], Dict[str, int]]:
        """
        Analyze character-level differences.

        Returns:
            Tuple of (char_diffs, error_counts)
        """
        char_diffs = []
        error_counts = {
            'substitution': 0,
            'insertion': 0,
            'deletion': 0,
            'correct': 0,
        }

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, ground_truth, prediction)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    char_diffs.append(('equal', ground_truth[i1 + k], ground_truth[i1 + k]))
                    error_counts['correct'] += 1

            elif tag == 'replace':
                # Substitution
                gt_chars = ground_truth[i1:i2]
                pred_chars = prediction[j1:j2]

                max_len = max(len(gt_chars), len(pred_chars))
                for k in range(max_len):
                    gt_char = gt_chars[k] if k < len(gt_chars) else ''
                    pred_char = pred_chars[k] if k < len(pred_chars) else ''
                    char_diffs.append(('replace', pred_char, gt_char))
                    error_counts['substitution'] += 1

            elif tag == 'insert':
                # Insertion (extra chars in prediction)
                for k in range(j2 - j1):
                    char_diffs.append(('insert', prediction[j1 + k], ''))
                    error_counts['insertion'] += 1

            elif tag == 'delete':
                # Deletion (missing chars in prediction)
                for k in range(i2 - i1):
                    char_diffs.append(('delete', '', ground_truth[i1 + k]))
                    error_counts['deletion'] += 1

        return char_diffs, error_counts

    def find_top_errors(
        self,
        results: List[Dict],
        n: int = 20,
        metric: str = 'cer'
    ) -> List[ErrorSample]:
        """
        Find top N samples with highest errors.

        Args:
            results: List of prediction results
            n: Number of samples to return
            metric: Metric to sort by ('cer' or 'wer')

        Returns:
            List of ErrorSample objects
        """
        error_samples = []

        for result in results:
            pred = result.get('prediction', '')
            gt = result.get('ground_truth', '')

            if not gt:
                continue

            cer = self.calculate_cer(pred, gt)
            wer = self.calculate_wer(pred, gt)

            char_diffs, error_counts = self.analyze_character_diffs(pred, gt)

            sample = ErrorSample(
                image_path=Path(result['image_path']),
                prediction=pred,
                ground_truth=gt,
                cer=cer,
                wer=wer,
                error_types=error_counts,
                char_diffs=char_diffs,
            )

            error_samples.append(sample)

        # Sort by metric
        if metric == 'cer':
            error_samples.sort(key=lambda x: x.cer, reverse=True)
        else:
            error_samples.sort(key=lambda x: x.wer, reverse=True)

        return error_samples[:n]

    def create_diff_visualization(
        self,
        sample: ErrorSample,
        max_width: int = 800
    ) -> Image.Image:
        """
        Create visual diff showing prediction vs ground truth.

        Color coding:
        - Green: Correct
        - Red: Substitution
        - Blue: Insertion
        - Yellow: Deletion
        """
        # Try to load font
        try:
            font_size = 20
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            char_width = font_size
            char_height = font_size + 4
        except:
            font = ImageFont.load_default()
            char_width = 10
            char_height = 14

        # Calculate image dimensions
        max_chars = max_width // char_width
        lines_needed = (len(sample.char_diffs) + max_chars - 1) // max_chars

        img_height = char_height * 4 * lines_needed + 100  # 4 rows per line (labels + diffs)
        img = Image.new('RGB', (max_width, img_height), color='white')
        draw = ImageDraw.Draw(img)

        y_offset = 10

        # Draw title
        title = f"CER: {sample.cer:.3f} | Errors: S={sample.error_types['substitution']} I={sample.error_types['insertion']} D={sample.error_types['deletion']}"
        draw.text((10, y_offset), title, fill='black', font=font)
        y_offset += char_height + 10

        # Process char diffs in chunks
        for line_idx in range(lines_needed):
            start_idx = line_idx * max_chars
            end_idx = min(start_idx + max_chars, len(sample.char_diffs))
            line_diffs = sample.char_diffs[start_idx:end_idx]

            # Draw ground truth label
            draw.text((10, y_offset), "GT:", fill='black', font=font)

            # Draw ground truth characters
            x_offset = 50
            for diff_type, pred_char, gt_char in line_diffs:
                if diff_type in ('equal', 'replace', 'delete'):
                    char = gt_char if gt_char else '_'

                    # Color based on type
                    if diff_type == 'equal':
                        color = (0, 150, 0)  # Green
                    elif diff_type == 'replace':
                        color = (200, 0, 0)  # Red
                    else:  # delete
                        color = (200, 150, 0)  # Yellow

                    draw.text((x_offset, y_offset), char, fill=color, font=font)
                    x_offset += char_width

            y_offset += char_height

            # Draw prediction label
            draw.text((10, y_offset), "Pred:", fill='black', font=font)

            # Draw prediction characters
            x_offset = 50
            for diff_type, pred_char, gt_char in line_diffs:
                if diff_type in ('equal', 'replace', 'insert'):
                    char = pred_char if pred_char else '_'

                    # Color based on type
                    if diff_type == 'equal':
                        color = (0, 150, 0)  # Green
                    elif diff_type == 'replace':
                        color = (200, 0, 0)  # Red
                    else:  # insert
                        color = (0, 100, 200)  # Blue

                    draw.text((x_offset, y_offset), char, fill=color, font=font)
                    x_offset += char_width

            y_offset += char_height + 20

        return img

    def generate_error_report(
        self,
        error_samples: List[ErrorSample],
        dataset_path: Optional[Path] = None,
        include_thumbnails: bool = True,
        thumbnail_size: int = 200,
    ):
        """
        Generate comprehensive error analysis report.

        Args:
            error_samples: List of error samples
            dataset_path: Path to dataset (for loading images)
            include_thumbnails: Whether to include image thumbnails
            thumbnail_size: Size of thumbnails
        """
        timestamp = Path(self.output_dir.name)

        # Create thumbnails directory
        if include_thumbnails:
            thumbnails_dir = self.output_dir / "thumbnails"
            thumbnails_dir.mkdir(exist_ok=True)

            diffs_dir = self.output_dir / "diffs"
            diffs_dir.mkdir(exist_ok=True)

        # Generate report data
        report_data = {
            'total_samples': len(error_samples),
            'samples': []
        }

        for idx, sample in enumerate(error_samples):
            sample_data = {
                'rank': idx + 1,
                'image_path': str(sample.image_path),
                'cer': sample.cer,
                'wer': sample.wer,
                'prediction': sample.prediction,
                'ground_truth': sample.ground_truth,
                'error_types': sample.error_types,
            }

            if include_thumbnails:
                # Load and save thumbnail
                try:
                    if dataset_path:
                        full_image_path = dataset_path / sample.image_path
                    else:
                        full_image_path = sample.image_path

                    if full_image_path.exists():
                        # Create thumbnail
                        img = Image.open(full_image_path)
                        img.thumbnail((thumbnail_size, thumbnail_size))

                        thumb_filename = f"thumb_{idx:03d}.jpg"
                        thumb_path = thumbnails_dir / thumb_filename
                        img.save(thumb_path)

                        sample_data['thumbnail'] = f"thumbnails/{thumb_filename}"

                        # Create diff visualization
                        diff_img = self.create_diff_visualization(sample)
                        diff_filename = f"diff_{idx:03d}.png"
                        diff_path = diffs_dir / diff_filename
                        diff_img.save(diff_path)

                        sample_data['diff_image'] = f"diffs/{diff_filename}"

                except Exception as e:
                    print(f"Warning: Could not create thumbnail for {sample.image_path}: {e}")

            report_data['samples'].append(sample_data)

        # Save JSON report
        json_path = self.output_dir / "error_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Generate markdown report
        self._generate_markdown_report(report_data)

        print(f"\n✓ Error analysis report generated:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {self.output_dir / 'error_report.md'}")
        if include_thumbnails:
            print(f"  Thumbnails: {thumbnails_dir}")
            print(f"  Diffs: {diffs_dir}")

    def _generate_markdown_report(self, report_data: Dict):
        """Generate markdown error report."""
        lines = [
            "# OCR Error Analysis Report",
            "",
            f"**Total Samples Analyzed:** {report_data['total_samples']}",
            "",
            "## Top Errors",
            "",
        ]

        for sample in report_data['samples']:
            lines.extend([
                f"### #{sample['rank']}: CER={sample['cer']:.3f}, WER={sample['wer']:.3f}",
                "",
                f"**Image:** `{sample['image_path']}`",
                "",
            ])

            if 'thumbnail' in sample:
                lines.append(f"![Thumbnail]({sample['thumbnail']})")
                lines.append("")

            lines.extend([
                "**Ground Truth:**",
                f"```\n{sample['ground_truth']}\n```",
                "",
                "**Prediction:**",
                f"```\n{sample['prediction']}\n```",
                "",
                "**Error Breakdown:**",
                f"- Substitutions: {sample['error_types']['substitution']}",
                f"- Insertions: {sample['error_types']['insertion']}",
                f"- Deletions: {sample['error_types']['deletion']}",
                f"- Correct: {sample['error_types']['correct']}",
                "",
            ])

            if 'diff_image' in sample:
                lines.extend([
                    "**Character-level Diff:**",
                    f"![Diff]({sample['diff_image']})",
                    "",
                ])

            lines.append("---")
            lines.append("")

        # Save markdown
        md_path = self.output_dir / "error_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def analyze_error_patterns(self, error_samples: List[ErrorSample]) -> Dict:
        """
        Analyze common error patterns.

        Returns:
            Dictionary with pattern statistics
        """
        patterns = {
            'total_errors': 0,
            'substitution_patterns': {},
            'insertion_patterns': {},
            'deletion_patterns': {},
            'avg_cer': 0,
            'avg_wer': 0,
        }

        total_cer = 0
        total_wer = 0

        for sample in error_samples:
            total_cer += sample.cer
            total_wer += sample.wer

            for diff_type, pred_char, gt_char in sample.char_diffs:
                if diff_type == 'replace':
                    key = f"{gt_char} → {pred_char}"
                    patterns['substitution_patterns'][key] = \
                        patterns['substitution_patterns'].get(key, 0) + 1
                    patterns['total_errors'] += 1

                elif diff_type == 'insert':
                    key = f"insert '{pred_char}'"
                    patterns['insertion_patterns'][key] = \
                        patterns['insertion_patterns'].get(key, 0) + 1
                    patterns['total_errors'] += 1

                elif diff_type == 'delete':
                    key = f"delete '{gt_char}'"
                    patterns['deletion_patterns'][key] = \
                        patterns['deletion_patterns'].get(key, 0) + 1
                    patterns['total_errors'] += 1

        if error_samples:
            patterns['avg_cer'] = total_cer / len(error_samples)
            patterns['avg_wer'] = total_wer / len(error_samples)

        # Sort patterns by frequency
        patterns['substitution_patterns'] = dict(
            sorted(patterns['substitution_patterns'].items(), key=lambda x: x[1], reverse=True)[:20]
        )
        patterns['insertion_patterns'] = dict(
            sorted(patterns['insertion_patterns'].items(), key=lambda x: x[1], reverse=True)[:20]
        )
        patterns['deletion_patterns'] = dict(
            sorted(patterns['deletion_patterns'].items(), key=lambda x: x[1], reverse=True)[:20]
        )

        return patterns


def main():
    """Example usage."""
    from pathlib import Path

    # Example results (would come from actual benchmark)
    example_results = [
        {
            'image_path': 'test1.jpg',
            'prediction': '안녕하세요',
            'ground_truth': '안녕하세요',
        },
        {
            'image_path': 'test2.jpg',
            'prediction': '한국어 인식 테스트',
            'ground_truth': '한국어 인식 테스트',
        },
        {
            'image_path': 'test3.jpg',
            'prediction': '테스듀 데이터',  # Intentional errors
            'ground_truth': '테스트 데이터',
        },
    ]

    analyzer = ErrorAnalyzer(output_dir="reports/error_analysis_demo")

    # Find top errors
    error_samples = analyzer.find_top_errors(example_results, n=10)

    print("Top Errors:")
    print("=" * 70)
    for idx, sample in enumerate(error_samples[:5]):
        print(f"\n#{idx+1}: CER={sample.cer:.3f}")
        print(f"  GT:   {sample.ground_truth}")
        print(f"  Pred: {sample.prediction}")
        print(f"  Errors: S={sample.error_types['substitution']} "
              f"I={sample.error_types['insertion']} "
              f"D={sample.error_types['deletion']}")

    # Analyze patterns
    patterns = analyzer.analyze_error_patterns(error_samples)
    print(f"\n\nError Patterns:")
    print("=" * 70)
    print(f"Total errors: {patterns['total_errors']}")
    print(f"Avg CER: {patterns['avg_cer']:.3f}")
    print(f"\nTop substitutions:")
    for pattern, count in list(patterns['substitution_patterns'].items())[:5]:
        print(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()
