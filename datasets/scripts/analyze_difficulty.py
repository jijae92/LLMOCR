#!/usr/bin/env python3
"""
Analyze dataset difficulty and generate statistics report.

Usage:
    python analyze_difficulty.py --input processed/synthdog_ko_small/train.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: matplotlib not available. Install for visualization:")
    print("  pip install matplotlib")


def analyze_dataset(input_file: Path, output_dir: Path = None):
    """Analyze dataset and generate statistics."""
    input_file = Path(input_file)

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return

    # Read annotations
    print(f"Reading annotations from {input_file}...")
    annotations = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    print(f"Total samples: {len(annotations)}")

    # Collect statistics
    stats = {
        "total_samples": len(annotations),
        "text_lengths": [],
        "ko_ratios": [],
        "en_ratios": [],
        "difficulties": [],
        "sources": [],
    }

    for ann in annotations:
        if "metadata" not in ann:
            continue

        metadata = ann["metadata"]
        stats["text_lengths"].append(metadata.get("length", 0))
        stats["ko_ratios"].append(metadata.get("ko_ratio", 0))
        stats["en_ratios"].append(metadata.get("en_ratio", 0))
        stats["difficulties"].append(metadata.get("difficulty", "unknown"))
        stats["sources"].append(ann.get("source", "unknown"))

    # Calculate summary statistics
    summary = generate_summary(stats)

    # Print report
    print_report(summary)

    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_file = output_dir / "dataset_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to {report_file}")

        # Save markdown report
        md_file = output_dir / "dataset_analysis.md"
        save_markdown_report(summary, md_file, input_file)
        print(f"✓ Markdown report saved to {md_file}")

        # Generate plots if available
        if PLOTTING_AVAILABLE:
            plot_file = output_dir / "dataset_analysis.png"
            plot_statistics(stats, plot_file)
            print(f"✓ Plots saved to {plot_file}")

    return summary


def generate_summary(stats: Dict) -> Dict:
    """Generate summary statistics."""
    text_lengths = stats["text_lengths"]
    ko_ratios = stats["ko_ratios"]
    en_ratios = stats["en_ratios"]

    difficulty_counter = Counter(stats["difficulties"])
    source_counter = Counter(stats["sources"])

    summary = {
        "total_samples": stats["total_samples"],
        "text_length": {
            "mean": float(np.mean(text_lengths)) if text_lengths else 0,
            "median": float(np.median(text_lengths)) if text_lengths else 0,
            "min": int(min(text_lengths)) if text_lengths else 0,
            "max": int(max(text_lengths)) if text_lengths else 0,
            "std": float(np.std(text_lengths)) if text_lengths else 0,
        },
        "korean_ratio": {
            "mean": float(np.mean(ko_ratios)) if ko_ratios else 0,
            "median": float(np.median(ko_ratios)) if ko_ratios else 0,
        },
        "english_ratio": {
            "mean": float(np.mean(en_ratios)) if en_ratios else 0,
            "median": float(np.median(en_ratios)) if en_ratios else 0,
        },
        "difficulty_distribution": dict(difficulty_counter),
        "source_distribution": dict(source_counter),
    }

    return summary


def print_report(summary: Dict):
    """Print analysis report to console."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)

    print(f"\nTotal Samples: {summary['total_samples']}")

    print("\nText Length Statistics:")
    tl = summary['text_length']
    print(f"  Mean: {tl['mean']:.1f} characters")
    print(f"  Median: {tl['median']:.1f} characters")
    print(f"  Min: {tl['min']} characters")
    print(f"  Max: {tl['max']} characters")
    print(f"  Std Dev: {tl['std']:.1f}")

    print("\nCharacter Distribution:")
    print(f"  Korean Ratio: {summary['korean_ratio']['mean']:.1%}")
    print(f"  English Ratio: {summary['english_ratio']['mean']:.1%}")

    print("\nDifficulty Distribution:")
    for difficulty, count in summary['difficulty_distribution'].items():
        pct = count / summary['total_samples'] * 100
        print(f"  {difficulty.capitalize()}: {count} ({pct:.1f}%)")

    print("\nSource Distribution:")
    for source, count in summary['source_distribution'].items():
        pct = count / summary['total_samples'] * 100
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("="*70)


def save_markdown_report(summary: Dict, output_file: Path, input_file: Path):
    """Save analysis report as markdown."""
    lines = [
        "# Dataset Analysis Report",
        "",
        f"**Dataset:** `{input_file}`",
        f"**Generated:** {input_file.stat().st_mtime}",
        "",
        "## Overview",
        "",
        f"- **Total Samples:** {summary['total_samples']:,}",
        "",
        "## Text Length Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean | {summary['text_length']['mean']:.1f} chars |",
        f"| Median | {summary['text_length']['median']:.1f} chars |",
        f"| Min | {summary['text_length']['min']} chars |",
        f"| Max | {summary['text_length']['max']} chars |",
        f"| Std Dev | {summary['text_length']['std']:.1f} |",
        "",
        "## Character Distribution",
        "",
        f"- **Korean Ratio:** {summary['korean_ratio']['mean']:.1%}",
        f"- **English Ratio:** {summary['english_ratio']['mean']:.1%}",
        "",
        "## Difficulty Distribution",
        "",
        "| Difficulty | Count | Percentage |",
        "|------------|-------|------------|",
    ]

    for difficulty, count in summary['difficulty_distribution'].items():
        pct = count / summary['total_samples'] * 100
        lines.append(f"| {difficulty.capitalize()} | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Source Distribution",
        "",
        "| Source | Count | Percentage |",
        "|--------|-------|------------|",
    ])

    for source, count in summary['source_distribution'].items():
        pct = count / summary['total_samples'] * 100
        lines.append(f"| {source} | {count:,} | {pct:.1f}% |")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_statistics(stats: Dict, output_file: Path):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Text length distribution
    axes[0, 0].hist(stats["text_lengths"], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Text Length Distribution')

    # Korean ratio distribution
    axes[0, 1].hist(stats["ko_ratios"], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Korean Character Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Korean Character Ratio Distribution')

    # Difficulty distribution
    difficulty_counter = Counter(stats["difficulties"])
    difficulties = list(difficulty_counter.keys())
    counts = list(difficulty_counter.values())
    axes[1, 0].bar(difficulties, counts, edgecolor='black')
    axes[1, 0].set_xlabel('Difficulty Level')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Difficulty Distribution')

    # Source distribution
    source_counter = Counter(stats["sources"])
    sources = list(source_counter.keys())
    source_counts = list(source_counter.values())
    axes[1, 1].bar(sources, source_counts, edgecolor='black')
    axes[1, 1].set_xlabel('Source')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Source Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset difficulty and statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports (default: same as input file)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_file.parent

    analyze_dataset(input_file, output_dir)


if __name__ == "__main__":
    main()
