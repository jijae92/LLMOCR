#!/usr/bin/env python3
"""
Example workflow demonstrating the complete pipeline.

This script shows how to:
1. Download and prepare data
2. Run benchmarks
3. Execute continuous learning loop

Usage:
    python example_workflow.py --quick_demo
"""

import argparse
import subprocess
from pathlib import Path


def print_step(step_num: int, title: str):
    """Print formatted step header."""
    print("\n" + "="*70)
    print(f"Step {step_num}: {title}")
    print("="*70 + "\n")


def quick_demo():
    """Run a quick demo with subset of data."""
    print("="*70)
    print("LLMOCR QUICK DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("1. Download 200 samples from SynthDoG-ko")
    print("2. Clean and process the data")
    print("3. Create train/val/test splits")
    print("4. Run benchmark evaluation (placeholder model)")
    print("5. Generate reports")
    print()

    # Step 1: Download sample data
    print_step(1, "Download Sample Data")

    cmd = [
        "python", "datasets/scripts/download_synthdog_ko.py",
        "--output_dir", "datasets/raw/synthdog_ko_demo",
        "--limit", "200",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Download failed. Make sure you have internet connection and dependencies installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        return

    # Step 2: Clean data
    print_step(2, "Clean and Process Data")

    cmd = [
        "python", "datasets/scripts/clean_data.py",
        "--source", "datasets/raw/synthdog_ko_demo",
        "--output", "datasets/processed/synthdog_ko_demo",
        "--copy_images",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Data cleaning failed.")
        return

    # Step 3: Create splits
    print_step(3, "Create Train/Val/Test Splits")

    cmd = [
        "python", "datasets/scripts/create_splits.py",
        "--input", "datasets/processed/synthdog_ko_demo",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Split creation failed.")
        return

    # Step 4: Analyze dataset
    print_step(4, "Analyze Dataset")

    cmd = [
        "python", "datasets/scripts/analyze_difficulty.py",
        "--input", "datasets/processed/synthdog_ko_demo/train.jsonl",
        "--output_dir", "datasets/processed/synthdog_ko_demo",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    # Step 5: Run benchmark (with placeholder model)
    print_step(5, "Run Benchmark Evaluation")

    print("⚠️  Note: This will use a placeholder model.")
    print("    To use real models, implement the model loading in benchmarks/run_bench.py\n")

    # Create placeholder model directory
    placeholder_model = Path("models/baseline_placeholder")
    placeholder_model.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "benchmarks/run_bench.py",
        "--models", "models/baseline_placeholder",
        "--datasets", "synthdog_ko_demo",
        "--limit", "50",  # Use small subset for demo
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  📁 datasets/raw/synthdog_ko_demo/          - Downloaded images")
    print("  📁 datasets/processed/synthdog_ko_demo/    - Cleaned data with splits")
    print("  📄 datasets/processed/.../train.jsonl      - Training data")
    print("  📄 datasets/processed/.../val.jsonl        - Validation data")
    print("  📄 datasets/processed/.../test.jsonl       - Test data")
    print("  📊 reports/benchmark_results_*.json        - Benchmark results (JSON)")
    print("  📊 reports/benchmark_report_*.md           - Benchmark report (Markdown)")
    print()
    print("Next steps:")
    print("  1. Implement model loading in benchmarks/run_bench.py")
    print("  2. Download full datasets (AI-Hub, larger SynthDoG-ko)")
    print("  3. Train models with your data")
    print("  4. Run continuous learning pipeline")
    print()


def full_pipeline():
    """Run full pipeline with complete datasets."""
    print("="*70)
    print("FULL PIPELINE")
    print("="*70)
    print("\nThis will run the complete workflow:")
    print("1. Download full SynthDoG-ko dataset (10,000 samples)")
    print("2. Process and clean data")
    print("3. Train LoRA model")
    print("4. Run full benchmarks")
    print("5. Generate comprehensive reports")
    print()

    response = input("This will take significant time and resources. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Step 1: Download full dataset
    print_step(1, "Download Full Dataset")

    cmd = [
        "python", "datasets/scripts/download_synthdog_ko.py",
        "--output_dir", "datasets/raw/synthdog_ko",
        "--limit", "10000",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)

    # Step 2: Clean and process
    print_step(2, "Clean and Process")

    cmd = [
        "python", "datasets/scripts/clean_data.py",
        "--source", "datasets/raw/synthdog_ko",
        "--output", "datasets/processed/synthdog_ko_small",
        "--copy_images",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)

    # Step 3: Create splits
    print_step(3, "Create Splits")

    cmd = [
        "python", "datasets/scripts/create_splits.py",
        "--input", "datasets/processed/synthdog_ko_small",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)

    # Step 4: Run continuous learning
    print_step(4, "Run Continuous Learning Pipeline")

    print("⚠️  Implement model training in continuous_learning.py before running this step.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Example workflow for LLMOCR pipeline"
    )
    parser.add_argument(
        "--quick_demo",
        action="store_true",
        help="Run quick demo with 200 samples"
    )
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Run full pipeline with complete datasets"
    )

    args = parser.parse_args()

    if args.quick_demo:
        quick_demo()
    elif args.full_pipeline:
        full_pipeline()
    else:
        print("Usage:")
        print("  Quick demo:     python example_workflow.py --quick_demo")
        print("  Full pipeline:  python example_workflow.py --full_pipeline")


if __name__ == "__main__":
    main()
