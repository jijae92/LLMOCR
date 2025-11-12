#!/usr/bin/env python3
"""
Comprehensive OCR benchmark suite for Korean datasets.

Features:
- Multiple benchmark sets (receipts, contracts, admin docs, synthdog)
- Metrics: CER, WER, throughput, p95 latency
- Multi-model comparison
- Automated report generation

Usage:
    python run_bench.py --model models/baseline --datasets ko_receipts,synthdog_ko_small
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import numpy as np
    from tqdm import tqdm
    import Levenshtein
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install numpy tqdm python-Levenshtein")
    exit(1)


@dataclass
class BenchmarkResult:
    """Result for a single model on a dataset."""
    model_name: str
    dataset_name: str
    total_samples: int
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    throughput: float  # images per second
    p50_latency: float  # median latency (ms)
    p95_latency: float  # 95th percentile latency (ms)
    p99_latency: float  # 99th percentile latency (ms)
    total_time: float  # total processing time (s)
    timestamp: str


class OCRBenchmark:
    """OCR Benchmark runner."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.model_name = self.model_path.name

    def load_model(self):
        """Load OCR model (placeholder - implement based on your model)."""
        print(f"Loading model from {self.model_path}...")

        # TODO: Replace with actual model loading code
        # Example implementations:

        # For TrOCR:
        # from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        # self.processor = TrOCRProcessor.from_pretrained(self.model_path)
        # self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        # self.model.to(self.device)
        # self.model.eval()

        # For EasyOCR:
        # import easyocr
        # self.model = easyocr.Reader(['ko', 'en'], gpu=self.device=='cuda')

        # For PaddleOCR:
        # from paddleocr import PaddleOCR
        # self.model = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=self.device=='cuda')

        # Placeholder
        print("⚠️  Warning: Using placeholder model. Implement model loading in load_model()")
        self.model = "placeholder"

    def predict(self, image_path: Path) -> str:
        """Run OCR prediction on image."""
        # TODO: Replace with actual prediction code
        # Example:
        # from PIL import Image
        # image = Image.open(image_path).convert("RGB")
        # pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # pixel_values = pixel_values.to(self.device)
        # generated_ids = self.model.generate(pixel_values)
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return generated_text

        # Placeholder - returns ground truth for testing
        return "placeholder_prediction"

    def run_benchmark(
        self,
        dataset_path: Path,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run benchmark on a dataset."""

        # Load dataset
        test_file = dataset_path / "test.jsonl"
        if not test_file.exists():
            # Fallback to train.jsonl for testing
            test_file = dataset_path / "train.jsonl"

        if not test_file.exists():
            raise FileNotFoundError(f"No test data found in {dataset_path}")

        print(f"\nLoading test data from {test_file}...")
        samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        if limit:
            samples = samples[:limit]

        print(f"Running inference on {len(samples)} samples...")

        # Run predictions
        predictions = []
        ground_truths = []
        latencies = []

        for sample in tqdm(samples, desc="Processing"):
            # Get image path
            image_rel_path = sample["image_path"]
            image_path = dataset_path / image_rel_path

            if not image_path.exists():
                # Try parent directory
                image_path = dataset_path.parent / image_rel_path

            if not image_path.exists():
                print(f"\n⚠️  Image not found: {image_path}")
                continue

            # Measure latency
            start_time = time.perf_counter()
            pred_text = self.predict(image_path)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Store results
            predictions.append(pred_text)
            ground_truths.append(sample["text"])

        # Calculate metrics
        total_time = sum(latencies) / 1000  # Convert to seconds
        throughput = len(samples) / total_time if total_time > 0 else 0

        cer = calculate_cer(predictions, ground_truths)
        wer = calculate_wer(predictions, ground_truths)

        # Latency percentiles
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        result = BenchmarkResult(
            model_name=self.model_name,
            dataset_name=dataset_path.name,
            total_samples=len(samples),
            cer=cer,
            wer=wer,
            throughput=throughput,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            total_time=total_time,
            timestamp=datetime.now().isoformat(),
        )

        return result


def calculate_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate Character Error Rate."""
    total_distance = 0
    total_length = 0

    for pred, gt in zip(predictions, ground_truths):
        distance = Levenshtein.distance(pred, gt)
        total_distance += distance
        total_length += len(gt)

    cer = total_distance / total_length if total_length > 0 else 0
    return cer


def calculate_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate Word Error Rate."""
    total_distance = 0
    total_words = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.split()
        gt_words = gt.split()

        distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
        total_distance += distance
        total_words += len(gt_words)

    wer = total_distance / total_words if total_words > 0 else 0
    return wer


def run_benchmarks(
    models: List[str],
    datasets: List[str],
    output_dir: str = "reports",
    limit: Optional[int] = None,
    device: str = "cuda",
) -> List[BenchmarkResult]:
    """Run benchmarks for multiple models and datasets."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_path in models:
        print("\n" + "="*70)
        print(f"Model: {model_path}")
        print("="*70)

        benchmark = OCRBenchmark(model_path, device=device)
        benchmark.load_model()

        for dataset_name in datasets:
            dataset_path = Path(f"datasets/processed/{dataset_name}")

            if not dataset_path.exists():
                print(f"\n⚠️  Dataset not found: {dataset_path}")
                continue

            print(f"\nDataset: {dataset_name}")

            try:
                result = benchmark.run_benchmark(dataset_path, limit=limit)
                all_results.append(result)

                # Print result
                print_result(result)

            except Exception as e:
                print(f"\n❌ Error running benchmark: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    save_results(all_results, output_path)

    return all_results


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\nResults:")
    print(f"  CER: {result.cer:.4f} ({result.cer*100:.2f}%)")
    print(f"  WER: {result.wer:.4f} ({result.wer*100:.2f}%)")
    print(f"  Throughput: {result.throughput:.2f} img/s")
    print(f"  Latency (p50): {result.p50_latency:.2f} ms")
    print(f"  Latency (p95): {result.p95_latency:.2f} ms")
    print(f"  Total time: {result.total_time:.2f} s")


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save benchmark results to files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    json_file = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {json_file}")

    # Save as CSV
    csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
    save_csv_report(results, csv_file)
    print(f"✓ CSV report saved to {csv_file}")

    # Save as Markdown
    md_file = output_dir / f"benchmark_report_{timestamp}.md"
    save_markdown_report(results, md_file)
    print(f"✓ Markdown report saved to {md_file}")


def save_csv_report(results: List[BenchmarkResult], output_file: Path):
    """Save results as CSV."""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if not results:
            return

        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_markdown_report(results: List[BenchmarkResult], output_file: Path):
    """Save results as Markdown report."""

    lines = [
        "# OCR Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total Models:** {len(set(r.model_name for r in results))}",
        f"- **Total Datasets:** {len(set(r.dataset_name for r in results))}",
        f"- **Total Benchmarks:** {len(results)}",
        "",
        "## Results by Dataset",
        "",
    ]

    # Group by dataset
    datasets = {}
    for result in results:
        if result.dataset_name not in datasets:
            datasets[result.dataset_name] = []
        datasets[result.dataset_name].append(result)

    for dataset_name, dataset_results in datasets.items():
        lines.extend([
            f"### {dataset_name}",
            "",
            "| Model | CER | WER | Throughput (img/s) | p95 Latency (ms) |",
            "|-------|-----|-----|-------------------|------------------|",
        ])

        for result in dataset_results:
            lines.append(
                f"| {result.model_name} | {result.cer:.4f} | {result.wer:.4f} | "
                f"{result.throughput:.2f} | {result.p95_latency:.2f} |"
            )

        lines.append("")

    # Add detailed table
    lines.extend([
        "## Detailed Results",
        "",
        "| Model | Dataset | Samples | CER | WER | Throughput | p50 Lat | p95 Lat | p99 Lat |",
        "|-------|---------|---------|-----|-----|------------|---------|---------|---------|",
    ])

    for result in results:
        lines.append(
            f"| {result.model_name} | {result.dataset_name} | {result.total_samples} | "
            f"{result.cer:.4f} | {result.wer:.4f} | {result.throughput:.2f} | "
            f"{result.p50_latency:.1f} | {result.p95_latency:.1f} | {result.p99_latency:.1f} |"
        )

    lines.append("")

    # Add notes
    lines.extend([
        "## Notes",
        "",
        "- **CER**: Character Error Rate (lower is better)",
        "- **WER**: Word Error Rate (lower is better)",
        "- **Throughput**: Images processed per second (higher is better)",
        "- **pXX Latency**: Percentile latency in milliseconds (lower is better)",
        "",
    ])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR benchmarks on Korean datasets"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model paths"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of dataset names"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per dataset (for quick testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]

    print("="*70)
    print("OCR BENCHMARK SUITE")
    print("="*70)
    print(f"\nModels: {models}")
    print(f"Datasets: {datasets}")
    print(f"Device: {args.device}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    print()

    results = run_benchmarks(
        models=models,
        datasets=datasets,
        output_dir=args.output_dir,
        limit=args.limit,
        device=args.device,
    )

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nTotal benchmarks run: {len(results)}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
