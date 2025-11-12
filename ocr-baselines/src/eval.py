"""Evaluation script for OCR models."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics

from PIL import Image
from jiwer import wer, cer
import Levenshtein

from src.pipelines import (
    TrOCRPipeline,
    DonutPipeline,
    Pix2StructPipeline,
    PaddleOCRPipeline,
)
from src.pipelines.paddleocr_pipeline import is_paddleocr_available


def load_dataset(data_dir: Path) -> List[Dict[str, Any]]:
    """Load test dataset from directory.

    Expected structure:
        data_dir/
            image1.jpg
            image1.txt  (ground truth)
            image2.jpg
            image2.txt

    Args:
        data_dir: Directory containing test images and ground truth

    Returns:
        List of dicts with 'image_path' and 'ground_truth'
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in data_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    dataset = []
    for image_path in sorted(image_files):
        # Look for corresponding ground truth file
        gt_path = image_path.with_suffix(".txt")
        if not gt_path.exists():
            print(f"Warning: No ground truth found for {image_path.name}, skipping")
            continue

        # Read ground truth
        ground_truth = gt_path.read_text(encoding="utf-8").strip()

        dataset.append({
            "image_path": image_path,
            "ground_truth": ground_truth,
        })

    if not dataset:
        raise ValueError(f"No valid image/text pairs found in {data_dir}")

    return dataset


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute CER, WER, and edit distance metrics.

    Args:
        predictions: List of predicted texts
        references: List of reference (ground truth) texts

    Returns:
        Dictionary with metrics
    """
    # Compute WER (Word Error Rate)
    wer_score = wer(references, predictions)

    # Compute CER (Character Error Rate)
    cer_score = cer(references, predictions)

    # Compute average Levenshtein distance
    edit_distances = [
        Levenshtein.distance(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    avg_edit_distance = statistics.mean(edit_distances)

    return {
        "wer": wer_score,
        "cer": cer_score,
        "avg_edit_distance": avg_edit_distance,
    }


def evaluate_model(
    model_name: str,
    dataset: List[Dict[str, Any]],
    device: str = "auto",
    lang: str = "korean",
) -> Dict[str, Any]:
    """Evaluate a single model on the dataset.

    Args:
        model_name: Model identifier
        dataset: Test dataset
        device: Device to use
        lang: Language for PaddleOCR

    Returns:
        Evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load pipeline
    try:
        if model_name == "trocr":
            pipeline = TrOCRPipeline(device=device)
        elif model_name == "donut":
            pipeline = DonutPipeline(device=device)
        elif model_name == "pix2struct":
            pipeline = Pix2StructPipeline(device=device)
        elif model_name == "paddleocr":
            if not is_paddleocr_available():
                print(f"Skipping {model_name}: PaddleOCR not available")
                return None
            pipeline = PaddleOCRPipeline(lang=lang, use_gpu=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    # Run inference on all samples
    predictions = []
    references = []
    latencies = []

    for i, sample in enumerate(dataset):
        image_path = sample["image_path"]
        ground_truth = sample["ground_truth"]

        print(f"Processing {i+1}/{len(dataset)}: {image_path.name}")

        try:
            # Load image
            image = Image.open(image_path)

            # Run inference
            result = pipeline.infer(image)

            predictions.append(result["text"])
            references.append(ground_truth)
            latencies.append(result["latency_ms"])

            print(f"  Predicted: {result['text'][:100]}...")
            print(f"  Reference: {ground_truth[:100]}...")
            print(f"  Latency: {result['latency_ms']:.2f} ms")

        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            # Add empty prediction to maintain alignment
            predictions.append("")
            references.append(ground_truth)
            latencies.append(0.0)

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Compute latency statistics
    latency_stats = {
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "min_ms": min(latencies) if latencies else 0.0,
        "max_ms": max(latencies) if latencies else 0.0,
    }

    # Compute throughput (images per second)
    if latency_stats["mean_ms"] > 0:
        throughput = 1000.0 / latency_stats["mean_ms"]
    else:
        throughput = 0.0

    results = {
        "model": model_name,
        "num_samples": len(dataset),
        "metrics": metrics,
        "latency": latency_stats,
        "throughput_imgs_per_sec": throughput,
    }

    # Print summary
    print(f"\n{'-'*60}")
    print(f"Results for {model_name}:")
    print(f"  CER: {metrics['cer']:.4f}")
    print(f"  WER: {metrics['wer']:.4f}")
    print(f"  Avg Edit Distance: {metrics['avg_edit_distance']:.2f}")
    print(f"  Mean Latency: {latency_stats['mean_ms']:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"{'-'*60}")

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on test dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test images and ground truth files",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="trocr,donut,pix2struct,paddleocr",
        help="Comma-separated list of models to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="korean",
        help="Language for PaddleOCR",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = load_dataset(Path(args.data_dir))
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    # Evaluate each model
    all_results = []
    for model in models:
        result = evaluate_model(
            model_name=model,
            dataset=dataset,
            device=args.device,
            lang=args.lang,
        )
        if result is not None:
            all_results.append(result)

    # Save results
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print comparison table
    if all_results:
        print(f"\n{'Model':<15} {'CER':<10} {'WER':<10} {'Latency (ms)':<15} {'Throughput (img/s)':<20}")
        print("-" * 70)
        for result in all_results:
            model = result["model"]
            cer_val = result["metrics"]["cer"]
            wer_val = result["metrics"]["wer"]
            latency = result["latency"]["mean_ms"]
            throughput = result["throughput_imgs_per_sec"]
            print(f"{model:<15} {cer_val:<10.4f} {wer_val:<10.4f} {latency:<15.2f} {throughput:<20.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
