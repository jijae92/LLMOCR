"""Evaluation script for OCR baselines."""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import jiwer


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        CER as a percentage
    """
    if not reference:
        return 100.0 if hypothesis else 0.0

    cer = jiwer.cer(reference, hypothesis)
    return cer * 100


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        WER as a percentage
    """
    if not reference:
        return 100.0 if hypothesis else 0.0

    wer = jiwer.wer(reference, hypothesis)
    return wer * 100


def get_pipeline(model: str, device: str, lang: str = "korean"):
    """Get OCR pipeline by model name."""
    if model == "trocr":
        from src.pipelines.trocr_pipeline import TrOCRPipeline
        return TrOCRPipeline(device=device)
    elif model == "donut":
        from src.pipelines.donut_pipeline import DonutPipeline
        return DonutPipeline(device=device)
    elif model == "pix2struct":
        from src.pipelines.pix2struct_pipeline import Pix2StructPipeline
        return Pix2StructPipeline(device=device)
    elif model == "paddleocr":
        from src.pipelines.paddleocr_pipeline import PaddleOCRPipeline
        return PaddleOCRPipeline(lang=lang, use_gpu=(device == "cuda"))
    else:
        raise ValueError(f"Unknown model: {model}")


def load_test_data(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load test data (image + ground truth text pairs).

    Expected structure:
        data_dir/
            image1.jpg
            image1.txt
            image2.jpg
            image2.txt
            ...

    Args:
        data_dir: Directory containing test data

    Returns:
        List of dicts with 'image_path' and 'text'
    """
    test_data = []

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for image_path in data_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        # Look for corresponding text file
        text_path = image_path.with_suffix(".txt")
        if not text_path.exists():
            print(f"Warning: No ground truth found for {image_path.name}, skipping")
            continue

        # Load ground truth text
        with open(text_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        test_data.append({
            "image_path": image_path,
            "ground_truth": ground_truth,
        })

    return test_data


def evaluate_model(
    model: str,
    test_data: List[Dict[str, Any]],
    device: str,
    lang: str,
) -> Dict[str, Any]:
    """
    Evaluate a single model on test data.

    Args:
        model: Model name
        test_data: List of test samples
        device: Device to use
        lang: Language for PaddleOCR

    Returns:
        Evaluation results
    """
    print(f"\nEvaluating {model}...")

    # Load pipeline
    try:
        pipeline = get_pipeline(model, device, lang)
    except Exception as e:
        print(f"Error loading {model}: {e}")
        return {
            "model": model,
            "error": str(e),
        }

    results = {
        "model": model,
        "device": device,
        "num_samples": len(test_data),
        "samples": [],
        "metrics": {},
    }

    cer_scores = []
    wer_scores = []
    latencies = []

    # Evaluate each sample
    for i, sample in enumerate(test_data):
        image_path = sample["image_path"]
        ground_truth = sample["ground_truth"]

        print(f"  Processing {i+1}/{len(test_data)}: {image_path.name}...")

        try:
            # Load image
            image = Image.open(image_path)

            # Run inference
            result = pipeline(image, return_metadata=True)
            prediction = result["text"]
            metadata = result.get("metadata", {})

            # Compute metrics
            cer = compute_cer(ground_truth, prediction)
            wer = compute_wer(ground_truth, prediction)
            latency_ms = metadata.get("latency_ms", 0)

            cer_scores.append(cer)
            wer_scores.append(wer)
            latencies.append(latency_ms)

            # Store sample result
            results["samples"].append({
                "image": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "cer": cer,
                "wer": wer,
                "latency_ms": latency_ms,
            })

        except Exception as e:
            print(f"    Error processing {image_path.name}: {e}")
            results["samples"].append({
                "image": image_path.name,
                "error": str(e),
            })

    # Compute aggregate metrics
    if cer_scores:
        results["metrics"] = {
            "avg_cer": sum(cer_scores) / len(cer_scores),
            "avg_wer": sum(wer_scores) / len(wer_scores),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_samples_per_sec": 1000 / (sum(latencies) / len(latencies)) if latencies else 0,
        }

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on test data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test images and ground truth texts",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of models to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on",
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
        help="Output JSON file for detailed results",
    )

    args = parser.parse_args()

    # Load test data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print(f"Loading test data from {data_dir}...")
    test_data = load_test_data(data_dir)
    if not test_data:
        print("Error: No test data found")
        return 1

    print(f"Found {len(test_data)} test samples")

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    valid_models = ["trocr", "donut", "pix2struct", "paddleocr"]
    for model in models:
        if model not in valid_models:
            print(f"Error: Invalid model '{model}'. Must be one of: {valid_models}")
            return 1

    # Evaluate each model
    all_results = []
    for model in models:
        results = evaluate_model(model, test_data, args.device, args.lang)
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for results in all_results:
        model = results["model"]
        print(f"\n{model.upper()}:")

        if "error" in results:
            print(f"  Error: {results['error']}")
            continue

        metrics = results.get("metrics", {})
        if metrics:
            print(f"  Average CER:      {metrics['avg_cer']:.2f}%")
            print(f"  Average WER:      {metrics['avg_wer']:.2f}%")
            print(f"  Average Latency:  {metrics['avg_latency_ms']:.2f} ms")
            print(f"  Min Latency:      {metrics['min_latency_ms']:.2f} ms")
            print(f"  Max Latency:      {metrics['max_latency_ms']:.2f} ms")
            print(f"  Throughput:       {metrics['throughput_samples_per_sec']:.2f} samples/sec")

    print("=" * 80)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
