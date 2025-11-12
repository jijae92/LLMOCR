"""CLI interface for OCR baselines."""
import argparse
import json
from pathlib import Path
from typing import Optional
from PIL import Image


def get_pipeline(
    model: str,
    device: str,
    lang: str = "korean",
    adapter_path: Optional[str] = None
):
    """
    Get OCR pipeline by model name.

    Args:
        model: Model name ('trocr', 'donut', 'pix2struct', 'paddleocr')
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        lang: Language for PaddleOCR
        adapter_path: Path to LoRA adapter (for trocr/donut)

    Returns:
        Initialized pipeline
    """
    if model == "trocr":
        from src.pipelines.trocr_pipeline import TrOCRPipeline
        return TrOCRPipeline(device=device, adapter_path=adapter_path)
    elif model == "donut":
        from src.pipelines.donut_pipeline import DonutPipeline
        return DonutPipeline(device=device, adapter_path=adapter_path)
    elif model == "pix2struct":
        from src.pipelines.pix2struct_pipeline import Pix2StructPipeline
        return Pix2StructPipeline(device=device)
    elif model == "paddleocr":
        from src.pipelines.paddleocr_pipeline import PaddleOCRPipeline
        return PaddleOCRPipeline(lang=lang, use_gpu=(device == "cuda"))
    else:
        raise ValueError(f"Unknown model: {model}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run OCR on images using various models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["trocr", "donut", "pix2struct", "paddleocr"],
        help="OCR model to use",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
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
        help="Language for PaddleOCR (korean, en, ch, etc.)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter checkpoint (for fine-tuned models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metadata",
    )

    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1

    # Get pipeline
    print(f"Loading {args.model} model on {args.device}...")
    if hasattr(args, 'adapter_path') and args.adapter_path:
        print(f"  With LoRA adapter: {args.adapter_path}")
    try:
        pipeline = get_pipeline(
            args.model,
            args.device,
            args.lang,
            getattr(args, 'adapter_path', None)
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run OCR
    print(f"Running OCR on {image_path.name}...")
    try:
        result = pipeline(image, return_metadata=True)
    except Exception as e:
        print(f"Error running OCR: {e}")
        return 1

    # Print results
    print("\n" + "=" * 80)
    print("OCR Result:")
    print("=" * 80)
    print(result["text"])
    print("=" * 80)

    if args.verbose and "metadata" in result:
        print("\nMetadata:")
        print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
