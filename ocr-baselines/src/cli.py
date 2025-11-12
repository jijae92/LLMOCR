"""Command-line interface for OCR inference."""

import argparse
import json
from pathlib import Path
from PIL import Image

from src.pipelines import (
    TrOCRPipeline,
    DonutPipeline,
    Pix2StructPipeline,
    PaddleOCRPipeline,
)
from src.pipelines.paddleocr_pipeline import is_paddleocr_available


# Model registry
MODEL_REGISTRY = {
    "trocr": {
        "class": TrOCRPipeline,
        "default_name": "microsoft/trocr-base-printed",
    },
    "donut": {
        "class": DonutPipeline,
        "default_name": "naver-clova-ix/donut-base",
    },
    "pix2struct": {
        "class": Pix2StructPipeline,
        "default_name": "google/pix2struct-base",
    },
    "paddleocr": {
        "class": PaddleOCRPipeline,
        "default_name": None,  # No HF model name for PaddleOCR
    },
}


def load_pipeline(model: str, model_name: str = None, device: str = "auto", lang: str = "korean"):
    """Load the specified OCR pipeline.

    Args:
        model: Model type ('trocr', 'donut', 'pix2struct', 'paddleocr')
        model_name: Optional custom model name (for HF models)
        device: Device to use
        lang: Language for PaddleOCR

    Returns:
        Initialized pipeline

    Raises:
        ValueError: If model type is invalid
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_config = MODEL_REGISTRY[model]
    pipeline_class = model_config["class"]

    # Special handling for PaddleOCR
    if model == "paddleocr":
        if not is_paddleocr_available():
            raise RuntimeError(
                "PaddleOCR is not available. Install with: pip install paddleocr paddlepaddle"
            )
        return pipeline_class(lang=lang, use_gpu=False)

    # For transformer-based models
    if model_name is None:
        model_name = model_config["default_name"]

    return pipeline_class(model_name=model_name, device=device)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OCR inference using various models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for OCR",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name (for HuggingFace models)",
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
        help="Language for PaddleOCR (korean, en, ch, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
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

    # Load pipeline
    try:
        pipeline = load_pipeline(
            model=args.model,
            model_name=args.model_name,
            device=args.device,
            lang=args.lang,
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return 1

    # Run inference
    try:
        result = pipeline.infer(image)
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1

    # Format output
    if args.json:
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = f"Text: {result['text']}\n"
        output += f"Latency: {result['latency_ms']:.2f} ms\n"
        output += f"Engine: {result['engine']}\n"
        output += f"Model: {result['model']}\n"

    # Save or print output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding="utf-8")
        print(f"Results saved to: {output_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    exit(main())
