"""
Export TrOCR model to ONNX format.

This script exports TrOCR (VisionEncoderDecoder) models to ONNX format
for optimized inference on CPU/MPS (macOS arm64).

Note: VisionEncoderDecoder models have known limitations with ONNX export:
- Dynamic input dimensions
- Past key values handling
- Cross-attention states

We use optimum.exporters.onnx as primary method with fallback options.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def export_with_optimum(
    model_name: str,
    output_dir: Path,
    opset: int = 14,
) -> bool:
    """
    Export using optimum.exporters.onnx (recommended).

    Args:
        model_name: HuggingFace model name or path
        output_dir: Output directory for ONNX model
        opset: ONNX opset version

    Returns:
        True if successful, False otherwise
    """
    try:
        from optimum.exporters.onnx import main_export

        print("=" * 80)
        print("Exporting with optimum.exporters.onnx...")
        print("=" * 80)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export using optimum
        # Note: This may fail for VisionEncoderDecoder models
        main_export(
            model_name_or_path=model_name,
            output=str(output_dir),
            opset=opset,
        )

        print(f"\n✅ Successfully exported to: {output_dir}")
        return True

    except Exception as e:
        print(f"\n❌ Optimum export failed: {e}")
        print("\nThis is a known limitation with VisionEncoderDecoder models.")
        print("See: https://github.com/huggingface/optimum/issues")
        return False


def export_with_torch_onnx(
    model_name: str,
    output_path: Path,
    opset: int = 14,
) -> bool:
    """
    Export using torch.onnx (fallback method).

    This exports only the encoder-decoder structure, not a complete
    generation pipeline. For full generation, use ONNX Runtime with
    custom decoding logic.

    Args:
        model_name: HuggingFace model name or path
        output_path: Output ONNX file path
        opset: ONNX opset version

    Returns:
        True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("Exporting with torch.onnx (fallback)...")
        print("=" * 80)

        # Load model and processor
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()

        # Create dummy inputs
        from PIL import Image
        import numpy as np

        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        )

        inputs = processor(images=dummy_image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n⚠️  Warning: Encoder-only export")
        print("Full generation pipeline requires custom ONNX Runtime logic")

        # Export encoder only (more stable)
        encoder_path = output_path.parent / f"{output_path.stem}_encoder.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model.encoder,
                pixel_values,
                str(encoder_path),
                opset_version=opset,
                input_names=["pixel_values"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "last_hidden_state": {0: "batch_size"},
                },
            )

        print(f"\n✅ Encoder exported to: {encoder_path}")
        print("\nNote: Decoder export is complex due to autoregressive generation.")
        print("For production, consider:")
        print("  1. Use PyTorch inference with torch.jit.script")
        print("  2. Use ONNX Runtime Execution Provider (ORT EP)")
        print("  3. Wait for improved optimum support")

        return True

    except Exception as e:
        print(f"\n❌ Torch ONNX export failed: {e}")
        return False


def export_with_torchscript(
    model_name: str,
    output_path: Path,
) -> bool:
    """
    Export using TorchScript (alternative to ONNX).

    TorchScript provides better compatibility with encoder-decoder models
    and can be used with ONNX Runtime Execution Provider.

    Args:
        model_name: HuggingFace model name or path
        output_path: Output .pt file path

    Returns:
        True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("Exporting with TorchScript (alternative)...")
        print("=" * 80)

        # Load model
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to script the model
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(output_path))
            print(f"\n✅ TorchScript model saved to: {output_path}")
            return True
        except Exception as script_error:
            print(f"\n⚠️  torch.jit.script failed: {script_error}")
            print("Trying torch.jit.trace...")

            # Fallback to trace
            from PIL import Image
            import numpy as np
            from transformers import TrOCRProcessor

            processor = TrOCRProcessor.from_pretrained(model_name)
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
            )
            inputs = processor(images=dummy_image, return_tensors="pt")

            # Trace only works for specific inputs
            print("\n⚠️  TorchScript trace is input-specific and not recommended")
            print("for generative models. Skipping...")
            return False

    except Exception as e:
        print(f"\n❌ TorchScript export failed: {e}")
        return False


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export TrOCR model to ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trocr-onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (for torch.onnx fallback)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "optimum", "torch", "torchscript"],
        help="Export method",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TrOCR ONNX Export")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"ONNX Opset: {args.opset}")

    if args.adapter_path:
        print(f"\n⚠️  Note: LoRA adapter export not yet supported")
        print("Merge adapter weights into base model first:")
        print("  from peft import PeftModel")
        print("  model = PeftModel.from_pretrained(base_model, adapter_path)")
        print("  merged_model = model.merge_and_unload()")
        print("  merged_model.save_pretrained('merged_model')")
        print("\nThen export the merged model.")
        return 1

    print("\n" + "=" * 80)
    print("IMPORTANT: VisionEncoderDecoder ONNX Export Limitations")
    print("=" * 80)
    print("""
TrOCR uses a VisionEncoderDecoder architecture which has known challenges
with ONNX export:

1. Dynamic Shapes: Input dimensions can vary
2. Autoregressive Generation: Decoder requires past key values
3. Cross-Attention: Complex attention mechanisms

Recommendations:
- For CPU inference: Use PyTorch with torch.compile() (PyTorch 2.0+)
- For deployment: Use ONNX Runtime Execution Provider (ORT EP) with PyTorch
- For maximum compatibility: Keep using PyTorch inference

We'll attempt export with the following priority:
1. optimum.exporters.onnx (recommended but may fail)
2. torch.onnx (encoder-only, limited)
3. TorchScript (alternative to ONNX)
""")

    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file) if args.output_file else output_dir / "model.onnx"

    success = False

    # Try methods in order
    if args.method == "auto":
        # Try optimum first
        if export_with_optimum(args.model, output_dir, args.opset):
            success = True
        # Fallback to torch.onnx
        elif export_with_torch_onnx(args.model, output_file, args.opset):
            success = True
        # Last resort: TorchScript
        else:
            ts_path = output_dir / "model_torchscript.pt"
            if export_with_torchscript(args.model, ts_path):
                success = True

    elif args.method == "optimum":
        success = export_with_optimum(args.model, output_dir, args.opset)

    elif args.method == "torch":
        success = export_with_torch_onnx(args.model, output_file, args.opset)

    elif args.method == "torchscript":
        ts_path = output_dir / "model_torchscript.pt"
        success = export_with_torchscript(args.model, ts_path)

    if success:
        print("\n" + "=" * 80)
        print("Export completed!")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"1. Test exported model")
        print(f"2. Quantize for faster inference (optional):")
        print(f"   python export/quantize_onnx.py --input {output_file}")
        print(f"3. Deploy with ONNX Runtime")
        return 0
    else:
        print("\n" + "=" * 80)
        print("Export failed!")
        print("=" * 80)
        print("\nRecommendation: Use PyTorch inference for TrOCR")
        print("PyTorch provides:")
        print("  - Better compatibility with encoder-decoder models")
        print("  - torch.compile() for optimization (PyTorch 2.0+)")
        print("  - MPS acceleration on macOS")
        print("\nFor serving, continue using src/pipelines/trocr_pipeline.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
