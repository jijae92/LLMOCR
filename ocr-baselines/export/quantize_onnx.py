"""
Quantize ONNX models to INT8 for faster inference.

This script performs Post-Training Quantization (PTQ) on ONNX models
using ONNX Runtime's quantization tools.

Supports:
- Dynamic Quantization: Fast, no calibration data needed
- Static Quantization: Better accuracy, requires calibration data

Optimized for CPU and macOS arm64.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List

try:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        CalibrationDataReader,
    )
    ONNX_QUANT_AVAILABLE = True
except ImportError:
    ONNX_QUANT_AVAILABLE = False
    print("Warning: onnxruntime quantization not available")


class DummyCalibrationDataReader(CalibrationDataReader):
    """
    Dummy calibration data reader for static quantization.

    In production, replace this with real calibration data from your dataset.
    """

    def __init__(self, num_samples: int = 100):
        """
        Initialize calibration data reader.

        Args:
            num_samples: Number of calibration samples
        """
        self.num_samples = num_samples
        self.current_sample = 0

    def get_next(self) -> Optional[dict]:
        """Get next calibration sample."""
        if self.current_sample >= self.num_samples:
            return None

        # Create dummy input (replace with real data)
        import numpy as np

        # TrOCR typical input shape: (1, 3, 384, 384)
        dummy_input = {
            "pixel_values": np.random.randn(1, 3, 384, 384).astype(np.float32)
        }

        self.current_sample += 1
        return dummy_input


def quantize_dynamic_model(
    input_model: Path,
    output_model: Path,
    op_types_to_quantize: Optional[List[str]] = None,
) -> bool:
    """
    Perform dynamic quantization on ONNX model.

    Dynamic quantization:
    - No calibration data needed
    - Fast quantization
    - Good for CPU inference
    - Slightly lower accuracy than static

    Args:
        input_model: Path to input ONNX model
        output_model: Path to output quantized model
        op_types_to_quantize: Operator types to quantize (default: MatMul, Gemm)

    Returns:
        True if successful
    """
    if not ONNX_QUANT_AVAILABLE:
        print("Error: onnxruntime quantization not installed")
        return False

    try:
        print("=" * 80)
        print("Dynamic INT8 Quantization")
        print("=" * 80)
        print(f"Input: {input_model}")
        print(f"Output: {output_model}")

        # Default ops to quantize
        if op_types_to_quantize is None:
            op_types_to_quantize = ["MatMul", "Gemm", "Attention"]

        print(f"Quantizing ops: {op_types_to_quantize}")

        # Perform dynamic quantization
        quantize_dynamic(
            model_input=str(input_model),
            model_output=str(output_model),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types_to_quantize,
        )

        print(f"\n✅ Dynamic quantization completed!")
        print(f"Output: {output_model}")

        # Compare sizes
        input_size = input_model.stat().st_size / (1024 * 1024)
        output_size = output_model.stat().st_size / (1024 * 1024)
        reduction = (1 - output_size / input_size) * 100

        print(f"\nModel size:")
        print(f"  Original: {input_size:.2f} MB")
        print(f"  Quantized: {output_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        print(f"\n❌ Dynamic quantization failed: {e}")
        return False


def quantize_static_model(
    input_model: Path,
    output_model: Path,
    calibration_data_reader: Optional[CalibrationDataReader] = None,
) -> bool:
    """
    Perform static quantization on ONNX model.

    Static quantization:
    - Requires calibration data
    - Better accuracy than dynamic
    - Slower quantization process
    - Recommended for production

    Args:
        input_model: Path to input ONNX model
        output_model: Path to output quantized model
        calibration_data_reader: Calibration data reader

    Returns:
        True if successful
    """
    if not ONNX_QUANT_AVAILABLE:
        print("Error: onnxruntime quantization not installed")
        return False

    try:
        print("=" * 80)
        print("Static INT8 Quantization")
        print("=" * 80)
        print(f"Input: {input_model}")
        print(f"Output: {output_model}")

        # Use dummy calibration data if none provided
        if calibration_data_reader is None:
            print("\n⚠️  Using dummy calibration data")
            print("For production, provide real calibration data from your dataset")
            calibration_data_reader = DummyCalibrationDataReader(num_samples=10)

        print("\nPerforming static quantization...")
        print("This may take a while...")

        # Perform static quantization
        quantize_static(
            model_input=str(input_model),
            model_output=str(output_model),
            calibration_data_reader=calibration_data_reader,
        )

        print(f"\n✅ Static quantization completed!")
        print(f"Output: {output_model}")

        # Compare sizes
        input_size = input_model.stat().st_size / (1024 * 1024)
        output_size = output_model.stat().st_size / (1024 * 1024)
        reduction = (1 - output_size / input_size) * 100

        print(f"\nModel size:")
        print(f"  Original: {input_size:.2f} MB")
        print(f"  Quantized: {output_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        print(f"\n❌ Static quantization failed: {e}")
        return False


def main():
    """Main quantization function."""
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output quantized model path (default: input_int8.onnx)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for static quantization",
    )

    args = parser.parse_args()

    if not ONNX_QUANT_AVAILABLE:
        print("Error: onnxruntime quantization not installed")
        print("\nInstall with:")
        print("  pip install onnxruntime")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_int8.onnx"

    print("\n" + "=" * 80)
    print("ONNX Model Quantization")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Perform quantization
    if args.mode == "dynamic":
        success = quantize_dynamic_model(input_path, output_path)
    else:  # static
        calibration_reader = DummyCalibrationDataReader(
            num_samples=args.calibration_samples
        )
        success = quantize_static_model(
            input_path,
            output_path,
            calibration_reader,
        )

    if success:
        print("\n" + "=" * 80)
        print("Quantization completed!")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"1. Test quantized model for accuracy")
        print(f"2. Benchmark latency improvement")
        print(f"3. Deploy with ONNX Runtime:")
        print(f"   python serve/onnx_app.py --model {output_path}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("Quantization failed!")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("  - Check if input model is valid ONNX")
        print("  - Try different quantization mode")
        print("  - Check onnxruntime version compatibility")
        return 1


if __name__ == "__main__":
    sys.exit(main())
