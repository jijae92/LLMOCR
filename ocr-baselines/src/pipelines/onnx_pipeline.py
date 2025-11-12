"""
ONNX Runtime pipeline for optimized OCR inference.

This pipeline uses ONNX Runtime for faster CPU/macOS arm64 inference.
Supports quantized INT8 models for additional speedup.
"""
import time
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXPipeline:
    """ONNX Runtime-based OCR pipeline."""

    def __init__(
        self,
        model_path: str,
        processor_name: str = "microsoft/trocr-base-printed",
        device: str = "cpu",
    ):
        """
        Initialize ONNX pipeline.

        Args:
            model_path: Path to ONNX model file
            processor_name: HuggingFace processor name for preprocessing
            device: Device to run on ('cpu' recommended for ONNX)
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )

        self.model_path = model_path
        self.processor_name = processor_name
        self.device = device

        # Load processor
        from transformers import TrOCRProcessor
        self.processor = TrOCRProcessor.from_pretrained(processor_name)

        # Create ONNX Runtime session
        self.session = self._create_session(model_path, device)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"ONNX model loaded: {model_path}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")

    def _create_session(self, model_path: str, device: str) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session.

        Args:
            model_path: Path to ONNX model
            device: Device to run on

        Returns:
            InferenceSession
        """
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Execution providers
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # For macOS arm64, ensure CPU provider is optimized
        print(f"Using execution providers: {providers}")

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        return session

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for ONNX inference.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed numpy array
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        max_dim = 1280
        width, height = image.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Use processor to get pixel values
        pixel_values = self.processor(
            images=image,
            return_tensors="np"  # Return numpy for ONNX
        ).pixel_values

        return pixel_values

    def postprocess(self, output_ids: np.ndarray) -> str:
        """
        Postprocess ONNX output.

        Args:
            output_ids: Generated token IDs from ONNX

        Returns:
            Decoded text
        """
        # Decode token IDs to text
        text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()

    def __call__(
        self,
        image: Image.Image,
        return_metadata: bool = False,
    ) -> Dict[str, Any]:
        """
        Run OCR on image using ONNX.

        Note: This is a simplified version. Full encoder-decoder
        generation may require custom logic.

        Args:
            image: Input PIL Image
            return_metadata: Whether to return inference metadata

        Returns:
            Dictionary with 'text' and optionally 'metadata'
        """
        start_time = time.time()

        # Preprocess
        pixel_values = self.preprocess(image)

        # Prepare inputs
        ort_inputs = {
            self.input_names[0]: pixel_values
        }

        # Run inference
        try:
            ort_outputs = self.session.run(
                self.output_names,
                ort_inputs
            )

            # Extract output (this depends on model structure)
            # For encoder-only: use output embedding
            # For full model: use generated IDs
            output_ids = ort_outputs[0]

            # Postprocess
            if isinstance(output_ids, np.ndarray):
                # If output is token IDs
                text = self.postprocess(output_ids)
            else:
                # If output is embeddings or other format
                text = "[ONNX inference - decoder not implemented]"

        except Exception as e:
            print(f"ONNX inference error: {e}")
            text = f"[Error: {str(e)}]"

        latency_ms = (time.time() - start_time) * 1000

        result = {"text": text}

        if return_metadata:
            result["metadata"] = {
                "model": self.model_path,
                "engine": "onnx",
                "device": self.device,
                "latency_ms": latency_ms,
                "providers": self.session.get_providers(),
            }

        return result


class ONNXEncoderOnlyPipeline:
    """
    ONNX pipeline for encoder-only models.

    This is a fallback when full encoder-decoder export fails.
    Extracts features from encoder, then uses PyTorch decoder.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_model_name: str = "microsoft/trocr-base-printed",
        device: str = "cpu",
    ):
        """
        Initialize hybrid ONNX/PyTorch pipeline.

        Args:
            encoder_path: Path to ONNX encoder model
            decoder_model_name: HuggingFace decoder model name
            device: Device for PyTorch decoder
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed")

        self.encoder_path = encoder_path
        self.decoder_model_name = decoder_model_name

        # Load ONNX encoder
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder_session = ort.InferenceSession(
            encoder_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load PyTorch decoder
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
        import torch

        self.processor = TrOCRProcessor.from_pretrained(decoder_model_name)
        full_model = VisionEncoderDecoderModel.from_pretrained(decoder_model_name)
        self.decoder = full_model.decoder

        # Move decoder to device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.decoder.to(device)
        self.decoder.eval()

        print(f"Hybrid ONNX/PyTorch pipeline:")
        print(f"  Encoder (ONNX): {encoder_path}")
        print(f"  Decoder (PyTorch): {decoder_model_name} on {device}")

    def __call__(
        self,
        image: Image.Image,
        return_metadata: bool = False,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """Run hybrid ONNX encoder + PyTorch decoder inference."""
        import torch

        start_time = time.time()

        # Preprocess
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="np"
        ).pixel_values

        # ONNX encoder inference
        encoder_outputs = self.encoder_session.run(
            None,
            {"pixel_values": pixel_values}
        )

        encoder_hidden_states = torch.from_numpy(encoder_outputs[0]).to(self.device)

        # PyTorch decoder generation
        with torch.no_grad():
            # Simple greedy decoding
            decoder_input_ids = torch.tensor(
                [[self.processor.tokenizer.bos_token_id]],
                device=self.device
            )

            for _ in range(max_length):
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                )

                next_token_logits = decoder_outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

                if next_token.item() == self.processor.tokenizer.eos_token_id:
                    break

        # Decode
        text = self.processor.tokenizer.decode(
            decoder_input_ids[0],
            skip_special_tokens=True
        )

        latency_ms = (time.time() - start_time) * 1000

        result = {"text": text.strip()}

        if return_metadata:
            result["metadata"] = {
                "model": f"{self.encoder_path} + {self.decoder_model_name}",
                "engine": "onnx+pytorch",
                "device": self.device,
                "latency_ms": latency_ms,
            }

        return result
