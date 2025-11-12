"""Pix2Struct pipeline for OCR inference."""
import time
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


class Pix2StructPipeline:
    """Pix2Struct-based OCR pipeline."""

    def __init__(
        self,
        model_name: str = "google/pix2struct-base",
        device: str = "auto",
        max_length: int = 512,
    ):
        """
        Initialize Pix2Struct pipeline.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._get_device(device)

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self, device: str) -> str:
        """Get appropriate device based on availability."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for OCR.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
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

        return image

    def postprocess(self, text: str) -> str:
        """
        Postprocess OCR output.

        Args:
            text: Raw OCR output

        Returns:
            Cleaned text
        """
        # Basic cleanup
        text = text.strip()
        return text

    def __call__(
        self,
        image: Image.Image,
        return_metadata: bool = False,
        prompt: str = "Extract all text from the image.",
    ) -> Dict[str, Any]:
        """
        Run OCR on image.

        Args:
            image: Input PIL Image
            return_metadata: Whether to return inference metadata
            prompt: Text prompt for the model

        Returns:
            Dictionary with 'text' and optionally 'metadata'
        """
        start_time = time.time()

        # Preprocess
        image = self.preprocess(image)

        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Postprocess
        text = self.postprocess(generated_text)

        latency_ms = (time.time() - start_time) * 1000

        result = {"text": text}

        if return_metadata:
            result["metadata"] = {
                "model": self.model_name,
                "engine": "pytorch",
                "device": str(self.device),
                "latency_ms": latency_ms,
                "decode_params": {
                    "max_length": self.max_length,
                    "prompt": prompt,
                },
            }

        return result
