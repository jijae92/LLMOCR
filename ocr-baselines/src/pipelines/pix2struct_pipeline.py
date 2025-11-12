"""Pix2Struct pipeline for OCR inference."""

import time
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


class Pix2StructPipeline:
    """Pix2Struct pipeline for image-to-text tasks.

    Designed for visual language understanding tasks including OCR.
    """

    def __init__(
        self,
        model_name: str = "google/pix2struct-base",
        device: str = "auto",
        max_length: int = 512,
    ):
        """Initialize Pix2Struct pipeline.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
            max_length: Maximum sequence length for generation
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading Pix2Struct model: {model_name} on {self.device}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Pix2Struct model loaded successfully")

    def preprocess_image(self, image: Image.Image, target_size: int = 1280) -> Image.Image:
        """Preprocess image: resize, normalize.

        Args:
            image: Input PIL image
            target_size: Target size for longest edge

        Returns:
            Preprocessed PIL image
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if needed (keep aspect ratio)
        width, height = image.size
        if max(width, height) > target_size:
            if width > height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def postprocess_text(self, text: str) -> str:
        """Basic postprocessing of output text.

        Args:
            text: Raw model output

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()

    def infer(
        self,
        image: Image.Image,
        text_prompt: str = "Extract all text from this image:",
        **generation_kwargs: Any
    ) -> Dict[str, Any]:
        """Run OCR inference on an image.

        Args:
            image: Input PIL image
            text_prompt: Optional text prompt to guide extraction
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with:
                - text: Recognized text
                - latency_ms: Inference time in milliseconds
                - engine: 'pytorch'
                - model: Model name
        """
        start_time = time.time()

        # Preprocess
        image = self.preprocess_image(image)

        # Process image and text
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                **generation_kwargs
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Postprocess
        text = self.postprocess_text(generated_text)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "text": text,
            "latency_ms": latency_ms,
            "engine": "pytorch",
            "model": self.model_name,
        }
