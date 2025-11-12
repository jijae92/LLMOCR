"""Donut pipeline for OCR inference."""
import time
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


class DonutPipeline:
    """Donut-based OCR pipeline (Swin encoder + BART decoder)."""

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        device: str = "auto",
        max_length: int = 512,
        task_prompt: str = "<s_cord-v2>",
    ):
        """
        Initialize Donut pipeline.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            max_length: Maximum generation length
            task_prompt: Task prompt for Donut model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.task_prompt = task_prompt
        self.device = self._get_device(device)

        # Load processor and model
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
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
        # Remove task prompt if present
        if self.task_prompt in text:
            text = text.replace(self.task_prompt, "")

        # Remove special tokens
        text = text.replace("</s>", "").replace("<s>", "")

        # Basic cleanup
        text = text.strip()
        return text

    def __call__(
        self,
        image: Image.Image,
        return_metadata: bool = False,
    ) -> Dict[str, Any]:
        """
        Run OCR on image.

        Args:
            image: Input PIL Image
            return_metadata: Whether to return inference metadata

        Returns:
            Dictionary with 'text' and optionally 'metadata'
        """
        start_time = time.time()

        # Preprocess
        image = self.preprocess(image)

        # Prepare inputs
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Prepare decoder input (task prompt)
        task_prompt_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                decoder_input_ids=task_prompt_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
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
                    "task_prompt": self.task_prompt,
                },
            }

        return result
