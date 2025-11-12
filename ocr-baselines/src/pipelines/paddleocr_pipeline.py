"""PaddleOCR pipeline for OCR inference."""
import time
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np


class PaddleOCRPipeline:
    """PaddleOCR-based OCR pipeline (CPU-optimized, supports Korean)."""

    def __init__(
        self,
        lang: str = "korean",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize PaddleOCR pipeline.

        Args:
            lang: Language code ('korean', 'en', 'ch', etc.)
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU (False for CPU-only)
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install with: pip install paddleocr"
            )

        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu

        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            show_log=False,
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for OCR.

        Args:
            image: Input PIL Image

        Returns:
            Numpy array (RGB format)
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

        # Convert to numpy array
        img_array = np.array(image)
        return img_array

    def postprocess(self, ocr_result: List) -> str:
        """
        Postprocess OCR output.

        Args:
            ocr_result: PaddleOCR result list

        Returns:
            Concatenated text from all detected boxes
        """
        if not ocr_result or ocr_result[0] is None:
            return ""

        # Extract text from all boxes
        texts = []
        for line in ocr_result[0]:
            if len(line) >= 2:
                text = line[1][0]  # line[1] is (text, confidence)
                texts.append(text)

        # Join with newlines
        full_text = "\n".join(texts)
        return full_text.strip()

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
        img_array = self.preprocess(image)

        # Run OCR
        try:
            ocr_result = self.ocr.ocr(img_array, cls=self.use_angle_cls)
        except Exception as e:
            # Handle PaddleOCR errors gracefully
            ocr_result = [[]]
            error_msg = str(e)
            if return_metadata:
                return {
                    "text": "",
                    "metadata": {
                        "model": "paddleocr",
                        "engine": "paddle",
                        "device": "gpu" if self.use_gpu else "cpu",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "error": error_msg,
                        "decode_params": {
                            "lang": self.lang,
                            "use_angle_cls": self.use_angle_cls,
                        },
                    },
                }
            return {"text": ""}

        # Postprocess
        text = self.postprocess(ocr_result)

        latency_ms = (time.time() - start_time) * 1000

        result = {"text": text}

        if return_metadata:
            result["metadata"] = {
                "model": "paddleocr",
                "engine": "paddle",
                "device": "gpu" if self.use_gpu else "cpu",
                "latency_ms": latency_ms,
                "decode_params": {
                    "lang": self.lang,
                    "use_angle_cls": self.use_angle_cls,
                },
            }

        return result
