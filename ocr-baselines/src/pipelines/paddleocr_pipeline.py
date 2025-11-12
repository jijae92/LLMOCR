"""PaddleOCR pipeline for OCR inference."""

import time
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    PADDLEOCR_IMPORT_ERROR = str(e)


class PaddleOCRPipeline:
    """PaddleOCR pipeline for lightweight CPU-based OCR.

    Supports Korean language pack and runs efficiently on CPU.
    """

    def __init__(
        self,
        lang: str = "korean",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        show_log: bool = False,
    ):
        """Initialize PaddleOCR pipeline.

        Args:
            lang: Language to use ('korean', 'en', 'ch', etc.)
            use_angle_cls: Use angle classification model
            use_gpu: Use GPU (typically False for macOS)
            show_log: Show PaddleOCR logs

        Raises:
            RuntimeError: If PaddleOCR is not available
        """
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError(
                f"PaddleOCR not available: {PADDLEOCR_IMPORT_ERROR}\n"
                "This is an optional dependency. Install with: pip install paddleocr paddlepaddle"
            )

        self.lang = lang
        self.model_name = f"paddleocr-{lang}"

        print(f"Initializing PaddleOCR with language: {lang}")

        try:
            self.ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                use_gpu=use_gpu,
                show_log=show_log,
            )
            print("PaddleOCR initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image: Image.Image, target_size: int = 1280) -> np.ndarray:
        """Preprocess image: resize, convert to numpy array.

        Args:
            image: Input PIL image
            target_size: Target size for longest edge

        Returns:
            Preprocessed numpy array
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

        # Convert to numpy array
        return np.array(image)

    def postprocess_results(self, results: List) -> str:
        """Extract and combine text from PaddleOCR results.

        Args:
            results: PaddleOCR raw results

        Returns:
            Combined text string
        """
        if not results or not results[0]:
            return ""

        # Extract text from each detection
        texts = []
        for line in results[0]:
            if len(line) >= 2:
                # line[0] is bounding box, line[1] is (text, confidence)
                text, confidence = line[1]
                texts.append(text)

        # Join texts with newline
        combined_text = "\n".join(texts)

        # Basic cleanup
        combined_text = combined_text.strip()

        return combined_text

    def infer(
        self,
        image: Image.Image,
        **ocr_kwargs: Any
    ) -> Dict[str, Any]:
        """Run OCR inference on an image.

        Args:
            image: Input PIL image
            **ocr_kwargs: Additional PaddleOCR parameters

        Returns:
            Dictionary with:
                - text: Recognized text
                - latency_ms: Inference time in milliseconds
                - engine: 'paddleocr'
                - model: Model name
        """
        start_time = time.time()

        # Preprocess
        img_array = self.preprocess_image(image)

        # Run OCR
        results = self.ocr.ocr(img_array, **ocr_kwargs)

        # Postprocess
        text = self.postprocess_results(results)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "text": text,
            "latency_ms": latency_ms,
            "engine": "paddleocr",
            "model": self.model_name,
        }


def is_paddleocr_available() -> bool:
    """Check if PaddleOCR is available.

    Returns:
        True if PaddleOCR can be imported, False otherwise
    """
    return PADDLEOCR_AVAILABLE
