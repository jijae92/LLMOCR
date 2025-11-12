"""FastAPI server for OCR inference."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

from src.pipelines import (
    TrOCRPipeline,
    DonutPipeline,
    Pix2StructPipeline,
    PaddleOCRPipeline,
)
from src.pipelines.paddleocr_pipeline import is_paddleocr_available


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="OCR Baselines API",
    description="Multi-model OCR inference API supporting TrOCR, Donut, Pix2Struct, and PaddleOCR",
    version="0.1.0",
)

# Global pipeline cache
_pipelines = {}

# Log file for requests
LOG_FILE = Path("ocr_requests.jsonl")


def get_pipeline(model: str, lang: str = "korean", device: str = "auto"):
    """Get or create a pipeline instance.

    Args:
        model: Model type ('trocr', 'donut', 'pix2struct', 'paddleocr')
        lang: Language for PaddleOCR
        device: Device to use

    Returns:
        Pipeline instance

    Raises:
        HTTPException: If model is invalid or unavailable
    """
    # Create cache key
    cache_key = f"{model}_{lang}_{device}"

    # Return cached pipeline if available
    if cache_key in _pipelines:
        return _pipelines[cache_key]

    # Create new pipeline
    try:
        if model == "trocr":
            pipeline = TrOCRPipeline(device=device)
        elif model == "donut":
            pipeline = DonutPipeline(device=device)
        elif model == "pix2struct":
            pipeline = Pix2StructPipeline(device=device)
        elif model == "paddleocr":
            if not is_paddleocr_available():
                raise HTTPException(
                    status_code=501,
                    detail="PaddleOCR is not available on this server",
                )
            pipeline = PaddleOCRPipeline(lang=lang, use_gpu=False)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model}. Available: trocr, donut, pix2struct, paddleocr",
            )

        # Cache the pipeline
        _pipelines[cache_key] = pipeline
        logger.info(f"Created new pipeline: {cache_key}")

        return pipeline

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating pipeline {model}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}",
        )


def log_request(result: dict, model: str, lang: str, device: str):
    """Log request to JSONL file.

    Args:
        result: Inference result
        model: Model name
        lang: Language
        device: Device used
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "lang": lang,
        "device": device,
        "latency_ms": result.get("latency_ms"),
        "engine": result.get("engine"),
        "text_length": len(result.get("text", "")),
    }

    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error writing log: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OCR Baselines API",
        "version": "0.1.0",
        "models": ["trocr", "donut", "pix2struct", "paddleocr"],
        "endpoints": {
            "/infer": "POST - Run OCR inference on an image",
            "/health": "GET - Health check",
            "/models": "GET - List available models",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded_models": list(_pipelines.keys()),
    }


@app.get("/models")
async def list_models():
    """List available models and their status."""
    models = {
        "trocr": {
            "available": True,
            "description": "Microsoft TrOCR - Vision Encoder Decoder",
        },
        "donut": {
            "available": True,
            "description": "Naver Donut - Document Understanding Transformer",
        },
        "pix2struct": {
            "available": True,
            "description": "Google Pix2Struct - Image-to-Text",
        },
        "paddleocr": {
            "available": is_paddleocr_available(),
            "description": "PaddleOCR - Lightweight CPU-based OCR",
        },
    }
    return models


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Query("trocr", description="Model to use"),
    lang: str = Query("korean", description="Language (for PaddleOCR)"),
    device: str = Query("auto", description="Device to use (auto, cpu, mps, cuda)"),
):
    """Run OCR inference on an uploaded image.

    Args:
        file: Image file to process
        model: Model to use ('trocr', 'donut', 'pix2struct', 'paddleocr')
        lang: Language for PaddleOCR
        device: Device to use

    Returns:
        JSON response with recognized text and metadata
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*",
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Get pipeline
        pipeline = get_pipeline(model=model, lang=lang, device=device)

        # Run inference
        result = pipeline.infer(image)

        # Log request
        log_request(result, model, lang, device)

        return JSONResponse(
            content={
                "success": True,
                "result": result,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
