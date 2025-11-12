"""FastAPI server for OCR inference."""
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="OCR Baselines API",
    description="REST API for OCR inference using multiple models",
    version="0.1.0",
)

# Global pipeline cache
_pipelines = {}

# Log file
LOG_FILE = Path("ocr_inference.jsonl")


def get_pipeline(model: str, device: str = "auto", lang: str = "korean"):
    """
    Get or create OCR pipeline (cached).

    Args:
        model: Model name
        device: Device to use
        lang: Language for PaddleOCR

    Returns:
        Pipeline instance
    """
    cache_key = f"{model}_{device}_{lang}"

    if cache_key not in _pipelines:
        if model == "trocr":
            from src.pipelines.trocr_pipeline import TrOCRPipeline
            _pipelines[cache_key] = TrOCRPipeline(device=device)
        elif model == "donut":
            from src.pipelines.donut_pipeline import DonutPipeline
            _pipelines[cache_key] = DonutPipeline(device=device)
        elif model == "pix2struct":
            from src.pipelines.pix2struct_pipeline import Pix2StructPipeline
            _pipelines[cache_key] = Pix2StructPipeline(device=device)
        elif model == "paddleocr":
            from src.pipelines.paddleocr_pipeline import PaddleOCRPipeline
            _pipelines[cache_key] = PaddleOCRPipeline(
                lang=lang,
                use_gpu=(device == "cuda")
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    return _pipelines[cache_key]


def log_inference(log_data: dict):
    """Append inference log to JSONL file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing log: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "OCR Baselines API",
        "version": "0.1.0",
        "models": ["trocr", "donut", "pix2struct", "paddleocr"],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Query(..., description="Model to use"),
    device: str = Query("auto", description="Device (auto/cpu/cuda/mps)"),
    lang: str = Query("korean", description="Language for PaddleOCR"),
):
    """
    Run OCR inference on uploaded image.

    Args:
        file: Image file
        model: Model name (trocr, donut, pix2struct, paddleocr)
        device: Device to run on
        lang: Language for PaddleOCR

    Returns:
        JSON response with OCR result and metadata
    """
    # Validate model
    valid_models = ["trocr", "donut", "pix2struct", "paddleocr"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Must be one of: {valid_models}"
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading image: {str(e)}"
        )

    # Get pipeline
    try:
        pipeline = get_pipeline(model, device, lang)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )

    # Run inference
    try:
        result = pipeline(image, return_metadata=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running inference: {str(e)}"
        )

    # Log inference
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "model": model,
        "device": device,
        "lang": lang,
        **result.get("metadata", {}),
    }
    log_inference(log_data)

    return JSONResponse(content=result)


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "trocr",
                "description": "TrOCR (Vision Encoder-Decoder)",
                "default_checkpoint": "microsoft/trocr-base-printed",
            },
            {
                "name": "donut",
                "description": "Donut (Swin + BART)",
                "default_checkpoint": "naver-clova-ix/donut-base",
            },
            {
                "name": "pix2struct",
                "description": "Pix2Struct (Image-to-Text)",
                "default_checkpoint": "google/pix2struct-base",
            },
            {
                "name": "paddleocr",
                "description": "PaddleOCR (CPU-optimized, multilingual)",
                "languages": ["korean", "en", "ch", "japan", "fr", "german"],
            },
        ]
    }


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    uvicorn.run(
        "src.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    start_server(reload=True)
