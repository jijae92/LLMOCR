"""
FastAPI server for ONNX-optimized OCR inference.

This server uses ONNX Runtime for faster CPU/macOS arm64 inference
with support for INT8 quantized models.

Features:
- ONNX Runtime backend (optimized for CPU/arm64)
- Batch inference support
- Multiprocess workers via uvicorn
- Compatible with quantized INT8 models

Usage:
    uvicorn serve.onnx_app:app --host 0.0.0.0 --port 8001 --workers 4
"""
import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Global model cache
_onnx_pipelines = {}

# Log file
LOG_FILE = Path("onnx_inference.jsonl")


app = FastAPI(
    title="ONNX OCR API",
    description="Optimized OCR inference with ONNX Runtime",
    version="0.1.0",
)


def get_onnx_pipeline(
    model_path: str,
    processor_name: str = "microsoft/trocr-base-printed",
    device: str = "cpu",
):
    """
    Get or create ONNX pipeline (cached).

    Args:
        model_path: Path to ONNX model file
        processor_name: HuggingFace processor name
        device: Device to use

    Returns:
        ONNX pipeline instance
    """
    cache_key = f"{model_path}_{processor_name}_{device}"

    if cache_key not in _onnx_pipelines:
        # Check if encoder-only model
        if "_encoder" in model_path:
            print(f"Loading hybrid ONNX/PyTorch pipeline: {model_path}")
            from src.pipelines.onnx_pipeline import ONNXEncoderOnlyPipeline

            # Extract decoder model name from processor
            _onnx_pipelines[cache_key] = ONNXEncoderOnlyPipeline(
                encoder_path=model_path,
                decoder_model_name=processor_name,
                device=device,
            )
        else:
            print(f"Loading ONNX pipeline: {model_path}")
            from src.pipelines.onnx_pipeline import ONNXPipeline

            _onnx_pipelines[cache_key] = ONNXPipeline(
                model_path=model_path,
                processor_name=processor_name,
                device=device,
            )

    return _onnx_pipelines[cache_key]


def log_inference(log_data: dict):
    """Append inference log to JSONL file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing log: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("=" * 80)
    print("ONNX OCR Server Starting")
    print("=" * 80)
    print(f"ONNX Runtime backend for optimized inference")
    print(f"Supports CPU and macOS arm64")
    print(f"Compatible with INT8 quantized models")
    print("=" * 80)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ONNX OCR API",
        "version": "0.1.0",
        "backend": "onnxruntime",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model_path: str = Query(..., description="Path to ONNX model file"),
    processor_name: str = Query(
        "microsoft/trocr-base-printed",
        description="HuggingFace processor name"
    ),
    device: str = Query("cpu", description="Device (cpu recommended)"),
):
    """
    Run OCR inference on uploaded image using ONNX model.

    Args:
        file: Image file
        model_path: Path to ONNX model file (e.g., models/trocr.onnx)
        processor_name: HuggingFace processor for preprocessing
        device: Device to run on (cpu recommended for ONNX)

    Returns:
        JSON response with OCR result and metadata
    """
    # Check if model exists
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"ONNX model not found: {model_path}. "
                   f"Export model first: python export/export_trocr_onnx.py"
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

    # Get ONNX pipeline
    try:
        pipeline = get_onnx_pipeline(model_path, processor_name, device)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading ONNX model: {str(e)}"
        )

    # Run inference
    try:
        result = pipeline(image, return_metadata=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running ONNX inference: {str(e)}"
        )

    # Log inference
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "model_path": model_path,
        "processor": processor_name,
        "device": device,
        **result.get("metadata", {}),
    }
    log_inference(log_data)

    return JSONResponse(content=result)


@app.post("/batch_infer")
async def batch_infer(
    files: List[UploadFile] = File(...),
    model_path: str = Query(..., description="Path to ONNX model file"),
    processor_name: str = Query(
        "microsoft/trocr-base-printed",
        description="HuggingFace processor name"
    ),
    device: str = Query("cpu", description="Device"),
):
    """
    Run batch OCR inference on multiple images.

    Args:
        files: List of image files
        model_path: Path to ONNX model file
        processor_name: HuggingFace processor
        device: Device to use

    Returns:
        List of OCR results
    """
    # Check model exists
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"ONNX model not found: {model_path}"
        )

    # Get pipeline
    try:
        pipeline = get_onnx_pipeline(model_path, processor_name, device)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading ONNX model: {str(e)}"
        )

    results = []

    # Process each image
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(BytesIO(contents))

            # Run inference
            result = pipeline(image, return_metadata=True)
            result["filename"] = file.filename

            results.append(result)

            # Log
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
                "model_path": model_path,
                **result.get("metadata", {}),
            }
            log_inference(log_data)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })

    return JSONResponse(content={"results": results, "count": len(results)})


def start_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    workers: int = 1,
    reload: bool = False,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to listen on
        workers: Number of worker processes
        reload: Enable auto-reload for development
    """
    import uvicorn

    print("\n" + "=" * 80)
    print("Starting ONNX OCR Server")
    print("=" * 80)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Backend: ONNX Runtime")
    print("=" * 80 + "\n")

    uvicorn.run(
        "serve.onnx_app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start ONNX OCR server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )
