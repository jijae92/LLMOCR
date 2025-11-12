# OCR Baselines

Multi-model OCR baseline comparison for Korean documents. This project provides a unified interface for running and evaluating various OCR models including TrOCR, Donut, Pix2Struct, and PaddleOCR.

## Features

- Multiple OCR models in a single repository
- Command-line interface (CLI) for quick inference
- REST API server for production deployment
- Comprehensive evaluation metrics (CER, WER, latency)
- Support for macOS MPS (Metal Performance Shaders) acceleration
- Easy model comparison and benchmarking

## Supported Models

| Model | Type | Engine | Description |
|-------|------|--------|-------------|
| **TrOCR** | Transformer | PyTorch | Microsoft's Vision Encoder-Decoder model for OCR |
| **Donut** | Transformer | PyTorch | Naver's Document Understanding Transformer (Swin + BART) |
| **Pix2Struct** | Transformer | PyTorch | Google's image-to-text model for visual language understanding |
| **PaddleOCR** | Lightweight | PaddlePaddle | CPU-optimized OCR with Korean language support |

## Installation

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3) for MPS support, or Linux/Windows

### Quick Start

```bash
# Clone the repository
cd LLMOCR/ocr-baselines

# Install dependencies
pip install -e .

# Or install from pyproject.toml
pip install -r <(python -c 'import tomllib; f=open("pyproject.toml","rb"); print("\n".join(tomllib.load(f)["project"]["dependencies"]))')
```

### Development Installation

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Command-Line Interface (CLI)

Run OCR inference on a single image:

```bash
# Using TrOCR (default model)
python -m src.cli --model trocr --image tests/data_samples/receipt_ko.jpg --device auto

# Using Donut
python -m src.cli --model donut --image tests/data_samples/invoice_ko.jpg --device auto

# Using Pix2Struct
python -m src.cli --model pix2struct --image tests/data_samples/receipt_ko.jpg --device mps

# Using PaddleOCR (Korean)
python -m src.cli --model paddleocr --image tests/data_samples/invoice_ko.jpg --lang korean

# Output as JSON
python -m src.cli --model trocr --image image.jpg --json --output result.json
```

#### CLI Options

- `--model`: Model to use (`trocr`, `donut`, `pix2struct`, `paddleocr`)
- `--image`: Path to input image
- `--device`: Device to use (`auto`, `cpu`, `mps`, `cuda`)
- `--lang`: Language for PaddleOCR (default: `korean`)
- `--output`: Save results to file (optional)
- `--json`: Output results as JSON

### REST API Server

Start the FastAPI server:

```bash
# Start server (default port 8000)
uvicorn src.server.app:app --port 8000 --reload

# Or run directly
python -m src.server.app
```

#### API Endpoints

**POST /infer** - Run OCR inference

```bash
# Using TrOCR
curl -F "file=@tests/data_samples/receipt_ko.jpg" \
     "http://127.0.0.1:8000/infer?model=trocr&device=auto"

# Using Donut
curl -F "file=@tests/data_samples/invoice_ko.jpg" \
     "http://127.0.0.1:8000/infer?model=donut"

# Using PaddleOCR with Korean
curl -F "file=@tests/data_samples/receipt_ko.jpg" \
     "http://127.0.0.1:8000/infer?model=paddleocr&lang=korean"
```

**GET /** - API information

```bash
curl http://127.0.0.1:8000/
```

**GET /health** - Health check

```bash
curl http://127.0.0.1:8000/health
```

**GET /models** - List available models

```bash
curl http://127.0.0.1:8000/models
```

### Evaluation

Run comprehensive evaluation on test dataset:

```bash
# Evaluate all models
python -m src.eval --data-dir tests/data_samples --models trocr,donut,pix2struct,paddleocr

# Evaluate specific models only
python -m src.eval --data-dir tests/data_samples --models trocr,donut --device mps

# Save results to custom file
python -m src.eval --data-dir tests/data_samples --output my_results.json
```

#### Evaluation Metrics

The evaluation script computes:

- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy
- **Edit Distance**: Average Levenshtein distance
- **Latency**: Mean, median, min, max inference time (ms)
- **Throughput**: Images processed per second

#### Test Data Format

Place test images and ground truth in the same directory:

```
tests/data_samples/
  invoice_ko.jpg
  invoice_ko.txt    # Ground truth text
  receipt_ko.jpg
  receipt_ko.txt    # Ground truth text
```

See [tests/data_samples/README.md](tests/data_samples/README.md) for more details.

## Device Support

### macOS (MPS)

Apple Silicon Macs (M1/M2/M3) support GPU acceleration via Metal Performance Shaders (MPS):

```python
import torch
print(torch.backends.mps.is_available())  # Should print True
```

Use `--device auto` or `--device mps` to enable MPS acceleration.

### CUDA (NVIDIA GPU)

On Linux/Windows with NVIDIA GPU:

```bash
python -m src.cli --model trocr --image image.jpg --device cuda
```

### CPU

Works on all platforms:

```bash
python -m src.cli --model trocr --image image.jpg --device cpu
```

## Architecture

### Project Structure

```
ocr-baselines/
├── pyproject.toml          # Dependencies and project metadata
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── pipelines/          # OCR model implementations
│   │   ├── __init__.py
│   │   ├── trocr_pipeline.py
│   │   ├── donut_pipeline.py
│   │   ├── pix2struct_pipeline.py
│   │   └── paddleocr_pipeline.py
│   ├── server/             # FastAPI server
│   │   ├── __init__.py
│   │   └── app.py
│   ├── cli.py              # Command-line interface
│   └── eval.py             # Evaluation script
└── tests/
    ├── __init__.py
    ├── test_cli.py
    └── data_samples/       # Test images and ground truth
        ├── README.md
        ├── invoice_ko.txt
        └── receipt_ko.txt
```

### Pipeline Interface

All pipelines implement a common interface:

```python
class OCRPipeline:
    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize the pipeline."""
        pass

    def infer(self, image: PIL.Image.Image) -> Dict[str, Any]:
        """Run OCR inference.

        Returns:
            {
                "text": str,           # Recognized text
                "latency_ms": float,   # Inference time
                "engine": str,         # "pytorch" or "paddleocr"
                "model": str,          # Model identifier
            }
        """
        pass
```

## Logging

The FastAPI server logs all requests to `ocr_requests.jsonl`:

```json
{
  "timestamp": "2024-01-20T14:35:00.000Z",
  "model": "trocr",
  "lang": "korean",
  "device": "mps",
  "latency_ms": 234.56,
  "engine": "pytorch",
  "text_length": 127
}
```

## Troubleshooting

### PaddleOCR Installation Issues

PaddleOCR may have compatibility issues on some macOS/Python combinations:

```bash
# Try specific versions
pip install paddleocr==2.6.0 paddlepaddle==2.5.0

# Or skip PaddleOCR and use transformer models only
python -m src.eval --data-dir tests/data_samples --models trocr,donut,pix2struct
```

### MPS Not Available

MPS requires:
- macOS 12.3 or later
- Apple Silicon (M1/M2/M3) chip

For Intel Macs, use `--device cpu` instead.

### Model Download Issues

Models are downloaded from HuggingFace Hub on first use. If you encounter network issues:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or download manually
from transformers import AutoModel
AutoModel.from_pretrained("microsoft/trocr-base-printed")
```

## Performance Comparison

Example results (will vary based on hardware and data):

| Model | CER | WER | Latency (ms) | Throughput (img/s) |
|-------|-----|-----|--------------|-------------------|
| TrOCR | 0.05 | 0.12 | 250 | 4.0 |
| Donut | 0.08 | 0.15 | 400 | 2.5 |
| Pix2Struct | 0.10 | 0.18 | 350 | 2.9 |
| PaddleOCR | 0.12 | 0.20 | 150 | 6.7 |

*Note: Lower CER/WER is better, higher throughput is better*

## References

### Models

- **TrOCR**: [Paper](https://arxiv.org/abs/2109.10282) | [HuggingFace](https://huggingface.co/microsoft/trocr-base-printed)
- **Donut**: [Paper](https://arxiv.org/abs/2111.15664) | [HuggingFace](https://huggingface.co/naver-clova-ix/donut-base)
- **Pix2Struct**: [Paper](https://arxiv.org/abs/2210.03347) | [HuggingFace](https://huggingface.co/google/pix2struct-base)
- **PaddleOCR**: [GitHub](https://github.com/PaddlePaddle/PaddleOCR) | [Docs](https://paddlepaddle.org.cn/)

### Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - HuggingFace model library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [jiwer](https://github.com/jitsi/jiwer) - WER/CER calculation

## Contributing

Contributions are welcome! Areas for improvement:

- Add more OCR models (EasyOCR, Tesseract, etc.)
- Support for additional languages
- Model fine-tuning scripts
- ONNX export and optimization
- Batch inference support
- Web UI for interactive testing

## License

This project follows the LLMOCR repository license.

## Next Steps

After establishing baselines:

1. **Fine-tuning**: Adapt models to your specific document types
2. **Optimization**: Export to ONNX, quantization, pruning
3. **Deployment**: Containerization, scaling, monitoring
4. **Data Collection**: Build custom datasets for your use case
