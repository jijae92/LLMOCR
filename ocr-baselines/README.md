# OCR Baselines

A comprehensive OCR baseline project with multiple state-of-the-art models for comparing performance, accuracy, and latency.

## Features

- **Multiple Models**: TrOCR, Donut, Pix2Struct, PaddleOCR
- **CLI Interface**: Easy command-line interface for quick inference
- **REST API**: FastAPI-based server for production use
- **Evaluation Framework**: Automated CER/WER and latency benchmarking
- **Device Support**: Auto-detection for CUDA, MPS (Apple Silicon), and CPU
- **Korean Language Support**: Optimized for Korean OCR with multilingual capabilities

## Supported Models

| Model | Description | Backend | Use Case |
|-------|-------------|---------|----------|
| **TrOCR** | Vision Encoder-Decoder (ViT + Transformer) | PyTorch | High-accuracy printed text |
| **Donut** | Swin Transformer + BART decoder | PyTorch | Document understanding |
| **Pix2Struct** | Image-to-text model | PyTorch | Structured documents |
| **PaddleOCR** | Lightweight OCR engine | PaddlePaddle | Fast CPU inference, multilingual |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
cd ocr-baselines
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Generate Sample Test Images

```bash
python tests/generate_sample_images.py
```

## Usage

### CLI Interface

Basic usage:

```bash
python -m src.cli --model trocr --image tests/data_samples/receipt_ko.jpg --device auto
```

With verbose output:

```bash
python -m src.cli --model trocr --image tests/data_samples/invoice_ko.jpg --device auto --verbose
```

Save results to file:

```bash
python -m src.cli --model paddleocr --image tests/data_samples/receipt_ko.jpg --device cpu --output results.json
```

#### CLI Arguments

- `--model`: Model to use (`trocr`, `donut`, `pix2struct`, `paddleocr`)
- `--image`: Path to input image
- `--device`: Device to run on (`auto`, `cpu`, `cuda`, `mps`)
- `--lang`: Language for PaddleOCR (default: `korean`)
- `--output`: Output JSON file (optional)
- `--verbose`: Print detailed metadata

### REST API

Start the server:

```bash
cd ocr-baselines
uvicorn src.server.app:app --port 8000 --reload
```

Or using Python:

```bash
python -m src.server.app
```

#### API Endpoints

##### Health Check

```bash
curl http://127.0.0.1:8000/
```

##### OCR Inference

```bash
curl -F "file=@tests/data_samples/receipt_ko.jpg" "http://127.0.0.1:8000/infer?model=trocr&device=auto"
```

With PaddleOCR (specify language):

```bash
curl -F "file=@tests/data_samples/invoice_ko.jpg" "http://127.0.0.1:8000/infer?model=paddleocr&lang=korean"
```

##### List Available Models

```bash
curl http://127.0.0.1:8000/models
```

#### API Response Format

```json
{
  "text": "Extracted text from image...",
  "metadata": {
    "model": "trocr",
    "engine": "pytorch",
    "device": "mps",
    "latency_ms": 234.5,
    "decode_params": {
      "max_length": 512
    }
  }
}
```

### Evaluation

Run evaluation on all models:

```bash
python -m src.eval --data-dir tests/data_samples --models trocr,donut,pix2struct,paddleocr --device auto
```

Save detailed results:

```bash
python -m src.eval --data-dir tests/data_samples --models trocr,paddleocr --device auto --output evaluation_results.json
```

#### Evaluation Metrics

- **CER (Character Error Rate)**: Percentage of character-level errors
- **WER (Word Error Rate)**: Percentage of word-level errors
- **Latency**: Time taken for inference (milliseconds)
- **Throughput**: Samples processed per second

#### Expected Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

TROCR:
  Average CER:      5.23%
  Average WER:      12.45%
  Average Latency:  234.56 ms
  Min Latency:      198.23 ms
  Max Latency:      287.34 ms
  Throughput:       4.26 samples/sec

PADDLEOCR:
  Average CER:      7.89%
  Average WER:      15.67%
  Average Latency:  123.45 ms
  Min Latency:      98.76 ms
  Max Latency:      156.78 ms
  Throughput:       8.10 samples/sec
================================================================================
```

## Project Structure

```
ocr-baselines/
├── pyproject.toml              # Dependencies and project metadata
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── eval.py                 # Evaluation script
│   ├── pipelines/              # Model pipelines
│   │   ├── __init__.py
│   │   ├── trocr_pipeline.py
│   │   ├── donut_pipeline.py
│   │   ├── pix2struct_pipeline.py
│   │   └── paddleocr_pipeline.py
│   └── server/                 # FastAPI server
│       ├── __init__.py
│       └── app.py
└── tests/
    ├── __init__.py
    ├── test_cli.py             # Basic tests
    ├── generate_sample_images.py  # Generate test images
    └── data_samples/           # Test data
        ├── invoice_ko.txt
        ├── invoice_ko.jpg
        ├── receipt_ko.txt
        └── receipt_ko.jpg
```

## Device Support

### Auto Device Selection

The `auto` device option automatically selects the best available device:

1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon) - if available on macOS
3. **CPU** - fallback

### Specific Device Selection

- `--device cpu`: Force CPU execution
- `--device cuda`: Use NVIDIA GPU
- `--device mps`: Use Apple Silicon GPU (macOS only)

## Adding Custom Test Data

1. Add your image to `tests/data_samples/`
2. Create a corresponding `.txt` file with ground truth text
3. Run evaluation

Example:

```
tests/data_samples/
├── my_document.jpg     # Your image
└── my_document.txt     # Ground truth text
```

## Logging

The REST API automatically logs all inference requests to `ocr_inference.jsonl`:

```json
{
  "timestamp": "2024-03-15T14:30:25.123456",
  "filename": "receipt_ko.jpg",
  "model": "trocr",
  "device": "mps",
  "lang": "korean",
  "latency_ms": 234.56,
  "engine": "pytorch"
}
```

## Performance Tips

### For Faster Inference

1. **Use PaddleOCR** for CPU-only environments
2. **Enable GPU** with `--device cuda` or `--device mps`
3. **Batch processing** via REST API for multiple images

### For Better Accuracy

1. **TrOCR** generally provides highest accuracy for printed text
2. **Donut** works well for structured documents
3. **PaddleOCR** is optimized for Asian languages including Korean

## Troubleshooting

### PaddleOCR Issues on macOS

If PaddleOCR fails on macOS, you can:

1. Use other models (TrOCR, Donut, Pix2Struct)
2. Disable PaddleOCR in evaluation: `--models trocr,donut,pix2struct`

### Out of Memory Errors

1. Reduce batch size (for custom implementations)
2. Use CPU instead of GPU: `--device cpu`
3. Use lighter models like PaddleOCR

### Slow Model Loading

First-time model loading downloads weights from HuggingFace. Subsequent runs use cached models.

## Citation

If you use this codebase, please cite the original model papers:

- **TrOCR**: [TrOCR: Transformer-based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- **Donut**: [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- **Pix2Struct**: [Pix2Struct: Screenshot Parsing as Pretraining](https://arxiv.org/abs/2210.03347)
- **PaddleOCR**: [PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941)

## License

This project is for research and educational purposes. Please refer to individual model licenses for commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and issues, please open a GitHub issue.
