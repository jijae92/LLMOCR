# OCR Baselines

A comprehensive OCR baseline project with multiple state-of-the-art models for comparing performance, accuracy, and latency.

## Features

- **Multiple Models**: TrOCR, Donut, Pix2Struct, PaddleOCR
- **LoRA Fine-tuning**: Memory-efficient domain adaptation with macOS/MPS support
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
- `--adapter-path`: Path to LoRA adapter for fine-tuned models (optional)
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

## Fine-tuning with LoRA (macOS/MPS Compatible)

Fine-tune OCR models on your domain-specific data using LoRA (Low-Rank Adaptation) for memory-efficient training.

### 🍎 macOS Support

**LoRA training is fully supported on Apple Silicon (M1/M2/M3) with MPS acceleration.**

⚠️  **Important for Mac Users:**
- **LoRA (FP16/FP32)**: ✅ Fully supported on MPS
- **QLoRA (4-bit)**: ❌ NOT supported on macOS (bitsandbytes limitation)
- For QLoRA: Train on cloud GPU, download adapters for local inference

### Quick Start

1. **Prepare your data** (JSONL, CSV, or directory format):
   ```bash
   # Example: data/train.jsonl
   {"image": "path/to/img1.jpg", "text": "ground truth text 1"}
   {"image": "path/to/img2.jpg", "text": "ground truth text 2"}
   ```

2. **Train TrOCR with LoRA** (Mac/MPS):
   ```bash
   python -m src.train.train_trocr_lora \
     --base_model microsoft/trocr-base-printed \
     --train_json data/train.jsonl \
     --val_json data/val.jsonl \
     --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
     --per_device_train_batch_size 2 \
     --lr 2e-4 \
     --epochs 3 \
     --output_dir runs/trocr-lora \
     --device auto
   ```

3. **Use fine-tuned model** for inference:
   ```bash
   python -m src.cli \
     --model trocr \
     --adapter-path runs/trocr-lora \
     --image path/to/document.jpg \
     --device auto \
     --verbose
   ```

### Training Scripts

#### TrOCR LoRA

```bash
python -m src.train.train_trocr_lora \
  --base_model microsoft/trocr-base-printed \
  --train_json data/train.jsonl \
  --val_json data/val.jsonl \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --lr 2e-4 \
  --epochs 3 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 200 \
  --output_dir runs/trocr-lora \
  --device auto
```

#### Donut LoRA

```bash
python -m src.train.train_donut_lora \
  --base_model naver-clova-ix/donut-base \
  --train_json data/train.jsonl \
  --val_json data/val.jsonl \
  --lora_r 16 \
  --lora_alpha 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --lr 2e-4 \
  --epochs 3 \
  --output_dir runs/donut-lora \
  --device auto
```

### LoRA Parameters

| Parameter | Description | Typical Values | Mac Recommendation |
|-----------|-------------|----------------|-------------------|
| `lora_r` | LoRA rank (controls adapter size) | 4-64 | 16 (good balance) |
| `lora_alpha` | Scaling factor | 8-128 | 32 (typically 2×r) |
| `lora_dropout` | Dropout rate | 0.0-0.1 | 0.05 |
| `batch_size` | Samples per batch | 1-8 | 2 for TrOCR, 1 for Donut |
| `gradient_accumulation_steps` | Accumulate gradients | 1-8 | 1-2 |
| `learning_rate` | Learning rate | 1e-5 to 5e-4 | 2e-4 |

### Memory Requirements (macOS Unified Memory)

**TrOCR LoRA:**
- Batch size 2, r=16: ~10-14GB
- Batch size 1, r=16: ~8-10GB
- Batch size 1, r=8: ~6-8GB

**Donut LoRA (larger model):**
- Batch size 1, r=16: ~12-16GB
- Batch size 1, r=8: ~10-12GB

**If Out of Memory:**
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to 2 or 4
3. Reduce `lora_r` to 8 or 4
4. Use `--no_mps` to fall back to CPU (slower)

### Data Preparation

The training scripts support three data formats:

**1. JSONL Format** (Recommended):
```jsonl
{"image": "path/to/img1.jpg", "text": "text 1"}
{"image": "path/to/img2.jpg", "text": "text 2"}
```

**2. CSV Format:**
```csv
image,text
path/to/img1.jpg,"text 1"
path/to/img2.jpg,"text 2"
```

**3. Directory Format:**
```
data/train/
├── img1.jpg
├── img1.txt
├── img2.jpg
├── img2.txt
```

See `data/README.md` for detailed data preparation guide.

### Configuration Files

Pre-configured YAML files are available in `configs/`:

- `trocr_lora_mac.yaml` - TrOCR LoRA for macOS/MPS
- `donut_lora_mac.yaml` - Donut LoRA for macOS/MPS
- `trocr_qlora_cloud.yaml` - TrOCR QLoRA for cloud GPU (CUDA only)

### QLoRA (4-bit) for Cloud GPU

**⚠️  NOT supported on macOS** - Requires CUDA GPU with bitsandbytes

For QLoRA training:

1. **Train on cloud GPU** (A10, A100, H100):
   ```bash
   # On cloud machine with CUDA
   python -m src.train.train_trocr_lora \
     --base_model microsoft/trocr-base-printed \
     --train_json data/train.jsonl \
     --val_json data/val.jsonl \
     --lora_r 64 \
     --lora_alpha 128 \
     --per_device_train_batch_size 8 \
     --device cuda \
     --fp16 \
     --output_dir runs/trocr-qlora
   ```

2. **Download adapter** (only ~tens of MB):
   ```bash
   scp cloud:/path/runs/trocr-qlora ./runs/
   ```

3. **Use locally on Mac**:
   ```bash
   python -m src.cli \
     --model trocr \
     --adapter-path runs/trocr-qlora \
     --image document.jpg
   ```

### Training Tips

#### For Best Results:
- **Data quality**: Clean, accurate ground truth is critical
- **Data quantity**: 1,000+ samples recommended (100 minimum for experiments)
- **Domain matching**: Use images similar to your target use case
- **Evaluation**: Always use a validation set to monitor overfitting

#### Mac-Specific:
- **MPS stability**: If crashes occur, try `--no_mps` to use CPU
- **PyTorch version**: Use PyTorch 2.0+ for best MPS support
- **Monitor memory**: Use Activity Monitor to track memory usage
- **Batch size**: Start small (1-2) and increase if memory allows

#### General:
- **LoRA rank**: Higher r = more parameters (better fit, more memory)
- **Learning rate**: Start with 2e-4, reduce if unstable
- **Epochs**: 3-5 epochs typically sufficient
- **Checkpointing**: Best model saved automatically based on validation CER

### Example Workflow

1. **Collect domain data** (100-1000+ images):
   ```bash
   data/receipts/
   ├── r001.jpg + r001.txt
   ├── r002.jpg + r002.txt
   ...
   ```

2. **Create JSONL dataset**:
   ```python
   import json
   from pathlib import Path

   with open("data/train.jsonl", "w") as f:
       for img in Path("data/receipts").glob("*.jpg"):
           txt = img.with_suffix(".txt")
           if txt.exists():
               text = txt.read_text()
               f.write(json.dumps({"image": str(img), "text": text}) + "\n")
   ```

3. **Train on Mac with MPS**:
   ```bash
   python -m src.train.train_trocr_lora \
     --train_json data/train.jsonl \
     --val_json data/val.jsonl \
     --output_dir runs/receipts-lora \
     --epochs 3 \
     --device auto
   ```

4. **Evaluate improvement**:
   ```bash
   # Baseline
   python -m src.eval --data-dir tests/data_samples --models trocr

   # Fine-tuned
   python -m src.cli \
     --model trocr \
     --adapter-path runs/receipts-lora \
     --image test_receipt.jpg
   ```

5. **Monitor training**:
   - Training metrics: `runs/receipts-lora/train_metrics.json`
   - Validation CER: Logged during training
   - Best checkpoint: Auto-saved based on lowest CER

### Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size and increase gradient accumulation
python -m src.train.train_trocr_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lora_r 8
```

**MPS Crashes:**
```bash
# Fall back to CPU
python -m src.train.train_trocr_lora --no_mps
```

**Slow Training:**
```bash
# Use smaller model or reduce data
# Consider cloud GPU for large-scale training
```

## Project Structure

```
ocr-baselines/
├── pyproject.toml              # Dependencies and project metadata
├── README.md                   # This file
├── configs/                    # Training configuration files
│   ├── trocr_lora_mac.yaml
│   ├── donut_lora_mac.yaml
│   └── trocr_qlora_cloud.yaml
├── data/                       # Training data
│   ├── README.md               # Data preparation guide
│   ├── train.jsonl             # Training samples
│   └── val.jsonl               # Validation samples
├── runs/                       # Training outputs (checkpoints)
│   └── (created during training)
├── src/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── eval.py                 # Evaluation script
│   ├── pipelines/              # Model pipelines (with LoRA support)
│   │   ├── __init__.py
│   │   ├── trocr_pipeline.py
│   │   ├── donut_pipeline.py
│   │   ├── pix2struct_pipeline.py
│   │   └── paddleocr_pipeline.py
│   ├── server/                 # FastAPI server
│   │   ├── __init__.py
│   │   └── app.py
│   └── train/                  # Training scripts (LoRA fine-tuning)
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       ├── train_trocr_lora.py # TrOCR LoRA training
│       └── train_donut_lora.py # Donut LoRA training
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
