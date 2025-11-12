# Model Export and Optimization

This directory contains scripts for exporting and optimizing OCR models for production deployment.

## Overview

### Why ONNX?

ONNX (Open Neural Network Exchange) provides:
- **Faster CPU inference**: Optimized kernels for CPU and macOS arm64
- **Model quantization**: INT8 quantization for 2-4x speedup
- **Cross-platform**: Works on Linux, macOS, Windows
- **Production-ready**: Industry-standard format for deployment

### macOS arm64 Support

ONNX Runtime has official support for macOS arm64 (Apple Silicon) with optimized CPU kernels.

## ⚠️  VisionEncoderDecoder Export Limitations

**Important**: TrOCR and Donut use `VisionEncoderDecoder` architecture, which has known challenges with ONNX export:

1. **Dynamic Shapes**: Input dimensions can vary
2. **Autoregressive Generation**: Decoder requires past key values
3. **Cross-Attention**: Complex attention mechanisms

### Current Status

- **Full model export**: May fail with `optimum.exporters.onnx`
- **Encoder-only export**: Works reliably via `torch.onnx`
- **Recommended approach**: Continue using PyTorch inference with optimizations

### Alternatives to ONNX

If ONNX export fails, consider:

1. **PyTorch with torch.compile()** (PyTorch 2.0+):
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

2. **ONNX Runtime Execution Provider** with PyTorch:
   ```python
   # Use ORT EP without exporting to ONNX
   import torch_ort
   model = torch_ort.ORTModule(model)
   ```

3. **TorchScript**:
   ```python
   scripted_model = torch.jit.script(model)
   ```

## Export Scripts

### 1. Export TrOCR to ONNX

```bash
# Try automatic export (may fail)
python export/export_trocr_onnx.py \
  --model microsoft/trocr-base-printed \
  --output-dir models/trocr-onnx

# Export encoder only (fallback)
python export/export_trocr_onnx.py \
  --model microsoft/trocr-base-printed \
  --output-file models/trocr_encoder.onnx \
  --method torch
```

### 2. Quantize ONNX Model

```bash
# Dynamic INT8 quantization (recommended)
python export/quantize_onnx.py \
  --input models/trocr_encoder.onnx \
  --output models/trocr_encoder_int8.onnx \
  --mode dynamic

# Static INT8 quantization (better accuracy)
python export/quantize_onnx.py \
  --input models/trocr_encoder.onnx \
  --output models/trocr_encoder_int8.onnx \
  --mode static \
  --calibration-samples 100
```

## Quantization Modes

### Dynamic Quantization

**Pros:**
- No calibration data needed
- Fast quantization process
- Good for CPU inference
- 2-3x speedup on average

**Cons:**
- Slightly lower accuracy than static

**Use when:** You want quick optimization without collecting calibration data

### Static Quantization

**Pros:**
- Better accuracy preservation
- Optimal INT8 performance
- 3-4x speedup potential

**Cons:**
- Requires calibration data
- Slower quantization process

**Use when:** You have representative calibration data and need maximum accuracy

## Expected Performance Gains

### Model Size Reduction

- **FP32 → INT8**: ~75% size reduction
- Example: 350MB → 90MB

### Inference Speedup (CPU)

| Operation | FP32 Baseline | INT8 Quantized | Speedup |
|-----------|---------------|----------------|---------|
| Encoder | 100ms | 30-40ms | 2.5-3.3x |
| Full model* | 250ms | 100-150ms | 1.7-2.5x |

*If full export succeeds

### macOS arm64 (Apple Silicon)

ONNX Runtime is optimized for Apple Silicon:
- Native ARM64 support
- Accelerate framework integration
- Efficient CPU kernels

## Serving with ONNX

### Start ONNX Server

```bash
# Start server
uvicorn serve.onnx_app:app --host 0.0.0.0 --port 8001 --workers 4

# Test inference
curl -F "file=@tests/data_samples/receipt_ko.jpg" \
  "http://127.0.0.1:8001/infer?model_path=models/trocr_encoder.onnx"
```

### Batch Inference

```bash
curl -F "files=@img1.jpg" -F "files=@img2.jpg" \
  "http://127.0.0.1:8001/batch_infer?model_path=models/trocr_encoder_int8.onnx"
```

## Production Deployment

### Recommended Setup

1. **Export encoder** (most reliable):
   ```bash
   python export/export_trocr_onnx.py --method torch
   ```

2. **Quantize to INT8**:
   ```bash
   python export/quantize_onnx.py --mode dynamic
   ```

3. **Deploy with workers**:
   ```bash
   uvicorn serve.onnx_app:app --workers 4
   ```

### OR: Use PyTorch with Optimizations

If ONNX export is problematic, use PyTorch with:

```python
import torch

# Enable inference mode
model.eval()

# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# Use torch.no_grad() context
with torch.no_grad():
    outputs = model(**inputs)
```

## Troubleshooting

### ONNX Export Fails

```
Error: ONNX export failed for VisionEncoderDecoder
```

**Solutions:**
1. Use encoder-only export: `--method torch`
2. Use hybrid ONNX/PyTorch pipeline (encoder ONNX + decoder PyTorch)
3. Continue with PyTorch inference

### Quantization Fails

```
Error: Quantization failed
```

**Solutions:**
1. Verify input model is valid ONNX
2. Try different quantization mode (`dynamic` vs `static`)
3. Check onnxruntime version compatibility

### ONNX Runtime Not Found

```
ImportError: onnxruntime is not installed
```

**Solution:**
```bash
pip install onnxruntime>=1.15.0
```

For macOS arm64, ensure you have the native wheel:
```bash
pip install onnxruntime --force-reinstall
```

## Why Not TensorRT-LLM or vLLM?

**TensorRT-LLM** and **vLLM** are optimized for:
- Decoder-only LLMs (GPT, LLaMA, etc.)
- Text generation workloads
- NVIDIA GPUs (CUDA required)

**Not suitable for:**
- Encoder-decoder models (TrOCR, Donut)
- Vision tasks
- macOS (no CUDA support)

For OCR with encoder-decoder architectures, stick with:
1. ONNX Runtime (CPU/arm64 optimized)
2. PyTorch with optimizations
3. Optimum (HuggingFace optimization toolkit)

## OpenVINO (Alternative for Intel CPUs)

For Intel CPUs, consider OpenVINO:

```bash
# Install OpenVINO
pip install openvino openvino-dev

# Convert to OpenVINO IR
mo --saved_model_dir models/trocr

# Inference with OpenVINO
from openvino.runtime import Core
ie = Core()
model = ie.read_model("models/trocr.xml")
```

**Note**: OpenVINO is primarily for Intel CPUs. For macOS arm64, use ONNX Runtime.

## Summary

| Approach | macOS arm64 | CPU Speedup | Ease of Use | Recommended |
|----------|-------------|-------------|-------------|-------------|
| PyTorch (as-is) | ✅ | 1x | ⭐⭐⭐ | ✅ Default |
| PyTorch + torch.compile | ✅ | 1.5-2x | ⭐⭐ | ✅ PyTorch 2.0+ |
| ONNX (encoder) | ✅ | 2-3x | ⭐⭐ | ✅ If export works |
| ONNX + INT8 | ✅ | 3-4x | ⭐ | ⚠️  Test accuracy |
| TensorRT-LLM | ❌ | N/A | N/A | ❌ Not suitable |
| vLLM | ❌ | N/A | N/A | ❌ Not suitable |
| OpenVINO | ❌ | N/A | ⭐ | ❌ Intel only |

**Recommendation for macOS arm64:**
1. Start with PyTorch + MPS
2. Try torch.compile() (PyTorch 2.0+)
3. If ONNX export succeeds, use ONNX + INT8

For production on Intel CPUs: ONNX + INT8 quantization
