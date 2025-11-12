# LLMOCR

**한국어 OCR 데이터 & 벤치마크 자동화 파이프라인**

A comprehensive Korean OCR data processing and benchmarking pipeline with continuous learning automation.

## 🎯 Overview

This project provides a complete infrastructure for:
- **Data Management**: Download, clean, and process Korean OCR datasets (AI-Hub, SynthDoG-ko)
- **Benchmarking**: Automated evaluation with multiple metrics (CER, WER, throughput, latency)
- **Continuous Learning**: Automated retraining and regression checking pipeline
- **GUI & Operations**: Streamlit interface with error analysis, audit logging, and visualization
- **Reproducibility**: Standardized data formats and evaluation protocols

## 📁 Project Structure

```
LLMOCR/
├── datasets/                     # Data management
│   ├── README.md                # Dataset documentation
│   ├── raw/                     # Downloaded raw data
│   ├── processed/               # Cleaned and processed data
│   ├── scripts/                 # Data processing scripts
│   │   ├── download_synthdog_ko.py
│   │   ├── download_aihub.py
│   │   ├── clean_data.py
│   │   ├── create_splits.py
│   │   └── analyze_difficulty.py
│   └── configs/
│       └── aihub_config.yaml
│
├── benchmarks/                   # Evaluation and training
│   ├── run_bench.py             # Main benchmark runner
│   ├── continuous_learning.py   # Continuous learning pipeline
│   └── example_workflow.py      # Example usage
│
├── gui/                          # Streamlit web interface
│   ├── streamlit_app.py         # Main GUI application
│   └── README.md                # GUI documentation
│
├── tools/                        # Operational tools
│   ├── audit_logging/           # Audit logging system
│   ├── error_analysis/          # Error analysis tools
│   └── demo_gui.py              # Demo script
│
├── utils/                        # Utility modules
│   └── visualization/           # Visualization tools
│
├── models/                       # Model storage
│   ├── baseline/                # Base models
│   ├── experiments/             # Experimental models
│   └── production/              # Production models
│
├── reports/                      # Generated reports
│   ├── *.json                   # Raw results
│   ├── *.csv                    # CSV reports
│   └── *.md                     # Markdown reports
│
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd LLMOCR

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Quick Demo

**Option A: Data Pipeline Demo**

Test the pipeline with a small dataset (200 samples):

```bash
python benchmarks/example_workflow.py --quick_demo
```

This will:
1. Download 200 samples from SynthDoG-ko
2. Clean and process the data
3. Create train/val/test splits
4. Run benchmark evaluation
5. Generate reports

**Option B: GUI & Operations Demo**

Test the GUI and operational features:

```bash
python tools/demo_gui.py
```

This demonstrates:
1. Audit logging system
2. Error analysis tools
3. Bounding box visualization
4. High DPI retry functionality

**Option C: Launch Streamlit GUI**

Start the interactive web interface:

```bash
streamlit run gui/streamlit_app.py
```

Access the GUI at `http://localhost:8501` to:
- Upload and process images
- Visualize predictions with bounding boxes
- Analyze errors interactively
- View audit logs
- Batch process multiple images

### 3. Download Full Datasets

#### SynthDoG-ko (Synthetic Korean Documents)

```bash
# Download 10,000 samples
python datasets/scripts/download_synthdog_ko.py \
    --output_dir datasets/raw/synthdog_ko \
    --limit 10000
```

#### AI-Hub (Korean Administrative Documents)

1. Register at [AI-Hub](https://aihub.or.kr)
2. Configure credentials in `datasets/configs/aihub_config.yaml`
3. Download dataset:

```bash
python datasets/scripts/download_aihub.py \
    --dataset admin_docs \
    --output_dir datasets/raw/aihub_admin
```

### 4. Process Data

```bash
# Clean data
python datasets/scripts/clean_data.py \
    --source datasets/raw/synthdog_ko \
    --output datasets/processed/synthdog_ko_small \
    --copy_images

# Create splits
python datasets/scripts/create_splits.py \
    --input datasets/processed/synthdog_ko_small

# Analyze dataset
python datasets/scripts/analyze_difficulty.py \
    --input datasets/processed/synthdog_ko_small/train.jsonl \
    --output_dir datasets/processed/synthdog_ko_small
```

### 5. Run Benchmarks

```bash
python benchmarks/run_bench.py \
    --models models/baseline,models/experiment1 \
    --datasets synthdog_ko_small,ko_receipts \
    --output_dir reports
```

## 📊 Benchmark Metrics

The benchmark suite evaluates models on:

- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy
- **Throughput**: Images processed per second
- **Latency Percentiles**: p50, p95, p99 (milliseconds)

## 🔄 Continuous Learning Pipeline

Automate the complete training and evaluation workflow:

```bash
python benchmarks/continuous_learning.py \
    --base_model models/baseline \
    --new_data datasets/raw/new_receipts \
    --dataset_name ko_receipts_v2 \
    --benchmark_datasets synthdog_ko_small,ko_receipts \
    --epochs 3 \
    --auto_promote
```

The pipeline:
1. ✅ Cleans and processes new data
2. ✅ Trains LoRA model on new data
3. ✅ Evaluates on benchmark sets
4. ✅ Checks for regressions vs baseline
5. ✅ Optionally promotes model if improved

## 📚 Available Datasets

### SynthDoG-ko
- **Source**: [Hugging Face](https://huggingface.co/datasets/naver-clova-ix/synthdog-ko)
- **Size**: ~100K synthetic Korean document images
- **License**: MIT
- **Use Case**: Pre-training, data augmentation

### AI-Hub Datasets

#### 공공행정문서 OCR
- **Source**: [AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=88)
- **License**: AI-Hub Terms (research use)
- **Use Case**: Fine-tuning on real documents

#### 한국어 글자체 이미지
- **Source**: [AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=89)
- **License**: AI-Hub Terms (research use)
- **Use Case**: Character-level recognition

## 🎨 Data Format

All datasets use standardized JSONL format:

```json
{
  "image_path": "images/sample001.jpg",
  "text": "인식할 텍스트 내용",
  "split": "train",
  "source": "synthdog_ko",
  "metadata": {
    "difficulty": "medium",
    "ko_ratio": 0.95,
    "en_ratio": 0.03,
    "length": 45
  }
}
```

## 🔧 Configuration

### Data Cleaning Parameters

Edit in `datasets/scripts/clean_data.py`:
- `min_length`: Minimum text length (default: 5)
- `max_length`: Maximum text length (default: 1000)
- `min_dimension`: Minimum image dimension (default: 32px)
- `blur_threshold`: Blur detection threshold (default: 100.0)

### Benchmark Parameters

Edit in `benchmarks/run_bench.py`:
- Model loading implementation
- Inference parameters
- Batch size
- Device selection (CPU/GPU)

### Continuous Learning

Edit in `benchmarks/continuous_learning.py`:
- LoRA configuration (rank, alpha, dropout)
- Training hyperparameters
- Regression threshold (default: 2% CER delta)

## 📈 Example Results

After running benchmarks, reports are generated in `reports/`:

**CSV Format** (`benchmark_results_YYYYMMDD_HHMMSS.csv`):
```csv
model_name,dataset_name,cer,wer,throughput,p95_latency
baseline,synthdog_ko_small,0.0234,0.0456,12.3,145.2
experiment1,synthdog_ko_small,0.0189,0.0398,11.8,152.1
```

**Markdown Format** (`benchmark_report_YYYYMMDD_HHMMSS.md`):
```markdown
# OCR Benchmark Report

## synthdog_ko_small

| Model | CER | WER | Throughput (img/s) | p95 Latency (ms) |
|-------|-----|-----|-------------------|------------------|
| baseline | 0.0234 | 0.0456 | 12.3 | 145.2 |
| experiment1 | 0.0189 | 0.0398 | 11.8 | 152.1 |
```

## 🛠️ Model Integration

To use your own models, implement the model loading in `benchmarks/run_bench.py`:

### Example: TrOCR

```python
def load_model(self):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    self.processor = TrOCRProcessor.from_pretrained(self.model_path)
    self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
    self.model.to(self.device)
    self.model.eval()

def predict(self, image_path: Path) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    pixel_values = self.processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(self.device)

    generated_ids = self.model.generate(pixel_values)
    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text
```

### Example: EasyOCR

```python
def load_model(self):
    import easyocr
    self.model = easyocr.Reader(['ko', 'en'], gpu=(self.device=='cuda'))

def predict(self, image_path: Path) -> str:
    results = self.model.readtext(str(image_path), detail=0)
    return ' '.join(results)
```

## 🖥️ GUI & Operational Features

### Streamlit Web Interface

Launch the interactive GUI:

```bash
streamlit run gui/streamlit_app.py
```

**Key Features:**

1. **📝 Single Image Processing**
   - Upload and process images in real-time
   - Visualize bounding boxes with confidence scores
   - **High DPI Retry**: Automatically retry with 2x DPI when confidence < 80%
   - Adjustable preprocessing (DPI scale, denoise, sharpen)
   - Color-coded confidence (Green: >90%, Yellow: 70-90%, Red: <70%)

2. **📊 Error Analysis Dashboard**
   - Upload benchmark results
   - View top N samples with highest CER/WER
   - Character-level diff visualization (color-coded)
   - Error pattern analysis (substitutions, insertions, deletions)
   - Identify systematic issues for model improvement

3. **📋 Audit Log Viewer**
   - Query historical OCR operations
   - Filter by date, model, engine, CER threshold
   - Performance statistics and trends
   - Export reports (Markdown, CSV)
   - Full traceability of model versions and parameters

4. **⚡ Batch Processing**
   - Process multiple images concurrently
   - Progress tracking
   - Export results as CSV
   - Configurable preprocessing per batch

See [gui/README.md](gui/README.md) for detailed documentation.

### Audit Logging

Track all OCR operations with full audit trail:

```python
from tools.audit_logging import AuditLogger, ModelInfo, PreprocessingParams, EngineType

# Initialize logger
logger = AuditLogger(log_dir="logs/audit")

# Model and preprocessing info
model_info = ModelInfo(
    model_name="trocr-korean",
    model_version="v1.0.0",
    model_path="models/trocr",
    adapter_name="receipts_lora",  # Optional
    engine=EngineType.PYTORCH,
)

preprocess_params = PreprocessingParams(
    dpi_scale=1.5,
    denoise=True,
    sharpen=False,
)

# Log inference
logger.log_inference(
    image_path=Path("input.jpg"),
    model_info=model_info,
    preprocessing_params=preprocess_params,
    prediction="예측 결과",
    confidence=0.95,
    inference_time_ms=123.45,
    ground_truth="정답",  # Optional
    cer=0.05,  # Optional
    wer=0.10,  # Optional
)

# Query logs
entries = logger.query_logs(
    start_date="2025-01-01",
    min_cer=0.1,  # Only errors > 10%
)

# Export report
logger.export_report(Path("audit_report.md"), entries)
```

**Logged Information:**
- Input hash (SHA256)
- Model name, version, adapter
- Engine type (PyTorch/ONNX/OpenVINO/TensorRT)
- Preprocessing parameters
- Prediction, confidence, metrics
- Timestamps and performance

### Error Analysis

Analyze and visualize OCR errors:

```python
from tools.error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer(output_dir="reports/errors")

# Find top errors
error_samples = analyzer.find_top_errors(
    results,
    n=20,
    metric='cer'
)

# Generate report with visualizations
analyzer.generate_error_report(
    error_samples,
    dataset_path=Path("datasets/processed/test"),
    include_thumbnails=True,
)

# Analyze patterns
patterns = analyzer.analyze_error_patterns(error_samples)
# Returns: substitution patterns, insertion/deletion frequencies
```

### Bounding Box Visualization

Visualize predictions with confidence-based highlighting:

```python
from utils.visualization import BBoxVisualizer

visualizer = BBoxVisualizer(low_confidence_threshold=0.7)

# Draw bounding boxes
annotated = visualizer.draw_bboxes(
    image,
    bboxes=[
        {'box': [x1, y1, x2, y2], 'text': '한글', 'confidence': 0.95},
        {'box': [x3, y3, x4, y4], 'text': 'ABC', 'confidence': 0.65},  # Low conf
    ],
    highlight_low_confidence=True,
)

# Create comparison view
comparison = visualizer.create_comparison_view(
    original=original_img,
    processed=annotated,
    prediction="예측 텍스트",
    ground_truth="정답 텍스트",
    confidence=0.87,
)
```

### High DPI Retry

Automatically improve low-confidence predictions:

```python
def process_with_retry(image, model, threshold=0.8):
    # Initial inference
    result = model.predict(image)

    if result.confidence < threshold:
        # Retry with higher DPI
        high_dpi_image = preprocess_image(image, dpi_scale=2.0)
        result = model.predict(high_dpi_image)

    return result
```

In the GUI, this is automatic with a single button click!

## 🧪 Testing

Run tests with:

```bash
pytest tests/ -v --cov=.
```

## 📖 Documentation

- **Datasets**: See [datasets/README.md](datasets/README.md)
- **GUI**: See [gui/README.md](gui/README.md)
- **Benchmarks**: See inline documentation in `benchmarks/run_bench.py`
- **API Reference**: Generate with `pydoc`

Run the demo:
```bash
python tools/demo_gui.py
```

## 🤝 Contributing

To add a new dataset:
1. Create download script in `datasets/scripts/`
2. Follow standard JSONL format
3. Update `datasets/README.md`
4. Add benchmark configuration

To add a new model:
1. Implement `load_model()` and `predict()` in `benchmarks/run_bench.py`
2. Test with `--limit 10` first
3. Document any special requirements

## 📝 License

[Specify your license here]

## 🙏 Acknowledgments

- **SynthDoG-ko**: [Naver Clova](https://github.com/clovaai/donut)
- **AI-Hub**: [NIA Korea](https://aihub.or.kr)
- **Korean OCR Community**: For valuable feedback and contributions

## 📧 Contact

[Your contact information]

## 🗓️ Roadmap

- [x] Data download and cleaning pipeline
- [x] Benchmark suite with multiple metrics
- [x] Continuous learning automation
- [x] Streamlit GUI with error analysis
- [x] Audit logging system
- [x] Bounding box visualization
- [x] High DPI retry functionality
- [ ] Multi-GPU training support
- [ ] Real-time inference API
- [ ] Pre-trained model zoo
- [ ] Additional Korean datasets integration
- [ ] Docker deployment support

## ⚡ Performance Tips

1. **Data Loading**: Use `--limit` for testing before full runs
2. **GPU Memory**: Reduce batch size if OOM errors occur
3. **Disk Space**: Use symlinks for processed data to save space
4. **Parallel Processing**: Process multiple datasets concurrently

## 🐛 Troubleshooting

### "No module named 'datasets'"
```bash
pip install datasets
```

### "CUDA out of memory"
- Reduce batch size
- Use `--device cpu`
- Process fewer samples with `--limit`

### "AI-Hub download fails"
- Check credentials in `datasets/configs/aihub_config.yaml`
- Verify dataset access approval on AI-Hub website
- Consider manual download and extraction

### "Image not found during benchmark"
- Ensure `--copy_images` was used during cleaning
- Check symlink support on your filesystem
- Verify image paths in JSONL files

## 📚 References

1. Kim et al. (2021). "OCR-free Document Understanding Transformer (Donut)"
2. AI-Hub Korean OCR Dataset Documentation
3. Levenshtein Distance for OCR Evaluation
4. LoRA: Low-Rank Adaptation of Large Language Models
