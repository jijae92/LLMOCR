# LLMOCR

**한국어 OCR 데이터 & 벤치마크 자동화 파이프라인**

A comprehensive Korean OCR data processing and benchmarking pipeline with continuous learning automation.

## 🎯 Overview

This project provides a complete infrastructure for:
- **Data Management**: Download, clean, and process Korean OCR datasets (AI-Hub, SynthDoG-ko)
- **Benchmarking**: Automated evaluation with multiple metrics (CER, WER, throughput, latency)
- **Continuous Learning**: Automated retraining and regression checking pipeline
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

## 🧪 Testing

Run tests with:

```bash
pytest tests/ -v --cov=.
```

## 📖 Documentation

- **Datasets**: See [datasets/README.md](datasets/README.md)
- **Benchmarks**: See inline documentation in `benchmarks/run_bench.py`
- **API Reference**: Generate with `pydoc`

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
- [ ] Multi-GPU training support
- [ ] Real-time inference API
- [ ] Web-based visualization dashboard
- [ ] Pre-trained model zoo
- [ ] Additional Korean datasets integration

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
